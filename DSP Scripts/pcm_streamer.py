"""Raw PCM audio streamer over UDP for feeding an ASR host.

Sends decoded PCM audio as UDP datagrams with a minimal custom header
for lowest latency delivery to the ASR model running on a separate host.

Protocol:
  - Each UDP datagram contains a 12-byte header + PCM payload
  - Header: sequence_number(4B) + timestamp_us(4B) + sample_count(2B) + channels(1B) + bits_per_sample(1B)
  - PCM payload: raw little-endian samples (s16le or f32le)
  - Max datagram size ~1400 bytes to stay within typical MTU
"""

import socket
import struct
import time
import numpy as np


# Header format: uint32 seq, uint32 timestamp_us, uint16 sample_count,
#                uint8 channels, uint8 bits_per_sample
HEADER_FORMAT = '<IIHBBx'  # x = 1 byte padding for alignment (total 12 bytes)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Max UDP payload to stay within typical 1500-byte Ethernet MTU
MAX_UDP_PAYLOAD = 1400
MAX_PCM_PAYLOAD = MAX_UDP_PAYLOAD - HEADER_SIZE


class PCMStreamer:
    """Stream raw PCM audio over UDP to an ASR host.

    Parameters
    ----------
    dest_ip : str
        Destination IP address (ASR host). Default '10.0.0.2'.
    dest_port : int
        Destination UDP port. Default 5000.
    sample_rate : int
        Audio sample rate. Default 48000 Hz.
    channels : int
        Number of audio channels. Default 1 (mono for ASR).
    sample_format : str
        's16' for 16-bit signed int, 'f32' for 32-bit float. Default 's16'.
    multicast : bool
        If True, use multicast (dest_ip should be 239.x.x.x). Default False.
    ttl : int
        Multicast TTL. Default 2.
    """

    def __init__(
        self,
        dest_ip: str = '10.0.0.2',
        dest_port: int = 5000,
        sample_rate: int = 48000,
        channels: int = 1,
        sample_format: str = 's16',
        multicast: bool = False,
        ttl: int = 2,
    ):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_format = sample_format
        self.multicast = multicast

        self.bits_per_sample = 16 if sample_format == 's16' else 32
        self.bytes_per_sample = self.bits_per_sample // 8
        self.frame_size = self.bytes_per_sample * channels

        # Max samples per datagram
        self.max_samples_per_packet = MAX_PCM_PAYLOAD // self.frame_size

        # State
        self._seq = 0
        self._sock = None
        self._start_time = None

        # Stats
        self.packets_sent = 0
        self.bytes_sent = 0

    def open(self):
        """Open the UDP socket."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if self.multicast:
            self._sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                struct.pack('b', 2),
            )

        self._start_time = time.monotonic()

    def send(self, pcm_samples: np.ndarray):
        """Send PCM samples as UDP datagrams.

        Parameters
        ----------
        pcm_samples : np.ndarray
            PCM samples. For mono: shape (N,). For stereo: shape (N, 2).
            dtype should match sample_format (int16 or float32).
        """
        if self._sock is None:
            self.open()

        # Flatten to bytes
        if self.sample_format == 's16':
            raw = pcm_samples.astype(np.int16).tobytes()
        else:
            raw = pcm_samples.astype(np.float32).tobytes()

        # Fragment into MTU-safe datagrams
        offset = 0
        while offset < len(raw):
            chunk_size = min(
                self.max_samples_per_packet * self.frame_size,
                len(raw) - offset,
            )
            chunk = raw[offset:offset + chunk_size]
            sample_count = chunk_size // self.frame_size

            # Build header
            elapsed_us = int((time.monotonic() - self._start_time) * 1e6) & 0xFFFFFFFF
            header = struct.pack(
                HEADER_FORMAT,
                self._seq & 0xFFFFFFFF,
                elapsed_us,
                sample_count,
                self.channels,
                self.bits_per_sample,
            )

            datagram = header + chunk
            self._sock.sendto(datagram, (self.dest_ip, self.dest_port))

            self._seq += 1
            self.packets_sent += 1
            self.bytes_sent += len(datagram)
            offset += chunk_size

    def close(self):
        """Close the UDP socket."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class PCMReceiver:
    """Receive raw PCM audio over UDP (runs on the ASR host side).

    Parameters
    ----------
    listen_ip : str
        IP to bind to. Default '0.0.0.0' (all interfaces).
    listen_port : int
        UDP port to listen on. Default 5000.
    multicast_group : str, optional
        Multicast group to join. None for unicast.
    """

    def __init__(
        self,
        listen_ip: str = '0.0.0.0',
        listen_port: int = 5000,
        multicast_group: str = None,
    ):
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.multicast_group = multicast_group

        self._sock = None
        self._expected_seq = 0
        self.packets_received = 0
        self.packets_dropped = 0

    def open(self):
        """Open the UDP receive socket."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.listen_ip, self.listen_port))

        if self.multicast_group:
            mreq = struct.pack(
                '4s4s',
                socket.inet_aton(self.multicast_group),
                socket.inet_aton(self.listen_ip),
            )
            self._sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq,
            )

    def receive(self, timeout: float = 1.0) -> tuple:
        """Receive one datagram of PCM audio.

        Parameters
        ----------
        timeout : float
            Socket timeout in seconds. Default 1.0.

        Returns
        -------
        samples : np.ndarray or None
            PCM samples, or None on timeout.
        metadata : dict
            Header metadata (seq, timestamp_us, sample_count, etc.).
        """
        self._sock.settimeout(timeout)
        try:
            data, addr = self._sock.recvfrom(65536)
        except socket.timeout:
            return None, {}

        if len(data) < HEADER_SIZE:
            return None, {}

        # Parse header
        seq, timestamp_us, sample_count, channels, bps = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )

        # Check for dropped packets
        if seq != self._expected_seq:
            gap = (seq - self._expected_seq) & 0xFFFFFFFF
            if gap < 0x7FFFFFFF:  # forward gap
                self.packets_dropped += gap

        self._expected_seq = (seq + 1) & 0xFFFFFFFF
        self.packets_received += 1

        # Parse PCM payload
        payload = data[HEADER_SIZE:]
        if bps == 16:
            samples = np.frombuffer(payload, dtype=np.int16)
        elif bps == 32:
            samples = np.frombuffer(payload, dtype=np.float32)
        else:
            return None, {}

        if channels > 1:
            samples = samples.reshape(-1, channels)

        metadata = {
            'seq': seq,
            'timestamp_us': timestamp_us,
            'sample_count': sample_count,
            'channels': channels,
            'bits_per_sample': bps,
            'source': addr,
        }

        return samples, metadata

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
