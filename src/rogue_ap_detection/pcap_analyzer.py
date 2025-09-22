from __future__ import annotations

from typing import Iterable, List

try:
    from scapy.all import rdpcap  # type: ignore
except Exception:  # pragma: no cover
    rdpcap = None  # type: ignore

def iter_pcap(path: str) -> Iterable:
    if rdpcap is None:
        raise RuntimeError("Scapy is not available. Install dependencies to analyze pcap.")
    for pkt in rdpcap(path):
        yield pkt

def analyze_pcap(path: str, detector) -> List:
    events = []
    for pkt in iter_pcap(path):
        events.extend(detector.process_packet(pkt))
    return events