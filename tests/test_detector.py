from __future__ import annotations

from typing import List

import types

from rogue_ap_detection.detector import Detector, DetectionEvent

try:
    from scapy.all import RadioTap, Dot11, Dot11Deauth  # type: ignore
except Exception:
    RadioTap = Dot11 = Dot11Deauth = None  # type: ignore


def fake_clock(start=0.0, step=0.1):
    t = start
    def _now():
        nonlocal t
        t += step
        return t
    return _now


def make_deauth(addr2="00:11:22:33:44:55", addr3="66:77:88:99:aa:bb"):
    if RadioTap is None:
        return types.SimpleNamespace(haslayer=lambda _: True)  # minimal stub
    return RadioTap()/Dot11(type=0, subtype=12, addr1="ff:ff:ff:ff:ff:ff", addr2=addr2, addr3=addr3)/Dot11Deauth(reason=7)


def test_deauth_flood_detection():
    clock = fake_clock(start=0.0, step=0.1)
    det = Detector(deauth_threshold=5, window_seconds=1, time_provider=clock)
    events: List[DetectionEvent] = []
    for _ in range(6):
        evs = det.process_packet(make_deauth())
        events.extend(evs)
    kinds = [e.kind for e in events]
    assert "deauth_flood" in kinds
