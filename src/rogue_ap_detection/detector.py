from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

try:
    # scapy is optional until runtime
    from scapy.all import Dot11, Dot11Beacon, Dot11Deauth, Dot11Elt, RadioTap  # type: ignore
except Exception:  # pragma: no cover - during tests without scapy installed
    Dot11 = Dot11Beacon = Dot11Deauth = Dot11Elt = RadioTap = object  # type: ignore

@dataclass
class DetectionEvent:
    kind: str
    message: str
    severity: str  # "info" | "warning" | "high"
    timestamp: float
    data: Dict[str, Any]


class Detector:
    def __init__(
        self,
        deauth_threshold: int = 30,
        window_seconds: int = 10,
        cluster_threshold: int = 6,
        whitelist_ssids: Optional[Set[str]] = None,
        whitelist_bssids: Optional[Set[str]] = None,
        time_provider=time.time,
    ) -> None:
        self.deauth_threshold = deauth_threshold
        self.window_seconds = window_seconds
        self.cluster_threshold = cluster_threshold
        self.whitelist_ssids = whitelist_ssids or set()
        self.whitelist_bssids = {b.lower() for b in (whitelist_bssids or set())}
        self.now = time_provider

        # Rolling window for deauth flood
        self._deauth_times: Deque[float] = deque()

        # BSSID -> observed SSIDs, channels
        self._bssid_to_ssids: Dict[str, Set[str]] = defaultdict(set)
        self._bssid_to_channels: Dict[str, Set[int]] = defaultdict(set)

        # SSID -> set of BSSIDs
        self._ssid_to_bssids: Dict[str, Set[str]] = defaultdict(set)

        # Track last event keys to reduce spam
        self._last_event_keys: Dict[Tuple[str, str], float] = {}

    def _prune_deauth(self, t: float) -> None:
        w = self.window_seconds
        while self._deauth_times and (t - self._deauth_times[0] > w):
            self._deauth_times.popleft()

    @staticmethod
    def _parse_beacon(pkt) -> Tuple[Optional[str], Optional[int]]:
        ssid: Optional[str] = None
        channel: Optional[int] = None
        try:
            elt = pkt.getlayer(Dot11Elt)
            while elt is not None:
                if getattr(elt, "ID", None) == 0:  # SSID
                    raw = getattr(elt, "info", b"")
                    try:
                        ssid = raw.decode(errors="ignore")
                    except Exception:
                        ssid = None
                if getattr(elt, "ID", None) == 3:  # DS Parameter set (channel)
                    if isinstance(getattr(elt, "info", b""), (bytes, bytearray)) and len(elt.info) >= 1:
                        channel = int(elt.info[0])
                elt = elt.payload if isinstance(getattr(elt, "payload", None), Dot11Elt) else elt.getlayer(Dot11Elt, 1)
        except Exception:
            pass
        return ssid, channel

    def process_packet(self, pkt) -> List[DetectionEvent]:
        events: List[DetectionEvent] = []
        t = self.now()

        try:
            dot11 = pkt.getlayer(Dot11)
        except Exception:
            dot11 = None

        if dot11 is None:
            return events

        # BSSID is typically addr3 for beacon and deauth management frames
        bssid = getattr(dot11, "addr3", None)
        if isinstance(bssid, str):
            bssid = bssid.lower()

        # Deauthentication flood detection
        try:
            if pkt.haslayer(Dot11Deauth):
                self._deauth_times.append(t)
                self._prune_deauth(t)
                count = len(self._deauth_times)
                if count >= self.deauth_threshold:
                    ev = DetectionEvent(
                        kind="deauth_flood",
                        message=f"Deauth flood detected: {count} frames in last {self.window_seconds}s",
                        severity="high",
                        timestamp=t,
                        data={"count": count, "window_s": self.window_seconds},
                    )
                    if self._dedupe(ev, key=("deauth", "window")):
                        events.append(ev)
        except Exception:
            pass

        # Beacon analysis
        try:
            if pkt.haslayer(Dot11Beacon):
                ssid, channel = self._parse_beacon(pkt)
                if bssid:
                    if ssid:
                        self._bssid_to_ssids[bssid].add(ssid)
                        self._ssid_to_bssids[ssid].add(bssid)
                    if isinstance(channel, int):
                        self._bssid_to_channels[bssid].add(channel)

                    # BSSID spoofing: one BSSID advertising multiple SSIDs/channels
                    if len(self._bssid_to_ssids[bssid]) > 1 and bssid not in self.whitelist_bssids:
                        ev = DetectionEvent(
                            kind="bssid_spoofing",
                            message=f"BSSID {bssid} advertises multiple SSIDs: {sorted(self._bssid_to_ssids[bssid])}",
                            severity="high",
                            timestamp=t,
                            data={"bssid": bssid, "ssids": sorted(self._bssid_to_ssids[bssid])},
                        )
                        if self._dedupe(ev, key=("bssid_spoof", bssid)):
                            events.append(ev)

                    if len(self._bssid_to_channels[bssid]) > 1 and bssid not in self.whitelist_bssids:
                        ev = DetectionEvent(
                            kind="bssid_channel_anomaly",
                            message=f"BSSID {bssid} seen on multiple channels: {sorted(self._bssid_to_channels[bssid])}",
                            severity="warning",
                            timestamp=t,
                            data={"bssid": bssid, "channels": sorted(self._bssid_to_channels[bssid])},
                        )
                        if self._dedupe(ev, key=("bssid_chan", bssid)):
                            events.append(ev)

                # Empty or suspicious SSID
                if ssid is None or ssid == "":
                    ev = DetectionEvent(
                        kind="beacon_anomaly",
                        message="Beacon with empty or undecodable SSID",
                        severity="warning",
                        timestamp=t,
                        data={"bssid": bssid or ""},
                    )
                    if self._dedupe(ev, key=("empty_ssid", bssid or "")):
                        events.append(ev)

                # Evil twin cluster: many BSSIDs advertising the same SSID
                if ssid and ssid not in self.whitelist_ssids:
                    bssids = self._ssid_to_bssids.get(ssid, set())
                    if len(bssids) >= self.cluster_threshold:
                        ev = DetectionEvent(
                            kind="evil_twin_cluster",
                            message=f"SSID '{ssid}' advertised by {len(bssids)} BSSIDs",
                            severity="warning",
                            timestamp=t,
                            data={"ssid": ssid, "bssids": sorted(bssids)},
                        )
                        if self._dedupe(ev, key=("cluster", ssid)):
                            events.append(ev)
        except Exception:
            pass

        return events

    def _dedupe(self, ev: DetectionEvent, key: Tuple[str, str], cooldown: float = 15.0) -> bool:
        last = self._last_event_keys.get(key, 0.0)
        now = ev.timestamp
        if now - last < cooldown:
            return False
        self._last_event_keys[key] = now
        return True

    def stats(self) -> Dict[str, Any]:
        return {
            "deauth_window": list(self._deauth_times),
            "bssid_to_ssids": {k: sorted(v) for k, v in self._bssid_to_ssids.items()},
            "bssid_to_channels": {k: sorted(v) for k, v in self._bssid_to_channels.items()},
            "ssid_to_bssids": {k: sorted(v) for k, v in self._ssid_to_bssids.items()},
        }