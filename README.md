# Rogue Access Detection

Detect rogue Wi‑Fi access points (evil twins, spoofed BSSIDs) and wireless attacks (deauth floods) using 802.11 frame heuristics with Scapy.

## Features
- Live monitoring on a wireless interface (requires root)
- Offline analysis of pcap files
- Heuristics:
  - Deauthentication flood detection (rate-based)
  - Beacon anomalies (suspicious/malformed beacons)
  - BSSID spoofing (same BSSID advertising different SSIDs/channels)
  - Possible evil twin cluster (many BSSIDs advertising the same SSID)
- Whitelisting of trusted SSIDs/BSSIDs
- CLI with rich, colorized output

## Quick start
### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Alternatively, using requirements:
```bash
pip install -r requirements.txt
```

### 2) Live monitoring (root required for sniffing)
```bash
sudo rogue-ap-detect monitor --iface wlan0
```
Options:
- `--deauth-threshold` (default: 30 per 10s window)
- `--cluster-threshold` (default: 6 BSSIDs per SSID)
- `--window-seconds` (default: 10)
- `--whitelist path.json` (JSON with keys: ssids[], bssids[])

### 3) Analyze an existing pcap
```bash
rogue-ap-detect analyze --pcap path/to/capture.pcap
```

## How it works (heuristics)
- Deauth flood: If number of Dot11Deauth frames within the rolling time window exceeds threshold.
- BSSID spoofing: A single BSSID advertises different SSIDs or channels over time.
- Evil twin cluster: Excessive number of distinct BSSIDs advertising the same SSID (may indicate spoofed APs). Note: large venues can legitimately have many APs per SSID.
- Beacon anomalies: Empty SSID, malformed info elements, or unusual capability combinations.

RSSI: If Radiotap headers are present, RSSI (dBm_AntSignal) is recorded when available.

## Limitations
- Requires monitor mode-capable interface and appropriate OS/driver support for live sniffing.
- RSSI extraction depends on Radiotap metadata in captures.
- Heuristics can produce false positives in dense enterprise environments; tune thresholds and use whitelists.

## Whitelist format (JSON)
```json
{
  "ssids": ["HomeWiFi", "CorpWiFi"],
  "bssids": ["00:11:22:33:44:55", "aa:bb:cc:dd:ee:ff"]
}
```

## Development
- Run unit tests:
```bash
pytest -q
```
- Lint/format as desired.

## License
This repository retains the existing LICENSE file.
