#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-wlan0mon}"
DEAUTH="${2:-30}"
CLUSTER="${3:-6}"
WINDOW="${4:-10}"
WHITELIST="${5:-}"

if [[ -n "${WHITELIST}" ]]; then
  rogue-ap-detect monitor --iface "${IFACE}" --deauth-threshold "${DEAUTH}" --cluster-threshold "${CLUSTER}" --window-seconds "${WINDOW}" --whitelist "${WHITELIST}"
else
  rogue-ap-detect monitor --iface "${IFACE}" --deauth-threshold "${DEAUTH}" --cluster-threshold "${CLUSTER}" --window-seconds "${WINDOW}"
fi
