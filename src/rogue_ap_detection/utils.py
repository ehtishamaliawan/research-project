from __future__ import annotations
import json
from typing import Any, Dict, Optional, Set

def load_whitelist(path: Optional[str]) -> Dict[str, Set[str]]:
    ssids: Set[str] = set()
    bssids: Set[str] = set()
    if not path:
        return {"ssids": ssids, "bssids": bssids}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        for s in data.get("ssids", []):
            if isinstance(s, str) and s:
                ssids.add(s)
        for b in data.get("bssids", []):
            if isinstance(b, str) and b:
                bssids.add(b.lower())
    except Exception:
        # If whitelist fails to load, default to empty
        pass
    return {"ssids": ssids, "bssids": bssids}