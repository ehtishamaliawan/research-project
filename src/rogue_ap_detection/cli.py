from __future__ import annotations

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .detector import Detector
from .pcap_analyzer import analyze_pcap
from .utils import load_whitelist

console = Console()

@click.group()
def main():
    """Rogue Wi‑Fi Access Point Detection CLI."""
    pass

@main.command("monitor")
@click.option("--iface", required=True, help="Wireless interface in monitor mode (e.g., wlan0mon)")
@click.option("--deauth-threshold", type=int, default=30, show_default=True)
@click.option("--cluster-threshold", type=int, default=6, show_default=True)
@click.option("--window-seconds", type=int, default=10, show_default=True)
@click.option("--whitelist", type=click.Path(exists=True), default=None, help="JSON with ssids[] and bssids[]")
def monitor_cmd(
    iface: str,
    deauth_threshold: int,
    cluster_threshold: int,
    window_seconds: int,
    whitelist: Optional[str],
):
    """Sniff live 802.11 traffic and detect anomalies."""
    try:
        from scapy.all import sniff  # type: ignore
    except Exception as e:
        console.print(f"[red]Scapy is required for live sniffing: {e}[/red]")
        sys.exit(1)

    wl = load_whitelist(whitelist)
    detector = Detector(
        deauth_threshold=deauth_threshold,
        window_seconds=window_seconds,
        cluster_threshold=cluster_threshold,
        whitelist_ssids=wl["ssids"],
        whitelist_bssids=wl["bssids"],
    )

    console.print(f"[bold]Monitoring[/bold] on [cyan]{iface}[/cyan]...")
    console.print(
        f"Deauth threshold: {deauth_threshold}/{window_seconds}s, Cluster threshold: {cluster_threshold}, "
        f"Whitelist SSIDs: {len(wl['ssids'])}, BSSIDs: {len(wl['bssids'])}"
    )

    def on_pkt(pkt):
        events = detector.process_packet(pkt)
        for ev in events:
            console.print(f"[yellow]{ev.severity.upper()}[/yellow] {ev.kind}: {ev.message}")

    try:
        sniff(iface=iface, prn=on_pkt, store=False, monitor=True)
    except KeyboardInterrupt:
        console.print("\n[bold]Stopping...[/bold]")
    finally:
        _print_summary(detector)


def _print_summary(detector: Detector):
    stats = detector.stats()
    table = Table(title="Summary")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("SSIDs observed", str(len(stats["ssid_to_bssids"])))
    table.add_row("BSSIDs observed", str(len(stats["bssid_to_ssids"])))
    console.print(table)

@main.command("analyze")
@click.option("--pcap", "pcap_path", required=True, type=click.Path(exists=True), help="Path to .pcap or .pcapng")
@click.option("--deauth-threshold", type=int, default=30, show_default=True)
@click.option("--cluster-threshold", type=int, default=6, show_default=True)
@click.option("--window-seconds", type=int, default=10, show_default=True)
@click.option("--whitelist", type=click.Path(exists=True), default=None, help="JSON with ssids[] and bssids[]")
def analyze_cmd(
    pcap_path: str,
    deauth_threshold: int,
    cluster_threshold: int,
    window_seconds: int,
    whitelist: Optional[str],
):
    """Analyze a capture file for anomalies."""
    wl = load_whitelist(whitelist)
    detector = Detector(
        deauth_threshold=deauth_threshold,
        window_seconds=window_seconds,
        cluster_threshold=cluster_threshold,
        whitelist_ssids=wl["ssids"],
        whitelist_bssids=wl["bssids"],
    )
    events = analyze_pcap(pcap_path, detector)
    if not events:
        console.print("[green]No anomalies detected.[/green]")
        return
    for ev in events:
        console.print(f"[yellow]{ev.severity.upper()}[/yellow] {ev.kind}: {ev.message}")
    _print_summary(detector)