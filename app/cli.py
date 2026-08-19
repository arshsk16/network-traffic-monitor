"""
app/cli.py — Command-line demonstration interface for the monitoring engine.

This module is the Step 11 "presentation layer". It calls the existing
run_monitoring_cycle() function and formats the results for human reading.

Architecture
────────────
    CLI (this file)
        ↓  calls
    core.cycle.run_monitoring_cycle()
        ↓  which calls
    core.monitor → core.scoring → core.failover
        ↓  which call
    core.probe / core.loss / core.stats / core.throughput

This file contains ZERO:
  - socket logic
  - scoring formulas
  - failover decisions
  - normalization code
  - RTT / loss / jitter / throughput calculations

It only reads values from the CycleResult and prints them.

Demo design
───────────
The CLI starts its own controlled local TCP servers on loopback (127.0.0.1).
This guarantees the demo works without any external network, external hosts,
or real SD-WAN infrastructure. The demo makes it explicit that these are
local demonstration paths, not real WAN paths.

The failure scenario is produced genuinely:
  - Cycles 1–2: both servers running → both paths healthy
  - Cycle 3: primary server stopped mid-run → primary becomes 100% loss
             → scoring marks primary unavailable → failover to backup
  - Cycle 4: primary still stopped → backup remains preferred → NO_CHANGE

This exercises the actual monitoring pipeline — not a mocked failure.

Running the CLI
───────────────
    .venv\\Scripts\\python -m app.cli

    # Or with explicit asyncio entrypoint:
    .venv\\Scripts\\python app/cli.py
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
from io import StringIO
from typing import IO

from core.cycle import CycleResult, run_monitoring_cycle
from core.failover import PathState, TransitionType
from core.path import Path
from core.scoring import ScoringWeights


# ── ANSI colour codes (disabled on Windows if unsupported) ───────────────────


def _ansi_supported() -> bool:
    """Check whether the Windows console supports ANSI escape sequences."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # Enable VIRTUAL_TERMINAL_PROCESSING (flag 0x0004)
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


_USE_COLOUR = sys.platform != "win32" or _ansi_supported()


# Colour helpers — each returns the string wrapped in ANSI codes if supported,
# or the string unchanged if colours are disabled.

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if _USE_COLOUR else s


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if _USE_COLOUR else s


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m" if _USE_COLOUR else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOUR else s


def _cyan(s: str) -> str:
    return f"\033[96m{s}\033[0m" if _USE_COLOUR else s


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if _USE_COLOUR else s


# ── Local demo server ─────────────────────────────────────────────────────────


class _DemoServer:
    """
    A local TCP dual-protocol server for the demo paths.

    Handles the same PING→PONG / SIZE:N→data→DONE protocol as the test
    fixtures. Started on an OS-assigned ephemeral port. Can be stopped
    mid-demo to simulate a path failure.

    This is infrastructure for the demo only — it does not represent any
    real network path.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self.host: str = ""
        self.port: int = 0

    def start(self) -> None:
        """Start listening on an ephemeral loopback port."""
        self._stop.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(64)
        sock.settimeout(0.05)
        self._sock = sock
        self.host, self.port = sock.getsockname()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def stop(self) -> None:
        """Signal the accept loop to exit and close the server socket."""
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    @staticmethod
    def _handle(conn: socket.socket) -> None:
        try:
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                reader = conn.makefile("rb")
                first_line = reader.readline()
                if first_line == b"PING\n":
                    conn.sendall(b"PONG\n")
                elif first_line.startswith(b"SIZE:"):
                    n_bytes = int(first_line.split(b":")[1].strip())
                    remaining = n_bytes
                    while remaining > 0:
                        chunk = reader.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                    conn.sendall(b"DONE\n")
        except OSError:
            pass

    @property
    def path(self) -> Path:
        """Return the Path object pointing at this server."""
        return Path(name=self.name, host=self.host, port=self.port)


# ── Formatting helpers ─────────────────────────────────────────────────────────


_COL_PATH = 14
_COL_RTT = 11
_COL_LOSS = 9
_COL_JITTER = 11
_COL_TPUT = 15
_COL_SCORE = 8
_COL_AVAIL = 13


def _fmt_rtt(v: float | None) -> str:
    if v is None:
        return _dim("  —")
    return f"{v:6.1f} ms"


def _fmt_loss(v: float | None) -> str:
    if v is None:
        return _dim("  —")
    if v == 0.0:
        return _green(f"{v:5.1f}%")
    if v >= 100.0:
        return _red(f"{v:5.1f}%")
    return _yellow(f"{v:5.1f}%")


def _fmt_jitter(v: float | None) -> str:
    if v is None:
        return _dim("  —")
    return f"{v:5.1f} ms"


def _fmt_tput(v: float | None) -> str:
    if v is None:
        return _dim("  —")
    return f"{v:8.1f} Mbps"


def _fmt_score(v: float, available: bool) -> str:
    if not available:
        return _red(" N/A")
    if v >= 70.0:
        return _green(f"{v:6.1f}")
    if v >= 40.0:
        return _yellow(f"{v:6.1f}")
    return _red(f"{v:6.1f}")


def _fmt_avail(available: bool) -> str:
    return _green("available") if available else _red("UNAVAILABLE")


def _event_label(event_type: TransitionType) -> str:
    labels = {
        TransitionType.INITIAL_SELECTION: _green("INITIAL_SELECTION"),
        TransitionType.NO_CHANGE:         _cyan("NO_CHANGE"),
        TransitionType.FAILOVER:          _yellow("FAILOVER"),
        TransitionType.NO_AVAILABLE_PATH: _red("NO_AVAILABLE_PATH"),
    }
    return labels.get(event_type, event_type.value)


# ── Main output functions ──────────────────────────────────────────────────────


def format_cycle_output(cycle_num: int, result: CycleResult, out: IO[str] = sys.stdout) -> None:
    """
    Format one CycleResult for terminal display.

    This function reads values from the CycleResult and formats them for
    human consumption. It performs no calculations of its own.

    Parameters
    ----------
    cycle_num:
        The 1-based cycle number shown in the header.
    result:
        The complete cycle result from run_monitoring_cycle().
    out:
        Output stream (default: sys.stdout). Accepts a StringIO for testing.
    """
    sep = "─" * 78

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{sep}", file=out)
    print(_bold(f" Cycle {cycle_num}"), file=out)
    print(sep, file=out)

    # ── Column headers ────────────────────────────────────────────────────────
    header = (
        f"  {'PATH':<{_COL_PATH}}"
        f"{'RTT':>{_COL_RTT}}"
        f"{'LOSS':>{_COL_LOSS}}"
        f"{'JITTER':>{_COL_JITTER}}"
        f"{'THROUGHPUT':>{_COL_TPUT}}"
        f"{'SCORE':>{_COL_SCORE}}"
        f"  STATUS"
    )
    print(_dim(header), file=out)
    print(_dim("  " + "·" * 76), file=out)

    # ── Per-path rows ─────────────────────────────────────────────────────────
    # Build a lookup: path name → ScoredPath
    scored_by_name = {sp.path.name: sp for sp in result.scoring_result.scored_paths}

    # Iterate paths in rank order
    for sp in sorted(result.scoring_result.scored_paths, key=lambda s: s.rank):
        name = sp.path.name
        pm = result.monitoring_result.metrics.get(name)

        # Extract raw metric values from monitoring result
        rtt_ms     = pm.rtt_stats.mean_ms    if (pm and pm.rtt_stats and pm.rtt_stats.count > 0) else None
        loss_pct   = pm.probe_stats.loss_rate if (pm and pm.probe_stats) else None
        jitter_ms  = pm.rtt_stats.jitter_ms  if (pm and pm.rtt_stats and pm.rtt_stats.count > 0) else None
        tput_mbps  = pm.throughput.throughput_mbps if (pm and pm.throughput) else None

        is_preferred = (
            result.scoring_result.preferred_path is not None
            and result.scoring_result.preferred_path.path.name == name
        )
        name_str = _bold(f"* {name}") if is_preferred else f"  {name}"

        row = (
            f"{name_str:<{_COL_PATH + 2}}"
            f"{_fmt_rtt(rtt_ms):>{_COL_RTT}}"
            f"{_fmt_loss(loss_pct):>{_COL_LOSS}}"
            f"{_fmt_jitter(jitter_ms):>{_COL_JITTER}}"
            f"{_fmt_tput(tput_mbps):>{_COL_TPUT}}"
            f"{_fmt_score(sp.total_score, sp.available):>{_COL_SCORE}}"
            f"  {_fmt_avail(sp.available)}"
        )
        print(row, file=out)

    print(file=out)

    # ── Preferred path summary ────────────────────────────────────────────────
    preferred = result.preferred_path_name
    preferred_str = _bold(_green(preferred)) if preferred else _red("none")
    print(f"  Preferred Path : {preferred_str}", file=out)

    # ── Event / transition ────────────────────────────────────────────────────
    event = result.transition.event_type
    print(f"  Event          : {_event_label(event)}", file=out)

    if event == TransitionType.FAILOVER:
        prev_name = result.transition.previous_path.name if result.transition.previous_path else "?"
        new_name  = result.transition.new_path.name      if result.transition.new_path      else "?"
        print(f"  Transition     : {_bold(prev_name)} → {_bold(new_name)}", file=out)

    if event == TransitionType.NO_CHANGE and result.transition.previous_path:
        prev_name = result.transition.previous_path.name
        print(f"  Previous       : {_dim(prev_name)}", file=out)

    print(sep, file=out)


def print_header(out: IO[str] = sys.stdout) -> None:
    """Print the application banner."""
    banner = r"""
  ███╗   ██╗███████╗████████╗███╗   ███╗ ██████╗ ███╗   ██╗
  ████╗  ██║██╔════╝╚══██╔══╝████╗ ████║██╔═══██╗████╗  ██║
  ██╔██╗ ██║█████╗     ██║   ██╔████╔██║██║   ██║██╔██╗ ██║
  ██║╚██╗██║██╔══╝     ██║   ██║╚██╔╝██║██║   ██║██║╚██╗██║
  ██║ ╚████║███████╗   ██║   ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║
  ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
  Network Traffic Monitor — SD-WAN Path Selection Demo
  Python 3.12  ·  local loopback demonstration only
"""
    print(_bold(_cyan(banner)), file=out)


def print_demo_legend(primary: _DemoServer, backup: _DemoServer, out: IO[str] = sys.stdout) -> None:
    """Print a legend explaining the demo paths."""
    print("  Demo paths (all traffic is local loopback — no real WAN links):", file=out)
    print(f"    primary  →  {primary.host}:{primary.port}", file=out)
    print(f"    backup   →  {backup.host}:{backup.port}", file=out)
    print(file=out)
    print(_dim("  NOTE: Throughput figures are loopback speed, NOT WAN bandwidth."), file=out)
    print(file=out)


# ── Demo sequence ─────────────────────────────────────────────────────────────


async def run_demo(out: IO[str] = sys.stdout) -> None:
    """
    Execute the four-cycle demonstration sequence.

    Cycle 1 — Initial selection (both paths healthy)
    Cycle 2 — No change (both paths healthy)
    Cycle 3 — Failover (primary server stopped → 100% loss → failover to backup)
    Cycle 4 — No change (backup remains preferred)

    The failure in Cycle 3 is genuine: the primary server's socket is closed
    before the cycle runs. The monitoring engine measures 100% probe loss on
    the primary, the scoring engine marks it unavailable, and the failover
    state manager produces a FAILOVER transition. No values are faked.
    """
    # ── Start demo servers ────────────────────────────────────────────────────
    primary_server = _DemoServer("primary")
    backup_server  = _DemoServer("backup")
    primary_server.start()
    backup_server.start()

    state = PathState()
    # Use fast probe settings so the demo completes quickly
    cycle_config = dict(
        probe_count=2,
        probe_timeout=1.0,
        transfer_bytes=32 * 1024,  # 32 KiB — fast local transfer
        throughput_timeout=5.0,
    )

    try:
        print_header(out)
        print_demo_legend(primary_server, backup_server, out)

        # ── Cycle 1: Both paths healthy ───────────────────────────────────────
        paths_healthy = [primary_server.path, backup_server.path]

        print(_bold("  Demo sequence:"), file=out)
        print("  Cycle 1 — Initial selection  (both paths healthy)", file=out)
        print("  Cycle 2 — No change          (both paths healthy)", file=out)
        print("  Cycle 3 — Failover           (primary server stopped)", file=out)
        print("  Cycle 4 — No change          (backup remains preferred)", file=out)
        print(file=out)

        print(_dim("  Running Cycle 1 …"), file=out)
        result1 = await run_monitoring_cycle(paths_healthy, state, **cycle_config)
        format_cycle_output(1, result1, out)

        # ── Cycle 2: Still healthy ────────────────────────────────────────────
        print(_dim("  Running Cycle 2 …"), file=out)
        result2 = await run_monitoring_cycle(paths_healthy, state, **cycle_config)
        format_cycle_output(2, result2, out)

        # ── Cycle 3: Primary fails ────────────────────────────────────────────
        # Stop the primary server before the cycle. The monitoring engine will
        # see connection refused → 100% loss → marks primary unavailable →
        # scoring selects backup as preferred → FAILOVER transition.
        primary_server.stop()

        print(_dim("  [Demo] Stopping primary server to simulate path failure …"), file=out)
        print(_dim("  Running Cycle 3 …"), file=out)

        # The path object still points to primary's old host:port (now refused)
        paths_with_dead_primary = [primary_server.path, backup_server.path]
        result3 = await run_monitoring_cycle(
            paths_with_dead_primary, state, **cycle_config
        )
        format_cycle_output(3, result3, out)

        # ── Cycle 4: Backup remains ───────────────────────────────────────────
        print(_dim("  Running Cycle 4 …"), file=out)
        result4 = await run_monitoring_cycle(
            paths_with_dead_primary, state, **cycle_config
        )
        format_cycle_output(4, result4, out)

        print(file=out)
        print(_bold("  Demo complete."), file=out)
        print(
            _dim("  In a real system the preferred path drives routing decisions.\n"
                 "  This prototype only determines the recommendation — no OS\n"
                 "  routing table modifications are made."),
            file=out,
        )
        print(file=out)

    finally:
        primary_server.stop()
        backup_server.stop()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Synchronous entry point — runs the async demo with asyncio.run()."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
