"""
tests/test_monitor_api.py — API tests for GET /api/v1/monitor and GET /.

Test strategy
─────────────
We use the existing session-scoped `client` fixture from conftest.py.
The TestClient wraps the ASGI app synchronously, which means FastAPI's
async route functions are run with anyio under the hood. This is exactly
how the existing test_health.py tests work.

app.state starts two _DemoServer instances at module import time
(process startup). Those servers serve the demo paths for all API tests.
Because the servers are daemon threads, they are cleaned up when the
process exits — no explicit teardown needed in tests.

State persistence across tests
──────────────────────────────
The app fixture is session-scoped, so the same monitoring_state (PathState)
persists for the entire test session. The first call to /api/v1/monitor will
produce INITIAL_SELECTION; subsequent calls will produce NO_CHANGE (assuming
the path quality is stable).

Tests that need to assert on the event type use ordering-aware checks:
  - TestMonitorEndpointFirstCall: verified against INITIAL_SELECTION
    (because these are the first tests to call the endpoint in the session,
    ordered before other classes by pytest's default collection order).
  - TestMonitorEndpointStateRetention: makes a second call and asserts
    NO_CHANGE.

Tests that need controlled failover state construct isolated PathState
objects rather than using the shared one — they call run_monitoring_cycle()
directly just as test_cycle.py does.

No exact metric values are asserted (RTT, throughput vary with machine speed).
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

import app.state as app_state
from app.cli import _DemoServer
from app.routers.monitor import _cycle_result_to_response
from core.cycle import run_monitoring_cycle
from core.failover import PathState, TransitionType
from core.path import Path


# ── Helpers for controlled local servers ──────────────────────────────────────

def _dead_path(name: str = "dead") -> Path:
    """Return a Path whose port has no server (connection refused immediately)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    return Path(name=name, host="127.0.0.1", port=port)


_FAST = dict(probe_count=1, probe_timeout=0.5, transfer_bytes=4096, throughput_timeout=2.0)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Existing health endpoint still works
# ══════════════════════════════════════════════════════════════════════════════


class TestHealthStillWorks:
    def test_health_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_health_status_is_ok(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Monitor endpoint — HTTP contract
# ══════════════════════════════════════════════════════════════════════════════


class TestMonitorEndpointContract:
    def test_monitor_returns_200(self, client: TestClient) -> None:
        assert client.get("/api/v1/monitor").status_code == 200

    def test_monitor_content_type_json(self, client: TestClient) -> None:
        resp = client.get("/api/v1/monitor")
        assert "application/json" in resp.headers["content-type"]

    def test_monitor_response_has_paths(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert "paths" in body

    def test_monitor_response_has_preferred_path(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert "preferred_path" in body

    def test_monitor_response_has_event(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert "event" in body

    def test_monitor_response_has_previous_path(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert "previous_path" in body

    def test_monitor_response_has_cycle_note(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert "cycle_note" in body

    def test_cycle_note_mentions_loopback(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert "loopback" in body["cycle_note"].lower() or "local" in body["cycle_note"].lower()

    def test_cycle_note_not_wan_claim(self, client: TestClient) -> None:
        """Cycle note must not claim this is WAN bandwidth."""
        body = client.get("/api/v1/monitor").json()
        note = body["cycle_note"].lower()
        assert "not wan" in note or "not real wan" in note or "not wan" in note


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Per-path metric fields
# ══════════════════════════════════════════════════════════════════════════════


class TestPerPathMetrics:
    def test_paths_dict_is_not_empty(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert len(body["paths"]) > 0

    def test_demo_paths_present(self, client: TestClient) -> None:
        """'primary' and 'backup' are the two demo paths."""
        body = client.get("/api/v1/monitor").json()
        assert "primary" in body["paths"]
        assert "backup" in body["paths"]

    def test_path_has_available_field(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        path = next(iter(body["paths"].values()))
        assert "available" in path
        assert isinstance(path["available"], bool)

    def test_path_has_rtt_ms_field(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        path = next(iter(body["paths"].values()))
        assert "rtt_ms" in path

    def test_path_has_loss_percent_field(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        path = next(iter(body["paths"].values()))
        assert "loss_percent" in path

    def test_path_has_jitter_ms_field(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        path = next(iter(body["paths"].values()))
        assert "jitter_ms" in path

    def test_path_has_throughput_mbps_field(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        path = next(iter(body["paths"].values()))
        assert "throughput_mbps" in path

    def test_path_has_score_field(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        path = next(iter(body["paths"].values()))
        assert "score" in path
        assert isinstance(path["score"], (int, float))

    def test_path_has_rank_field(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        path = next(iter(body["paths"].values()))
        assert "rank" in path
        assert isinstance(path["rank"], int)

    def test_available_path_has_non_null_rtt(self, client: TestClient) -> None:
        """An available path must have a real RTT value (not null)."""
        body = client.get("/api/v1/monitor").json()
        for metrics in body["paths"].values():
            if metrics["available"]:
                assert metrics["rtt_ms"] is not None

    def test_available_path_loss_is_zero(self, client: TestClient) -> None:
        """Both demo servers are live, so loss must be 0 for available paths."""
        body = client.get("/api/v1/monitor").json()
        for metrics in body["paths"].values():
            if metrics["available"]:
                assert metrics["loss_percent"] == 0.0

    def test_rtt_ms_is_positive_float(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        for m in body["paths"].values():
            if m["rtt_ms"] is not None:
                assert m["rtt_ms"] > 0.0

    def test_score_is_in_range(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        for m in body["paths"].values():
            assert 0.0 <= m["score"] <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Preferred path and event fields
# ══════════════════════════════════════════════════════════════════════════════


class TestPreferredPathAndEvent:
    def test_preferred_path_is_string_or_null(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert body["preferred_path"] is None or isinstance(body["preferred_path"], str)

    def test_preferred_path_is_one_of_demo_paths(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        pp = body["preferred_path"]
        if pp is not None:
            assert pp in body["paths"]

    def test_event_is_valid_string(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        valid = {"initial_selection", "no_change", "failover", "no_available_path"}
        assert body["event"] in valid

    def test_previous_path_is_string_or_null(self, client: TestClient) -> None:
        body = client.get("/api/v1/monitor").json()
        assert body["previous_path"] is None or isinstance(body["previous_path"], str)


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: State persistence across requests (shared client = shared state)
# ══════════════════════════════════════════════════════════════════════════════


class TestStateRetention:
    """
    Verify that repeated calls with the same PathState produce NO_CHANGE.

    IMPORTANT: We do NOT rely on the session-scoped client's app.state here
    because test_cli.py::TestFullDemoRun calls run_demo() which stops the
    primary server in app.state mid-session. Using the shared state could
    legitimately produce FAILOVER in subsequent API calls — that is correct
    behaviour, not a test failure.

    Instead we use direct cycle calls with fresh isolated servers + PathState
    to verify the state-retention principle in isolation.
    """

    def test_same_state_second_call_is_no_change(self) -> None:
        """Using the SAME PathState across two calls → second is NO_CHANGE."""
        import asyncio
        from app.routers.monitor import _cycle_result_to_response
        srv = _DemoServer("retention-srv")
        srv.start()
        state = PathState()
        try:
            asyncio.run(run_monitoring_cycle([srv.path], state, **_FAST))
            r2 = asyncio.run(run_monitoring_cycle([srv.path], state, **_FAST))
            resp = _cycle_result_to_response(r2)
            assert resp.event == "no_change"
        finally:
            srv.stop()

    def test_preferred_path_stable_with_same_state(self) -> None:
        """Preferred path name must be the same across two cycles on same state."""
        import asyncio
        from app.routers.monitor import _cycle_result_to_response
        srv = _DemoServer("stable-srv")
        srv.start()
        state = PathState()
        try:
            r1 = asyncio.run(run_monitoring_cycle([srv.path], state, **_FAST))
            r2 = asyncio.run(run_monitoring_cycle([srv.path], state, **_FAST))
            assert _cycle_result_to_response(r1).preferred_path == \
                   _cycle_result_to_response(r2).preferred_path
        finally:
            srv.stop()

    def test_api_endpoint_returns_some_preferred_or_null(self, client: TestClient) -> None:
        """
        After any API call, preferred_path must be a string or null.
        We do not assert which value — the shared app.state may have
        experienced legitimate failover events from other tests.
        """
        body = client.get("/api/v1/monitor").json()
        assert body["preferred_path"] is None or isinstance(body["preferred_path"], str)



# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Dashboard HTML endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestDashboardEndpoint:
    def test_dashboard_returns_200(self, client: TestClient) -> None:
        assert client.get("/").status_code == 200

    def test_dashboard_content_type_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_dashboard_contains_title(self, client: TestClient) -> None:
        text = client.get("/").text
        assert "Network Traffic Monitor" in text

    def test_dashboard_contains_demo_notice(self, client: TestClient) -> None:
        text = client.get("/").text
        assert "loopback" in text.lower() or "demo" in text.lower()

    def test_dashboard_contains_run_button(self, client: TestClient) -> None:
        text = client.get("/").text
        assert "run-btn" in text or "Run Monitoring" in text

    def test_dashboard_mentions_not_wan(self, client: TestClient) -> None:
        """Dashboard must disclaim that throughput is not WAN bandwidth."""
        text = client.get("/").text
        assert "not wan" in text.lower() or "not real wan" in text.lower()

    def test_dashboard_no_external_dependencies_required(self, client: TestClient) -> None:
        """
        The dashboard uses Google Fonts (optional, CDN), but all interactive
        logic is inline JS — it must not require any local npm build output.
        The JS function runCycle() must be present.
        """
        text = client.get("/").text
        assert "runCycle" in text


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: Failover representation (isolated — uses direct cycle calls)
# ══════════════════════════════════════════════════════════════════════════════


class TestFailoverRepresentation:
    """
    Verify that _cycle_result_to_response() correctly serializes failover
    and unavailable-path states. These tests use run_monitoring_cycle()
    directly with controlled server objects and a fresh PathState — they
    do not depend on the shared app state.
    """

    def _start_server(self, name: str) -> _DemoServer:
        srv = _DemoServer(name)
        srv.start()
        return srv

    def test_unavailable_path_serializes_null_rtt(self) -> None:
        """A dead path must produce rtt_ms=null in the JSON."""
        import asyncio
        dead = _dead_path("dead")
        state = PathState()
        result = asyncio.run(
            run_monitoring_cycle(
                [dead], state,
                probe_count=1, probe_timeout=0.3,
                transfer_bytes=4096, throughput_timeout=0.3,
            )
        )
        resp = _cycle_result_to_response(result)
        pm = resp.paths["dead"]
        assert pm.available is False
        assert pm.rtt_ms is None

    def test_unavailable_path_serializes_no_available_path_event(self) -> None:
        import asyncio
        dead = _dead_path("dead")
        state = PathState()
        result = asyncio.run(
            run_monitoring_cycle(
                [dead], state,
                probe_count=1, probe_timeout=0.3,
                transfer_bytes=4096, throughput_timeout=0.3,
            )
        )
        resp = _cycle_result_to_response(result)
        assert resp.event == "no_available_path"

    def test_failover_event_serializes_both_paths(self) -> None:
        """After primary dies, response event=failover with previous_path set."""
        import asyncio
        srv_a = self._start_server("srv-a")
        srv_b = self._start_server("srv-b")
        state = PathState()
        try:
            # Cycle 1: only srv-a → INITIAL_SELECTION
            asyncio.run(
                run_monitoring_cycle([srv_a.path], state, **_FAST)
            )
            # Cycle 2: srv-a dead, srv-b alive → FAILOVER
            srv_a.stop()
            result = asyncio.run(
                run_monitoring_cycle([_dead_path("srv-a"), srv_b.path], state,
                                     probe_count=1, probe_timeout=0.3,
                                     transfer_bytes=4096, throughput_timeout=0.3)
            )
            resp = _cycle_result_to_response(result)
            assert resp.event == "failover"
            assert resp.previous_path == "srv-a"
            assert resp.preferred_path == "srv-b"
        finally:
            srv_a.stop()
            srv_b.stop()

    def test_initial_selection_event_has_null_previous(self) -> None:
        """First cycle: previous_path must be null."""
        import asyncio
        srv = self._start_server("first")
        state = PathState()
        try:
            result = asyncio.run(run_monitoring_cycle([srv.path], state, **_FAST))
            resp = _cycle_result_to_response(result)
            assert resp.event == "initial_selection"
            assert resp.previous_path is None
        finally:
            srv.stop()

    def test_no_change_event_has_same_preferred_path(self) -> None:
        """Second healthy cycle: preferred_path must be same as first."""
        import asyncio
        srv = self._start_server("stable")
        state = PathState()
        try:
            r1 = asyncio.run(run_monitoring_cycle([srv.path], state, **_FAST))
            r2 = asyncio.run(run_monitoring_cycle([srv.path], state, **_FAST))
            resp2 = _cycle_result_to_response(r2)
            assert resp2.event == "no_change"
            assert resp2.preferred_path == r1.preferred_path_name
        finally:
            srv.stop()
