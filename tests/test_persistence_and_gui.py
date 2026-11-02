"""
Tests for persistence, checkpoint resume, JPY pip accounting,
marker remapping within bounds, GUI compilation, and HTTP server endpoints (§ 11 ETAP E7).
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

import gui_server
from build_gui import build_standalone_gui
from compile_timeframe_payload import compile_56_markets_payload
from engine.costs import get_pair_specs, price_to_pips
from storage.state import StorageManager


def test_jpy_spread_and_pip_precision() -> None:
    """
    Verifies that for JPY pairs (USDJPY, EURJPY), spread in pips is strictly < 20.0
    and never falsely inflated by an incorrect 10000.0 multiplier.
    """
    for jpy_pair in ["USDJPY", "EURJPY"]:
        spec = get_pair_specs(jpy_pair)
        assert spec["pip_mult"] == 100.0
        assert spec["pip_unit"] == 0.01
        assert spec["dec"] == 3

        # Sample market spread in absolute price (e.g. 0.015 = 1.5 pips)
        abs_spread = 0.015
        pips = price_to_pips(jpy_pair, abs_spread)
        assert pips == pytest.approx(1.5, rel=1e-3)
        assert pips < 20.0, f"Spread for {jpy_pair} in pips must be < 20.0 pips, got {pips}"

        # Ensure that multiplying by 10000 would falsely report 150 pips
        false_pips = abs_spread * 10000.0
        assert false_pips > 100.0, "Multiplication by 10000 produces invalid spread for JPY"


def test_chart_markers_bounds_and_remapping() -> None:
    """
    Verifies that all markers in output/live_signals_multi_tf.json have indices
    strictly within the valid bounds [0, len(candles) - 1].
    """
    payload_path = Path("output/live_signals_multi_tf.json")
    if not payload_path.exists():
        compile_56_markets_payload(output_json_path=payload_path, n_bars_window=100)

    assert payload_path.exists()
    with open(payload_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    markets = data.get("markets", {})
    assert len(markets) == 56, f"Expected 56 markets, got {len(markets)}"

    for m_key, m_val in markets.items():
        candles = m_val.get("candles", [])
        n_candles = len(candles)
        assert n_candles > 0, f"Candles empty for market {m_key}"

        strategies = m_val.get("strategies", {})
        for s_id, s_data in strategies.items():
            markers = s_data.get("chart_markers", [])
            for marker in markers:
                idx = marker.get("index")
                assert 0 <= idx < n_candles, (
                    f"Marker index {idx} out of bounds for market {m_key}, strat {s_id} (n_candles={n_candles})"
                )

            paths = s_data.get("transaction_paths", [])
            for p in paths:
                e_idx = p.get("entry_idx")
                x_idx = p.get("exit_idx")
                assert e_idx is not None
                assert x_idx is not None
                assert 0 <= e_idx < n_candles
                assert 0 <= x_idx < n_candles
                assert x_idx >= e_idx


def test_checkpoint_persistence_and_resume(tmp_path: Path) -> None:
    """
    Verifies saving and loading of atomic checkpoints to resume interrupted runs.
    """
    storage = StorageManager(run_id="TEST_RUN_RESUME", base_dir=tmp_path)

    state = {
        "iteration": 42,
        "best_fitness": 1.875,
        "qd_archive_size": 25,
        "active_strategies": ["STRAT_ALPHA", "STRAT_BETA"],
    }
    storage.save_checkpoint(state)

    # Re-initialize another manager pointing to same run_id
    resumed_storage = StorageManager(run_id="TEST_RUN_RESUME", base_dir=tmp_path)
    loaded_state = resumed_storage.load_latest_checkpoint()

    assert loaded_state is not None
    assert loaded_state["iteration"] == 42
    assert loaded_state["best_fitness"] == pytest.approx(1.875)
    assert loaded_state["qd_archive_size"] == 25
    assert loaded_state["active_strategies"] == ["STRAT_ALPHA", "STRAT_BETA"]


def test_gui_html_and_canvas_elements(tmp_path: Path) -> None:
    """
    Verifies compilation of live_trfx_gui.html and presence of all interactive elements.
    """
    payload_path = Path("output/live_signals_multi_tf.json")
    if not payload_path.exists():
        compile_56_markets_payload(output_json_path=payload_path, n_bars_window=100)

    html_out = tmp_path / "test_gui.html"
    build_standalone_gui(payload_path, html_out)

    assert html_out.exists()
    content = html_out.read_text(encoding="utf-8")

    # Verify Dual-Pane HTML5 Canvas
    assert "upperCanvas" in content
    assert "lowerCanvas" in content

    # Verify What-If simulator
    assert "sliderSlAtr" in content
    assert "sliderTpAtr" in content
    assert "sliderRiskPct" in content

    # Verify Lot Calculator
    assert "calcEquity" in content
    assert "calcSlPips" in content
    assert "calcLots" in content

    # Verify 56-Market Radar Matrix Modal
    assert "modalMatrix" in content
    assert "btnMatrix" in content

    # Verify Web Audio API Synth
    assert "AudioContext" in content
    assert "playAlertSound" in content

    # Verify Run Selector and Switcher
    assert "runSelector" in content
    assert "loadAvailableRuns" in content
    assert "switchRun" in content
    assert "loadingToast" in content


class MockHttpServer:
    server_address = ("127.0.0.1", 8050)


def create_mock_handler(path: str) -> tuple[gui_server.ProTerminalHandler, BytesIO]:
    handler = gui_server.ProTerminalHandler.__new__(gui_server.ProTerminalHandler)
    handler.server = MockHttpServer()
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.rfile = BytesIO()
    handler.wfile = BytesIO()
    handler.command = "GET"
    handler.request_version = "HTTP/1.1"
    handler.headers = {}
    return handler, handler.wfile


def test_gui_server_rest_api() -> None:
    """
    Verifies server socket initialization and REST endpoints (/api/health, /api/signals, /api/scan, /).
    """
    # 1. Test server socket binding & clean close
    server = gui_server.start_server(host="127.0.0.1", port=0)
    bound_port = server.server_address[1]
    assert bound_port > 0
    server.server_close()

    # 2. Test GET /api/health
    h_health, out_health = create_mock_handler("/api/health")
    h_health.do_GET()
    raw_health = out_health.getvalue()
    assert b"200 OK" in raw_health
    assert b"Access-Control-Allow-Origin: *" in raw_health
    body_health = raw_health.split(b"\r\n\r\n", 1)[1]
    data_health = json.loads(body_health.decode("utf-8"))
    assert data_health["status"] == "ok"
    assert data_health["markets_count"] == 56

    # 3. Test GET /api/signals
    h_sig, out_sig = create_mock_handler("/api/signals")
    h_sig.do_GET()
    raw_sig = out_sig.getvalue()
    assert b"200 OK" in raw_sig
    body_sig = raw_sig.split(b"\r\n\r\n", 1)[1]
    data_sig = json.loads(body_sig.decode("utf-8"))
    assert "pairs" in data_sig
    assert len(data_sig["pairs"]) == 7

    # 4. Test GET /api/scan
    h_scan, out_scan = create_mock_handler("/api/scan")
    h_scan.do_GET()
    raw_scan = out_scan.getvalue()
    assert b"200 OK" in raw_scan
    body_scan = raw_scan.split(b"\r\n\r\n", 1)[1]
    data_scan = json.loads(body_scan.decode("utf-8"))
    assert data_scan["total_markets"] == 56
    assert "radar_matrix" in data_scan

    # 5. Test GET / (serves HTML)
    h_root, out_root = create_mock_handler("/")
    h_root.do_GET()
    raw_root = out_root.getvalue()
    assert b"200 OK" in raw_root
    assert b"text/html" in raw_root
    assert b"upperCanvas" in raw_root or b"TRFX PRO TERMINAL" in raw_root

    # 6. Test GET /api/runs (ensure test run exists in output/runs)
    test_run_dir = Path("output/runs/run_test_gui")
    test_run_dir.mkdir(parents=True, exist_ok=True)
    (test_run_dir / "leaderboard").mkdir(parents=True, exist_ok=True)
    lb_file = test_run_dir / "leaderboard" / "current.json"
    if not lb_file.exists():
        lb_file.write_text(json.dumps([{"strategy_id": "TEST_STRAT", "family": "F1_Trend", "fitness": 1.5, "sharpe_holdout": 2.0}]), encoding="utf-8")

    h_runs, out_runs = create_mock_handler("/api/runs")
    h_runs.do_GET()
    raw_runs = out_runs.getvalue()
    assert b"200 OK" in raw_runs
    body_runs = raw_runs.split(b"\r\n\r\n", 1)[1]
    data_runs = json.loads(body_runs.decode("utf-8"))
    assert isinstance(data_runs, list)
    assert len(data_runs) > 0
    run_ids = [r["run_id"] for r in data_runs]
    assert len(run_ids) > 0
    target_run_id = run_ids[0]
    first_run = data_runs[0]
    assert "run_id" in first_run
    assert "created_at" in first_run
    assert "top_strategy" in first_run

    # 7. Test GET /api/select_run?run_id=target_run_id
    h_sel, out_sel = create_mock_handler(f"/api/select_run?run_id={target_run_id}")
    h_sel.do_GET()
    raw_sel = out_sel.getvalue()
    assert b"200 OK" in raw_sel
    body_sel = raw_sel.split(b"\r\n\r\n", 1)[1]
    data_sel = json.loads(body_sel.decode("utf-8"))
    assert data_sel["run_id"] == target_run_id
    assert "strategies" in data_sel
    assert "markets" in data_sel

    # 8. Test GET /api/select_run with invalid run_id
    h_inv, out_inv = create_mock_handler("/api/select_run?run_id=invalid_bad_id")
    h_inv.do_GET()
    raw_inv = out_inv.getvalue()
    assert b"400 Bad Request" in raw_inv or b"404 Not Found" in raw_inv
