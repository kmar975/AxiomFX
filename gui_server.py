"""
Forex Quant AI Lab & Pro Live Terminal HTTP Server (§ 10.2).
Serves output/live_trfx_gui.html on http://localhost:8050 with REST API:
  - GET /            -> Serves live_trfx_gui.html
  - GET /index.html  -> Serves live_trfx_gui.html
  - GET /api/health  -> Healthcheck & server metadata
  - GET /api/signals -> Returns output/live_signals_multi_tf.json
  - GET /api/scan    -> Triggers or returns live scanner state across all 56 markets
"""

from __future__ import annotations

import argparse
import datetime
import http.server
import json
import re
import socketserver
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from compile_timeframe_payload import compile_56_markets_payload, get_or_compile_run_payload

DEFAULT_PORT = 8050
DEFAULT_HOST = "127.0.0.1"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" if (BASE_DIR / "output").exists() else Path("output")
GUI_HTML_PATH = OUTPUT_DIR / "live_trfx_gui.html"
PAYLOAD_JSON_PATH = OUTPUT_DIR / "live_signals_multi_tf.json"


class ProTerminalHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving Pro Terminal Canvas GUI and REST API endpoints.
    """

    def send_cors_headers(self, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_cors_headers()

    def do_GET(self) -> None:
        parsed_path = self.path.split("?")[0]

        if parsed_path in ("/", "/index.html"):
            self.handle_serve_gui()
        elif parsed_path == "/api/health":
            self.handle_api_health()
        elif parsed_path == "/api/signals":
            self.handle_api_signals()
        elif parsed_path == "/api/scan":
            self.handle_api_scan()
        elif parsed_path == "/api/runs":
            self.handle_api_runs()
        elif parsed_path == "/api/select_run":
            self.handle_api_select_run()
        else:
            # Safely serve local static files from output/ strictly preventing Path Traversal
            try:
                candidate_path = (OUTPUT_DIR / parsed_path.lstrip("/")).resolve()
                output_root = OUTPUT_DIR.resolve()
                if candidate_path.is_relative_to(output_root) and candidate_path.is_file():
                    self.serve_static_file(candidate_path)
                else:
                    self.send_cors_headers(content_type="application/json", status=404)
                    self.wfile.write(json.dumps({"error": "Not Found", "path": parsed_path}).encode("utf-8"))
            except Exception:
                self.send_cors_headers(content_type="application/json", status=404)
                self.wfile.write(json.dumps({"error": "Not Found", "path": parsed_path}).encode("utf-8"))

    def handle_serve_gui(self) -> None:
        if not GUI_HTML_PATH.exists():
            # If not built, attempt to build or return placeholder
            from build_gui import build_standalone_gui
            build_standalone_gui(PAYLOAD_JSON_PATH, GUI_HTML_PATH)

        if GUI_HTML_PATH.exists():
            content = GUI_HTML_PATH.read_bytes()
            self.send_cors_headers(content_type="text/html; charset=utf-8", status=200)
            self.wfile.write(content)
        else:
            self.send_cors_headers(content_type="text/plain", status=500)
            self.wfile.write(b"Error: live_trfx_gui.html not found.")

    def handle_api_health(self) -> None:
        health_data = {
            "status": "ok",
            "server": "Forex Quant AI Lab Pro Terminal v3.5",
            "markets_count": 56,
            "port": self.server.server_address[1],
            "gui_available": GUI_HTML_PATH.exists(),
            "payload_available": PAYLOAD_JSON_PATH.exists(),
        }
        self.send_cors_headers(content_type="application/json; charset=utf-8", status=200)
        self.wfile.write(json.dumps(health_data, indent=2).encode("utf-8"))

    def handle_api_signals(self) -> None:
        if not PAYLOAD_JSON_PATH.exists():
            compile_56_markets_payload(output_json_path=PAYLOAD_JSON_PATH)

        if PAYLOAD_JSON_PATH.exists():
            content = PAYLOAD_JSON_PATH.read_bytes()
            self.send_cors_headers(content_type="application/json; charset=utf-8", status=200)
            self.wfile.write(content)
        else:
            self.send_cors_headers(content_type="application/json", status=500)
            self.wfile.write(json.dumps({"error": "Failed to compile signals payload"}).encode("utf-8"))

    def handle_api_scan(self) -> None:
        # Returns current radar matrix summary across 56 markets
        if not PAYLOAD_JSON_PATH.exists():
            compile_56_markets_payload(output_json_path=PAYLOAD_JSON_PATH)

        try:
            with open(PAYLOAD_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            radar = data.get("radar_matrix", {})
            scan_summary = {
                "generated_at": data.get("generated_at"),
                "total_markets": 56,
                "radar_matrix": radar,
                "active_signals": {
                    pair: {tf: state for tf, state in tf_map.items() if state in ("BUY", "SELL", "ARMED")}
                    for pair, tf_map in radar.items()
                },
            }
            self.send_cors_headers(content_type="application/json; charset=utf-8", status=200)
            self.wfile.write(json.dumps(scan_summary, indent=2).encode("utf-8"))
        except (json.JSONDecodeError, OSError, ValueError, KeyError) as e:
            self.send_cors_headers(content_type="application/json", status=500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def handle_api_runs(self) -> None:
        runs_dir = OUTPUT_DIR / "runs"
        runs_list = []

        current_run_id = None
        if PAYLOAD_JSON_PATH.exists():
            try:
                with open(PAYLOAD_JSON_PATH, "r", encoding="utf-8") as f:
                    cur_p = json.load(f)
                    current_run_id = cur_p.get("run_id")
            except (json.JSONDecodeError, OSError, KeyError):
                current_run_id = None

        if runs_dir.exists():
            for p in sorted(runs_dir.iterdir(), reverse=True):
                if p.is_dir() and p.name.startswith("run_"):
                    lb_file = p / "leaderboard" / "current.json"
                    cards = list(p.glob("experiments/*/card.md"))
                    cached = (p / "live_signals_multi_tf.json").exists()

                    n_cand = len(cards)
                    top_id = "—"
                    top_fam = "—"
                    top_status = "C"
                    top_fitness = 0.0
                    top_sharpe = 0.0

                    if lb_file.exists():
                        try:
                            lb_data = json.loads(lb_file.read_text(encoding="utf-8"))
                            if lb_data:
                                n_cand = len(lb_data)
                                top_item = lb_data[0]
                                top_id = top_item.get("strategy_id", top_id)
                                top_fam = top_item.get("family", top_fam)
                                top_status = top_item.get("status", top_status)
                                top_fitness = float(top_item.get("fitness", 0.0))
                                top_sharpe = float(top_item.get("sharpe_holdout", 0.0))
                        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                            pass
                    elif cards:
                        top_id = cards[0].parent.name

                    mtime = p.stat().st_mtime
                    dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

                    is_cur = (p.name == current_run_id) if current_run_id else False

                    runs_list.append({
                        "run_id": p.name,
                        "created_at": dt_str,
                        "candidates_count": n_cand,
                        "top_strategy": top_id,
                        "top_family": top_fam,
                        "top_status": top_status,
                        "top_fitness": round(top_fitness, 3),
                        "top_sharpe": round(top_sharpe, 2),
                        "cached": cached,
                        "is_current": is_cur,
                    })

        if runs_list and not any(r["is_current"] for r in runs_list):
            runs_list[0]["is_current"] = True

        self.send_cors_headers(content_type="application/json; charset=utf-8", status=200)
        self.wfile.write(json.dumps(runs_list, indent=2).encode("utf-8"))

    def handle_api_select_run(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        run_id = qs.get("run_id", [None])[0]

        if not run_id:
            self.send_cors_headers(content_type="application/json", status=400)
            self.wfile.write(json.dumps({"error": "Missing 'run_id' query parameter"}).encode("utf-8"))
            return

        if not re.match(r"^run_[a-zA-Z0-9_-]+$", run_id):
            self.send_cors_headers(content_type="application/json", status=400)
            self.wfile.write(json.dumps({"error": f"Invalid run_id format: {run_id}"}).encode("utf-8"))
            return

        run_dir = OUTPUT_DIR / "runs" / run_id
        if not run_dir.exists():
            self.send_cors_headers(content_type="application/json", status=404)
            self.wfile.write(json.dumps({"error": f"Run directory not found: {run_id}"}).encode("utf-8"))
            return

        try:
            payload = get_or_compile_run_payload(run_id, runs_dir=OUTPUT_DIR / "runs", timeframes_dir=OUTPUT_DIR / "timeframes")
            with open(PAYLOAD_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.send_cors_headers(content_type="application/json; charset=utf-8", status=200)
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            self.send_cors_headers(content_type="application/json", status=500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def serve_static_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".js": "application/javascript",
            ".css": "text/css",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }
        ct = content_types.get(suffix, "application/octet-stream")
        content = path.read_bytes()
        self.send_cors_headers(content_type=ct, status=200)
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        # Concise logging to stdout
        sys.stdout.write(f"[HTTP {self.address_string()}] {format % args}\n")
        sys.stdout.flush()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> http.server.HTTPServer:
    """
    Initializes and starts the HTTP server.
    """
    server_address = (host, port)
    server = ThreadingHTTPServer(server_address, ProTerminalHandler)
    print("\n========================================================")
    print("  PRO LIVE TERMINAL v3.5 SERVER STARTED")
    print(f"  URL: http://{host}:{port}/")
    print("  REST API Endpoints:")
    print(f"    - http://{host}:{port}/api/health")
    print(f"    - http://{host}:{port}/api/signals")
    print(f"    - http://{host}:{port}/api/scan")
    print("========================================================\n")
    return server


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Pro Live Terminal HTTP Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default: 8050)")
    args = parser.parse_args()

    server = start_server(host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[HTTP SERVER] Shutting down gracefully...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    run_cli()
