"""
Generator for Pro Live Terminal v3.5 HTML5 Canvas Single-Page Application (§ 10.3).
Generates output/live_trfx_gui.html with Dual-Pane Canvas (60 FPS),
TradingView-Style Anchored Zoom & Pan Engine, 600-Bar Historical Buffer,
Transaction Paths, What-If Simulator (<2ms in JS), Lot Calculator,
Global 56-Market Radar Matrix, Strategy Cards with SVG Sparklines, and Web Audio API alerts.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_gui_html(
    payload_json_path: Path = Path("output/live_signals_multi_tf.json"),
    output_html_path: Path = Path("output/live_trfx_gui.html"),
) -> None:
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Load compiled payload or use embedded minimal structure
    payload_data = {}
    if payload_json_path.exists():
        with open(payload_json_path, "r", encoding="utf-8") as f:
            payload_data = json.load(f)

    json_str = json.dumps(payload_data)

    html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRFX PRO TERMINAL v3.5 — Institutional TradingView Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        :root {{
            --bg-main: #090d16;
            --bg-panel: #111622;
            --bg-panel-elevated: #161c2b;
            --border-color: #1e2538;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
        }}
        body {{
            background-color: var(--bg-main);
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
            user-select: none;
        }}
        .badge-buy {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .badge-sell {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-arm {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .badge-neutral {{ background: rgba(100, 116, 139, 0.12); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.25); }}
        .card-active {{ border-color: #3b82f6 !important; background: rgba(59, 130, 246, 0.08) !important; box-shadow: 0 0 12px rgba(59, 130, 246, 0.15); }}
        .tab-btn-active {{ background: #1e2538; color: #ffffff; border-bottom: 2px solid #3b82f6; font-weight: 600; }}
        .pulse-dot {{
            animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}
        @keyframes pulse-ring {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.4; transform: scale(1.2); }}
        }}
        /* Custom scrollbar */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: #0c101a; }}
        ::-webkit-scrollbar-thumb {{ background: #1e2538; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #2a334d; }}
    </style>
</head>
<body class="h-screen overflow-hidden flex flex-col">

    <!-- TOP HEADER NAVBAR -->
    <header class="h-13 bg-[#111622] border-b border-[#1e2538] px-3.5 flex items-center justify-between z-20 shrink-0">
        <div class="flex items-center space-x-3">
            <!-- Brand -->
            <div class="flex items-center space-x-1.5 pr-2 border-r border-[#1e2538]">
                <span class="text-blue-500 font-black text-lg tracking-wider">TRFX</span>
                <span class="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold bg-blue-950/80 text-blue-300 border border-blue-800/40">PRO v3.5</span>
            </div>

            <!-- RUN SELECTOR -->
            <div class="flex items-center space-x-1.5 bg-[#161c2b] px-2 py-1 rounded-md border border-[#1e2538]">
                <span class="text-[10px] font-bold text-gray-400">RUN:</span>
                <select id="runSelector" class="bg-[#111622] text-xs text-yellow-400 font-mono font-bold border border-[#1e2538] rounded px-2 py-0.5 focus:outline-none focus:border-blue-500 cursor-pointer max-w-[200px] truncate" title="Wybierz run badawczy do załadowania">
                    <option value="">Wczytywanie runów...</option>
                </select>
                <span id="runSpinner" class="hidden text-xs text-blue-400 animate-spin" title="Przeliczanie rynków...">⏳</span>
                <span id="runBadge" class="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold bg-blue-950 text-blue-300 border border-blue-800/50">LIVE</span>
            </div>

            <!-- PAIR SELECTOR WITH SIGNAL STATUS DOTS -->
            <div class="flex items-center bg-[#161c2b] rounded-md border border-[#1e2538] p-0.5" id="pairButtons">
                <!-- Dynamically populated pairs -->
            </div>

            <!-- TIMEFRAME SELECTOR -->
            <div class="flex items-center space-x-1" id="tfButtons">
                <!-- Dynamically populated TFs -->
            </div>
        </div>

        <!-- CENTER / RIGHT CONTROLS & TICKER HUD -->
        <div class="flex items-center space-x-3.5">
            <!-- Ticker HUD -->
            <div class="flex items-center space-x-2 bg-[#161c2b] px-3 py-1 rounded-md border border-[#1e2538] text-xs font-mono" id="marketTickerHud">
                <span class="font-bold text-white text-sm" id="tickerPair">EURUSD</span>
                <span class="font-bold text-emerald-400 text-sm ml-1" id="tickerPrice">1.11760</span>
                <span class="text-gray-400 text-[11px] ml-2">Spread: <b id="tickerSpread" class="text-blue-300">1.1p</b></span>
            </div>

            <!-- Radar 56 Markets Button -->
            <button id="btnMatrix" class="px-2.5 py-1 bg-indigo-600/25 hover:bg-indigo-600/40 border border-indigo-500/40 rounded-md text-xs text-indigo-300 font-medium flex items-center space-x-1.5 transition">
                <span>🌐 Radar 56 (M)</span>
                <span id="radarActiveBadge" class="hidden text-[10px] px-1 py-0.2 rounded-full bg-emerald-500 text-white font-bold font-mono">0</span>
            </button>

            <!-- Sound alert toggle -->
            <button id="btnAudio" class="px-2 py-1 bg-[#161c2b] hover:bg-[#1e2538] border border-[#1e2538] rounded-md text-xs flex items-center space-x-1 text-gray-300 transition">
                <span id="audioIcon">🔇</span>
                <span>Dźwięk</span>
            </button>

            <!-- Clock -->
            <div class="text-xs text-gray-400 font-mono" id="serverClock">--:--:-- UTC</div>
        </div>
    </header>

    <!-- MAIN WORKSPACE -->
    <div class="flex-1 flex overflow-hidden">

        <!-- LEFT / CENTER DUAL-PANE CANVAS VIEW -->
        <div class="flex-1 flex flex-col min-w-0 bg-[#090d16] relative overflow-hidden">

            <!-- OVERLAYS TOOLBAR (CLEAN, MODERN WITH ZOOM WIDGET) -->
            <div class="h-9 bg-[#111622]/90 backdrop-blur border-b border-[#1e2538] px-3.5 flex items-center justify-between text-xs shrink-0 z-10">
                <div class="flex items-center space-x-3.5">
                    <label class="flex items-center space-x-1.5 cursor-pointer text-gray-300 hover:text-white select-none">
                        <input type="checkbox" id="chkEMA" checked class="rounded bg-[#161c2b] border-[#1e2538] text-blue-600 focus:ring-0">
                        <span>EMA 20/50</span>
                    </label>
                    <label class="flex items-center space-x-1.5 cursor-pointer text-gray-300 hover:text-white select-none">
                        <input type="checkbox" id="chkBB" class="rounded bg-[#161c2b] border-[#1e2538] text-blue-600 focus:ring-0">
                        <span>Bollinger Bands</span>
                    </label>
                    <label class="flex items-center space-x-1.5 cursor-pointer text-gray-300 hover:text-white select-none" title="Pokaż wejścia i wyjścia z pozycji">
                        <input type="checkbox" id="chkPaths" checked class="rounded bg-[#161c2b] border-[#1e2538] text-blue-600 focus:ring-0">
                        <span>Ścieżki Transakcji (+/-pips)</span>
                    </label>
                    <label class="flex items-center space-x-1.5 cursor-pointer text-gray-300 hover:text-white select-none" title="Rysuj strefy zysku i straty w trybie skupienia">
                        <input type="checkbox" id="chkBoxRR" checked class="rounded bg-[#161c2b] border-[#1e2538] text-blue-600 focus:ring-0">
                        <span>Strefy R:R (Focus / Hover)</span>
                    </label>

                    <!-- Trade filter selector -->
                    <div class="flex items-center space-x-1 ml-1 pl-2.5 border-l border-[#1e2538]">
                        <span class="text-[11px] text-gray-400">Pokaż:</span>
                        <select id="selTradeFilter" class="bg-[#161c2b] text-[11px] text-gray-200 border border-[#1e2538] rounded px-1.5 py-0.5 focus:outline-none cursor-pointer">
                            <option value="last5">Ostatnie 5 transakcji</option>
                            <option value="all">Wszystkie transakcje</option>
                            <option value="wins">Tylko zyskowne (TP)</option>
                        </select>
                    </div>

                    <!-- TradingView Zoom Controls -->
                    <div class="flex items-center space-x-1 ml-1 pl-2.5 border-l border-[#1e2538]">
                        <button id="btnZoomOut" class="w-5 h-5 rounded bg-[#161c2b] hover:bg-[#1e2538] border border-[#1e2538] text-gray-300 font-bold flex items-center justify-center text-xs transition" title="Oddal (Zoom Out / Kółko myszy w dół)">−</button>
                        <button id="btnZoomIn" class="w-5 h-5 rounded bg-[#161c2b] hover:bg-[#1e2538] border border-[#1e2538] text-gray-300 font-bold flex items-center justify-center text-xs transition" title="Przybliż (Zoom In / Kółko myszy w górę)">+</button>
                        <button id="btnResetView" class="px-1.5 h-5 rounded bg-[#161c2b] hover:bg-[#1e2538] border border-[#1e2538] text-gray-300 font-medium flex items-center justify-center text-[10px] transition" title="Przejdź do najnowszej świecy (Reset widoku)">⟲ Bieżąca</button>
                        <span id="zoomBarInfo" class="text-[10px] text-gray-400 font-mono ml-1">120 / 600 świec</span>
                    </div>
                </div>

                <!-- Live Candle OHLC Header -->
                <div class="text-[11px] text-gray-300 font-mono tracking-tight" id="candleInfo">
                    O: <span class="text-white">--</span>  H: <span class="text-white">--</span>  L: <span class="text-white">--</span>  C: <span class="text-white">--</span> | Spread: --
                </div>
            </div>

            <!-- CANVAS CONTAINER -->
            <div class="flex-1 flex flex-col min-h-0 w-full relative overflow-hidden" id="canvasWrapper">
                <!-- TOP CANVAS (75% HEIGHT): CANDLES + PATHS -->
                <div class="flex-[3] min-h-0 w-full relative overflow-hidden bg-[#090d16]">
                    <canvas id="upperCanvas" class="absolute inset-0 w-full h-full block cursor-crosshair"></canvas>
                </div>

                <!-- BOTTOM CANVAS (25% HEIGHT): SYNCHRONIZED EQUITY & DRAWDOWN -->
                <div class="flex-[1] min-h-0 w-full border-t border-[#1e2538] relative bg-[#0c101a] flex flex-col overflow-hidden">
                    <!-- Bottom Pane Control Header -->
                    <div class="h-6 bg-[#111622]/95 border-b border-[#1e2538] px-3 flex items-center justify-between text-[10px] text-gray-400 font-mono z-10 shrink-0">
                        <div class="flex items-center space-x-2">
                            <span class="font-bold text-gray-300 uppercase tracking-wider">Krzywa Wyników:</span>
                            <div class="flex space-x-1 bg-[#161c2b] rounded p-0.5">
                                <button id="btnViewEquity" class="px-2 py-0.5 rounded text-[10px] font-bold bg-[#1e2538] text-blue-400">📈 Equity %</button>
                                <button id="btnViewDrawdown" class="px-2 py-0.5 rounded text-[10px] text-gray-400 hover:text-white">📉 Drawdown %</button>
                            </div>
                        </div>

                        <!-- Live Metrics HUD Synchronized with active strategy -->
                        <div class="flex items-center space-x-3" id="lowerHudStats">
                            <span>Zwrot: <b id="eqMetricRet" class="text-emerald-400 font-bold">--%</b></span>
                            <span>Max DD: <b id="eqMetricDD" class="text-rose-400 font-bold">--%</b></span>
                            <span>Win Rate: <b id="eqMetricWR" class="text-blue-400 font-bold">--%</b></span>
                            <span>Trades: <b id="eqMetricTrades" class="text-gray-300 font-bold">--</b></span>
                            <span>Sharpe: <b id="eqMetricSharpe" class="text-purple-400 font-bold">--</b></span>
                        </div>
                    </div>

                    <div class="flex-1 relative w-full h-full min-h-0 overflow-hidden">
                        <canvas id="lowerCanvas" class="absolute inset-0 w-full h-full block cursor-crosshair"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT SIDEBAR: TABBED WORKSPACE (STRATEGIES, WHAT-IF, CALCULATOR) -->
        <div class="w-84 bg-[#111622] border-l border-[#1e2538] flex flex-col overflow-hidden shrink-0">

            <!-- TAB HEADER SWITCHER -->
            <div class="flex border-b border-[#1e2538] bg-[#0d111a] text-xs font-semibold shrink-0">
                <button id="tabBtnStrategies" class="flex-1 py-2.5 text-center tab-btn-active transition flex items-center justify-center space-x-1">
                    <span>🎯 Strategie</span>
                    <span id="stratCountBadge" class="text-[10px] px-1.5 py-0.2 rounded-full bg-blue-600/30 text-blue-300 font-mono">0</span>
                </button>
                <button id="tabBtnWhatIf" class="flex-1 py-2.5 text-center text-gray-400 hover:text-white transition flex items-center justify-center space-x-1">
                    <span>⚡ What-If</span>
                </button>
                <button id="tabBtnCalculator" class="flex-1 py-2.5 text-center text-gray-400 hover:text-white transition flex items-center justify-center space-x-1">
                    <span>🧮 Kalkulator</span>
                </button>
            </div>

            <!-- TAB 1: STRATEGIES LIST -->
            <div id="tabContentStrategies" class="flex-1 flex flex-col min-h-0 p-3 overflow-hidden">
                <!-- Search & Filter Bar -->
                <div class="flex flex-col space-y-2 mb-2.5 shrink-0">
                    <div class="flex items-center justify-between">
                        <div class="flex space-x-1 bg-[#161c2b] p-0.5 rounded text-[11px]">
                            <button id="filterAll" class="px-2 py-0.5 rounded font-medium bg-[#1e2538] text-white">Wszystkie</button>
                            <button id="filterSignals" class="px-2 py-0.5 rounded font-medium text-gray-400 hover:text-white">Sygnały</button>
                            <button id="filterProfitable" class="px-2 py-0.5 rounded font-medium text-gray-400 hover:text-white">Zyskowne</button>
                        </div>
                        <label class="flex items-center space-x-1 text-[10px] text-gray-400 cursor-pointer hover:text-white select-none">
                            <input type="checkbox" id="chkHideEmpty" checked class="rounded bg-[#161c2b] border-[#1e2538] text-blue-600 focus:ring-0">
                            <span>Ukryj puste</span>
                        </label>
                    </div>
                </div>

                <!-- Scrollable Strategy Cards Container -->
                <div class="flex-1 overflow-y-auto space-y-2 pr-1" id="strategyCardsContainer">
                    <!-- Populated dynamically by JavaScript -->
                </div>
            </div>

            <!-- TAB 2: WHAT-IF SIMULATOR -->
            <div id="tabContentWhatIf" class="flex-1 flex flex-col min-h-0 p-3.5 overflow-y-auto hidden space-y-4">
                <div class="flex items-center justify-between pb-2 border-b border-[#1e2538]">
                    <span class="text-xs font-bold text-yellow-500 uppercase tracking-wider flex items-center space-x-1">
                        <span>⚡ Symulator Parametrów</span>
                    </span>
                    <span class="text-[10px] text-gray-400 font-mono">&lt;2ms JS Engine</span>
                </div>

                <div class="space-y-3.5 text-xs">
                    <div>
                        <div class="flex justify-between text-gray-300 mb-1.5 font-medium">
                            <span>Stop-Loss (x ATR)</span>
                            <span id="lblSlAtr" class="font-mono text-blue-400 font-bold">1.5</span>
                        </div>
                        <input type="range" id="sliderSlAtr" min="0.5" max="4.0" step="0.1" value="1.5" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
                    </div>

                    <div>
                        <div class="flex justify-between text-gray-300 mb-1.5 font-medium">
                            <span>Take-Profit (x ATR)</span>
                            <span id="lblTpAtr" class="font-mono text-emerald-400 font-bold">2.5</span>
                        </div>
                        <input type="range" id="sliderTpAtr" min="0.5" max="6.0" step="0.1" value="2.5" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-500">
                    </div>

                    <div>
                        <div class="flex justify-between text-gray-300 mb-1.5 font-medium">
                            <span>Ryzyko / Transakcję (%)</span>
                            <span id="lblRiskPct" class="font-mono text-purple-400 font-bold">1.0%</span>
                        </div>
                        <input type="range" id="sliderRiskPct" min="0.2" max="5.0" step="0.1" value="1.0" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500">
                    </div>

                    <!-- SIMULATOR OUTCOME METRICS -->
                    <div class="pt-3 border-t border-[#1e2538]">
                        <div class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2">Przeliczone Wyniki Symulacji</div>
                        <div class="grid grid-cols-2 gap-2 text-center font-mono">
                            <div class="bg-[#161c2b] p-2.5 rounded-lg border border-[#1e2538]">
                                <div class="text-[10px] text-gray-400">Win Rate</div>
                                <div id="simWinRate" class="text-base font-bold text-emerald-400">--%</div>
                            </div>
                            <div class="bg-[#161c2b] p-2.5 rounded-lg border border-[#1e2538]">
                                <div class="text-[10px] text-gray-400">Profit Factor</div>
                                <div id="simPf" class="text-base font-bold text-blue-400">--</div>
                            </div>
                            <div class="bg-[#161c2b] p-2.5 rounded-lg border border-[#1e2538]">
                                <div class="text-[10px] text-gray-400">Net Return</div>
                                <div id="simReturn" class="text-base font-bold text-emerald-400">--%</div>
                            </div>
                            <div class="bg-[#161c2b] p-2.5 rounded-lg border border-[#1e2538]">
                                <div class="text-[10px] text-gray-400">Liczba Pozycji</div>
                                <div id="simTrades" class="text-base font-bold text-gray-200">--</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 3: POSITION SIZING & LOT CALCULATOR -->
            <div id="tabContentCalculator" class="flex-1 flex flex-col min-h-0 p-3.5 overflow-y-auto hidden space-y-4">
                <div class="pb-2 border-b border-[#1e2538]">
                    <div class="text-xs font-bold text-gray-300 uppercase tracking-wider">🧮 Kalkulator Wielkości Pozycji</div>
                    <div class="text-[10px] text-gray-400 mt-0.5">Precyzyjne przeliczenie lotów z walutą kwotowaną na USD</div>
                </div>

                <div class="space-y-3 text-xs">
                    <div class="flex justify-between items-center bg-[#161c2b] p-2 rounded-lg border border-[#1e2538]">
                        <span class="text-gray-300 font-medium">Kapitał Rachunku ($):</span>
                        <input type="number" id="calcEquity" value="10000" class="w-24 bg-[#111622] border border-[#1e2538] rounded px-2 py-1 text-right font-mono text-white focus:outline-none focus:border-blue-500">
                    </div>

                    <div class="flex justify-between items-center bg-[#161c2b] p-2 rounded-lg border border-[#1e2538]">
                        <span class="text-gray-300 font-medium">Odległość SL (pips):</span>
                        <input type="number" id="calcSlPips" value="25.0" step="0.5" class="w-24 bg-[#111622] border border-[#1e2538] rounded px-2 py-1 text-right font-mono text-white focus:outline-none focus:border-blue-500">
                    </div>

                    <!-- Calculated Result Card -->
                    <div class="p-3.5 rounded-xl bg-gradient-to-br from-[#161c2b] to-[#1c2336] border border-blue-500/30 mt-2">
                        <div class="text-[11px] font-semibold text-gray-300 uppercase tracking-wider">Rekomendowany Wolumen:</div>
                        <div class="flex items-baseline justify-between mt-1">
                            <span id="calcLots" class="text-2xl font-black font-mono text-emerald-400">0.40 Lot</span>
                            <span id="calcRiskCash" class="text-xs text-gray-300 font-mono font-bold">Ryzyko: $100.00</span>
                        </div>

                        <div class="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-[#1e2538] text-[10px] text-gray-400 font-mono">
                            <div>Wartość 1 pipsa: <b id="calcPipValue" class="text-gray-200">$10.00</b></div>
                            <div>Depozyt (1:30): <b id="calcMarginReq" class="text-gray-200">~$1,333</b></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- 56 MARKETS RADAR MATRIX MODAL -->
    <div id="modalMatrix" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
        <div class="bg-[#111622] border border-[#1e2538] rounded-xl max-w-4xl w-full p-5 shadow-2xl flex flex-col max-h-[90vh]">
            <div class="flex justify-between items-center pb-3 border-b border-[#1e2538]">
                <div>
                    <h2 class="text-base font-bold text-white flex items-center space-x-2">
                        <span>🌐 Globalna Macierz Radarowa 56 Rynków (7 Par × 8 TF)</span>
                    </h2>
                    <p class="text-xs text-gray-400 mt-0.5">Kliknij dowolną komórkę, aby natychmiast załadować wykres i przeliczyć strategię.</p>
                </div>
                <button id="btnCloseMatrix" class="text-gray-400 hover:text-white text-xl font-bold px-2 py-1">&times;</button>
            </div>

            <!-- RADAR MATRIX GRID TABLE -->
            <div class="flex-1 overflow-auto my-4">
                <table class="w-full text-xs text-center border-collapse" id="matrixTable">
                    <thead>
                        <tr class="bg-[#161c2b] text-gray-300">
                            <th class="p-2.5 border border-[#1e2538] text-left">Para FX</th>
                            <th class="p-2.5 border border-[#1e2538]">M1</th>
                            <th class="p-2.5 border border-[#1e2538]">M5</th>
                            <th class="p-2.5 border border-[#1e2538]">M15</th>
                            <th class="p-2.5 border border-[#1e2538]">H1</th>
                            <th class="p-2.5 border border-[#1e2538]">H4</th>
                            <th class="p-2.5 border border-[#1e2538]">D1</th>
                            <th class="p-2.5 border border-[#1e2538]">W1</th>
                            <th class="p-2.5 border border-[#1e2538]">MN</th>
                        </tr>
                    </thead>
                    <tbody id="matrixTableBody">
                        <!-- Populated dynamically -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- CLIENT JAVASCRIPT ENGINE -->
    <script>
        const PAYLOAD = {json_str};

        let currentPair = "EURUSD";
        let currentTf = "H1";
        let currentStrategyId = PAYLOAD.strategy_ids ? PAYLOAD.strategy_ids[0] : "";
        let audioEnabled = false;
        let audioCtx = null;
        let lowerPaneMode = "equity"; // "equity" or "drawdown"
        let activeTab = "strategies"; // "strategies", "whatif", "calculator"
        let currentFilter = "all"; // "all", "signals", "profitable"

        // TRADINGVIEW VIEWPORT STATE (MATHEMATICALLY BOUNDED PAN & ZOOM)
        let viewOffset = 0;          // Bars scrolled back into history (0 = at the newest bar)
        let visibleBars = 120;        // Visible candle count on screen (range 20 to 500)
        const rightPaddingBars = 6;  // Space on right edge before price axis (TradingView standard)
        let isDragging = false;
        let dragStartX = 0;
        let dragStartOffset = 0;
        let hoveredTradeIndex = -1;

        // Web Audio API Synth
        function playAlertSound(type) {{
            if (!audioEnabled) return;
            try {{
                if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (audioCtx.state === 'suspended') audioCtx.resume();

                const now = audioCtx.currentTime;
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.connect(gain);
                gain.connect(audioCtx.destination);

                if (type === "BUY") {{
                    osc.frequency.setValueAtTime(523.25, now); // C5
                    osc.frequency.setValueAtTime(659.25, now + 0.1); // E5
                    osc.frequency.setValueAtTime(783.99, now + 0.2); // G5
                    gain.gain.setValueAtTime(0.15, now);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
                    osc.start(now);
                    osc.stop(now + 0.5);
                }} else if (type === "SELL") {{
                    osc.frequency.setValueAtTime(440, now);
                    osc.frequency.setValueAtTime(330, now + 0.15);
                    gain.gain.setValueAtTime(0.15, now);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
                    osc.start(now);
                    osc.stop(now + 0.4);
                }} else if (type === "ARMED") {{
                    osc.frequency.setValueAtTime(587.33, now); // D5
                    osc.frequency.setValueAtTime(880, now + 0.1); // A5
                    gain.gain.setValueAtTime(0.10, now);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.6);
                    osc.start(now);
                    osc.stop(now + 0.6);
                }}
            }} catch(e) {{
                console.error("Audio error:", e);
            }}
        }}

        // Load Available Runs from Server API (§ 10.2)
        async function loadAvailableRuns() {{
            const selector = document.getElementById("runSelector");
            if (!selector) return;
            try {{
                const res = await fetch("/api/runs");
                if (!res.ok) throw new Error("Status " + res.status);
                const runs = await res.json();
                if (!Array.isArray(runs) || runs.length === 0) return;

                selector.innerHTML = "";
                let selectedVal = PAYLOAD.run_id || "";

                runs.forEach(r => {{
                    const opt = document.createElement("option");
                    opt.value = r.run_id;
                    const cachedMark = r.cached ? "⚡" : "⚙️";
                    const statusMark = r.top_status ? `[Tier ${{r.top_status}}]` : "";
                    const countTxt = r.candidates_count > 0 ? `${{r.candidates_count}} strat` : "brak";
                    opt.textContent = `${{cachedMark}} ${{r.run_id}} | ${{statusMark}} ${{r.top_strategy}} (Fit: ${{r.top_fitness}}, ${{countTxt}})`;
                    if (r.is_current || r.run_id === selectedVal) {{
                        opt.selected = true;
                        selectedVal = r.run_id;
                    }}
                    selector.appendChild(opt);
                }});

                if (selectedVal) {{
                    selector.value = selectedVal;
                    const badge = document.getElementById("runBadge");
                    if (badge) badge.textContent = selectedVal.replace("run_", "R_");
                }}
            }} catch (err) {{
                console.warn("Run selector offline or standalone mode:", err);
                selector.innerHTML = `<option value="${{PAYLOAD.run_id || 'latest'}}">${{PAYLOAD.run_id || 'LATEST ACTIVE'}}</option>`;
            }}
        }}

        async function switchRun(runId) {{
            if (!runId) return;
            const spinner = document.getElementById("runSpinner");
            const toast = document.getElementById("loadingToast");
            const toastTxt = document.getElementById("loadingToastText");

            if (spinner) spinner.classList.remove("hidden");
            if (toast) {{
                toastTxt.textContent = `Przeliczanie 56 rynków dla ${{runId}}...`;
                toast.classList.remove("opacity-0");
            }}

            try {{
                const res = await fetch(`/api/select_run?run_id=${{encodeURIComponent(runId)}}`);
                if (!res.ok) throw new Error(`Błąd HTTP ${{res.status}}`);
                const newPayload = await res.json();

                // Update properties of global PAYLOAD
                Object.assign(PAYLOAD, newPayload);
                PAYLOAD.markets = newPayload.markets;
                PAYLOAD.radar_matrix = newPayload.radar_matrix;
                PAYLOAD.strategies = newPayload.strategies;
                PAYLOAD.strategy_ids = newPayload.strategy_ids;
                PAYLOAD.strategy_meta = newPayload.strategy_meta || {{}};
                PAYLOAD.pair_signals = newPayload.pair_signals || {{}};
                PAYLOAD.run_id = newPayload.run_id;

                if (!PAYLOAD.strategy_ids || !PAYLOAD.strategy_ids.includes(currentStrategyId)) {{
                    currentStrategyId = (PAYLOAD.strategy_ids && PAYLOAD.strategy_ids.length > 0) ? PAYLOAD.strategy_ids[0] : "";
                }}

                const badge = document.getElementById("runBadge");
                if (badge) badge.textContent = runId.replace("run_", "R_");

                // Reset viewport offset to newest bar
                viewOffset = 0;

                // Re-render UI
                renderPairButtons();
                renderStrategyCards();
                renderMatrixTable();
                onWhatIfChange();
                updateToolbarZoomInfo();
                drawCanvases();

                if (toastTxt) toastTxt.textContent = `✅ Załadowano ${{runId}} (${{PAYLOAD.strategy_ids?.length || 0}} strategii)`;
                playAlertSound("BUY");
            }} catch (err) {{
                console.error("Błąd przełączania runa:", err);
                if (toastTxt) toastTxt.textContent = `❌ Błąd ładowania: ${{err.message}}`;
            }} finally {{
                if (spinner) spinner.classList.add("hidden");
                if (toast) {{
                    setTimeout(() => {{
                        toast.classList.add("opacity-0");
                    }}, 2200);
                }}
            }}
        }}

        // Render Pair Selector with Signal Indicator Dots
        function renderPairButtons() {{
            const pairContainer = document.getElementById("pairButtons");
            if (!pairContainer) return;
            pairContainer.innerHTML = "";

            (PAYLOAD.pairs || ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURGBP", "EURJPY"]).forEach(p => {{
                const btn = document.createElement("button");
                const isSelected = (p === currentPair);
                const pSignal = PAYLOAD.pair_signals ? PAYLOAD.pair_signals[p] : "NEUTRAL";

                let dotHtml = "";
                if (pSignal === "BUY") {{
                    dotHtml = `<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 inline-block pulse-dot"></span>`;
                }} else if (pSignal === "SELL") {{
                    dotHtml = `<span class="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5 inline-block pulse-dot"></span>`;
                }} else if (pSignal === "ARMED") {{
                    dotHtml = `<span class="w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5 inline-block"></span>`;
                }}

                btn.className = `px-2.5 py-1 text-xs font-semibold rounded flex items-center transition ${{isSelected ? 'bg-blue-600 text-white font-bold' : 'text-gray-400 hover:text-white'}}`;
                btn.innerHTML = `${{dotHtml}}<span>${{p}}</span>`;
                btn.onclick = () => {{ currentPair = p; updateMarket(); }};
                pairContainer.appendChild(btn);
            }});
        }}

        function updateToolbarZoomInfo() {{
            const mKey = `${{currentPair}}_${{currentTf}}`;
            const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;
            const total = mData && mData.candles ? mData.candles.length : 0;
            const el = document.getElementById("zoomBarInfo");
            if (el) {{
                el.textContent = `${{visibleBars}} / ${{total}} świec`;
            }}
        }}

        function zoomByFactor(factor, clientX = -1) {{
            const mKey = `${{currentPair}}_${{currentTf}}`;
            const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;
            if (!mData || !mData.candles || mData.candles.length === 0) return;
            const total = mData.candles.length;

            const uCanvas = document.getElementById("upperCanvas");
            const rect = uCanvas.getBoundingClientRect();
            const mouseX = (clientX >= 0) ? (clientX - rect.left) : (rect.width * 0.5);
            const dpr = window.devicePixelRatio || 1;
            const chartW = Math.max(40, (uCanvas.width / dpr) - 72);
            const mouseRatio = Math.max(0, Math.min(1, mouseX / chartW));

            // Current bar under anchor
            const currentEnd = total - viewOffset;
            const currentStart = Math.max(0, currentEnd - visibleBars);
            const barUnderAnchor = currentStart + mouseRatio * (currentEnd - currentStart);

            let newVisible = Math.round(visibleBars * factor);
            newVisible = Math.max(20, Math.min(Math.min(500, total), newVisible));

            if (newVisible !== visibleBars) {{
                visibleBars = newVisible;
                const newStart = barUnderAnchor - mouseRatio * visibleBars;
                const newEnd = newStart + visibleBars;
                const maxOffset = Math.max(0, total - Math.min(total, visibleBars));
                viewOffset = Math.max(0, Math.min(maxOffset, Math.round(total - newEnd)));
                updateToolbarZoomInfo();
                drawCanvases();
            }}
        }}

        // Initialize UI Elements
        function initUI() {{
            // 1. Pairs
            renderPairButtons();

            // 2. Timeframes
            const tfContainer = document.getElementById("tfButtons");
            tfContainer.innerHTML = "";
            (PAYLOAD.timeframes || ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN"]).forEach(tf => {{
                const btn = document.createElement("button");
                btn.className = `px-2 py-1 text-xs rounded border border-[#1e2538] transition ${{tf === currentTf ? 'bg-[#1e2538] text-white font-bold' : 'text-gray-400 hover:text-white'}}`;
                btn.textContent = tf;
                btn.onclick = () => {{ currentTf = tf; updateMarket(); }};
                tfContainer.appendChild(btn);
            }});

            // 3. Audio toggle
            document.getElementById("btnAudio").onclick = () => {{
                audioEnabled = !audioEnabled;
                document.getElementById("audioIcon").textContent = audioEnabled ? "🔊" : "🔇";
                if (audioEnabled && !audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }};

            // 4. Matrix Modal
            const modal = document.getElementById("modalMatrix");
            document.getElementById("btnMatrix").onclick = () => modal.classList.remove("hidden");
            document.getElementById("btnCloseMatrix").onclick = () => modal.classList.add("hidden");

            // 5. Tab Switching
            setupTabNavigation();

            // 6. Strategy Filtering
            setupStrategyFilters();

            // 7. Lower Pane View Switcher
            document.getElementById("btnViewEquity").onclick = () => setLowerPaneMode("equity");
            document.getElementById("btnViewDrawdown").onclick = () => setLowerPaneMode("drawdown");

            // 8. Sliders What-If Realtime Listener
            ["sliderSlAtr", "sliderTpAtr", "sliderRiskPct"].forEach(id => {{
                document.getElementById(id).addEventListener("input", onWhatIfChange);
            }});

            // 9. Calculator Inputs
            ["calcEquity", "calcSlPips"].forEach(id => {{
                document.getElementById(id).addEventListener("input", updateLotCalculator);
            }});

            // 10. Run Selector Listener
            const runSel = document.getElementById("runSelector");
            if (runSel) {{
                runSel.onchange = () => switchRun(runSel.value);
            }}
            loadAvailableRuns();

            // 11. Zoom Controls in Toolbar
            document.getElementById("btnZoomIn").onclick = () => zoomByFactor(0.82);
            document.getElementById("btnZoomOut").onclick = () => zoomByFactor(1.22);
            document.getElementById("btnResetView").onclick = () => {{
                viewOffset = 0;
                visibleBars = 120;
                updateToolbarZoomInfo();
                drawCanvases();
            }};

            // 12. Keyboard Controls (Arrow keys, Home, End, Zoom)
            window.addEventListener("keydown", (e) => {{
                if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

                if (e.key === "m" || e.key === "M") modal.classList.toggle("hidden");
                if (e.key >= "1" && e.key <= "8") {{
                    const tfs = ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN"];
                    currentTf = tfs[parseInt(e.key) - 1];
                    updateMarket();
                }}

                const mKey = `${{currentPair}}_${{currentTf}}`;
                const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;
                const total = mData && mData.candles ? mData.candles.length : 0;
                const maxOffset = Math.max(0, total - Math.min(total, visibleBars));

                if (e.key === "ArrowLeft") {{
                    // Pan back in history (left)
                    viewOffset = Math.min(maxOffset, viewOffset + 6);
                    drawCanvases();
                }} else if (e.key === "ArrowRight") {{
                    // Pan forward (right)
                    viewOffset = Math.max(0, viewOffset - 6);
                    drawCanvases();
                }} else if (e.key === "ArrowUp" || e.key === "+") {{
                    zoomByFactor(0.85);
                }} else if (e.key === "ArrowDown" || e.key === "-") {{
                    zoomByFactor(1.18);
                }} else if (e.key === "End") {{
                    viewOffset = 0;
                    drawCanvases();
                }} else if (e.key === "Home") {{
                    viewOffset = maxOffset;
                    drawCanvases();
                }}
            }});

            // 13. Canvas Mouse Controls (TradingView Pan & Zoom Engine)
            setupCanvasInteractions();

            // 14. Checkboxes listener for canvas redraw
            ["chkEMA", "chkBB", "chkPaths", "chkBoxRR", "selTradeFilter"].forEach(id => {{
                const el = document.getElementById(id);
                if (el) el.addEventListener("change", () => drawCanvases());
            }});

            updateMarket();
            renderMatrixTable();
            updateToolbarZoomInfo();
        }}

        function setupTabNavigation() {{
            const tabs = [
                {{ id: "strategies", btn: document.getElementById("tabBtnStrategies"), content: document.getElementById("tabContentStrategies") }},
                {{ id: "whatif", btn: document.getElementById("tabBtnWhatIf"), content: document.getElementById("tabContentWhatIf") }},
                {{ id: "calculator", btn: document.getElementById("tabBtnCalculator"), content: document.getElementById("tabContentCalculator") }},
            ];

            tabs.forEach(t => {{
                t.btn.onclick = () => {{
                    activeTab = t.id;
                    tabs.forEach(other => {{
                        if (other.id === t.id) {{
                            other.btn.className = "flex-1 py-2.5 text-center tab-btn-active transition flex items-center justify-center space-x-1";
                            other.content.classList.remove("hidden");
                        }} else {{
                            other.btn.className = "flex-1 py-2.5 text-center text-gray-400 hover:text-white transition flex items-center justify-center space-x-1";
                            other.content.classList.add("hidden");
                        }}
                    }});
                }};
            }});
        }}

        function setupStrategyFilters() {{
            const filters = [
                {{ id: "filterAll", type: "all" }},
                {{ id: "filterSignals", type: "signals" }},
                {{ id: "filterProfitable", type: "profitable" }},
            ];

            filters.forEach(f => {{
                const btn = document.getElementById(f.id);
                btn.onclick = () => {{
                    currentFilter = f.type;
                    filters.forEach(o => {{
                        const oBtn = document.getElementById(o.id);
                        if (o.type === f.type) {{
                            oBtn.className = "px-2 py-0.5 rounded font-medium bg-[#1e2538] text-white";
                        }} else {{
                            oBtn.className = "px-2 py-0.5 rounded font-medium text-gray-400 hover:text-white";
                        }}
                    }});
                    renderStrategyCards();
                }};
            }});

            document.getElementById("chkHideEmpty").onchange = () => renderStrategyCards();
        }}

        function setLowerPaneMode(mode) {{
            lowerPaneMode = mode;
            const bEq = document.getElementById("btnViewEquity");
            const bDd = document.getElementById("btnViewDrawdown");
            if (mode === "equity") {{
                bEq.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-[#1e2538] text-blue-400";
                bDd.className = "px-2 py-0.5 rounded text-[10px] text-gray-400 hover:text-white";
            }} else {{
                bDd.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-[#1e2538] text-rose-400";
                bEq.className = "px-2 py-0.5 rounded text-[10px] text-gray-400 hover:text-white";
            }}
            drawCanvases();
        }}

        function onWhatIfChange() {{
            const slAtr = parseFloat(document.getElementById("sliderSlAtr").value);
            const tpAtr = parseFloat(document.getElementById("sliderTpAtr").value);
            const riskPct = parseFloat(document.getElementById("sliderRiskPct").value);

            document.getElementById("lblSlAtr").textContent = slAtr.toFixed(1);
            document.getElementById("lblTpAtr").textContent = tpAtr.toFixed(1);
            document.getElementById("lblRiskPct").textContent = riskPct.toFixed(1) + "%";

            // Instantly re-simulate transaction paths in <2ms
            recalculateWhatIfSimulation(slAtr, tpAtr, riskPct);
            drawCanvases();
            updateLotCalculator();
        }}

        function recalculateWhatIfSimulation(slAtr, tpAtr, riskPct) {{
            const mKey = `${{currentPair}}_${{currentTf}}`;
            const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;
            if (!mData || !mData.candles) return;

            const candles = mData.candles;
            const stratData = mData.strategies[currentStrategyId];
            if (!stratData || !stratData.chart_markers) return;

            const pipMult = mData.pip_mult || 10000.0;
            const pipUnit = mData.pip_unit || 0.0001;

            let wins = 0;
            let losses = 0;
            let grossProfit = 0;
            let grossLoss = 0;
            let currentEquity = 100000;

            stratData.chart_markers.forEach(m => {{
                const eIdx = m.index;
                if (eIdx >= candles.length - 1) return;

                const isLong = (m.type === "BUY");
                const ePrice = candles[eIdx].open;

                // 20-bar causal ATR prior to signal bar
                let atrSum = 0;
                let atrCount = 0;
                for (let a = Math.max(0, eIdx - 20); a < eIdx; a++) {{
                    const prevC = (a > 0) ? candles[a - 1].close : candles[a].open;
                    const tr = Math.max(
                        candles[a].high - candles[a].low,
                        Math.max(Math.abs(candles[a].high - prevC), Math.abs(candles[a].low - prevC))
                    );
                    atrSum += tr;
                    atrCount++;
                }}
                const barAtr = atrCount > 0 ? Math.max(pipUnit * 10, atrSum / atrCount) : Math.max(pipUnit * 10, candles[eIdx].high - candles[eIdx].low);

                const slPrice = isLong ? (ePrice - slAtr * barAtr) : (ePrice + slAtr * barAtr);
                const tpPrice = isLong ? (ePrice + tpAtr * barAtr) : (ePrice - tpAtr * barAtr);

                const maxHoldBars = stratData.max_holding_bars || 24;
                const lastBarIdx = Math.min(candles.length - 1, eIdx + maxHoldBars);
                let exitPrice = candles[lastBarIdx].close;

                for (let k = eIdx + 1; k <= lastBarIdx; k++) {{
                    const kh = candles[k].high;
                    const kl = candles[k].low;
                    if (isLong) {{
                        if (kl <= slPrice) {{ exitPrice = slPrice; break; }}
                        if (kh >= tpPrice) {{ exitPrice = tpPrice; break; }}
                    }} else {{
                        if (kh >= slPrice) {{ exitPrice = slPrice; break; }}
                        if (kl <= tpPrice) {{ exitPrice = tpPrice; break; }}
                    }}
                }}

                const pnlPips = (isLong ? (exitPrice - ePrice) : (ePrice - exitPrice)) * pipMult;
                const slPips = Math.max(1.0, slAtr * barAtr * pipMult);
                const riskCash = currentEquity * (riskPct / 100.0);
                const tradeUsd = riskCash * (pnlPips / slPips);
                currentEquity += tradeUsd;

                if (pnlPips > 0) {{
                    wins++;
                    grossProfit += pnlPips;
                }} else {{
                    losses++;
                    grossLoss += Math.abs(pnlPips);
                }}
            }});

            const total = wins + losses;
            const wr = total > 0 ? (wins / total * 100) : 0;
            const pf = grossLoss > 0 ? (grossProfit / grossLoss) : (grossProfit > 0 ? 99 : 0);
            const totRet = ((currentEquity - 100000) / 100000) * 100;

            document.getElementById("simWinRate").textContent = wr.toFixed(1) + "%";
            document.getElementById("simPf").textContent = pf.toFixed(2);
            document.getElementById("simReturn").textContent = (totRet >= 0 ? "+" : "") + totRet.toFixed(1) + "%";
            document.getElementById("simTrades").textContent = total;
        }}

        function updateLotCalculator() {{
            const equity = parseFloat(document.getElementById("calcEquity").value) || 10000;
            const slPips = parseFloat(document.getElementById("calcSlPips").value) || 25;
            const riskPct = parseFloat(document.getElementById("sliderRiskPct").value) || 1.0;

            const riskCash = equity * (riskPct / 100.0);
            let pipValuePerLot = 10.0; // standard for USD quote

            // Dynamic live quote-to-USD conversion from PAYLOAD market candles
            if (currentPair.includes("JPY")) {{
                let usdjpyRate = 150.0;
                const mkt = (PAYLOAD.markets && PAYLOAD.markets["USDJPY_" + currentTf]) || (PAYLOAD.markets && PAYLOAD.markets["USDJPY_H1"]);
                if (mkt && mkt.candles && mkt.candles.length > 0) {{
                    usdjpyRate = mkt.candles[mkt.candles.length - 1].close || 150.0;
                }}
                pipValuePerLot = 1000.0 / usdjpyRate;
            }} else if (currentPair.includes("CHF")) {{
                let usdchfRate = 0.90;
                const mkt = (PAYLOAD.markets && PAYLOAD.markets["USDCHF_" + currentTf]) || (PAYLOAD.markets && PAYLOAD.markets["USDCHF_H1"]);
                if (mkt && mkt.candles && mkt.candles.length > 0) {{
                    usdchfRate = mkt.candles[mkt.candles.length - 1].close || 0.90;
                }}
                pipValuePerLot = 10.0 / usdchfRate;
            }} else if (currentPair === "EURGBP") {{
                let gbpusdRate = 1.30;
                const mkt = (PAYLOAD.markets && PAYLOAD.markets["GBPUSD_" + currentTf]) || (PAYLOAD.markets && PAYLOAD.markets["GBPUSD_H1"]);
                if (mkt && mkt.candles && mkt.candles.length > 0) {{
                    gbpusdRate = mkt.candles[mkt.candles.length - 1].close || 1.30;
                }}
                pipValuePerLot = 10.0 * gbpusdRate;
            }}

            const lots = (riskCash / (slPips * pipValuePerLot));
            document.getElementById("calcLots").textContent = Math.max(0.01, lots).toFixed(2) + " Lot";
            document.getElementById("calcRiskCash").textContent = `Ryzyko: $${{riskCash.toFixed(2)}}`;
            document.getElementById("calcPipValue").textContent = `$${{pipValuePerLot.toFixed(2)}} / lot`;
            document.getElementById("calcMarginReq").textContent = `~$${{(lots * 100000 / 30).toFixed(0)}}`;
        }}

        function updateMarket() {{
            renderPairButtons();
            document.querySelectorAll("#tfButtons button").forEach(b => {{
                b.className = b.textContent === currentTf ? 'px-2 py-1 text-xs rounded border border-[#1e2538] bg-[#1e2538] text-white font-bold' : 'px-2 py-1 text-xs rounded border border-[#1e2538] text-gray-400 hover:text-white';
            }});

            const mKey = `${{currentPair}}_${{currentTf}}`;
            const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;

            if (mData && mData.candles && mData.candles.length > 0) {{
                const lastC = mData.candles[mData.candles.length - 1];
                const dec = mData.dec || 5;
                document.getElementById("tickerPair").textContent = currentPair;
                document.getElementById("tickerPrice").textContent = lastC.close.toFixed(dec);
                document.getElementById("tickerSpread").textContent = `${{lastC.spread}}p`;
            }}

            renderStrategyCards();
            onWhatIfChange();
            updateToolbarZoomInfo();
            drawCanvases();
        }}

        // Mini SVG Sparkline generator for Strategy Cards
        function generateSvgSparkline(pts, isPositive) {{
            if (!pts || pts.length < 2) {{
                return `<svg class="w-full h-7 text-gray-600" viewBox="0 0 100 24"><line x1="0" y1="12" x2="100" y2="12" stroke="currentColor" stroke-dasharray="2,2"/></svg>`;
            }}

            const min = Math.min(...pts);
            const max = Math.max(...pts);
            const span = Math.max(0.01, max - min);
            const w = 100;
            const h = 24;

            let pathD = "";
            pts.forEach((p, i) => {{
                const x = (i / (pts.length - 1)) * w;
                const y = h - ((p - min) / span) * (h - 4) - 2;
                pathD += (i === 0 ? "M" : "L") + `${{x.toFixed(1)}},${{y.toFixed(1)}} `;
            }});

            const color = isPositive ? "#10b981" : "#ef4444";
            return `
                <svg class="w-full h-7 overflow-visible" viewBox="0 0 100 24">
                    <path d="${{pathD}}" fill="none" stroke="${{color}}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            `;
        }}

        function renderStrategyCards() {{
            const container = document.getElementById("strategyCardsContainer");
            container.innerHTML = "";
            const mKey = `${{currentPair}}_${{currentTf}}`;
            const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;
            const hideEmpty = document.getElementById("chkHideEmpty")?.checked;

            let visibleCount = 0;

            (PAYLOAD.strategy_ids || []).forEach((sId, idx) => {{
                const sMeta = mData ? mData.strategies[sId] : null;
                const gMeta = PAYLOAD.strategy_meta ? PAYLOAD.strategy_meta[sId] : null;
                const state = sMeta ? sMeta.live_state : "NEUTRAL";
                const isSelected = (sId === currentStrategyId);
                const nTrades = sMeta && sMeta.metrics ? sMeta.metrics.n_trades : 0;
                const retPct = sMeta && sMeta.metrics ? sMeta.metrics.total_return_pct : 0.0;

                // Filtering rules
                if (hideEmpty && nTrades === 0 && state === "NEUTRAL") return;
                if (currentFilter === "signals" && state === "NEUTRAL") return;
                if (currentFilter === "profitable" && retPct <= 0) return;

                visibleCount++;

                let badgeClass = "badge-neutral";
                if (state === "BUY") badgeClass = "badge-buy font-bold";
                if (state === "SELL") badgeClass = "badge-sell font-bold";
                if (state === "ARMED") badgeClass = "badge-arm font-bold";

                // Family icon
                const fam = gMeta ? gMeta.family : "F1_Trend";
                let famIcon = "⚡ Trend";
                if (fam.includes("Breakout")) famIcon = "💥 Breakout";
                if (fam.includes("MeanRev")) famIcon = "🔄 MeanRev";
                if (fam.includes("Candle")) famIcon = "🕯️ Candle";

                const tier = gMeta ? gMeta.tier : "C";
                let tierBadge = `<span class="text-[9px] px-1 py-0.2 rounded font-mono font-bold bg-purple-950 text-purple-300 border border-purple-800/40">Tier ${{tier}}</span>`;
                if (tier === "A") tierBadge = `<span class="text-[9px] px-1 py-0.2 rounded font-mono font-bold bg-amber-950 text-amber-300 border border-amber-800/40">Tier A</span>`;
                if (tier === "B") tierBadge = `<span class="text-[9px] px-1 py-0.2 rounded font-mono font-bold bg-blue-950 text-blue-300 border border-blue-800/40">Tier B</span>`;

                const sparkSvg = generateSvgSparkline(sMeta?.sparkline, retPct >= 0);

                const card = document.createElement("div");
                card.className = `p-2.5 rounded-lg border border-[#1e2538] cursor-pointer transition select-none ${{isSelected ? 'card-active' : 'bg-[#161c2b]/70 hover:bg-[#161c2b]'}}`;
                card.onclick = () => {{
                    currentStrategyId = sId;
                    renderStrategyCards();
                    onWhatIfChange();
                }};

                card.innerHTML = `
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-1.5">
                            <span class="font-bold text-xs text-white">${{sId}}</span>
                            <span class="text-[9px] px-1 py-0.2 rounded text-gray-400 bg-gray-800">${{famIcon}}</span>
                            ${{tierBadge}}
                        </div>
                        <span class="text-[10px] px-1.5 py-0.5 rounded font-mono ${{badgeClass}}">${{state}}</span>
                    </div>

                    <div class="my-1.5">
                        ${{sparkSvg}}
                    </div>

                    <div class="flex justify-between items-center text-[10px] text-gray-400 font-mono border-t border-[#1e2538] pt-1.5">
                        <span>WR: <b class="${{sMeta?.metrics?.win_rate >= 0.5 ? 'text-emerald-400' : 'text-gray-300'}}">${{sMeta ? (sMeta.metrics.win_rate*100).toFixed(0) : '0'}}%</b></span>
                        <span>PF: <b class="${{sMeta?.metrics?.profit_factor >= 1.0 ? 'text-blue-400' : 'text-gray-300'}}">${{sMeta ? sMeta.metrics.profit_factor.toFixed(2) : '0.00'}}</b></span>
                        <span>Zwrot: <b class="${{retPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}}">${{retPct >= 0 ? '+' : ''}}${{retPct.toFixed(1)}}%</b></span>
                        <span>Poz: <b>${{nTrades}}</b></span>
                    </div>
                `;
                container.appendChild(card);
            }});

            const countBadge = document.getElementById("stratCountBadge");
            if (countBadge) countBadge.textContent = visibleCount;
        }}

        function renderMatrixTable() {{
            const tbody = document.getElementById("matrixTableBody");
            tbody.innerHTML = "";
            const tfs = ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN"];
            let activeOpportunities = 0;

            (PAYLOAD.pairs || []).forEach(p => {{
                const tr = document.createElement("tr");
                tr.className = "hover:bg-[#161c2b]/60 transition";
                let html = `<td class="p-2.5 border border-[#1e2538] text-left font-bold text-white">${{p}}</td>`;

                tfs.forEach(tf => {{
                    const state = PAYLOAD.radar_matrix && PAYLOAD.radar_matrix[p] ? PAYLOAD.radar_matrix[p][tf] : "NEUTRAL";
                    let badgeClass = "text-gray-500";
                    let symbol = "•";
                    if (state === "BUY") {{ badgeClass = "text-emerald-400 font-bold bg-emerald-950/30"; symbol = "▲ BUY"; activeOpportunities++; }}
                    if (state === "SELL") {{ badgeClass = "text-rose-400 font-bold bg-rose-950/30"; symbol = "▼ SELL"; activeOpportunities++; }}
                    if (state === "ARMED") {{ badgeClass = "text-amber-400 font-bold bg-amber-950/30"; symbol = "🟠 ARM"; activeOpportunities++; }}

                    html += `<td class="p-2 border border-[#1e2538] cursor-pointer ${{badgeClass}}" onclick="switchFromMatrix('${{p}}', '${{tf}}')">${{symbol}}</td>`;
                }});

                tr.innerHTML = html;
                tbody.appendChild(tr);
            }});

            const rBadge = document.getElementById("radarActiveBadge");
            if (rBadge) {{
                rBadge.textContent = activeOpportunities;
                if (activeOpportunities > 0) rBadge.classList.remove("hidden");
                else rBadge.classList.add("hidden");
            }}
        }}

        window.switchFromMatrix = function(p, tf) {{
            currentPair = p;
            currentTf = tf;
            document.getElementById("modalMatrix").classList.add("hidden");
            updateMarket();
        }};

        // Canvas Rendering Core
        let hoverX = -1;
        let hoverY = -1;

        function setupCanvasInteractions() {{
            const uCanvas = document.getElementById("upperCanvas");
            const lCanvas = document.getElementById("lowerCanvas");

            window.addEventListener("resize", () => {{ resizeCanvases(); drawCanvases(); }});
            resizeCanvases();

            // TRADINGVIEW PAN ENGINE (MOUSE DOWN -> DRAG -> MOUSE UP)
            uCanvas.addEventListener("mousedown", (e) => {{
                isDragging = true;
                dragStartX = e.clientX;
                dragStartOffset = viewOffset;
                uCanvas.style.cursor = "grabbing";
            }});

            window.addEventListener("mouseup", () => {{
                if (isDragging) {{
                    isDragging = false;
                    uCanvas.style.cursor = "crosshair";
                }}
            }});

            uCanvas.addEventListener("mousemove", (e) => {{
                const rect = uCanvas.getBoundingClientRect();
                hoverX = e.clientX - rect.left;
                hoverY = e.clientY - rect.top;

                if (isDragging) {{
                    const dx = e.clientX - dragStartX;
                    const dpr = window.devicePixelRatio || 1;
                    const chartW = Math.max(40, (uCanvas.width / dpr) - 72);
                    const curCandleW = chartW / (visibleBars + rightPaddingBars);

                    // Moving mouse RIGHT (dx > 0) pulls past candles right -> viewOffset increases (looking at older candles)
                    // Moving mouse LEFT (dx < 0) pulls future candles left -> viewOffset decreases (looking at newer candles)
                    const barsShifted = Math.round(dx / curCandleW);

                    const mKey = `${{currentPair}}_${{currentTf}}`;
                    const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;
                    const total = mData && mData.candles ? mData.candles.length : 0;
                    const maxOffset = Math.max(0, total - Math.min(total, visibleBars));

                    viewOffset = Math.max(0, Math.min(maxOffset, dragStartOffset + barsShifted));
                }}
                drawCanvases();
            }});

            uCanvas.addEventListener("mouseleave", () => {{
                hoverX = -1;
                hoverY = -1;
                document.getElementById("candleInfo").innerHTML = "O: <span class='text-white'>--</span>  H: <span class='text-white'>--</span>  L: <span class='text-white'>--</span>  C: <span class='text-white'>--</span> | Spread: --";
                drawCanvases();
            }});

            // ANCHORED MOUSE WHEEL ZOOM (TRADINGVIEW STANDARD)
            uCanvas.addEventListener("wheel", (e) => {{
                e.preventDefault();
                zoomByFactor(e.deltaY < 0 ? 0.85 : 1.18, e.clientX);
            }}, {{ passive: false }});
        }}

        function resizeCanvases() {{
            const uCanvas = document.getElementById("upperCanvas");
            const lCanvas = document.getElementById("lowerCanvas");
            const dpr = window.devicePixelRatio || 1;

            const uRect = uCanvas.getBoundingClientRect();
            uCanvas.width = Math.floor(uRect.width * dpr);
            uCanvas.height = Math.floor(uRect.height * dpr);

            const lRect = lCanvas.getBoundingClientRect();
            lCanvas.width = Math.floor(lRect.width * dpr);
            lCanvas.height = Math.floor(lRect.height * dpr);
        }}

        function drawCanvases() {{
            const mKey = `${{currentPair}}_${{currentTf}}`;
            const mData = PAYLOAD.markets ? PAYLOAD.markets[mKey] : null;
            if (!mData || !mData.candles) return;

            const candles = mData.candles;
            const stratData = mData.strategies[currentStrategyId];

            drawUpperPane(candles, stratData, mData);
            drawLowerPane(candles, stratData, mData);
        }}

        function calcEMA(candles, period) {{
            const res = [];
            const k = 2 / (period + 1);
            let prev = candles[0].close;
            res.push(prev);
            for (let i = 1; i < candles.length; i++) {{
                const val = candles[i].close * k + prev * (1 - k);
                res.push(val);
                prev = val;
            }}
            return res;
        }}

        function calcBollingerBands(candles, period, mult) {{
            const upper = [];
            const lower = [];
            const middle = [];

            for (let i = 0; i < candles.length; i++) {{
                if (i < period - 1) {{
                    middle.push(candles[i].close);
                    upper.push(candles[i].close);
                    lower.push(candles[i].close);
                    continue;
                }}
                let sum = 0;
                for (let j = 0; j < period; j++) sum += candles[i - j].close;
                const mean = sum / period;
                let varSum = 0;
                for (let j = 0; j < period; j++) varSum += Math.pow(candles[i - j].close - mean, 2);
                const std = Math.sqrt(varSum / period);
                middle.push(mean);
                upper.push(mean + mult * std);
                lower.push(mean - mult * std);
            }}
            return {{ upper, lower, middle }};
        }}

        function drawUpperPane(candles, stratData, mData) {{
            const canvas = document.getElementById("upperCanvas");
            const ctx = canvas.getContext("2d");
            const dpr = window.devicePixelRatio || 1;
            const w = canvas.width / dpr;
            const h = canvas.height / dpr;
            const dec = mData.dec || 5;

            ctx.save();
            ctx.scale(dpr, dpr);
            ctx.clearRect(0, 0, w, h);

            // Hard clip bounds
            ctx.beginPath();
            ctx.rect(0, 0, w, h);
            ctx.clip();

            const priceAxisW = 72;
            const chartW = Math.max(40, w - priceAxisW);

            // TRADINGVIEW BOUNDED VIEWPORT SLICE
            const total = candles.length;
            const maxOffset = Math.max(0, total - Math.min(total, visibleBars));
            viewOffset = Math.max(0, Math.min(maxOffset, viewOffset));

            const endIndex = Math.min(total, total - viewOffset);
            const startIndex = Math.max(0, endIndex - visibleBars);
            const viewCandles = candles.slice(startIndex, endIndex);

            if (viewCandles.length === 0) {{ ctx.restore(); return; }}

            // Candle width is STRICTLY fixed by visibleBars + rightPaddingBars (never ballooning on drag!)
            const candleW = chartW / (visibleBars + rightPaddingBars);

            // Price scale min/max strictly from visible candles
            let minP = 1e9, maxP = -1e9;
            viewCandles.forEach(c => {{
                if (c.low < minP) minP = c.low;
                if (c.high > maxP) maxP = c.high;
            }});

            const pipUnit = mData.pip_unit || 0.0001;
            const span = Math.max(pipUnit * 20, maxP - minP);
            const pad = span * 0.12;
            minP -= pad;
            maxP += pad;

            const priceToY = (p) => {{
                const y = h * (1 - (p - minP) / (maxP - minP));
                return Math.max(4, Math.min(h - 4, y));
            }};

            // Background grid lines (Institutional subtle dark)
            ctx.strokeStyle = "#141a28";
            ctx.lineWidth = 1;
            for (let i = 0; i <= 6; i++) {{
                const y = Math.round(h * (i / 6));
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(chartW, y);
                ctx.stroke();
            }}

            // Right Price Axis Background
            ctx.fillStyle = "#0c101a";
            ctx.fillRect(chartW, 0, priceAxisW, h);
            ctx.strokeStyle = "#1e2538";
            ctx.beginPath();
            ctx.moveTo(chartW, 0);
            ctx.lineTo(chartW, h);
            ctx.stroke();

            // Price Labels on Right Scale
            ctx.fillStyle = "#787b86";
            ctx.font = "10px monospace";
            ctx.textAlign = "left";
            for (let i = 1; i <= 5; i++) {{
                const y = h * (i / 6);
                const pVal = maxP - (i / 6) * (maxP - minP);
                ctx.fillText(pVal.toFixed(dec), chartW + 8, y + 3);
            }}

            // Bollinger Bands
            if (document.getElementById("chkBB").checked && candles.length >= 20) {{
                const bb = calcBollingerBands(candles, 20, 2.0);
                // Fill
                ctx.fillStyle = "rgba(59, 130, 246, 0.04)";
                ctx.beginPath();
                for (let i = 0; i < viewCandles.length; i++) {{
                    const globalIdx = startIndex + i;
                    const x = i * candleW + candleW * 0.5;
                    const y = priceToY(bb.upper[globalIdx]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                for (let i = viewCandles.length - 1; i >= 0; i--) {{
                    const globalIdx = startIndex + i;
                    const x = i * candleW + candleW * 0.5;
                    const y = priceToY(bb.lower[globalIdx]);
                    ctx.lineTo(x, y);
                }}
                ctx.closePath();
                ctx.fill();

                // Upper & lower lines
                ctx.strokeStyle = "rgba(59, 130, 246, 0.35)";
                ctx.lineWidth = 1.0;
                ctx.beginPath();
                for (let i = 0; i < viewCandles.length; i++) {{
                    const globalIdx = startIndex + i;
                    const x = i * candleW + candleW * 0.5;
                    const y = priceToY(bb.upper[globalIdx]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();

                ctx.beginPath();
                for (let i = 0; i < viewCandles.length; i++) {{
                    const globalIdx = startIndex + i;
                    const x = i * candleW + candleW * 0.5;
                    const y = priceToY(bb.lower[globalIdx]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();
            }}

            // EMA 20 & EMA 50
            if (document.getElementById("chkEMA").checked && candles.length >= 20) {{
                const ema20All = calcEMA(candles, 20);
                const ema50All = calcEMA(candles, 50);

                // EMA 20 (Cyan / Blue)
                ctx.strokeStyle = "#3b82f6";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (let i = 0; i < viewCandles.length; i++) {{
                    const globalIdx = startIndex + i;
                    const x = i * candleW + candleW * 0.5;
                    const y = priceToY(ema20All[globalIdx]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();

                // EMA 50 (Amber)
                ctx.strokeStyle = "#f59e0b";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (let i = 0; i < viewCandles.length; i++) {{
                    const globalIdx = startIndex + i;
                    const x = i * candleW + candleW * 0.5;
                    const y = priceToY(ema50All[globalIdx]);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();
            }}

            // Draw Candlesticks (Clean Institutional Styling)
            viewCandles.forEach((c, i) => {{
                const x = i * candleW + candleW * 0.5;
                const isGreen = c.close >= c.open;
                const color = isGreen ? "#10b981" : "#ef4444";

                const yOpen = priceToY(c.open);
                const yClose = priceToY(c.close);
                const yHigh = priceToY(c.high);
                const yLow = priceToY(c.low);

                // Wick
                ctx.strokeStyle = color;
                ctx.lineWidth = Math.max(1, Math.min(2, candleW * 0.15));
                ctx.beginPath();
                ctx.moveTo(x, yHigh);
                ctx.lineTo(x, yLow);
                ctx.stroke();

                // Body
                const bodyTop = Math.min(yOpen, yClose);
                const bodyH = Math.max(2, Math.abs(yClose - yOpen));
                ctx.fillStyle = color;
                ctx.fillRect(x - candleW * 0.36, bodyTop, Math.max(1, candleW * 0.72), bodyH);
            }});

            // Draw Transaction Paths & R:R Boxes (INSTITUTIONAL ZERO-CLUTTER ENGINE)
            if (document.getElementById("chkPaths").checked && stratData && stratData.transaction_paths) {{
                const chkBox = document.getElementById("chkBoxRR").checked;
                const filterMode = document.getElementById("selTradeFilter")?.value || "last5";

                let pathsToDraw = stratData.transaction_paths;
                if (filterMode === "wins") {{
                    pathsToDraw = pathsToDraw.filter(p => p.is_win);
                }} else if (filterMode === "last5") {{
                    pathsToDraw = pathsToDraw.slice(-5);
                }}

                // Find active or hovered trade to highlight R:R box
                let targetBoxTrade = (pathsToDraw.length > 0) ? pathsToDraw[pathsToDraw.length - 1] : null;

                pathsToDraw.forEach((tp, tpIdx) => {{
                    const relEntry = tp.entry_idx - startIndex;
                    const relExit = tp.exit_idx - startIndex;

                    // Visible in window?
                    if (relEntry < viewCandles.length && relExit >= 0) {{
                        const entryClamped = Math.max(0, relEntry);
                        const exitClamped = Math.min(viewCandles.length - 1, relExit);

                        const x1 = entryClamped * candleW + candleW * 0.5;
                        const y1 = priceToY(tp.entry_price);
                        const x2 = exitClamped * candleW + candleW * 0.5;
                        const y2 = priceToY(tp.exit_price);

                        const isWin = tp.is_win;
                        const isSelectedFocus = (chkBox && (tp === targetBoxTrade || tpIdx === hoveredTradeIndex));

                        // 1. Subtle, elegant Focus R:R Box (ONLY for selected/last trade, never 50 overlapping blocks!)
                        if (isSelectedFocus) {{
                            const boxW = Math.max(8, x2 - x1);
                            const topY = Math.min(y1, y2);
                            const boxH = Math.max(4, Math.abs(y2 - y1));

                            ctx.fillStyle = isWin ? "rgba(16, 185, 129, 0.09)" : "rgba(239, 68, 68, 0.09)";
                            ctx.fillRect(x1, topY, boxW, boxH);

                            ctx.strokeStyle = isWin ? "rgba(16, 185, 129, 0.4)" : "rgba(239, 68, 68, 0.4)";
                            ctx.lineWidth = 1;
                            ctx.strokeRect(x1, topY, boxW, boxH);
                        }}

                        // 2. Dashed Transaction Connector Path
                        ctx.strokeStyle = isWin ? "#10b981" : "#ef4444";
                        ctx.lineWidth = isSelectedFocus ? 1.8 : 1.2;
                        ctx.setLineDash([3, 3]);
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        // 3. Entry Triangle Marker
                        if (relEntry >= 0 && relEntry < viewCandles.length) {{
                            const isLong = (tp.type === "BUY");
                            ctx.fillStyle = isLong ? "#10b981" : "#ef4444";
                            ctx.beginPath();
                            if (isLong) {{
                                const triY = y1 + 10;
                                ctx.moveTo(x1, triY);
                                ctx.lineTo(x1 - 4, triY + 7);
                                ctx.lineTo(x1 + 4, triY + 7);
                            }} else {{
                                const triY = y1 - 10;
                                ctx.moveTo(x1, triY);
                                ctx.lineTo(x1 - 4, triY - 7);
                                ctx.lineTo(x1 + 4, triY - 7);
                            }}
                            ctx.closePath();
                            ctx.fill();
                        }}

                        // 4. Exit Mark & Anti-Collision Pips Badge
                        if (relExit >= 0 && relExit < viewCandles.length) {{
                            // Exit cross 'x'
                            ctx.strokeStyle = isWin ? "#10b981" : "#ef4444";
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.moveTo(x2 - 3, y2 - 3);
                            ctx.lineTo(x2 + 3, y2 + 3);
                            ctx.moveTo(x2 + 3, y2 - 3);
                            ctx.lineTo(x2 - 3, y2 + 3);
                            ctx.stroke();

                            // Clean badge with rounded corners
                            const badgeTxt = (tp.pnl_pips >= 0 ? "+" : "") + tp.pnl_pips.toFixed(1) + "p";
                            ctx.font = "bold 9px monospace";
                            const txtW = ctx.measureText(badgeTxt).width;
                            const bW = txtW + 8;
                            const bH = 14;
                            const bX = Math.min(chartW - bW - 2, Math.max(2, x2 - bW / 2));
                            const bY = Math.min(h - bH - 4, Math.max(4, y2 - 18));

                            ctx.fillStyle = isWin ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)";
                            ctx.fillRect(bX, bY, bW, bH);
                            ctx.strokeStyle = isWin ? "#10b981" : "#ef4444";
                            ctx.lineWidth = 1;
                            ctx.strokeRect(bX, bY, bW, bH);

                            ctx.fillStyle = isWin ? "#10b981" : "#ef4444";
                            ctx.textAlign = "center";
                            ctx.fillText(badgeTxt, bX + bW / 2, bY + 10);
                        }}
                    }}
                }});
            }}

            // Latest Price Marker on Axis (With glowing pulse tag)
            const lastCandle = viewCandles[viewCandles.length - 1];
            const lastY = priceToY(lastCandle.close);
            const isLastGreen = lastCandle.close >= lastCandle.open;

            ctx.fillStyle = isLastGreen ? "#10b981" : "#ef4444";
            ctx.fillRect(chartW + 1, lastY - 9, priceAxisW - 2, 18);
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 10px monospace";
            ctx.textAlign = "center";
            ctx.fillText(lastCandle.close.toFixed(dec), chartW + (priceAxisW / 2), lastY + 3);

            // Crosshair & Tooltip
            if (hoverX >= 0 && hoverX < chartW && hoverY >= 0 && hoverY < h) {{
                ctx.strokeStyle = "#475569";
                ctx.lineWidth = 0.8;
                ctx.setLineDash([2, 2]);

                // Vertical
                ctx.beginPath();
                ctx.moveTo(hoverX, 0);
                ctx.lineTo(hoverX, h);
                ctx.stroke();

                // Horizontal
                ctx.beginPath();
                ctx.moveTo(0, hoverY);
                ctx.lineTo(chartW, hoverY);
                ctx.stroke();
                ctx.setLineDash([]);

                // Hovered price label on axis
                const hoverP = maxP - (hoverY / h) * (maxP - minP);
                ctx.fillStyle = "#3b82f6";
                ctx.fillRect(chartW + 1, hoverY - 8, priceAxisW - 2, 16);
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 9px monospace";
                ctx.textAlign = "center";
                ctx.fillText(hoverP.toFixed(dec), chartW + (priceAxisW / 2), hoverY + 3);

                // Update candleInfo header
                const hIdx = Math.floor(hoverX / candleW);
                if (hIdx >= 0 && hIdx < viewCandles.length) {{
                    const hc = viewCandles[hIdx];
                    const isCGreen = hc.close >= hc.open;
                    const cColor = isCGreen ? "text-emerald-400" : "text-rose-400";
                    document.getElementById("candleInfo").innerHTML =
                        `O: <span class="text-white">${{hc.open.toFixed(dec)}}</span>  H: <span class="text-white">${{hc.high.toFixed(dec)}}</span>  L: <span class="text-white">${{hc.low.toFixed(dec)}}</span>  C: <span class="${{cColor}} font-bold">${{hc.close.toFixed(dec)}}</span> | Spread: <span class="text-blue-300 font-bold">${{hc.spread}}p</span> | <span class="text-gray-400">${{hc.time.substring(0, 16)}}</span>`;
                }}
            }}

            ctx.restore();
        }}

        function drawLowerPane(candles, stratData, mData) {{
            const canvas = document.getElementById("lowerCanvas");
            const ctx = canvas.getContext("2d");
            const dpr = window.devicePixelRatio || 1;
            const w = canvas.width / dpr;
            const h = canvas.height / dpr;

            ctx.save();
            ctx.scale(dpr, dpr);
            ctx.clearRect(0, 0, w, h);

            ctx.beginPath();
            ctx.rect(0, 0, w, h);
            ctx.clip();

            const priceAxisW = 72;
            const chartW = Math.max(40, w - priceAxisW);

            // Synchronized bar slice matching upper pane exactly
            const total = candles.length;
            const maxOffset = Math.max(0, total - Math.min(total, visibleBars));
            viewOffset = Math.max(0, Math.min(maxOffset, viewOffset));

            const endIndex = Math.min(total, total - viewOffset);
            const startIndex = Math.max(0, endIndex - visibleBars);
            const candleW = chartW / (visibleBars + rightPaddingBars);

            // Axis gutter background
            ctx.fillStyle = "#0c101a";
            ctx.fillRect(chartW, 0, priceAxisW, h);
            ctx.strokeStyle = "#1e2538";
            ctx.beginPath();
            ctx.moveTo(chartW, 0);
            ctx.lineTo(chartW, h);
            ctx.stroke();

            // Update Lower Pane HUD Metrics
            if (stratData && stratData.metrics) {{
                const m = stratData.metrics;
                const ret = m.total_return_pct;
                document.getElementById("eqMetricRet").textContent = (ret >= 0 ? "+" : "") + ret.toFixed(2) + "%";
                document.getElementById("eqMetricRet").className = ret >= 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold";
                document.getElementById("eqMetricDD").textContent = m.max_dd_pct.toFixed(2) + "%";
                document.getElementById("eqMetricWR").textContent = (m.win_rate * 100).toFixed(0) + "%";
                document.getElementById("eqMetricTrades").textContent = m.n_trades;
                document.getElementById("eqMetricSharpe").textContent = m.sharpe.toFixed(2);
            }}

            // Extract visible series (1:1 synchronized with candle indices)
            let series = [];
            if (stratData) {{
                if (lowerPaneMode === "drawdown" && stratData.drawdown_series) {{
                    series = stratData.drawdown_series.slice(startIndex, endIndex);
                }} else if (stratData.equity_series) {{
                    series = stratData.equity_series.slice(startIndex, endIndex);
                }}
            }}

            if (series.length < 2) {{
                // Baseline 0.00%
                ctx.strokeStyle = "#252d40";
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(0, h * 0.5);
                ctx.lineTo(chartW, h * 0.5);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = "#787b86";
                ctx.font = "9px monospace";
                ctx.textAlign = "left";
                ctx.fillText("0.00%", chartW + 8, h * 0.5 + 3);
                ctx.restore();
                return;
            }}

            // Scale calculations
            let minVal = Math.min(0, ...series);
            let maxVal = Math.max(0, ...series);
            if (lowerPaneMode === "drawdown") {{
                minVal = 0;
                maxVal = Math.max(1.0, ...series);
            }}

            const valSpan = Math.max(0.2, maxVal - minVal);
            const pad = valSpan * 0.15;
            const topVal = maxVal + pad;
            const botVal = minVal - pad;

            const valToY = (val) => {{
                return h * (1 - (val - botVal) / (topVal - botVal));
            }};

            const zeroY = valToY(0.0);

            // Baseline 0.00%
            ctx.strokeStyle = "#252d40";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(0, zeroY);
            ctx.lineTo(chartW, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = "#787b86";
            ctx.font = "9px monospace";
            ctx.textAlign = "left";
            ctx.fillText("0.00%", chartW + 8, Math.max(10, Math.min(h - 4, zeroY + 3)));

            // Draw Area under curve
            const isProfit = (series[series.length - 1] >= 0);
            const grad = ctx.createLinearGradient(0, 0, 0, h);

            if (lowerPaneMode === "drawdown") {{
                grad.addColorStop(0, "rgba(239, 68, 68, 0.0)");
                grad.addColorStop(1, "rgba(239, 68, 68, 0.25)");
            }} else {{
                grad.addColorStop(0, isProfit ? "rgba(16, 185, 129, 0.22)" : "rgba(239, 68, 68, 0.22)");
                grad.addColorStop(1, "rgba(16, 185, 129, 0.0)");
            }}

            ctx.beginPath();
            ctx.moveTo(0, zeroY);
            series.forEach((val, idx) => {{
                const x = idx * candleW + candleW * 0.5;
                const y = valToY(val);
                ctx.lineTo(x, y);
            }});
            const lastSeriesX = (series.length - 1) * candleW + candleW * 0.5;
            ctx.lineTo(lastSeriesX, zeroY);
            ctx.closePath();
            ctx.fillStyle = grad;
            ctx.fill();

            // Draw Curve Line
            ctx.strokeStyle = lowerPaneMode === "drawdown" ? "#ef4444" : (isProfit ? "#10b981" : "#ef4444");
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            series.forEach((val, idx) => {{
                const x = idx * candleW + candleW * 0.5;
                const y = valToY(val);
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();

            // Latest Value Tag on Right Axis
            const finalVal = series[series.length - 1];
            const finalY = valToY(finalVal);
            const tagTxt = (finalVal >= 0 ? "+" : "") + finalVal.toFixed(2) + "%";

            ctx.fillStyle = (lowerPaneMode === "drawdown") ? "#ef4444" : (finalVal >= 0 ? "#10b981" : "#ef4444");
            ctx.fillRect(chartW + 1, finalY - 8, priceAxisW - 2, 16);
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 9px monospace";
            ctx.textAlign = "center";
            ctx.fillText(tagTxt, chartW + (priceAxisW / 2), finalY + 3);

            // Synchronized Crosshair with Upper Pane
            if (hoverX >= 0 && hoverX < chartW) {{
                ctx.strokeStyle = "#475569";
                ctx.lineWidth = 0.8;
                ctx.setLineDash([2, 2]);
                ctx.beginPath();
                ctx.moveTo(hoverX, 0);
                ctx.lineTo(hoverX, h);
                ctx.stroke();
                ctx.setLineDash([]);

                const hIdx = Math.floor(hoverX / candleW);
                if (hIdx >= 0 && hIdx < series.length) {{
                    const hVal = series[hIdx];
                    const hY = valToY(hVal);
                    ctx.fillStyle = "#3b82f6";
                    ctx.fillRect(chartW + 1, hY - 7, priceAxisW - 2, 14);
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "bold 9px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText((hVal >= 0 ? "+" : "") + hVal.toFixed(2) + "%", chartW + (priceAxisW / 2), hY + 3);
                }}
            }}

            ctx.restore();
        }}

        // Clock
        setInterval(() => {{
            const d = new Date();
            document.getElementById("serverClock").textContent = d.toISOString().substring(11, 19) + " UTC";
        }}, 1000);

        window.onload = initUI;
    </script>

    <!-- TOAST NOTIFICATION -->
    <div id="loadingToast" class="fixed bottom-4 right-4 bg-[#161c2b] border border-blue-500 text-white text-xs px-4 py-2.5 rounded-lg shadow-2xl flex items-center space-x-3 transition-opacity duration-300 opacity-0 pointer-events-none z-50">
        <span class="animate-spin text-base">⚙️</span>
        <span id="loadingToastText">Przeliczanie strategii...</span>
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[BUILD_GUI] Generated Pro Live Terminal HTML5 Canvas application -> {output_html_path}")


build_standalone_gui = generate_gui_html


if __name__ == "__main__":
    generate_gui_html()
