"""
Standalone lightweight HTTP server delivering a web-based dashboard
without external UI dependencies.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import numpy as np

from robotics_platform.fleet.fleet_simulator import FleetSimulator
from robotics_platform.ingest.pipeline import IngestionPipeline


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    pipeline = IngestionPipeline()
    fleet = FleetSimulator(num_robots=12, seed=42)
    current_time = 0.0

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = self.render_html()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            summary = self.pipeline.get_metrics_summary()
            self.wfile.write(json.dumps(summary).encode("utf-8"))
        elif self.path == "/api/step":
            # Advance simulation 1 second
            events = self.fleet.simulate_step(self.current_time, dt=1.0)
            self.pipeline.process_batch(events)
            self.current_time += 1.0
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            summary = self.pipeline.get_metrics_summary()
            self.wfile.write(json.dumps(summary).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def render_html(self) -> str:
        summary = self.pipeline.get_metrics_summary()
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Robotics Data RL Cluster Scheduler</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; margin-bottom: 4px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px; }}
        .metric-box {{ background: #0f172a; padding: 16px; border-radius: 6px; text-align: center; border: 1px solid #1e293b; }}
        .metric-val {{ font-size: 28px; font-weight: bold; color: #38bdf8; }}
        .metric-label {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
        button {{ background: #0284c7; color: white; border: none; padding: 10px 20px; font-size: 15px; border-radius: 6px; cursor: pointer; }}
        button:hover {{ background: #0369a1; }}
    </style>
</head>
<body>
    <h1>🤖 Robotics Data Platform RL Cluster Scheduler</h1>
    <p style="color: #94a3b8;">Discrete-event multi-rate telemetry ingestion, timestamp synchronization, and collection dispatch</p>
    
    <div class="card">
        <button onclick="fetch('/api/step').then(r=>r.json()).then(()=>location.reload())">Run Simulation Step (+1.0s)</button>
        <div class="grid">
            <div class="metric-box">
                <div class="metric-val">{summary.get('total_received', 0)}</div>
                <div class="metric-label">Packets Ingested</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{summary.get('valid_count', 0)}</div>
                <div class="metric-label">Valid In-Order Packets</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{summary.get('drop_rate_pct', 0.0):.1f}%</div>
                <div class="metric-label">Stale Drop Rate (&gt;500ms)</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{summary.get('latency_p50_ms', 0.0):.1f} ms</div>
                <div class="metric-label">p50 Ingest Latency</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{summary.get('latency_p95_ms', 0.0):.1f} ms</div>
                <div class="metric-label">p95 Ingest Latency</div>
            </div>
        </div>
    </div>
</body>
</html>
"""


def start_server(port: int = 8501):
    server = HTTPServer(("0.0.0.0", port), DashboardHTTPHandler)
    print(f"Robotics Dashboard running on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    start_server()
