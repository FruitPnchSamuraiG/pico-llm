#!/usr/bin/env python3
"""Lightweight web UI to browse interpretability outputs.

Serves a given output directory (from scripts/interpret_transformer.py) and
renders a small HTML UI that lets you:
- browse attention PNGs
- view logit-lens JSON
- view neuron top-activations JSON
- view summary.json

Usage:
  source /scratch/kk6081/ml_fall25/venv/bin/activate
  python scripts/interpretability_viewer.py --root /scratch/kk6081/picollm_extend/interpretability_test --port 8000

Notes:
- Uses only Python stdlib (http.server) for portability.
- Intended for local forwarding (ssh -L) rather than public deployment.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Interpretability output directory (contains summary.json)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def _safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        return {"error": str(e)}


def _html_page(title: str, body: str) -> bytes:
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fff; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; font-size: 13px; white-space: pre-wrap; }}
    .topnav {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; }}
    .muted {{ color: #6b7280; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    return doc.encode("utf-8")


class Handler(SimpleHTTPRequestHandler):
    # Set by main()
    root_dir: Path

    def translate_path(self, path: str) -> str:
        # Serve static files from root_dir
        raw = urlparse(path).path
        rel = raw.lstrip("/")
        fs_path = (self.root_dir / rel).resolve()
        # prevent path traversal
        if not str(fs_path).startswith(str(self.root_dir.resolve())):
            return str(self.root_dir)
        return str(fs_path)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            return self._serve_index()
        if parsed.path == "/json":
            return self._serve_json_view(parsed)
        if parsed.path == "/attention":
            return self._serve_attention()

        # Default: static file
        return super().do_GET()

    def _serve_index(self):
        summary = _safe_read_json(self.root_dir / "summary.json")
        ckpt = "(missing summary.json)"
        analyses = []
        if isinstance(summary, dict) and summary:
            ckpt = summary.get("checkpoint", ckpt)
            analyses = summary.get("analyses_run", []) or []

        body = """
<div class='topnav'>
  <a href='/'>Home</a>
  <a href='/attention'>Attention PNGs</a>
  <a href='/json?path=logit_lens/results.json'>Logit lens JSON</a>
  <a href='/json?path=neurons/top_neurons.json'>Top neurons JSON</a>
  <a href='/json?path=summary.json'>summary.json</a>
</div>
"""

        body += f"""<h1>Interpretability Viewer</h1>
<p class='muted'>Root: <code>{html.escape(str(self.root_dir))}</code></p>
<p>Checkpoint: <code>{html.escape(str(ckpt))}</code></p>
<p>Analyses: <code>{html.escape(', '.join(map(str, analyses)) if analyses else 'unknown')}</code></p>

<div class='grid'>
  <div class='card'>
    <h3>Attention</h3>
    <p>Browse heatmaps saved under <code>attention/</code>.</p>
    <p><a href='/attention'>Open attention gallery →</a></p>
  </div>
  <div class='card'>
    <h3>Logit lens</h3>
    <p>View decoded intermediate predictions.</p>
    <p><a href='/json?path=logit_lens/results.json'>Open JSON →</a></p>
  </div>
  <div class='card'>
    <h3>Neurons</h3>
    <p>Top activating neurons + contexts.</p>
    <p><a href='/json?path=neurons/top_neurons.json'>Open JSON →</a></p>
  </div>
  <div class='card'>
    <h3>Patching (if present)</h3>
    <p><a href='/json?path=patching/results.json'>Open JSON →</a></p>
  </div>
</div>
"""

        data = _html_page("Interpretability Viewer", body)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json_view(self, parsed):
        qs = parse_qs(parsed.query)
        rel = (qs.get("path", [""]) or [""])[0]
        rel = rel.lstrip("/")
        path = (self.root_dir / rel).resolve()
        if not str(path).startswith(str(self.root_dir.resolve())):
            self.send_error(400, "Invalid path")
            return

        payload = _safe_read_json(path)
        body = """
<div class='topnav'>
  <a href='/'>Home</a>
  <a href='/attention'>Attention PNGs</a>
</div>
"""
        body += f"<h2>JSON: <code>{html.escape(rel)}</code></h2>"
        body += "<div class='card mono'>"
        body += html.escape(json.dumps(payload, indent=2))
        body += "</div>"

        data = _html_page(f"JSON - {rel}", body)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_attention(self):
        attn_dir = self.root_dir / "attention"
        imgs = []
        if attn_dir.exists():
            imgs = sorted([p for p in attn_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

        body = """
<div class='topnav'>
  <a href='/'>Home</a>
  <a href='/json?path=summary.json'>summary.json</a>
</div>
"""
        body += "<h2>Attention heatmaps</h2>"
        if not imgs:
            body += "<p class='muted'>No images found in <code>attention/</code>.</p>"
        else:
            body += "<div class='grid'>"
            for p in imgs:
                rel = "attention/" + p.name
                body += "<div class='card'>"
                body += f"<div class='muted'><code>{html.escape(p.name)}</code></div>"
                body += f"<a href='/{html.escape(rel)}'><img src='/{html.escape(rel)}' loading='lazy' /></a>"
                body += "</div>"
            body += "</div>"

        data = _html_page("Attention heatmaps", body)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    # Ensure common mimetypes
    mimetypes.add_type("application/json", ".json")
    mimetypes.add_type("image/png", ".png")

    Handler.root_dir = root

    # Make the server serve from the root directory for static assets
    os.chdir(str(root))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"✅ Interpretability viewer running")
    print(f"  root: {root}")
    print(f"  url:  http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
