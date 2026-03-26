#!/usr/bin/env python3
"""
fileserver.py — HTTP file server with Archive.org-style directory listing.
Usage: python fileserver.py [port] [directory]
       Defaults: port=8080, directory=current working dir
"""

import http.server
import os
import sys
import urllib.parse
import html
from datetime import datetime


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
SERVE_DIR = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.getcwd())


def human_size(size: int) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(size) < 1024:
            return f"{size:.1f}{unit}" if unit else f"{size}B"
        size /= 1024
    return f"{size:.1f}P"


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Index of {title}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: "Helvetica Neue", Arial, sans-serif;
      font-size: 14px;
      background: #f5f5f5;
      color: #222;
    }}

    /* ── top bar ── */
    header {{
      background: #2c2c2c;
      color: #fff;
      text-align: center;
      padding: 8px 16px;
      font-size: 13px;
      letter-spacing: 0.03em;
    }}

    /* ── page heading ── */
    h1 {{
      text-align: center;
      font-size: 28px;
      font-weight: 700;
      padding: 28px 16px 18px;
      color: #111;
    }}

    /* ── card ── */
    .container {{
      max-width: 1100px;
      margin: 0 auto 40px;
      padding: 0 16px;
    }}

    .listing-box {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 4px;
      overflow: hidden;
    }}

    /* ── table ── */
    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    thead tr {{
      border-bottom: 2px solid #bbb;
    }}

    th {{
      text-align: left;
      padding: 10px 14px;
      font-size: 14px;
      font-weight: 700;
      color: #111;
    }}

    th.col-modified {{ width: 200px; }}
    th.col-size     {{ width: 100px; text-align: right; }}

    tbody tr {{
      border-bottom: 1px solid #eee;
    }}
    tbody tr:last-child {{ border-bottom: none; }}
    tbody tr:hover {{ background: #f0f4ff; }}

    td {{
      padding: 7px 14px;
      vertical-align: middle;
    }}

    td.col-modified {{
      font-size: 13px;
      color: #555;
      white-space: nowrap;
    }}

    td.col-size {{
      font-size: 13px;
      color: #555;
      text-align: right;
      white-space: nowrap;
    }}

    /* monospace blue links — Archive.org feel */
    a {{
      font-family: "Courier New", Courier, monospace;
      font-size: 13px;
      color: #2a55a0;
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}

    /* parent-dir row */
    .parent-icon {{ margin-right: 6px; }}
  </style>
</head>
<body>

<header>fileserver.py</header>

<h1>Index of {title}</h1>

<div class="container">
  <div class="listing-box">
    <table>
      <thead>
        <tr>
          <th class="col-name">Name</th>
          <th class="col-modified">Last modified</th>
          <th class="col-size">Size</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>
</div>

</body>
</html>
"""

PARENT_ROW = """\
        <tr>
          <td colspan="3">
            <span class="parent-icon">&#x2B06;</span>
            <a href="{href}">Go to parent directory</a>
          </td>
        </tr>"""

FILE_ROW = """\
        <tr>
          <td><a href="{href}">{name}</a></td>
          <td class="col-modified">{modified}</td>
          <td class="col-size">{size}</td>
        </tr>"""


class ArchiveHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # cleaner console output
        print(f"  {self.address_string()} [{self.log_date_time_string()}] {fmt % args}")

    # ------------------------------------------------------------------ #
    def do_GET(self):
        path = urllib.parse.unquote(self.path.split("?")[0])
        fs_path = os.path.normpath(os.path.join(SERVE_DIR, path.lstrip("/")))

        # Security: stay inside SERVE_DIR
        if not (fs_path == SERVE_DIR or fs_path.startswith(SERVE_DIR + os.sep)):
            self._send_error(403, "Forbidden")
            return

        if os.path.isdir(fs_path):
            self._serve_directory(path, fs_path)
        elif os.path.isfile(fs_path):
            self._serve_file(fs_path)
        else:
            self._send_error(404, "Not Found")

    # ------------------------------------------------------------------ #
    def _serve_directory(self, url_path: str, fs_path: str):
        try:
            entries = os.listdir(fs_path)
        except PermissionError:
            self._send_error(403, "Forbidden")
            return

        # Separate dirs and files, sort each alphabetically
        dirs = sorted([e for e in entries if os.path.isdir(os.path.join(fs_path, e))])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(fs_path, e))])

        rows = []

        # Parent directory link 
        if url_path.rstrip("/") not in ("", "/"):
            parent = os.path.dirname(url_path.rstrip("/")) or "/"
            if parent != "/":
                parent += "/"
            rows.append(PARENT_ROW.format(href=parent))

        def _row(name, is_dir):
            entry_fs = os.path.join(fs_path, name)
            stat = os.stat(entry_fs)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%d-%b-%Y %H:%M")
            size_str = "-" if is_dir else human_size(stat.st_size)
            href_name = urllib.parse.quote(name) + ("/" if is_dir else "")
            base_url = url_path if url_path.endswith("/") else url_path + "/"
            display = html.escape(name) + ("/" if is_dir else "")
            return FILE_ROW.format(
                href=base_url + href_name,
                name=display,
                modified=modified,
                size=size_str,
            )

        for d in dirs:
            rows.append(_row(d, is_dir=True))
        for f in files:
            rows.append(_row(f, is_dir=False))

        title = html.escape(url_path or "/")
        body = HTML_TEMPLATE.format(title=title, rows="\n".join(rows))
        encoded = body.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    # ------------------------------------------------------------------ #
    def _serve_file(self, fs_path: str):
        import mimetypes

        mime, _ = mimetypes.guess_type(fs_path)
        mime = mime or "application/octet-stream"

        try:
            with open(fs_path, "rb") as f:
                data = f.read()
        except PermissionError:
            self._send_error(403, "Forbidden")
            return

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------ #
    def _send_error(self, code: int, message: str):
        body = f"<h1>{code} {message}</h1>".encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    os.chdir(SERVE_DIR)
    server = http.server.HTTPServer(("", PORT), ArchiveHandler)
    print(f"Serving  : {SERVE_DIR}")
    print(f"Address  : http://localhost:{PORT}")
    print("Stop     : Ctrl+C\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
