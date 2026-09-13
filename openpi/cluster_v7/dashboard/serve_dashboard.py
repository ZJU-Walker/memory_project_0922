"""Fully live dashboard on the workstation: serves the dashboard page and re-reads the logs on every data request.
    python3 serve_dashboard.py [port]          (default 8020, binds 127.0.0.1)
Then from the laptop:  ssh -L 8020:localhost:8020 iris-ws-18.stanford.edu   ->  http://localhost:8020
`/data.json` rebuilds DATA from the logs each time it is fetched (milliseconds); a background thread refreshes the
GPU/job/disk snapshot (`gpu.json`, needs a live Kerberos cache for ssh) every 30 s via refresh.sh."""
import http.server
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8020


def rebuild() -> bytes:
    html = subprocess.run([sys.executable, str(HERE / "build_dashboard.py"), str(HERE / "gpu.json")], capture_output=True, text=True, cwd=HERE)
    return html.stdout.encode("utf-8")


def snapshot_loop() -> None:
    while True:
        subprocess.run(["bash", str(HERE / "refresh.sh")], capture_output=True, text=True)
        time.sleep(30)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/data.json"):
            rebuild()
            body = (HERE / "dash_data.json").read_bytes()
            ctype = "application/json"
        else:
            body = rebuild()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass


if __name__ == "__main__":
    threading.Thread(target=snapshot_loop, daemon=True).start()
    print(f"serving the dashboard on http://127.0.0.1:{PORT}  (ssh -L {PORT}:localhost:{PORT} iris-ws-18.stanford.edu)", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
