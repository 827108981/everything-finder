"""Loopback report viewer; native source navigation uses an allowlisted manifest."""
import argparse
import json
import secrets
import hashlib
import tempfile
import zipfile
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from local_full_text_search.core.open_location import open_file_at_location


def physical_source(source):
    if not source.get('archive'):
        return Path(source['path'])
    archive = Path(source['archive'])
    member = source['member']
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(member)
        if info.file_size > 128 * 1024 * 1024 or Path(member).suffix.lower() not in {'.pdf', '.ppt', '.pptx', '.doc', '.docx', '.xlsx', '.xls'}:
            raise ValueError('Unsupported archive document')
        key = hashlib.sha256((str(archive) + member + str(info.CRC)).encode()).hexdigest()
        cache = Path(tempfile.gettempdir()) / 'rag_report_sources' / key
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / Path(member.replace('\\', '/')).name
        # Only a basename is written; never extract an archive-controlled path.
        target.write_bytes(z.read(info))
        return target


def make_server(report, opener=open_file_at_location):
    report = Path(report).resolve()
    sources = json.loads(report.with_name('report_sources.json').read_text(encoding='utf8'))
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {'error': 'invalid host'})
                return
            name = urlsplit(self.path).path
            allowed = {'/': report, '/report.html': report, '/conclusion.md': report.with_name('conclusion.md'), '/final_acceptance.json': report.with_name('final_acceptance.json')}
            path = allowed.get(name)
            if path is None or not path.is_file():
                self.reply(404, {'error': 'not found'})
                return
            self.send_response(200)
            self.send_header('Content-Type', ('text/html' if path.suffix == '.html' else 'text/plain') + '; charset=utf-8')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(path.read_bytes())

        def do_POST(self):
            origin = self.headers.get('Origin')
            if not self.valid_host() or self.headers.get('X-Report-Token') != token or (origin and origin != f'http://127.0.0.1:{self.server.server_port}'):
                self.reply(403, {'error': 'invalid request'})
                return
            if self.path != '/open-source':
                self.reply(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 4096:
                    raise ValueError('invalid body length')
                body = json.loads(self.rfile.read(length))
                source = sources.get(body.get('id'))
                if source is None:
                    self.reply(404, {'error': 'source not in report manifest'})
                    return
                message = opener(physical_source(source), source['location'])
                self.reply(200, {'message': message})
            except Exception as exc:
                self.reply(400, {'error': str(exc)})

    server = HTTPServer(('127.0.0.1', 0), Handler)
    return server, f'http://127.0.0.1:{server.server_port}/report.html?token={token}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server, url = make_server(args.report)
    print(url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
