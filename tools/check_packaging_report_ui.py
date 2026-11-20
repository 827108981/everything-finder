"""Launch one isolated report server for the browser check and close it."""
import subprocess
import threading
from tools.prepare_vector_packaging_20261116 import OUT,ROOT
from tools.serve_rag_report import make_server
server,url=make_server(OUT/'report.html',opener=lambda p,l:'UI test: original source position resolved')
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    subprocess.run(['C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe',
                    str(ROOT/'tools/check_vector_packaging_report.cjs'),url],cwd=ROOT,check=True)
finally:server.shutdown();server.server_close()
