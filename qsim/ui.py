"""Chatbot web page. You ask in the page and get the answer there; the similarity report prints in the terminal.

    py -m qsim ui        ->  http://localhost:8000
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .bot import DEFAULT_KB, QABot
from .checker import InvalidQuestionError

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRONMAN Chatbot</title><style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2230;--mut:#6b7385;--line:#e3e6ec;--acc:#2f6bff}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2030;--ink:#e8ebf2;--mut:#98a1b5;--line:#2a3045;--acc:#6c93ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card)}header h1{font-size:16px;margin:0}header small{color:var(--mut)}
#log{flex:1;overflow:auto;padding:20px;display:flex;flex-direction:column;gap:12px;max-width:820px;width:100%;margin:0 auto}
.msg{max-width:85%;padding:10px 14px;border-radius:14px;word-wrap:break-word}.user{align-self:flex-end;background:var(--acc);color:#fff;border-bottom-right-radius:4px}
.bot{align-self:flex-start;background:var(--card);border:1px solid var(--line);border-bottom-left-radius:4px}.tag{display:block;margin-top:6px;font-size:12px;color:var(--mut)}
form{padding:14px 20px;border-top:1px solid var(--line);background:var(--card)}form div{display:flex;gap:10px;max-width:820px;margin:0 auto}
input{flex:1;padding:12px 14px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
button{padding:0 20px;border:0;border-radius:10px;background:var(--acc);color:#fff;font:inherit;font-weight:600;cursor:pointer}
</style></head><body>
<header><h1>IRONMAN Chatbot</h1><small>Similarity details are shown in the terminal</small></header>
<div id="log"><div class="msg bot">Hi! Ask me anything about IRONMAN races.</div></div>
<form id="f"><div><input id="q" autocomplete="off" placeholder="Type your question…" autofocus><button>Send</button></div></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q');
const add=(c,t,tag)=>{const m=document.createElement('div');m.className='msg '+c;m.textContent=t;
 if(tag){const s=document.createElement('span');s.className='tag';s.textContent=tag;m.append(s)}log.append(m);log.scrollTop=log.scrollHeight};
const TAG={session:'↺ answered earlier for a similar question',knowledge_base:'from the predefined answers',none:''};
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const t=q.value.trim();if(!t)return;q.value='';add('user',t);
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:t})});
 const d=await r.json();add('bot',d.error?('⚠ '+d.error):d.answer,d.error?'':TAG[d.source])}catch(err){add('bot','⚠ Could not reach the server')}};
</script></body></html>"""


def make_handler(bot: QABot):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: dict, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                return self._json({"error": "not found"}, 404)
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                turn = bot.ask(str(data.get("question", "")))
                self._json({"answer": turn.answer, "source": turn.source})
            except InvalidQuestionError as exc:
                self._json({"error": str(exc)})
            except (ValueError, json.JSONDecodeError):
                self._json({"error": "bad request"}, 400)
            except Exception as exc:   # show the error in the page, don't drop the connection
                import traceback
                traceback.print_exc()
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def log_message(self, *args) -> None:  # keep the terminal for the similarity reports
            pass

    return Handler


def serve(kb: str = str(DEFAULT_KB), session_dir: str = "sessions", port: int = 8000,
          open_browser: bool = True, wordnet: str = "auto") -> None:
    bot = QABot(kb, session_dir, wordnet=wordnet)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(bot))
    url = f"http://localhost:{port}"
    print(f"\nChatbot UI: {url}   (similarity reports print here; Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
