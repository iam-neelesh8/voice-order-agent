"""A local demo you can actually watch.

`voice-order serve` puts a page on localhost where you type an order and see
every layer working: what the model asked for, what the catalog returned, what
the confidence policy decided, and what happened to the cart.

The point is not that it is a web app. It is that the hybrid split is
invisible from a transcript -- "I added two spark plugs" looks the same
whether the model was allowed to write to the cart or had to ask. This shows
the asking.

Built on stdlib http.server. A dependency to route three URLs would be a
dependency to pin, upgrade and explain, and this project already avoids one
for the catalog download and the LLM client.

Single-user, single-session, localhost only. It is a demo, not a server: no
auth, no concurrency story, and it says so rather than pretending otherwise.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>voice-order-agent</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --text:#e6e9ef;
    --dim:#8b93a7; --accent:#7aa2f7; --ok:#9ece6a; --warn:#e0af68; --bad:#f7768e;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f7f9; --panel:#fff; --line:#e2e5ea; --text:#1a1d23;
            --dim:#6b7280; --accent:#2c5fd4; --ok:#2f7d32; --warn:#a86a00;
            --bad:#c1344e; }
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5
         ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex;
           align-items:baseline; gap:14px; flex-wrap:wrap }
  h1 { font-size:17px; margin:0; font-weight:650 }
  .sub { color:var(--dim); font-size:13px }
  main { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,420px);
         gap:0; height:calc(100vh - 62px) }
  @media (max-width:900px){ main{grid-template-columns:1fr; height:auto} }
  #chat { padding:22px 24px; overflow-y:auto; display:flex; flex-direction:column; gap:14px }
  #side { border-left:1px solid var(--line); background:var(--panel);
          padding:18px 20px; overflow-y:auto }
  @media (max-width:900px){ #side{border-left:0;border-top:1px solid var(--line)} }
  .msg { max-width:78%; padding:10px 14px; border-radius:12px; white-space:pre-wrap }
  .you { align-self:flex-end; background:var(--accent); color:#fff }
  .agent { align-self:flex-start; background:var(--panel); border:1px solid var(--line) }
  .sys { align-self:center; color:var(--dim); font-size:13px; font-style:italic }
  form { display:flex; gap:10px; padding:16px 24px; border-top:1px solid var(--line) }
  input { flex:1; padding:11px 14px; border-radius:9px; border:1px solid var(--line);
          background:var(--bg); color:var(--text); font:inherit }
  button { padding:11px 18px; border-radius:9px; border:0; background:var(--accent);
           color:#fff; font:inherit; font-weight:600; cursor:pointer }
  button.ghost { background:transparent; color:var(--dim); border:1px solid var(--line) }
  button:disabled { opacity:.5; cursor:default }
  h2 { font-size:11px; text-transform:uppercase; letter-spacing:.09em;
       color:var(--dim); margin:22px 0 9px }
  h2:first-child { margin-top:0 }
  .tool { border:1px solid var(--line); border-radius:9px; padding:9px 11px;
          margin-bottom:8px; font-size:13px }
  .tool b { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--accent) }
  .tool .args { color:var(--dim); font-size:12px; word-break:break-all;
                font-family:ui-monospace,Menlo,monospace }
  .tool.err { border-color:var(--bad) }
  .tool.err .args { color:var(--bad) }
  table { width:100%; border-collapse:collapse; font-size:13px }
  td { padding:5px 0; border-bottom:1px solid var(--line); vertical-align:top }
  td.q { color:var(--dim); width:34px }
  td.p { text-align:right; white-space:nowrap; color:var(--dim) }
  .total { display:flex; justify-content:space-between; margin-top:11px;
           font-weight:650; font-size:15px }
  .flag { color:var(--warn); font-size:12px; margin-top:7px; line-height:1.45 }
  .empty { color:var(--dim); font-size:13px }
  .pill { display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px;
          font-weight:650; text-transform:uppercase; letter-spacing:.05em }
  .commit{background:rgba(158,206,106,.16);color:var(--ok)}
  .confirm{background:rgba(224,175,104,.16);color:var(--warn)}
  .clarify{background:rgba(224,175,104,.16);color:var(--warn)}
  .reask{background:rgba(247,118,142,.16);color:var(--bad)}
</style>

<header>
  <h1>voice-order-agent</h1>
  <span class="sub">the model asks &middot; the code decides &middot; type an order below</span>
</header>

<main>
  <div style="display:flex;flex-direction:column;min-height:0">
    <div id="chat"></div>
    <form id="f">
      <input id="t" autocomplete="off" placeholder="I need two AC Delco 41-993 spark plugs" autofocus>
      <button id="send">Send</button>
      <button type="button" class="ghost" id="reset">New call</button>
    </form>
  </div>
  <div id="side">
    <h2>Tools the model called</h2>
    <div id="tools"><p class="empty">Nothing yet.</p></div>
    <h2>Order</h2>
    <div id="cart"><p class="empty">Empty.</p></div>
  </div>
</main>

<script>
const chat=document.getElementById('chat'), tools=document.getElementById('tools'),
      cart=document.getElementById('cart'), form=document.getElementById('f'),
      input=document.getElementById('t'), send=document.getElementById('send');

function bubble(cls, text){
  const d=document.createElement('div'); d.className='msg '+cls; d.textContent=text;
  chat.appendChild(d); chat.scrollTop=chat.scrollHeight;
}

function renderTools(list){
  if(!list || !list.length){ tools.innerHTML='<p class="empty">The model called nothing this turn.</p>'; return; }
  tools.innerHTML='';
  for(const t of list){
    const err = t.result && t.result.error;
    const d=document.createElement('div'); d.className='tool'+(err?' err':'');
    let extra='';
    if(t.result && t.result.guidance){
      const kind = t.result.guidance.startsWith('Strong') ? 'commit'
                 : t.result.guidance.startsWith('No usable') ? 'reask'
                 : t.result.guidance.startsWith('Several') ? 'clarify' : 'confirm';
      extra = '<div style="margin-top:6px"><span class="pill '+kind+'">'+kind+'</span></div>';
    }
    d.innerHTML='<b>'+t.name+'</b><div class="args">'+
      (err ? err : JSON.stringify(t.arguments))+'</div>'+extra;
    tools.appendChild(d);
  }
}

function renderCart(c){
  if(!c.lines.length){ cart.innerHTML='<p class="empty">Empty.</p>'; return; }
  let html='<table>';
  for(const l of c.lines){
    html+='<tr><td class="q">'+l.quantity+'&times;</td><td>'+
          l.name.replace(/</g,'&lt;')+'</td><td class="p">'+
          (l.unit_price==null ? 'no price' : '$'+l.subtotal.toFixed(2))+'</td></tr>';
  }
  html+='</table><div class="total"><span>'+
        (c.total_is_partial ? 'Priced items' : 'Total')+
        '</span><span>$'+c.total.toFixed(2)+'</span></div>';
  if(c.total_is_partial){
    html+='<div class="flag">'+c.unpriced_items+' of '+c.lines.length+
          ' item(s) have no price on file. This is not the order total.</div>';
  }
  cart.innerHTML=html;
}

async function post(url, body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
                          body:JSON.stringify(body||{})});
  return r.json();
}

form.onsubmit=async e=>{
  e.preventDefault();
  const text=input.value.trim(); if(!text) return;
  bubble('you', text); input.value=''; send.disabled=true;
  const thinking=document.createElement('div');
  thinking.className='msg sys'; thinking.textContent='thinking...';
  chat.appendChild(thinking); chat.scrollTop=chat.scrollHeight;
  try{
    const d=await post('/api/say',{text});
    thinking.remove();
    if(d.error){ bubble('sys','error: '+d.error); }
    else { bubble('agent', d.reply); renderTools(d.tools); renderCart(d.cart); }
  }catch(err){ thinking.remove(); bubble('sys','error: '+err); }
  send.disabled=false; input.focus();
};

document.getElementById('reset').onclick=async()=>{
  const d=await post('/api/reset');
  chat.innerHTML=''; renderTools([]); renderCart({lines:[],total:0});
  bubble('agent', d.greeting);
};

(async()=>{ const d=await post('/api/reset'); bubble('agent', d.greeting); })();
</script>
"""


class _State:
    """One call at a time. This is a demo, not a server."""

    lock = threading.Lock()
    agent = None

    @classmethod
    def reset(cls):
        from voice_order.agent.loop import OrderAgent

        cls.agent = OrderAgent(persist=True)
        return cls.agent


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "bad json"}, 400)
            return

        if self.path == "/api/reset":
            with _State.lock:
                agent = _State.reset()
                self._send({"greeting": agent.greeting()})
            return

        if self.path == "/api/say":
            text = str(payload.get("text") or "").strip()
            if not text:
                self._send({"error": "nothing said"}, 400)
                return
            with _State.lock:
                agent = _State.agent or _State.reset()
                try:
                    reply = agent.handle(text)
                except Exception as exc:
                    self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
                    return
                last = agent.brain.tool_log[-1] if agent.brain.tool_log else {}
                self._send(
                    {
                        "reply": reply,
                        "tools": last.get("tools", []),
                        "cart": agent.session._snapshot(),
                    }
                )
            return

        self.send_error(404)

    def log_message(self, *args):
        """Quiet. The interesting log is the one on the page."""


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    from voice_order import config

    cfg = config.load("agent")
    print(f"voice-order demo on http://{host}:{port}")
    print(f"  model     {cfg.get('llm.model')} via {cfg.get('llm.base_url')}")
    print("  localhost only, one call at a time -- this is a demo, not a server")
    print("  Ctrl-C to stop")
    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
