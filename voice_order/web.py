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
  .modelpick { margin-left:auto; color:var(--dim); font-size:12px;
               display:flex; align-items:center; gap:6px }
  .modelpick select { background:var(--panel); color:var(--text);
    border:1px solid var(--line); border-radius:7px; padding:5px 8px; font:inherit;
    font-size:12px; cursor:pointer }
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
  #mic { background:var(--panel); border:1px solid var(--line); font-size:18px; padding:9px 13px }
  #mic.rec { background:var(--bad); color:#fff; border-color:var(--bad);
             animation:pulse 1.1s ease-in-out infinite }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }
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
  <label class="modelpick">model
    <select id="model"><option>...</option></select>
  </label>
</header>

<main>
  <div style="display:flex;flex-direction:column;min-height:0">
    <div id="chat"></div>
    <form id="f">
      <button type="button" id="mic" title="hold a conversation by voice">&#127908;</button>
      <input id="t" autocomplete="off" placeholder="type, or press the mic to speak" autofocus>
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

const mic=document.getElementById('mic');

// ---- continuous, hands-free call --------------------------------------
// Press the mic once to START a call. It listens; when you speak it records,
// and ~1.2s of silence ends your turn and sends it. While the agent speaks,
// listening pauses so it does not hear its own voice. Press again to END.

let call=null;
const THRESH=0.020, SILENCE_MS=1200, TICK=90, MIN_SPEECH_MS=350;

function playAndWait(b64){
  return new Promise(res=>{
    if(!b64) return res();
    const a=new Audio('data:audio/wav;base64,'+b64);
    a.onended=res; a.onerror=res; a.play().catch(res);
  });
}

function rms(){
  call.analyser.getByteTimeDomainData(call.buf);
  let s=0; for(const v of call.buf){ const x=(v-128)/128; s+=x*x; }
  return Math.sqrt(s/call.buf.length);
}

function beginTurn(){
  call.chunks=[]; call.startedAt=Date.now();
  call.rec=new MediaRecorder(call.stream);
  call.rec.ondataavailable=e=>{ if(e.data.size) call.chunks.push(e.data); };
  call.rec.onstop=endTurn;
  call.rec.start();
}

async function endTurn(){
  const spoke=Date.now()-call.startedAt;
  const blob=new Blob(call.chunks,{type:'audio/webm'});
  if(spoke < MIN_SPEECH_MS){ call.silence=0; return; }   // a blip, not speech
  call.paused=true;                                       // stop listening
  const think=document.createElement('div');
  think.className='msg sys'; think.textContent='...'; chat.appendChild(think);
  chat.scrollTop=chat.scrollHeight;
  try{
    const r=await fetch('/api/voice',{method:'POST',body:blob});
    const d=await r.json(); think.remove();
    if(d.error){ bubble('sys','error: '+d.error); }
    else if(!d.transcript){ /* heard nothing usable -- stay quiet, keep listening */ }
    else {
      bubble('you', d.transcript); bubble('agent', d.reply);
      renderTools(d.tools); renderCart(d.cart);
      await playAndWait(d.audio);
    }
  }catch(err){ think.remove(); bubble('sys','error: '+err); }
  call.silence=0; call.paused=false;                     // resume listening
}

function tick(){
  if(!call || !call.active) return;
  if(!call.paused){
    const level=rms();
    if(level>THRESH){
      call.silence=0;
      if(!call.speaking){ call.speaking=true; beginTurn(); }
    } else if(call.speaking){
      call.silence+=TICK;
      if(call.silence>=SILENCE_MS){ call.speaking=false; call.rec.stop(); }
    }
  }
  call.timer=setTimeout(tick, TICK);
}

async function startCall(){
  let stream;
  try{ stream=await navigator.mediaDevices.getUserMedia({audio:true}); }
  catch(e){ bubble('sys','microphone blocked -- allow mic access and retry'); return; }
  const ctx=new (window.AudioContext||window.webkitAudioContext)();
  const src=ctx.createMediaStreamSource(stream);
  const analyser=ctx.createAnalyser(); analyser.fftSize=512; src.connect(analyser);
  call={stream, ctx, analyser, buf:new Uint8Array(analyser.fftSize),
        active:true, speaking:false, paused:false, silence:0, chunks:[], rec:null};
  mic.classList.add('rec'); mic.textContent='⏹';
  mic.title='end call';
  bubble('sys','call started -- just speak, pause when done. Press the mic to end.');
  tick();
}

function endCall(){
  if(!call) return;
  call.active=false; clearTimeout(call.timer);
  try{ call.rec && call.rec.state==='recording' && call.rec.stop(); }catch(e){}
  call.stream.getTracks().forEach(t=>t.stop());
  call.ctx.close();
  call=null;
  mic.classList.remove('rec'); mic.innerHTML='&#127908;'; mic.title='start a voice call';
  bubble('sys','call ended');
}

mic.onclick=()=>{ call ? endCall() : startCall(); };

const modelSel=document.getElementById('model');const modelSel=document.getElementById('model');

function fillModels(info){
  modelSel.innerHTML='';
  for(const p of info.profiles){
    const o=document.createElement('option');
    o.value=p.name; o.textContent=p.name+' ('+p.model+')';
    if(p.name===info.current) o.selected=true;
    modelSel.appendChild(o);
  }
}

modelSel.onchange=async()=>{
  send.disabled=true;
  bubble('sys','switching to '+modelSel.value+'...');
  const d=await post('/api/model',{profile:modelSel.value});
  if(d.error){ bubble('sys','error: '+d.error); }
  else {
    chat.innerHTML=''; renderTools([]); renderCart({lines:[],total:0});
    fillModels(d); bubble('agent', d.greeting);
  }
  send.disabled=false; input.focus();
};

(async()=>{
  const info=await post('/api/model',{});   // populate the dropdown
  fillModels(info);
  const d=await post('/api/reset'); bubble('agent', d.greeting);
})();
</script>
"""


class _State:
    """One call at a time. This is a demo, not a server."""

    lock = threading.Lock()
    agent = None
    profile: str | None = None   # None = the config default

    @classmethod
    def reset(cls, profile: str | None = None):
        from voice_order.agent.loop import OrderAgent
        from voice_order.llm.client import from_config

        if profile is not None:
            cls.profile = profile
        client = from_config(profile=cls.profile) if cls.profile else None
        cls.agent = OrderAgent(client=client, persist=True)
        return cls.agent

    @classmethod
    def model_info(cls) -> dict:
        from voice_order import config
        from voice_order.llm.client import active_profile

        profiles = config.load("agent").get("llm.profiles", {})
        return {
            "current": cls.profile or active_profile(),
            "profiles": [
                {"name": name, "model": spec.get("model")}
                for name, spec in profiles.items()
            ],
        }

    # Heavy models, loaded once on the first voice turn. Whisper reads the mic
    # audio (with the part-number biasing prompt from config); Piper speaks the
    # reply. Both are cached because loading them per turn would dominate.
    _transcriber = None
    _speaker = None

    @classmethod
    def transcriber(cls):
        if cls._transcriber is None:
            from voice_order.asr.transcribe import Transcriber

            cls._transcriber = Transcriber(n_best=1)  # 1-best is enough live
            cls._transcriber._loaded()
        return cls._transcriber

    @classmethod
    def speaker(cls):
        if cls._speaker is None:
            from voice_order.tts.speak import Speaker

            cls._speaker = Speaker()
            cls._speaker._loaded()
        return cls._speaker


def _voice_turn(audio_bytes: bytes) -> dict:
    """Mic audio in -> transcript -> agent -> spoken reply out.

    The whole loop the project is about, live. Whisper here is the same
    Transcriber the eval uses, biasing prompt and all, so a part number spoken
    into the mic gets the same treatment the measured pipeline gives it.
    """
    import base64
    import io
    import tempfile
    from pathlib import Path

    import soundfile as sf

    agent = _State.agent or _State.reset()

    # faster-whisper decodes via PyAV, which reads the browser's webm/opus
    # directly -- no ffmpeg, no conversion.
    tmp = Path(tempfile.gettempdir()) / f"voice_{threading.get_ident()}.webm"
    tmp.write_bytes(audio_bytes)
    try:
        transcript = _State.transcriber().transcribe_file(tmp).best
    except Exception as exc:
        return {"error": f"could not read the audio: {exc}"}
    finally:
        tmp.unlink(missing_ok=True)

    if not transcript.strip():
        return {"transcript": "", "reply": "Sorry, I didn't catch that -- try again?",
                "audio": None, "tools": [], "cart": agent.session._snapshot()}

    reply = agent.handle(transcript)

    audio, rate = _State.speaker().synthesize(reply)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV")
    spoken = base64.b64encode(buf.getvalue()).decode("ascii")

    last = agent.brain.tool_log[-1] if agent.brain.tool_log else {}
    return {
        "transcript": transcript,
        "reply": reply,
        "audio": spoken,
        "tools": last.get("tools", []),
        "cart": agent.session._snapshot(),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        # The browser auto-requests /favicon.ico; answer it quietly so it does
        # not show up as a scary 404.
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # No caching: the page is small and served locally, and a cached copy
        # after a code change is exactly the confusion this avoids.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)

        # Voice audio is raw bytes, not JSON. Handle it before parsing.
        if self.path == "/api/voice":
            with _State.lock:
                try:
                    result = _voice_turn(raw)
                except Exception as exc:
                    self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
                    return
            self._send(result)
            return

        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "bad json"}, 400)
            return

        if self.path == "/api/model":
            profile = payload.get("profile")
            with _State.lock:
                if profile:
                    try:
                        agent = _State.reset(profile)
                    except Exception as exc:  # e.g. missing API key
                        self._send({"error": f"{type(exc).__name__}: {exc}"}, 400)
                        return
                    self._send({**_State.model_info(), "greeting": agent.greeting()})
                else:
                    self._send(_State.model_info())
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
    from voice_order.llm.client import active_profile, from_config

    client = from_config()
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        print(f"\n  ! could not start on port {port}: {exc}")
        print("  ! an old server is probably still running there. Stop it")
        print("    (Ctrl-C in its window) or use a different port:")
        print(f"       python -m voice_order serve --port {port + 1}")
        return

    print(f"voice-order demo on http://{host}:{port}")
    print(f"  model     {client.model}  ({active_profile()}) via {client.base_url}")
    print("  localhost only, one call at a time -- this is a demo, not a server")
    print("  Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
