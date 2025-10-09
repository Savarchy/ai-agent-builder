(() => {
  const ds = document.currentScript.dataset;
  const ENDPOINT = ds.endpoint;
  const BOT_ID   = ds.botId || "default";
  const TOKEN    = ds.siteToken;
  const TITLE    = ds.title || "Chat";
  const PRIMARY  = ds.primary || "#3b82f6";
  const POS      = (ds.position || "bottom-right").toLowerCase();

  const host = document.createElement("div");
  host.style.position = "fixed";
  host.style.zIndex = 2147483647;
  host.style[POS.includes("bottom") ? "bottom" : "top"] = "16px";
  host.style[POS.includes("right") ? "right" : "left"] = "16px";
  const shadow = host.attachShadow({ mode: "open" });
  document.body.appendChild(host);

  const style = document.createElement("style");
  style.textContent = `
    .bubble{width:56px;height:56px;border-radius:9999px;background:${PRIMARY};color:#fff;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 8px 24px rgba(0,0,0,.15);font:14px system-ui}
    .panel{position:fixed;width:min(380px,90vw);height:min(560px,80vh);background:#fff;border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.2);display:flex;flex-direction:column;overflow:hidden}
    .header{background:${PRIMARY};color:#fff;padding:12px 16px;font-weight:600}
    .msgs{flex:1;padding:12px;overflow:auto;display:flex;flex-direction:column;gap:8px;background:#f8fafc}
    .m{max-width:85%;padding:10px 12px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
    .m.user{align-self:flex-end;background:#e2e8f0}
    .m.assistant{align-self:flex-start;background:#fff;border:1px solid #e5e7eb}
    form{display:flex;gap:8px;padding:10px;border-top:1px solid #e5e7eb;background:#fff}
    input{flex:1;padding:10px;border-radius:10px;border:1px solid #d1d5db;font:14px system-ui}
    button{padding:10px 14px;border-radius:10px;border:none;background:${PRIMARY};color:#fff;cursor:pointer}
  `;
  shadow.appendChild(style);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "💬";
  shadow.appendChild(bubble);

  const panel = document.createElement("div");
  panel.className = "panel";
  panel.style.display = "none";
  panel.style[POS.includes("bottom") ? "bottom" : "top"] = "80px";
  panel.style[POS.includes("right") ? "right" : "left"] = "0px";
  panel.innerHTML = `
    <div class="header">${TITLE}</div>
    <div class="msgs"></div>
    <form><input placeholder="Type a message..."/><button type="submit">Send</button></form>
  `;
  shadow.appendChild(panel);

  const msgs = panel.querySelector(".msgs");
  const form = panel.querySelector("form");
  const input = panel.querySelector("input");

  const append = (role, text) => {
    const d = document.createElement("div");
    d.className = `m ${role}`;
    d.textContent = text || "";
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  };

  bubble.addEventListener("click", () => {
    panel.style.display = panel.style.display === "none" ? "block" : "none";
    if (panel.style.display === "block") input.focus();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    append("user", q);
    const para = document.createElement("div");
    para.className = "m assistant";
    para.textContent = "";
    msgs.appendChild(para);

    const resp = await fetch(`${ENDPOINT}/ask/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${TOKEN}`
      },
      body: JSON.stringify({ bot_id: BOT_ID, question: q })
    });

    if (!resp.body) { para.textContent = "Error: no response"; return; }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const chunk of parts) {
        if (!chunk.startsWith("data: ")) continue;
        const data = chunk.slice(6);
        if (data === "[DONE]") return;
        para.textContent += data;
        msgs.scrollTop = msgs.scrollHeight;
      }
    }
  });
})();
