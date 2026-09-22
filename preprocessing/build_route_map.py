import json

GTS = "/tmp/claude-1003/-home-ncrc/4d46e732-0f0f-4fb4-b2b9-74749bd74d73/scratchpad/gts"

v2 = json.load(open(f"{GTS}/sensor_topk_od_v2.json"))
sensor_data = v2["sensors"]
dong_boundaries = v2["dong_boundaries"]
examples = json.load(open(f"{GTS}/example_paths.json"))["examples"]

DATA_JSON = json.dumps({"sensors": sensor_data, "dong_boundaries": dong_boundaries, "examples": examples})

html = """<title>OD 라우팅 지도</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap">
<style>
:root{
  --bg:#f3f4ef; --paper:#fbfbf9; --ink:#1b1f1a; --ink-dim:#5b6259; --rule:#dadfd6;
  --accent:#3a6ea5; --accent-bg:#e3ebf3; --warm:#b3790d; --warm-bg:#fbf1de;
  --origin:#3a8f5e; --dest:#b3452e;
  --sans:'Space Grotesk',-apple-system,BlinkMacSystemFont,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#131513; --paper:#1a1d1a; --ink:#e9ece7; --ink-dim:#96a092; --rule:#2e332c;
    --accent:#7aa8d8; --accent-bg:#1b2733; --warm:#e0ab4c; --warm-bg:#2b230f;
    --origin:#5cc98a; --dest:#e0806a;
  }
}
:root[data-theme="dark"]{
  --bg:#131513; --paper:#1a1d1a; --ink:#e9ece7; --ink-dim:#96a092; --rule:#2e332c;
  --accent:#7aa8d8; --accent-bg:#1b2733; --warm:#e0ab4c; --warm-bg:#2b230f;
  --origin:#5cc98a; --dest:#e0806a;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans)}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:18px;margin:0 0 4px}
p.sub{font-size:12px;color:var(--ink-dim);margin:0 0 16px;line-height:1.6;font-family:var(--mono)}
.layout{display:grid;grid-template-columns:1fr 340px;gap:16px;align-items:start}
@media (max-width:820px){.layout{grid-template-columns:1fr}}
.mapbox{background:var(--paper);border:1px solid var(--rule);border-radius:10px;padding:10px;position:relative}
#mapCanvas{width:100%;height:640px;display:block;cursor:pointer;border-radius:6px}
.panel{background:var(--paper);border:1px solid var(--rule);border-radius:10px;padding:14px;font-family:var(--mono);font-size:12px;min-height:640px;max-height:640px;overflow-y:auto}
.panel h3{font-family:var(--sans);font-size:13px;margin:0 0 8px}
.panel .hint{color:var(--ink-dim);line-height:1.6}
.odrow{display:flex;justify-content:space-between;gap:8px;padding:7px 0;border-bottom:1px solid var(--rule);cursor:pointer}
.odrow:hover{background:var(--accent-bg)}
.odrow.active{background:var(--accent-bg)}
.odrow:last-child{border-bottom:none}
.odrow .names{flex:1}
.odrow .arrow{color:var(--ink-dim);margin:0 4px}
.odrow .share{color:var(--warm);font-weight:700;flex:0 0 auto}
.legend{display:flex;gap:14px;font-size:11px;color:var(--ink-dim);margin-bottom:10px;font-family:var(--mono);flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:5px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.sw{width:14px;height:3px;display:inline-block}
.rankbadge{font-family:var(--mono);font-size:10px;color:var(--ink-dim);flex:0 0 18px}
</style>
<div class="wrap">
  <h1>OD 라우팅 확인 지도 — 센서 클릭 → Top-K 기여 OD</h1>
  <p class="sub">실제 A* 도로망 배정(alpha=0.9, 2023년 train 기준) 결과. 점 = 396개 속도 센서(TOPIS 링크), 클릭하면 그 센서를 지나가는 상위 기여 OD 쌍 전체(최대 8개)와 각 경로, 출발/도착 행정동 경계를 보여줌. 목록의 항목을 클릭하면 해당 순위 경로만 강조됩니다.</p>
  <div class="layout">
    <div class="mapbox">
      <div class="legend">
        <span><span class="dot" style="background:var(--warm)"></span>센서 (클릭 가능)</span>
        <span><span class="dot" style="background:var(--accent)"></span>선택된 센서</span>
        <span><span class="sw" style="background:var(--accent)"></span>라우팅 경로</span>
        <span><span class="dot" style="background:var(--origin);border:1px solid var(--origin)"></span>출발동(O) 경계</span>
        <span><span class="dot" style="background:var(--dest);border:1px solid var(--dest)"></span>도착동(D) 경계</span>
      </div>
      <canvas id="mapCanvas" width="1600" height="1280"></canvas>
    </div>
    <div class="panel" id="odPanel">
      <h3>센서를 클릭하세요</h3>
      <p class="hint">지도 위 점을 클릭하면 그 센서 링크를 지나가는 OD(출발동→도착동) 쌍이 기여도(share) 순으로 나열되고, 전체 경로가 지도에 표시됩니다. share = 그 출발동 총 유출량 중 해당 목적지로 가는 비율. 목록 항목을 클릭하면 그 경로/동 경계만 강조됩니다.</p>
    </div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
const sensors = DATA.sensors;
const dongB = DATA.dong_boundaries;
const sensorIds = Object.keys(sensors);
const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const dpr = Math.min(devicePixelRatio||1, 2);
const W = 1600, H = 1280;
canvas.width = W*dpr; canvas.height = H*dpr; canvas.style.height = "640px";
ctx.scale(dpr, dpr);

let lats = sensorIds.map(id => sensors[id].latlon[0]);
let lons = sensorIds.map(id => sensors[id].latlon[1]);
const latMin = Math.min(...lats), latMax = Math.max(...lats);
const lonMin = Math.min(...lons), lonMax = Math.max(...lons);
const pad = 40;
function project(lat, lon) {
  const x = pad + (lon - lonMin) / (lonMax - lonMin) * (W - 2*pad);
  const y = H - pad - (lat - latMin) / (latMax - latMin) * (H - 2*pad);
  return [x, y];
}

const good = getComputedStyle(document.documentElement).getPropertyValue("--warm").trim();
const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
const rule = getComputedStyle(document.documentElement).getPropertyValue("--rule").trim();
const paper = getComputedStyle(document.documentElement).getPropertyValue("--paper").trim();
const ink = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim();
const originC = getComputedStyle(document.documentElement).getPropertyValue("--origin").trim();
const destC = getComputedStyle(document.documentElement).getPropertyValue("--dest").trim();

let selected = null;
let activeRank = null;  // null = show all K paths+boundaries faded, else highlight one rank

function drawPoly(rings, color, fillAlpha, lineAlpha) {
  rings.forEach(ring => {
    ctx.beginPath();
    ring.forEach((ll, i) => {
      const [x, y] = project(ll[0], ll[1]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.globalAlpha = fillAlpha; ctx.fillStyle = color; ctx.fill();
    ctx.globalAlpha = lineAlpha; ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.globalAlpha = 1;
  });
}

function drawPath(path, color, width, alpha) {
  if (!path || path.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha;
  ctx.beginPath();
  path.forEach((ll, i) => {
    const [x, y] = project(ll[0], ll[1]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function draw() {
  ctx.fillStyle = paper; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = rule; ctx.strokeRect(pad, pad, W-2*pad, H-2*pad);

  if (selected) {
    const s = sensors[selected];
    s.topk.forEach((o, i) => {
      const isActive = activeRank === null || activeRank === i;
      const pathAlpha = activeRank === null ? 0.55 : (isActive ? 1 : 0.12);
      const pathW = isActive && activeRank !== null ? 3 : 1.8;
      drawPath(o.path, accent, pathW, pathAlpha);
    });
    // dong boundaries: for the active rank (or all, faded, if none selected)
    s.topk.forEach((o, i) => {
      const isActive = activeRank === null || activeRank === i;
      const a = activeRank === null ? 0.5 : (isActive ? 0.85 : 0.06);
      const fillA = activeRank === null ? 0.05 : (isActive ? 0.14 : 0.01);
      if (dongB[o.origin_code]) drawPoly(dongB[o.origin_code], originC, fillA, a);
      if (dongB[o.dest_code]) drawPoly(dongB[o.dest_code], destC, fillA, a);
    });
  }

  sensorIds.forEach(id => {
    const s = sensors[id];
    const [x, y] = project(s.latlon[0], s.latlon[1]);
    const n = s.topk.length;
    const r = n === 0 ? 2 : Math.min(3 + n * 0.9, 9);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 7);
    ctx.fillStyle = id === selected ? accent : (n === 0 ? rule : good);
    ctx.globalAlpha = id === selected ? 1 : 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
  });
}
draw();

function findNearest(mx, my) {
  let best = null, bestD = 14;
  sensorIds.forEach(id => {
    const s = sensors[id];
    const [x, y] = project(s.latlon[0], s.latlon[1]);
    const d = Math.hypot(x-mx, y-my);
    if (d < bestD) { bestD = d; best = id; }
  });
  return best;
}

function fmtShare(s) { return (s*100).toFixed(1) + "%"; }

function showPanel(id) {
  const s = sensors[id];
  const panel = document.getElementById("odPanel");
  if (!s.topk.length) {
    panel.innerHTML = `<h3>센서 ${id}</h3><p class="hint">이 센서를 지나가는 OD 쌍이 없습니다 (도로망은 있지만 상위-공유 OD 경로가 안 지나감).</p>`;
    return;
  }
  const rows = s.topk.map((o, i) => `
    <div class="odrow" data-rank="${i}">
      <span class="rankbadge">#${i+1}</span>
      <span class="names"><span style="color:var(--origin)">${o.origin_name}</span> <span class="arrow">→</span> <span style="color:var(--dest)">${o.dest_name}</span><br><span style="color:var(--ink-dim);font-size:10px">${o.hops} hops${o.path.length ? "" : " (경로 데이터 없음)"}</span></span>
      <span class="share">${fmtShare(o.share)}</span>
    </div>`).join("");
  panel.innerHTML = `<h3>센서 ${id} — 상위 ${s.topk.length}개 기여 OD</h3>${rows}
    <p class="hint" style="margin-top:10px">모든 경로가 옅게 표시됩니다. 항목을 클릭하면 그 경로/동 경계만 강조됩니다. (녹색=출발동, 적색=도착동)</p>`;
  panel.querySelectorAll(".odrow").forEach(el => {
    el.addEventListener("click", () => {
      const r = parseInt(el.dataset.rank);
      activeRank = activeRank === r ? null : r;
      panel.querySelectorAll(".odrow").forEach(e2 => e2.classList.remove("active"));
      if (activeRank === r) el.classList.add("active");
      draw();
    });
  });
}

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) / rect.width * W;
  const my = (e.clientY - rect.top) / rect.height * H;
  const id = findNearest(mx, my);
  if (id) { selected = id; activeRank = null; draw(); showPanel(id); }
});
</script>
"""
html = html.replace("__DATA_JSON__", DATA_JSON)
with open(f"{GTS}/route_map.html", "w") as f:
    f.write(html)
print("wrote route_map.html", len(html), "bytes")
