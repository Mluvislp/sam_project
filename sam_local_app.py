from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template_string, request, send_from_directory

ROOT = Path(r"D:\Python\sam_project")
REPO = ROOT / "segment-anything"
sys.path.insert(0, str(REPO))

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402

CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
UPLOAD_DIR = ROOT / "uploads_gui"
OUTPUT_DIR = ROOT / "outputs_gui"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SAM Local Demo</title>
  <style>
    :root {
      --bg:#0f172a; --panel:#111827; --panel2:#1f2937; --soft:#0b1220; --text:#e5e7eb; --muted:#94a3b8;
      --blue:#2563eb; --blue2:#1d4ed8; --green:#22c55e; --red:#ef4444; --border:#334155; --yellow:#f59e0b;
    }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Arial,sans-serif; background:linear-gradient(180deg,#0b1220 0%,#0f172a 100%); color:var(--text); }
    .wrap { max-width:1380px; margin:0 auto; padding:24px; }
    .box { background:rgba(17,24,39,.96); border:1px solid rgba(148,163,184,.12); border-radius:20px; padding:24px; box-shadow:0 14px 45px rgba(0,0,0,.28); }
    .hero { display:flex; justify-content:flex-end; gap:16px; align-items:flex-start; flex-wrap:wrap; }
    .pill { display:inline-block; background:#0b1220; border:1px solid #223048; color:#93c5fd; padding:6px 10px; border-radius:999px; font-size:13px; }
    .main-grid { display:grid; grid-template-columns:390px 1fr; gap:20px; margin-top:22px; }
    @media (max-width:1050px) { .main-grid { grid-template-columns:1fr; } }
    .sidebar, .viewer, .result-card { background:var(--panel2); border-radius:16px; padding:18px; }
    .section-title { margin:0 0 14px 0; font-size:18px; }
    .input-group { margin-bottom:14px; }
    .input-group label { display:block; margin-bottom:6px; color:#cbd5e1; font-size:14px; }
    input[type=file] { width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:var(--soft); color:white; }
    button { background:var(--blue); color:white; border:none; padding:11px 16px; border-radius:10px; cursor:pointer; font-weight:700; transition:.16s ease; }
    button:hover { background:var(--blue2); transform:translateY(-1px); }
    button.secondary { background:#334155; }
    button.secondary:hover { background:#475569; }
    button.ghost { background:#0b1220; border:1px solid var(--border); }
    button.ghost:hover { background:#162033; }
    .btn-row { display:flex; gap:10px; flex-wrap:wrap; }
    .hint { background:var(--soft); border-left:4px solid #38bdf8; padding:12px; border-radius:10px; color:#cbd5e1; font-size:14px; }
    .status-box { margin-top:14px; background:var(--soft); border:1px solid #1e293b; padding:12px; border-radius:10px; }
    .status-box strong { color:#93c5fd; }
    .viewer-head { display:flex; justify-content:space-between; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
    .canvas-wrap { position:relative; display:inline-block; max-width:100%; background:#020617; border-radius:14px; overflow:hidden; border:1px solid #1e293b; }
    #preview, #overlay { max-width:100%; max-height:74vh; display:none; user-select:none; -webkit-user-drag:none; }
    #preview.visible, #overlay.visible { display:block; }
    #overlay { position:absolute; left:0; top:0; cursor:crosshair; }
    .click-badge { font-size:13px; color:#a5f3fc; background:#0b1220; padding:6px 10px; border-radius:999px; border:1px solid #223048; }
    .points-list { margin-top:12px; max-height:200px; overflow:auto; background:var(--soft); border:1px solid #1e293b; border-radius:12px; padding:10px; font-size:14px; }
    .point-item { padding:6px 8px; border-radius:8px; margin-bottom:6px; display:flex; justify-content:space-between; gap:8px; align-items:center; }
    .point-item.pos { background:rgba(34,197,94,.12); }
    .point-item.neg { background:rgba(239,68,68,.12); }
    .point-meta { color:#e2e8f0; }
    .point-type.pos { color:#86efac; }
    .point-type.neg { color:#fca5a5; }
    .small-btn { padding:6px 10px; font-size:12px; border-radius:8px; }
    .legend { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; color:#cbd5e1; font-size:13px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:6px; }
    .result-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px; }
    @media (max-width:980px) { .result-grid { grid-template-columns:1fr; } }
    .result-card img { max-width:100%; border-radius:12px; border:1px solid #1e293b; background:white; }
    .meta, .error { white-space:pre-line; padding:12px; border-radius:12px; margin-top:16px; }
    .meta { background:var(--soft); color:#dbeafe; border:1px solid #1e293b; }
    .error { background:#2a1115; color:#fecaca; border:1px solid #7f1d1d; }
    a { color:#93c5fd; }
    .spinner { display:none; width:16px; height:16px; border:2px solid rgba(255,255,255,.25); border-top-color:#fff; border-radius:50%; animation:spin .8s linear infinite; }
    .loading .spinner { display:inline-block; }
    .mode-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
    .mode-btn.active { outline:2px solid #93c5fd; }
    .box-info { margin-top:10px; padding:10px; border-radius:10px; background:var(--soft); border:1px solid #1e293b; font-size:14px; }
    @keyframes spin { to { transform:rotate(360deg); } }
  </style>
</head>
<body>
<div class="wrap">
  <div class="box">
    <div class="hero">
      <div class="pill">Checkpoint: sam_vit_b_01ec64.pth • CPU</div>
    </div>

    <div class="main-grid">
      <div class="sidebar">
        <h3 class="section-title">Bảng điều khiển</h3>
        <div class="input-group">
          <label>Ảnh đầu vào</label>
          <input type="file" id="imageInput" accept="image/*">
        </div>
        <div class="btn-row">
          <button id="extractBtn"><span class="spinner"></span> <span class="btn-text">Extract image embedding</span></button>
          <button class="secondary" id="segmentBtn" disabled>Phân vùng</button>
          <button class="ghost" id="clearPointsBtn" disabled>Xóa điểm</button>
          <button class="ghost" id="clearBoxBtn" disabled>Xóa box</button>
        </div>
        <div class="mode-row">
          <button class="ghost mode-btn active" id="modePointBtn">Chế độ Point</button>
          <button class="ghost mode-btn" id="modeBoxBtn">Chế độ Box</button>
        </div>
        <div class="status-box"><strong>Trạng thái:</strong><br><span id="statusText">Chưa chọn ảnh.</span></div>
        <div class="legend">
          <div><span class="dot" style="background:#22c55e"></span>Điểm dương</div>
          <div><span class="dot" style="background:#ef4444"></span>Điểm âm</div>
          <div><span class="dot" style="background:#f59e0b"></span>Box</div>
        </div>
        <div class="box-info" id="boxInfo">Chưa có box nào.</div>
        <div class="points-list" id="pointsList">Chưa có điểm nào.</div>
      </div>

      <div class="viewer">
        <div class="viewer-head">
          <h3 class="section-title" style="margin:0;">Khung ảnh tương tác</h3>
          <div class="click-badge" id="coordBadge">Chưa sẵn sàng</div>
        </div>
        <div class="canvas-wrap" id="canvasWrap">
          <img id="preview">
          <canvas id="overlay"></canvas>
        </div>
      </div>
    </div>

    <div id="errorBox" class="error" style="display:none"></div>

    <div id="resultSection" style="display:none">
      <div class="result-grid">
        <div class="result-card">
          <h3 class="section-title">Ảnh gốc + prompt</h3>
          <img id="inputResult">
        </div>
        <div class="result-card">
          <h3 class="section-title">Kết quả mask SAM</h3>
          <img id="maskResult">
          <p><a id="downloadLink" href="#" download>Tải ảnh kết quả</a></p>
        </div>
      </div>
      <div class="result-grid">
        <div class="result-card">
          <h3 class="section-title">Ảnh đã tách khỏi nền</h3>
          <img id="cutoutTransparentResult">
          <p><a id="downloadTransparentLink" href="#" download>Tải ảnh đã tách khỏi nền</a></p>
        </div>
      </div>
      <div class="meta" id="metaBox"></div>
    </div>
  </div>
</div>
<script>
(() => {
  const imageInput = document.getElementById('imageInput');
  const extractBtn = document.getElementById('extractBtn');
  const segmentBtn = document.getElementById('segmentBtn');
  const clearPointsBtn = document.getElementById('clearPointsBtn');
  const clearBoxBtn = document.getElementById('clearBoxBtn');
  const modePointBtn = document.getElementById('modePointBtn');
  const modeBoxBtn = document.getElementById('modeBoxBtn');
  const statusText = document.getElementById('statusText');
  const coordBadge = document.getElementById('coordBadge');
  const boxInfo = document.getElementById('boxInfo');
  const preview = document.getElementById('preview');
  const overlay = document.getElementById('overlay');
  const pointsList = document.getElementById('pointsList');
  const errorBox = document.getElementById('errorBox');
  const resultSection = document.getElementById('resultSection');
  const inputResult = document.getElementById('inputResult');
  const maskResult = document.getElementById('maskResult');
  const cutoutTransparentResult = document.getElementById('cutoutTransparentResult');
  const metaBox = document.getElementById('metaBox');
  const downloadLink = document.getElementById('downloadLink');
  const downloadTransparentLink = document.getElementById('downloadTransparentLink');

  let currentFile = null;
  let points = [];
  let box = null;
  let embeddingReady = false;
  let mode = 'point';
  let isDragging = false;
  let dragStart = null;
  let tempBox = null;

  function setStatus(text) { statusText.textContent = text; }
  function setError(text='') {
    if (!text) { errorBox.style.display = 'none'; errorBox.textContent = ''; return; }
    errorBox.style.display = 'block'; errorBox.textContent = text;
  }
  function toggleBtnLoading(btn, loading, textWhenLoading) {
    const txt = btn.querySelector('.btn-text');
    if (loading) { btn.classList.add('loading'); btn.disabled = true; if (txt && textWhenLoading) txt.textContent = textWhenLoading; }
    else { btn.classList.remove('loading'); btn.disabled = false; if (txt) txt.textContent = 'Extract image embedding'; }
  }
  function updateModeUI() {
    modePointBtn.classList.toggle('active', mode === 'point');
    modeBoxBtn.classList.toggle('active', mode === 'box');
    if (mode === 'point') setStatus(embeddingReady ? 'Chế độ Point: click trái thêm điểm dương, click phải thêm điểm âm.' : 'Chưa chọn ảnh.');
    else setStatus(embeddingReady ? 'Chế độ Box: giữ và kéo chuột trái để vẽ box.' : 'Chưa chọn ảnh.');
  }
  function updateCoordBadge() {
    const pieces = [];
    pieces.push(embeddingReady ? 'Sẵn sàng' : 'Chưa sẵn sàng');
    pieces.push(`${points.length} điểm`);
    if (box) pieces.push('1 box');
    coordBadge.textContent = pieces.join(' • ');
  }
  function updateBoxInfo() {
    if (!box) boxInfo.textContent = 'Chưa có box nào.';
    else boxInfo.textContent = `Box hiện tại: [x1=${box[0]}, y1=${box[1]}, x2=${box[2]}, y2=${box[3]}]`;
  }
  function imageCoordsFromEvent(e) {
    const rect = preview.getBoundingClientRect();
    const scaleX = preview.naturalWidth / rect.width;
    const scaleY = preview.naturalHeight / rect.height;
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY)
    };
  }
  function canvasCoords(x, y) {
    const sx = preview.clientWidth / preview.naturalWidth;
    const sy = preview.clientHeight / preview.naturalHeight;
    return { x: x * sx, y: y * sy };
  }
  function redrawOverlay() {
    if (!preview.naturalWidth) return;
    overlay.width = preview.clientWidth;
    overlay.height = preview.clientHeight;
    overlay.style.width = preview.clientWidth + 'px';
    overlay.style.height = preview.clientHeight + 'px';
    overlay.classList.add('visible');
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0,0,overlay.width,overlay.height);

    points.forEach((p, idx) => {
      const c = canvasCoords(p.x, p.y);
      ctx.beginPath();
      ctx.arc(c.x, c.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? '#22c55e' : '#ef4444';
      ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = '#ffffff'; ctx.stroke();
      ctx.fillStyle = '#ffffff'; ctx.font = 'bold 12px Arial'; ctx.fillText(String(idx + 1), c.x + 10, c.y - 10);
    });

    const drawBox = (b, dashed=false) => {
      if (!b) return;
      const p1 = canvasCoords(b[0], b[1]);
      const p2 = canvasCoords(b[2], b[3]);
      const x = Math.min(p1.x, p2.x), y = Math.min(p1.y, p2.y), w = Math.abs(p2.x-p1.x), h = Math.abs(p2.y-p1.y);
      ctx.save();
      if (dashed) ctx.setLineDash([6,4]);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    };
    drawBox(box, false);
    drawBox(tempBox, true);
  }
  function renderPointsList() {
    if (!points.length) {
      pointsList.textContent = embeddingReady ? 'Chưa có điểm nào. Hãy click lên ảnh.' : 'Chưa có điểm nào.';
    } else {
      pointsList.innerHTML = '';
      points.forEach((p, idx) => {
        const row = document.createElement('div');
        row.className = 'point-item ' + (p.label === 1 ? 'pos' : 'neg');
        row.innerHTML = `<div class="point-meta"><span class="point-type ${p.label === 1 ? 'pos' : 'neg'}">${p.label === 1 ? 'Dương' : 'Âm'}</span> • (#${idx+1}) (${p.x}, ${p.y})</div>`;
        const btn = document.createElement('button');
        btn.className = 'small-btn secondary'; btn.textContent = 'Xóa';
        btn.onclick = () => { points.splice(idx, 1); renderPointsList(); updateCoordBadge(); redrawOverlay(); updateButtons(); };
        row.appendChild(btn); pointsList.appendChild(row);
      });
    }
    updateCoordBadge(); updateBoxInfo(); redrawOverlay();
  }
  function updateButtons() {
    segmentBtn.disabled = !(embeddingReady && (points.length > 0 || box));
    clearPointsBtn.disabled = !(embeddingReady && points.length > 0);
    clearBoxBtn.disabled = !(embeddingReady && box);
  }

  modePointBtn.addEventListener('click', () => { mode = 'point'; updateModeUI(); redrawOverlay(); });
  modeBoxBtn.addEventListener('click', () => { mode = 'box'; updateModeUI(); redrawOverlay(); });

  imageInput.addEventListener('change', () => {
    const file = imageInput.files && imageInput.files[0]; if (!file) return;
    currentFile = file; embeddingReady = false; points = []; box = null; tempBox = null; resultSection.style.display = 'none'; setError('');
    const url = URL.createObjectURL(file); preview.src = url; preview.classList.add('visible');
    preview.onload = () => { redrawOverlay(); setStatus('Ảnh đã hiển thị. Bấm Extract image embedding để chuẩn bị mô hình.'); updateCoordBadge(); updateBoxInfo(); updateButtons(); };
  });

  extractBtn.addEventListener('click', async () => {
    if (!currentFile) { setError('Bạn cần chọn ảnh trước.'); return; }
    setError(''); toggleBtnLoading(extractBtn, true, 'Đang extracting...'); setStatus('Extracting image embedding...');
    try {
      const form = new FormData(); form.append('image', currentFile);
      const res = await fetch('/api/upload', { method:'POST', body: form });
      const data = await res.json(); if (!data.ok) throw new Error(data.error || 'Upload failed');
      embeddingReady = true; points = []; box = null; tempBox = null; preview.src = data.preview_url + '?t=' + Date.now(); preview.classList.add('visible');
      updateModeUI(); renderPointsList(); updateButtons();
    } catch (e) { setError('Lỗi khi extracting image embedding: ' + e.message); setStatus('Extract embedding thất bại.'); }
    finally { toggleBtnLoading(extractBtn, false); }
  });

  preview.addEventListener('contextmenu', (e) => { if (embeddingReady) e.preventDefault(); });

  overlay.addEventListener('click', (e) => {
    if (!embeddingReady || mode !== 'point' || isDragging) return;
    const {x,y} = imageCoordsFromEvent(e);
    points.push({x, y, label: 1});
    setStatus(`Đã thêm điểm dương tại (${x}, ${y}).`);
    renderPointsList(); updateButtons();
  });

  overlay.addEventListener('mousedown', (e) => {
    if (!embeddingReady) return;
    if (mode === 'point') {
      if (e.button === 2) {
        e.preventDefault();
        const {x,y} = imageCoordsFromEvent(e);
        points.push({x, y, label: 0});
        setStatus(`Đã thêm điểm âm tại (${x}, ${y}).`);
        renderPointsList(); updateButtons();
      }
      return;
    }
    if (mode === 'box' && e.button === 0) {
      isDragging = true;
      const {x,y} = imageCoordsFromEvent(e);
      dragStart = {x,y};
      tempBox = [x,y,x,y];
      redrawOverlay();
    }
  });

  overlay.addEventListener('mousemove', (e) => {
    if (!embeddingReady || mode !== 'box' || !isDragging || !dragStart) return;
    const {x,y} = imageCoordsFromEvent(e);
    tempBox = [dragStart.x, dragStart.y, x, y];
    redrawOverlay();
  });

  const finishBox = (e) => {
    if (!embeddingReady || mode !== 'box' || !isDragging || !dragStart) return;
    const {x,y} = imageCoordsFromEvent(e);
    let x1 = Math.min(dragStart.x, x), y1 = Math.min(dragStart.y, y), x2 = Math.max(dragStart.x, x), y2 = Math.max(dragStart.y, y);
    if (Math.abs(x2-x1) > 4 && Math.abs(y2-y1) > 4) {
      box = [x1,y1,x2,y2];
      setStatus(`Đã tạo box: [${x1}, ${y1}, ${x2}, ${y2}].`);
    }
    isDragging = false; dragStart = null; tempBox = null;
    renderPointsList(); updateButtons();
  };
  overlay.addEventListener('mouseup', finishBox);
  overlay.addEventListener('mouseleave', (e) => { if (isDragging) finishBox(e); });

  clearPointsBtn.addEventListener('click', () => { points = []; setStatus('Đã xóa toàn bộ điểm.'); renderPointsList(); updateButtons(); });
  clearBoxBtn.addEventListener('click', () => { box = null; tempBox = null; setStatus('Đã xóa box.'); renderPointsList(); updateButtons(); });

  segmentBtn.addEventListener('click', async () => {
    if (!embeddingReady) { setError('Bạn cần extract image embedding trước.'); return; }
    if (!(points.length || box)) { setError('Bạn cần chọn ít nhất một điểm hoặc một box.'); return; }
    setError(''); segmentBtn.disabled = true; setStatus('Đang phân vùng từ prompt đã chọn...');
    try {
      const res = await fetch('/api/predict', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ points, box })
      });
      const data = await res.json(); if (!data.ok) throw new Error(data.error || 'Predict failed');
      inputResult.src = data.input_url + '?t=' + Date.now();
      maskResult.src = data.result_url + '?t=' + Date.now();
      cutoutTransparentResult.src = data.cutout_transparent_url + '?t=' + Date.now();
      downloadLink.href = data.result_url + '?t=' + Date.now();
      downloadTransparentLink.href = data.cutout_transparent_url + '?t=' + Date.now();
      metaBox.textContent = data.info; resultSection.style.display = 'block';
      setStatus('Đã phân vùng xong. Bạn có thể chỉnh point hoặc box rồi chạy lại.');
    } catch (e) { setError('Lỗi khi phân vùng: ' + e.message); setStatus('Phân vùng thất bại.'); }
    finally { updateButtons(); }
  });

  window.addEventListener('resize', redrawOverlay);
  updateModeUI(); updateCoordBadge(); updateBoxInfo(); updateButtons();
})();
</script>
</body>
</html>
'''


def build_model():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device="cpu")
    return sam

SAM_MODEL = build_model()
PREDICTOR = SamPredictor(SAM_MODEL)
app = Flask(__name__)
STATE = {"image_path": None, "image_rgb": None, "width": None, "height": None, "embedding_ready": False}


def make_overlay(image_rgb, mask, points, box=None):
    overlay = image_rgb.copy()
    color = np.array([30, 144, 255], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    for p in points:
        x, y, label = p["x"], p["y"], p["label"]
        pt_color = (0, 255, 0) if label == 1 else (255, 64, 64)
        cv2.circle(overlay, (x, y), 8, pt_color, -1)
        cv2.circle(overlay, (x, y), 12, (255, 255, 255), 2)
    if box:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (245, 158, 11), 3)
    return overlay


def make_prompts_image(image_rgb, points, box=None):
    vis = image_rgb.copy()
    for p in points:
        x, y, label = p["x"], p["y"], p["label"]
        pt_color = (0, 255, 0) if label == 1 else (255, 64, 64)
        cv2.circle(vis, (x, y), 8, pt_color, -1)
        cv2.circle(vis, (x, y), 12, (255, 255, 255), 2)
    if box:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (245, 158, 11), 3)
    return vis


def make_cutout_transparent(image_rgb, mask):
    return np.dstack([image_rgb, np.where(mask, 255, 0).astype(np.uint8)])


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    file = request.files.get('image')
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Chưa chọn ảnh."}), 400
    try:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        safe_name = Path(file.filename).name
        in_name = f'{stamp}-input-{safe_name}'
        in_path = UPLOAD_DIR / in_name
        file.save(in_path)
        image_bgr = cv2.imread(str(in_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        PREDICTOR.set_image(image_rgb)
        STATE.update({"image_path": str(in_path), "image_rgb": image_rgb, "width": w, "height": h, "embedding_ready": True})
        return jsonify({"ok": True, "preview_url": f'/file/upload/{in_name}', "width": w, "height": h})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not STATE["embedding_ready"] or STATE["image_rgb"] is None:
        return jsonify({"ok": False, "error": "Bạn cần extract image embedding trước."}), 400
    try:
        data = request.get_json(force=True)
        points = data.get('points', []) or []
        box = data.get('box')
        if not points and not box:
            return jsonify({"ok": False, "error": "Bạn cần chọn ít nhất một điểm hoặc một box."}), 400

        kwargs = {"multimask_output": True}
        if points:
            kwargs["point_coords"] = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.float32)
            kwargs["point_labels"] = np.array([int(p['label']) for p in points], dtype=np.int32)
        if box:
            kwargs["box"] = np.array([int(v) for v in box], dtype=np.float32)

        masks, scores, _ = PREDICTOR.predict(**kwargs)
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]
        image_rgb = STATE["image_rgb"]
        stamp = time.strftime('%Y%m%d-%H%M%S')
        out_name = f'{stamp}-sam-output.png'
        input_vis_name = f'{stamp}-input-annotated.png'
        cutout_name = f'{stamp}-sam-cutout-transparent.png'
        out_path = OUTPUT_DIR / out_name
        input_vis_path = OUTPUT_DIR / input_vis_name
        cutout_path = OUTPUT_DIR / cutout_name

        Image.fromarray(make_overlay(image_rgb, best_mask, points, box)).save(out_path)
        Image.fromarray(make_prompts_image(image_rgb, points, box)).save(input_vis_path)
        Image.fromarray(make_cutout_transparent(image_rgb, best_mask), mode='RGBA').save(cutout_path)

        pos_count = sum(1 for p in points if int(p['label']) == 1)
        neg_count = sum(1 for p in points if int(p['label']) == 0)
        box_text = 'Có' if box else 'Không'
        info = (
            f'Thiết bị chạy: CPU\n'
            f'Checkpoint: {CHECKPOINT}\n'
            f'Ảnh đầu vào: {STATE["image_path"]}\n'
            f'Kích thước ảnh: {STATE["width"]}x{STATE["height"]}\n'
            f'Số điểm dương: {pos_count}\n'
            f'Số điểm âm: {neg_count}\n'
            f'Có box: {box_text}\n'
            f'Score mask tốt nhất: {scores[best_idx]:.4f}\n'
            f'Ảnh mask: {out_path}\n'
            f'Ảnh đã tách khỏi nền: {cutout_path}'
        )
        return jsonify({
            "ok": True,
            "result_url": f'/file/output/{out_name}',
            "input_url": f'/file/output/{input_vis_name}',
            "cutout_transparent_url": f'/file/output/{cutout_name}',
            "info": info,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/file/output/<path:name>')
def output_file(name):
    return send_from_directory(OUTPUT_DIR, name)


@app.route('/file/upload/<path:name>')
def upload_file(name):
    return send_from_directory(UPLOAD_DIR, name)


if __name__ == '__main__':
    print('SAM local demo running at http://127.0.0.1:7861')
    app.run(host='127.0.0.1', port=7861, debug=False)

