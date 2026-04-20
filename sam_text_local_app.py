from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT = Path(r"D:\Python\sam_project")
REPO = ROOT / "segment-anything"
sys.path.insert(0, str(REPO))
from segment_anything import sam_model_registry, SamPredictor  # noqa: E402

CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cpu"
UPLOAD_DIR = ROOT / "uploads_text_gui"
OUTPUT_DIR = ROOT / "outputs_text_gui"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SAM Text Prompt Experimental</title>
  <style>
    :root { --bg:#0f172a; --panel:#111827; --panel2:#1f2937; --soft:#0b1220; --text:#e5e7eb; --muted:#94a3b8; --blue:#2563eb; --blue2:#1d4ed8; --border:#334155; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Arial,sans-serif; background:linear-gradient(180deg,#0b1220 0%,#0f172a 100%); color:var(--text); }
    .wrap { max-width:1380px; margin:0 auto; padding:24px; }
    .box { background:rgba(17,24,39,.96); border:1px solid rgba(148,163,184,.12); border-radius:20px; padding:24px; box-shadow:0 14px 45px rgba(0,0,0,.28); }
    .hero { display:flex; justify-content:space-between; gap:12px; align-items:center; flex-wrap:wrap; }
    .main-grid { display:grid; grid-template-columns:380px 1fr; gap:20px; margin-top:22px; }
    @media (max-width:1050px) { .main-grid { grid-template-columns:1fr; } }
    .sidebar, .viewer, .result-card { background:var(--panel2); border-radius:16px; padding:18px; }
    .section-title { margin:0 0 14px 0; font-size:18px; }
    .input-group { margin-bottom:14px; }
    .input-group label { display:block; margin-bottom:6px; color:#cbd5e1; font-size:14px; }
    input[type=file], input[type=text], input[type=number] { width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:var(--soft); color:white; }
    button { background:var(--blue); color:white; border:none; padding:11px 16px; border-radius:10px; cursor:pointer; font-weight:700; transition:.16s ease; }
    button:hover { background:var(--blue2); transform:translateY(-1px); }
    button.secondary { background:#334155; }
    button.secondary:hover { background:#475569; }
    .btn-row { display:flex; gap:10px; flex-wrap:wrap; }
    .status-box, .meta, .error { padding:12px; border-radius:12px; margin-top:14px; white-space:pre-line; }
    .status-box, .meta { background:var(--soft); border:1px solid #1e293b; }
    .error { background:#2a1115; color:#fecaca; border:1px solid #7f1d1d; }
    .viewer img, .result-card img { max-width:100%; border-radius:12px; border:1px solid #1e293b; background:white; }
    .result-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px; }
    @media (max-width:980px) { .result-grid { grid-template-columns:1fr; } }
    a { color:#93c5fd; }
    .small { color:#94a3b8; font-size:13px; }
    .pill { display:inline-block; background:#0b1220; border:1px solid #223048; color:#93c5fd; padding:6px 10px; border-radius:999px; font-size:13px; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="box">
    <div class="hero">
      <div><h2 style="margin:0">SAM Text Prompt Experimental</h2></div>
      <div class="pill">Flow: image → extract embedding → text → Grounding DINO → SAM</div>
    </div>

    <div class="main-grid">
      <div class="sidebar">
        <h3 class="section-title">Điều khiển</h3>
        <div class="input-group">
          <label>Ảnh đầu vào</label>
          <input type="file" id="imageInput" accept="image/*">
        </div>
        <div class="btn-row">
          <button id="extractBtn">Extract image embedding</button>
        </div>
        <div class="input-group" style="margin-top:14px;">
          <label>Text prompt</label>
          <input type="text" id="textPrompt" placeholder="Ví dụ: person, dog, bottle, red car">
        </div>
        <div class="input-group">
          <label>Ngưỡng box (box threshold)</label>
          <input type="number" id="boxThreshold" value="0.35" step="0.05">
        </div>
        <div class="input-group">
          <label>Ngưỡng text (text threshold)</label>
          <input type="number" id="textThreshold" value="0.25" step="0.05">
        </div>
        <div class="btn-row">
          <button class="secondary" id="runBtn" disabled>Chạy text prompt</button>
        </div>
        <div class="status-box"><b>Trạng thái:</b><br><span id="statusText">Chưa chọn ảnh.</span></div>
        <div id="errorBox" class="error" style="display:none"></div>
      </div>

      <div class="viewer">
        <h3 class="section-title">Xem trước ảnh</h3>
        <img id="preview" style="display:none">
        <div id="resultSection" style="display:none">
          <div class="result-grid">
            <div class="result-card">
              <h3 class="section-title">Ảnh gốc + box Grounding DINO</h3>
              <img id="groundedInput">
            </div>
            <div class="result-card">
              <h3 class="section-title">Kết quả mask SAM</h3>
              <img id="maskResult">
              <p><a id="maskLink" href="#" download>Tải ảnh kết quả</a></p>
            </div>
          </div>
          <div class="result-grid">
            <div class="result-card">
              <h3 class="section-title">Ảnh đã tách khỏi nền</h3>
              <img id="cutoutResult">
              <p><a id="cutoutLink" href="#" download>Tải ảnh đã tách khỏi nền</a></p>
            </div>
          </div>
          <div class="meta" id="metaBox"></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
(() => {
  const imageInput = document.getElementById('imageInput');
  const extractBtn = document.getElementById('extractBtn');
  const textPrompt = document.getElementById('textPrompt');
  const boxThreshold = document.getElementById('boxThreshold');
  const textThreshold = document.getElementById('textThreshold');
  const runBtn = document.getElementById('runBtn');
  const statusText = document.getElementById('statusText');
  const errorBox = document.getElementById('errorBox');
  const preview = document.getElementById('preview');
  const resultSection = document.getElementById('resultSection');
  const groundedInput = document.getElementById('groundedInput');
  const maskResult = document.getElementById('maskResult');
  const cutoutResult = document.getElementById('cutoutResult');
  const maskLink = document.getElementById('maskLink');
  const cutoutLink = document.getElementById('cutoutLink');
  const metaBox = document.getElementById('metaBox');

  let currentFile = null;
  let embeddingReady = false;
  function setStatus(t){ statusText.textContent = t; }
  function setError(t=''){ if(!t){ errorBox.style.display='none'; errorBox.textContent=''; return; } errorBox.style.display='block'; errorBox.textContent=t; }
  function updateButtons(){ runBtn.disabled = !embeddingReady; }

  imageInput.addEventListener('change', ()=>{
    const file = imageInput.files && imageInput.files[0];
    if(!file) return;
    currentFile = file;
    embeddingReady = false;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = 'block';
    resultSection.style.display = 'none';
    setError('');
    setStatus('Chưa extract');
    updateButtons();
  });

  extractBtn.addEventListener('click', async ()=>{
    if(!currentFile){ setError('Bạn cần chọn ảnh trước.'); return; }
    setError('');
    setStatus('Đang extract...');
    extractBtn.disabled = true;
    try {
      const form = new FormData();
      form.append('image', currentFile);
      const res = await fetch('/api/extract_image', { method:'POST', body: form });
      const data = await res.json();
      if(!data.ok) throw new Error(data.error || 'Extract failed');
      embeddingReady = true;
      preview.src = data.preview_url + '?t=' + Date.now();
      preview.style.display = 'block';
      setStatus('Extract xong');
      updateButtons();
    } catch(e) {
      setError('Lỗi extract image embedding: ' + e.message);
      setStatus('Extract lỗi');
      embeddingReady = false;
      updateButtons();
    } finally {
      extractBtn.disabled = false;
    }
  });

  runBtn.addEventListener('click', async ()=>{
    if(!currentFile){ setError('Bạn cần chọn ảnh trước.'); return; }
    if(!embeddingReady){ setError('Bạn cần extract image embedding trước.'); return; }
    if(!textPrompt.value.trim()){ setError('Bạn cần nhập text prompt.'); return; }
    setError('');
    setStatus('Đang chạy...');
    runBtn.disabled = true;
    try {
      const form = new FormData();
      form.append('text_prompt', textPrompt.value.trim());
      form.append('box_threshold', boxThreshold.value);
      form.append('text_threshold', textThreshold.value);
      const res = await fetch('/api/text_predict', { method:'POST', body: form });
      const data = await res.json();
      if(!data.ok) throw new Error(data.error || 'Text prompt failed');
      groundedInput.src = data.input_url + '?t=' + Date.now();
      maskResult.src = data.result_url + '?t=' + Date.now();
      cutoutResult.src = data.cutout_url + '?t=' + Date.now();
      maskLink.href = data.result_url + '?t=' + Date.now();
      cutoutLink.href = data.cutout_url + '?t=' + Date.now();
      metaBox.textContent = data.info;
      resultSection.style.display = 'block';
      setStatus('Hoàn tất');
    } catch(e) {
      setError('Lỗi text prompt: ' + e.message);
      setStatus('Có lỗi');
    } finally {
      runBtn.disabled = false;
    }
  });
})();
</script>
</body>
</html>
'''


def build_sam():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device=DEVICE)
    return sam

SAM_MODEL = build_sam()
PREDICTOR = SamPredictor(SAM_MODEL)
GD_PROCESSOR = None
GD_MODEL = None
app = Flask(__name__)
STATE = {"image_path": None, "image": None, "image_rgb": None, "width": None, "height": None, "embedding_ready": False}


def load_grounding_dino():
    global GD_PROCESSOR, GD_MODEL
    if GD_PROCESSOR is None or GD_MODEL is None:
        GD_PROCESSOR = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
        GD_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL_ID).to(DEVICE)
        GD_MODEL.eval()


def make_box_image(image_rgb, box, label_text):
    vis = image_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (245, 158, 11), 3)
    tag = label_text[:60]
    cv2.putText(vis, tag, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 158, 11), 2, cv2.LINE_AA)
    return vis


def make_overlay(image_rgb, mask, box=None):
    overlay = image_rgb.copy()
    color = np.array([30, 144, 255], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (245, 158, 11), 3)
    return overlay


def make_cutout_transparent(image_rgb, mask):
    return np.dstack([image_rgb, np.where(mask, 255, 0).astype(np.uint8)])


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)


@app.route('/api/extract_image', methods=['POST'])
def api_extract_image():
    file = request.files.get('image')
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Chưa chọn ảnh."}), 400
    try:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        safe_name = Path(file.filename).name
        in_name = f'{stamp}-input-{safe_name}'
        in_path = UPLOAD_DIR / in_name
        file.save(in_path)

        image = Image.open(in_path).convert('RGB')
        image_rgb = np.array(image)
        h, w = image_rgb.shape[:2]

        PREDICTOR.set_image(image_rgb)
        STATE.update({
            "image_path": str(in_path),
            "image": image,
            "image_rgb": image_rgb,
            "width": w,
            "height": h,
            "embedding_ready": True,
        })

        return jsonify({
            "ok": True,
            "preview_url": f'/file/upload/{in_name}',
            "width": w,
            "height": h,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/text_predict', methods=['POST'])
def api_text_predict():
    text_prompt = (request.form.get('text_prompt') or '').strip()
    if not STATE['embedding_ready'] or STATE['image'] is None or STATE['image_rgb'] is None:
        return jsonify({"ok": False, "error": "Bạn cần extract image embedding trước."}), 400
    if not text_prompt:
        return jsonify({"ok": False, "error": "Chưa nhập text prompt."}), 400
    try:
        box_threshold = float(request.form.get('box_threshold', '0.35'))
        text_threshold = float(request.form.get('text_threshold', '0.25'))
        image = STATE['image']
        image_rgb = STATE['image_rgb']
        h, w = image_rgb.shape[:2]

        load_grounding_dino()
        inputs = GD_PROCESSOR(images=image, text=[[text_prompt]], return_tensors='pt')
        inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = GD_MODEL(**inputs)

        results = GD_PROCESSOR.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        result = results[0]
        boxes = result.get('boxes', [])
        scores = result.get('scores', [])
        labels = result.get('labels', [])
        if len(boxes) == 0:
            return jsonify({"ok": False, "error": "Grounding DINO không tìm thấy box phù hợp. Hãy thử prompt khác hoặc giảm threshold."}), 200

        best_idx = int(torch.argmax(scores).item()) if hasattr(scores, 'shape') and len(scores) > 0 else 0
        best_box = boxes[best_idx].tolist() if hasattr(boxes[best_idx], 'tolist') else list(boxes[best_idx])
        best_score = float(scores[best_idx].item()) if hasattr(scores[best_idx], 'item') else float(scores[best_idx])
        best_label = labels[best_idx] if isinstance(labels, list) else str(text_prompt)

        masks, mask_scores, _ = PREDICTOR.predict(box=np.array(best_box, dtype=np.float32), multimask_output=True)
        best_mask_idx = int(np.argmax(mask_scores))
        best_mask = masks[best_mask_idx]

        stamp = time.strftime('%Y%m%d-%H%M%S')
        grounded_name = f'{stamp}-grounded-input.png'
        mask_name = f'{stamp}-sam-mask.png'
        cutout_name = f'{stamp}-sam-cutout.png'
        grounded_path = OUTPUT_DIR / grounded_name
        mask_path = OUTPUT_DIR / mask_name
        cutout_path = OUTPUT_DIR / cutout_name

        Image.fromarray(make_box_image(image_rgb, best_box, str(best_label))).save(grounded_path)
        Image.fromarray(make_overlay(image_rgb, best_mask, best_box)).save(mask_path)
        Image.fromarray(make_cutout_transparent(image_rgb, best_mask), mode='RGBA').save(cutout_path)

        info = (
            f'Thiết bị chạy: CPU\n'
            f'SAM checkpoint: {CHECKPOINT}\n'
            f'Grounding model: {GROUNDING_MODEL_ID}\n'
            f'Ảnh đầu vào: {STATE["image_path"]}\n'
            f'Kích thước ảnh: {w}x{h}\n'
            f'Text prompt: {text_prompt}\n'
            f'Grounding label tốt nhất: {best_label}\n'
            f'Grounding score: {best_score:.4f}\n'
            f'Box tốt nhất: [{round(best_box[0],2)}, {round(best_box[1],2)}, {round(best_box[2],2)}, {round(best_box[3],2)}]\n'
            f'SAM mask score tốt nhất: {float(mask_scores[best_mask_idx]):.4f}\n'
            f'Ảnh mask: {mask_path}\n'
            f'Ảnh đã tách khỏi nền: {cutout_path}'
        )
        return jsonify({
            'ok': True,
            'input_url': f'/file/output/{grounded_name}',
            'result_url': f'/file/output/{mask_name}',
            'cutout_url': f'/file/output/{cutout_name}',
            'info': info,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/file/output/<path:name>')
def output_file(name):
    return send_from_directory(OUTPUT_DIR, name)


@app.route('/file/upload/<path:name>')
def upload_file(name):
    return send_from_directory(UPLOAD_DIR, name)


if __name__ == '__main__':
    print('SAM text experimental running at http://127.0.0.1:7862')
    app.run(host='127.0.0.1', port=7862, debug=False)

