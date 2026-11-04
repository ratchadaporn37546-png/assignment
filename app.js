// =============================================================
// Pet & People Finder - ทายภาพด้วยโมเดล YOLO11n-seg บนเบราว์เซอร์
// ขั้นตอน: โหลดโมเดล -> เตรียมภาพ -> รันโมเดล -> แปลผล -> วาดผล
// =============================================================

// ---------- ตั้งค่า ----------
// ที่อยู่ไฟล์โมเดล: ไฟล์ในโฟลเดอร์ model/ หรือใส่ลิงก์ Supabase Storage แทนได้
const MODEL_URL = "model/best.onnx";
const SIZE = 640;                                   // ขนาดภาพที่โมเดลรับ
const NAMES = ["Dog", "Person", "Cat", "Bunny"];    // ลำดับต้องตรงกับตอนเทรน
const THAI = { Dog: "สุนัข", Person: "คน", Cat: "แมว", Bunny: "กระต่าย" };
const COLORS = ["#f59e0b", "#3b82f6", "#ec4899", "#14b8a6"];

let session = null;    // โมเดลที่โหลดแล้ว
let currentImg = null; // รูปที่กำลังแสดง
let lastOutput = null; // ผลดิบจากโมเดล เก็บไว้ใช้ตอนขยับ slider (ไม่ต้องรันใหม่)

const $ = (id) => document.getElementById(id);


// ---------- 1) โหลดโมเดล ----------
async function loadModel() {
  try {
    setStatus("กำลังโหลดโมเดล…", "busy");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    setStatus("พร้อมใช้งาน", "ok");
    showList([]);
  } catch (err) {
    console.error(err);
    setStatus("โหลดโมเดลไม่สำเร็จ: " + err.message, "err");
  }
}


// ---------- 2) เตรียมภาพ (letterbox) ----------
// ย่อภาพให้ด้านยาวเท่ากับ 640 แล้วเติมขอบสีเทาให้เป็นสี่เหลี่ยมจัตุรัส
// แปลงเป็นตัวเลข 0-1 เรียงแบบ [สี R ทั้งภาพ, สี G ทั้งภาพ, สี B ทั้งภาพ]
function preprocess(img) {
  const scale = Math.min(SIZE / img.naturalWidth, SIZE / img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const padX = Math.floor((SIZE - w) / 2);
  const padY = Math.floor((SIZE - h) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(img, padX, padY, w, h);

  const pixels = ctx.getImageData(0, 0, SIZE, SIZE).data;   // RGBA RGBA ...
  const input = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    input[i] = pixels[i * 4] / 255;                         // R
    input[i + SIZE * SIZE] = pixels[i * 4 + 1] / 255;       // G
    input[i + 2 * SIZE * SIZE] = pixels[i * 4 + 2] / 255;   // B
  }
  return { tensor: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]), scale, padX, padY };
}


// ---------- 3) รันโมเดล ----------
async function predict(img) {
  if (!session) return;
  setStatus("กำลังทำนาย…", "busy");
  const prep = preprocess(img);

  const t0 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: prep.tensor });
  $("ms").textContent = (performance.now() - t0).toFixed(0) + " ms";

  // output0 = [1, 4+4+32, 8400]  กรอบ 4 ค่า + คะแนน 4 คลาส + mask coefficient 32 ค่า
  // output1 = [1, 32, 160, 160]  prototype mask 32 แผ่น
  lastOutput = {
    prep,
    boxes: out[session.outputNames[0]],
    protos: out[session.outputNames[1]],
  };
  showResult();
  setStatus("พร้อมใช้งาน", "ok");
}


// ---------- 4) แปลผล ----------
function decode() {
  const { boxes } = lastOutput;
  const data = boxes.data;
  const rows = boxes.dims[1];                   // 40
  const n = boxes.dims[2];                      // 8400 จุดที่โมเดลทาย
  const numMask = lastOutput.protos.dims[1];    // 32
  const numClass = rows - 4 - numMask;          // 4
  const confMin = +$("conf").value;

  // ข้อมูลเรียงแบบ [แถวที่ r][จุดที่ j] => data[r * n + j]
  const found = [];
  for (let j = 0; j < n; j++) {
    let best = 0, cls = 0;
    for (let c = 0; c < numClass; c++) {
      const score = data[(4 + c) * n + j];
      if (score > best) { best = score; cls = c; }
    }
    if (best < confMin) continue;

    const cx = data[j], cy = data[n + j], w = data[2 * n + j], h = data[3 * n + j];
    const coef = [];
    for (let m = 0; m < numMask; m++) coef.push(data[(4 + numClass + m) * n + j]);
    found.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], score: best, cls, coef });
  }
  return nms(found, +$("iou").value);
}

// NMS: กรอบที่ทับกันมากและเป็นคลาสเดียวกัน เก็บไว้แค่อันที่คะแนนสูงสุด
function nms(list, iouMax) {
  list.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of list) {
    const overlap = keep.some((k) => k.cls === d.cls && iou(k.box, d.box) > iouMax);
    if (!overlap) keep.push(d);
  }
  return keep;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}


// ---------- 5) วาดผล ----------
function showResult() {
  const dets = decode();
  drawImage(dets);
  showList(dets);
}

function drawImage(dets) {
  const img = currentImg;
  const { prep, protos } = lastOutput;
  const W = img.naturalWidth, H = img.naturalHeight;
  const canvas = $("cv");
  const ctx = canvas.getContext("2d");
  canvas.width = W;
  canvas.height = H;
  ctx.drawImage(img, 0, 0);

  if ($("showMask").checked && dets.length > 0) {
    drawMasks(ctx, dets, prep, protos, W, H);
  }

  // กรอบ + ชื่อคลาส (แปลงพิกัดจากภาพ 640 กลับเป็นภาพจริง)
  const fontSize = Math.max(14, Math.round(Math.max(W, H) / 40));
  ctx.font = `600 ${fontSize}px sans-serif`;
  ctx.lineWidth = Math.max(2, Math.round(Math.max(W, H) / 320));
  for (const d of dets) {
    const x1 = (d.box[0] - prep.padX) / prep.scale;
    const y1 = (d.box[1] - prep.padY) / prep.scale;
    const x2 = (d.box[2] - prep.padX) / prep.scale;
    const y2 = (d.box[3] - prep.padY) / prep.scale;
    const color = COLORS[d.cls];

    if ($("showBox").checked) {
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    const label = `${NAMES[d.cls]} ${(d.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width;
    const top = Math.max(0, y1 - fontSize - 8);
    ctx.fillStyle = color;
    ctx.fillRect(x1, top, tw + 12, fontSize + 8);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 6, top + fontSize + 1);
  }
}

// mask ของแต่ละวัตถุ = coef (32 ค่า) คูณกับ prototype (32 แผ่น ขนาด 160x160) แล้วรวมกัน
// ถ้าผลรวม > 0 (คือ sigmoid > 0.5) ถือว่าเป็นส่วนของวัตถุ — คิดเฉพาะในกรอบของมัน
function drawMasks(ctx, dets, prep, protos, W, H) {
  const P = protos.data;
  const num = protos.dims[1], mh = protos.dims[2], mw = protos.dims[3];
  const r = mw / SIZE;   // 160/640 = 0.25

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = mw;
  maskCanvas.height = mh;
  const mctx = maskCanvas.getContext("2d");
  const imgData = mctx.createImageData(mw, mh);

  for (const d of dets) {
    const [cr, cg, cb] = hexToRgb(COLORS[d.cls]);
    const x1 = Math.max(0, Math.floor(d.box[0] * r)), y1 = Math.max(0, Math.floor(d.box[1] * r));
    const x2 = Math.min(mw, Math.ceil(d.box[2] * r)), y2 = Math.min(mh, Math.ceil(d.box[3] * r));
    for (let y = y1; y < y2; y++) {
      for (let x = x1; x < x2; x++) {
        const p = y * mw + x;
        let sum = 0;
        for (let k = 0; k < num; k++) sum += d.coef[k] * P[k * mh * mw + p];
        if (sum > 0) {
          imgData.data[p * 4] = cr;
          imgData.data[p * 4 + 1] = cg;
          imgData.data[p * 4 + 2] = cb;
          imgData.data[p * 4 + 3] = 125;   // โปร่งใสครึ่งหนึ่ง
        }
      }
    }
  }
  mctx.putImageData(imgData, 0, 0);

  // ตัดขอบเทาที่เติมไว้ตอน letterbox ออก แล้วยืดให้เท่าภาพจริง
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(maskCanvas,
    prep.padX * r, prep.padY * r, (SIZE - 2 * prep.padX) * r, (SIZE - 2 * prep.padY) * r,
    0, 0, W, H);
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function showList(dets) {
  $("counts").innerHTML = NAMES.map((name, i) => {
    const count = dets.filter((d) => d.cls === i).length;
    return `<div><b style="color:${COLORS[i]}">${count}</b><small>${THAI[name]}</small></div>`;
  }).join("");

  if (dets.length === 0) {
    $("dets").innerHTML = `<li class="empty">ยังไม่พบวัตถุ — เลือกรูป หรือลองลดค่า Confidence</li>`;
    return;
  }
  $("dets").innerHTML = dets.map((d) => {
    const pct = (d.score * 100).toFixed(1);
    return `<li class="det"><i class="sw" style="background:${COLORS[d.cls]}"></i>
      <div>${NAMES[d.cls]} <span class="note">${THAI[NAMES[d.cls]]}</span>
        <div class="bar"><i style="width:${pct}%;background:${COLORS[d.cls]}"></i></div></div>
      <span class="num">${pct}%</span></li>`;
  }).join("");
}

function setStatus(text, type) {
  $("status").textContent = text;
  $("dot").className = "dot " + (type || "");
}


// ---------- ปุ่มและการเลือกรูป ----------
function openImage(src) {
  const img = new Image();
  img.onload = () => {
    currentImg = img;
    $("drop").hidden = true;
    predict(img);
  };
  img.src = src;
}

$("file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) openImage(URL.createObjectURL(file));
});
$("rerun").addEventListener("click", () => currentImg && predict(currentImg));

// เปลี่ยน slider แล้วแปลผลใหม่จากผลเดิม ไม่ต้องรันโมเดลซ้ำ
for (const id of ["conf", "iou"]) {
  $(id).addEventListener("input", () => {
    $(id + "v").textContent = (+$(id).value).toFixed(2);
    if (lastOutput) showResult();
  });
}
for (const id of ["showMask", "showBox"]) {
  $(id).addEventListener("change", () => lastOutput && showResult());
}

// ลากรูปมาวาง / วางรูปด้วย Ctrl+V
const plate = $("plate");
plate.addEventListener("dragover", (e) => { e.preventDefault(); plate.classList.add("drag"); });
plate.addEventListener("dragleave", () => plate.classList.remove("drag"));
plate.addEventListener("drop", (e) => {
  e.preventDefault();
  plate.classList.remove("drag");
  const file = e.dataTransfer.files[0];
  if (file) openImage(URL.createObjectURL(file));
});
window.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) openImage(URL.createObjectURL(item.getAsFile()));
});

// legend สีของแต่ละคลาส
$("legend").innerHTML = NAMES.map((n, i) =>
  `<span><i class="sw" style="background:${COLORS[i]}"></i>${n} · ${THAI[n]}</span>`).join("");

loadModel();
