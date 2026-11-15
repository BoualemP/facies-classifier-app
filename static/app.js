"use strict";

const FACIES_NAMES = {
  SS: "Non-marine sandstone",
  CSiS: "Non-marine coarse siltstone",
  FSiS: "Non-marine fine siltstone",
  SiSh: "Marine siltstone and shale",
  MS: "Mudstone",
  WS: "Wackestone",
  D: "Dolomite",
  PS: "Packstone-grainstone",
  BS: "Phylloid-algal bafflestone",
};
const FACIES_COLORS = ["#F4D03F", "#F5B041", "#DC7633", "#6E2C00", "#1B4F72",
  "#2E86C1", "#AED6F1", "#A569BD", "#196F3D"];
const CURVES = ["GR", "ILD_log10", "PHIND", "PE"];
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);
let current = null;   // last result from the API
let cursor = 0;       // index of the sample under the cursor
let geometry = null;  // scales of the drawn display

/* ---------- Loading ---------- */

function setMessage(text, kind = "") {
  const box = $("message");
  box.textContent = text;
  box.className = "message" + (kind ? " " + kind : "");
}

function setBusy(busy) {
  document.querySelectorAll("#demo-buttons button").forEach((b) => { b.disabled = busy; });
}

async function request(url, options) {
  setBusy(true);
  setMessage("Classifying…", "busy");
  try {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail : "The request failed (" + response.status + ").";
      throw new Error(detail);
    }
    setMessage("");
    show(body);
    return true;
  } catch (err) {
    setMessage(err.message || "The server could not be reached.", "error");
    return false;
  } finally {
    setBusy(false);
  }
}

function loadDemo(id, scroll = false) {
  document.querySelectorAll("#demo-buttons button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.well === id));
  });
  return request("/api/demo/" + encodeURIComponent(id)).then((ok) => {
    if (ok) {
      current.demo = true;
      $("demo-note").hidden = false;
      $("well-title").textContent = document.querySelector(`#demo-buttons [data-well="${id}"]`).dataset.title;
      if (scroll) showResults();
    }
    return ok;
  });
}

function upload(file) {
  if (!file) return;
  if (!/\.csv$/i.test(file.name)) {
    setMessage("Please choose a .csv file.", "error");
    return;
  }
  document.querySelectorAll("#demo-buttons button").forEach((b) => b.setAttribute("aria-pressed", "false"));
  const form = new FormData();
  form.append("file", file);
  request("/api/predict", { method: "POST", body: form }).then((ok) => {
    if (ok && !current.well) $("well-title").textContent = file.name;
    if (ok) showResults();
  });
}

/* ---------- Results ---------- */

function showResults() {
  $("results").scrollIntoView({ block: "start" });
  $("results-title").focus({ preventScroll: true });
}

function show(result) {
  current = result;
  current.names = Object.values(result.facies);
  $("results").hidden = false;
  const depth = result.depth;
  $("well-title").textContent = result.well ? "Well " + result.well : "Uploaded well";
  $("demo-note").hidden = !result.demo;
  $("well-meta").textContent = `${result.samples} samples · ${fmt(depth[0], 1)}–${fmt(depth[depth.length - 1], 1)} ft`;
  renderMetrics(result);
  renderComposition(result);
  cursor = Math.floor(result.samples / 2);
  drawLog();
  updateReadout();
}

function fmt(value, digits) {
  return Number(value).toFixed(digits);
}

function renderMetrics(r) {
  const conf = r.confidence;
  const mean = conf.reduce((a, b) => a + b, 0) / conf.length;
  const low = conf.filter((c) => c < 0.5).length / conf.length;
  const scored = r.scores !== null;
  const items = [
    ["Accuracy", scored ? fmt(r.scores.accuracy, 3) : "—", scored ? "share of samples correct" : "add a Facies column to score"],
    ["Macro F1", scored ? fmt(r.scores.macro_f1, 3) : "—", scored ? "every facies weighs the same" : "add a Facies column to score"],
    ["Mean confidence", fmt(mean, 2), "highest probability per sample"],
    ["Low confidence", Math.round(low * 100) + " %", "samples below 0.5"],
  ];
  $("metrics").innerHTML = items.map(([label, value, note]) =>
    `<dl class="metric"><dt>${label}</dt><dd>${value}</dd><small>${note}</small></dl>`).join("");
}

function shares(codes, n) {
  const counts = new Array(n).fill(0);
  codes.forEach((c) => { counts[c - 1] += 1; });
  return counts.map((c) => c / codes.length);
}

function renderComposition(r) {
  const n = r.names.length;
  const rows = [["Predicted", shares(r.predicted, n)]];
  if (r.true) rows.push(["True", shares(r.true, n)]);
  const bar = (values) => values.map((v, i) => v > 0
    ? `<span style="width:${(v * 100).toFixed(2)}%;background:${r.colors[i]}" title="${r.names[i]} ${(v * 100).toFixed(1)} %"></span>` : "").join("");
  let html = rows.map(([label, values]) =>
    `<div class="comp-row"><span>${label}</span><div class="comp-bar" role="img" aria-label="${label} facies shares">${bar(values)}</div></div>`).join("");
  html += `<table class="comp-table"><thead><tr><th>Facies</th>${rows.map(([l]) => `<th>${l} %</th>`).join("")}</tr></thead><tbody>`;
  r.names.forEach((name, i) => {
    html += `<tr><td><span class="chip"><span class="swatch" style="background:${r.colors[i]}"></span>${name}</span></td>` +
      rows.map(([, v]) => `<td>${(v[i] * 100).toFixed(1)}</td>`).join("") + "</tr>";
  });
  $("composition").innerHTML = html + "</tbody></table>";
}

function renderLegend() {
  $("legend").innerHTML = Object.keys(FACIES_NAMES).map((name, i) =>
    `<span class="chip"><span class="swatch" style="background:${FACIES_COLORS[i]}"></span><strong>${name}</strong> ${FACIES_NAMES[name]}</span>`).join("");
}

/* ---------- Log display ---------- */

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
  if (parent) parent.appendChild(node);
  return node;
}

function niceRange(values) {
  let lo = Math.min(...values), hi = Math.max(...values);
  const pad = (hi - lo) * 0.05 || 1;
  return [lo >= 0 ? Math.max(0, lo - pad) : lo - pad, hi + pad];
}

function drawLog() {
  const r = current;
  const svg = $("logplot");
  svg.innerHTML = "";
  const width = Math.max(svg.clientWidth || 900, 760);
  const height = svg.clientHeight || 760;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const top = 48, bottom = height - 10, axisW = 58, gap = 8;
  const faciesTracks = [["Predicted", r.predicted]];
  if (r.true) faciesTracks.push(["True", r.true]);
  const faciesW = 58, confW = 70;
  const fixed = axisW + faciesTracks.length * (faciesW + gap) + confW + gap;
  const curveW = (width - fixed - gap * CURVES.length) / CURVES.length;

  const d = r.depth, d0 = d[0], d1 = d[d.length - 1];
  const step = d.length > 1 ? (d1 - d0) / (d.length - 1) : 1;
  const y = (depth) => top + ((depth - d0 + step / 2) / (d1 - d0 + step)) * (bottom - top);

  // Depth axis
  const tick = (d1 - d0) > 300 ? 50 : (d1 - d0) > 120 ? 20 : 10;
  for (let t = Math.ceil(d0 / tick) * tick; t <= d1; t += tick) {
    el("text", { x: axisW - 8, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = t;
    el("line", { x1: axisW - 4, x2: width, y1: y(t), y2: y(t), class: "grid" }, svg);
  }
  el("text", { x: 4, y: 14, class: "track-title" }, svg).textContent = "Depth (ft)";

  let x = axisW;
  const tracks = [];

  // Curve tracks
  CURVES.forEach((log) => {
    const values = r.logs[log];
    const [lo, hi] = niceRange(values);
    const sx = (v) => x + ((v - lo) / (hi - lo)) * curveW;
    el("rect", { x, y: top, width: curveW, height: bottom - top, class: "frame" }, svg);
    for (let k = 1; k < 4; k++) {
      el("line", { x1: x + (curveW * k) / 4, x2: x + (curveW * k) / 4, y1: top, y2: bottom, class: "grid" }, svg);
    }
    el("text", { x: x + curveW / 2, y: 16, "text-anchor": "middle", class: "track-title" }, svg).textContent = log;
    el("text", { x, y: 36 }, svg).textContent = fmt(lo, log === "ILD_log10" ? 2 : 0);
    el("text", { x: x + curveW, y: 36, "text-anchor": "end" }, svg).textContent = fmt(hi, log === "ILD_log10" ? 2 : 0);
    const points = values.map((v, i) => `${sx(v).toFixed(1)},${y(d[i]).toFixed(1)}`).join(" ");
    el("polyline", { points, class: "curve" }, svg);
    tracks.push({ x, w: curveW });
    x += curveW + gap;
  });

  // Facies tracks: merge consecutive samples with the same facies
  faciesTracks.forEach(([label, codes]) => {
    el("text", { x: x + faciesW / 2, y: 16, "text-anchor": "middle", class: "track-title" }, svg).textContent = label;
    let start = 0;
    for (let i = 1; i <= codes.length; i++) {
      if (i === codes.length || codes[i] !== codes[start]) {
        const yTop = y(d[start]) - (y(d0 + step) - y(d0)) / 2;
        const yBot = y(d[i - 1]) + (y(d0 + step) - y(d0)) / 2;
        el("rect", { x, y: yTop, width: faciesW, height: Math.max(yBot - yTop, 0.5), fill: r.colors[codes[start] - 1] }, svg);
        start = i;
      }
    }
    el("rect", { x, y: top, width: faciesW, height: bottom - top, class: "frame" }, svg);
    x += faciesW + gap;
  });

  // Confidence track
  el("text", { x: x + confW / 2, y: 16, "text-anchor": "middle", class: "track-title" }, svg).textContent = "Confidence";
  el("text", { x, y: 36 }, svg).textContent = "0";
  el("text", { x: x + confW, y: 36, "text-anchor": "end" }, svg).textContent = "1";
  const area = [`${x},${y(d[0])}`]
    .concat(r.confidence.map((c, i) => `${(x + c * confW).toFixed(1)},${y(d[i]).toFixed(1)}`))
    .concat([`${x},${y(d[d.length - 1])}`]).join(" ");
  el("polygon", { points: area, fill: "var(--accent)", "fill-opacity": 0.18, stroke: "var(--accent)", "stroke-width": 1 }, svg);
  el("line", { x1: x + confW / 2, x2: x + confW / 2, y1: top, y2: bottom, stroke: "var(--muted)", "stroke-dasharray": "3 3" }, svg);
  el("rect", { x, y: top, width: confW, height: bottom - top, class: "frame" }, svg);

  // Cursor
  const line = el("line", { x1: axisW, x2: width, class: "cursor" }, svg);
  geometry = { y, top, bottom, line };
  placeCursor();
}

function placeCursor() {
  if (!geometry) return;
  const yy = geometry.y(current.depth[cursor]);
  geometry.line.setAttribute("y1", yy);
  geometry.line.setAttribute("y2", yy);
}

function nearestIndex(depth) {
  const d = current.depth;
  let lo = 0, hi = d.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (d[mid] < depth) lo = mid; else hi = mid;
  }
  return Math.abs(d[lo] - depth) <= Math.abs(d[hi] - depth) ? lo : hi;
}

function moveCursor(index) {
  cursor = Math.max(0, Math.min(current.samples - 1, index));
  placeCursor();
  updateReadout();
}

function updateReadout() {
  const r = current, i = cursor;
  $("r-depth").textContent = fmt(r.depth[i], 1);
  $("r-logs").innerHTML = Object.entries(r.logs).map(([log, v]) =>
    `<tr><td>${log}</td><td>${fmt(v[i], log === "ILD_log10" ? 3 : 2)}</td></tr>`).join("");
  const chip = (code) => `<span class="chip"><span class="swatch" style="background:${r.colors[code - 1]}"></span>${r.names[code - 1]}</span>`;
  let html = `<div>Predicted: ${chip(r.predicted[i])} <span class="small">(${fmt(r.confidence[i], 2)})</span></div>`;
  if (r.true) {
    const ok = r.true[i] === r.predicted[i];
    html += `<div>True: ${chip(r.true[i])} <span class="small">${ok ? "✓ match" : "✗ differs"}</span></div>`;
  }
  $("r-facies").innerHTML = html;
  $("r-proba").innerHTML = r.probabilities[i].map((p, k) =>
    `<div class="bar"><span>${r.names[k]}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${(p * 100).toFixed(1)}%;background:${r.colors[k]}"></span></span><span>${fmt(p, 2)}</span></div>`).join("");
}

function depthFromEvent(event) {
  const svg = $("logplot");
  const rect = svg.getBoundingClientRect();
  const box = svg.viewBox.baseVal;
  const yy = ((event.clientY - rect.top) / rect.height) * box.height;
  const d = current.depth;
  const frac = (yy - geometry.top) / (geometry.bottom - geometry.top);
  return d[0] + frac * (d[d.length - 1] - d[0]);
}

/* ---------- Download ---------- */

function downloadCsv() {
  const r = current;
  const header = ["Depth", "Predicted", "Predicted_name", "Confidence"].concat(r.true ? ["True", "True_name"] : []);
  const lines = [header.join(",")];
  r.depth.forEach((depth, i) => {
    const row = [depth, r.predicted[i], r.names[r.predicted[i] - 1], r.confidence[i]];
    if (r.true) row.push(r.true[i], r.names[r.true[i] - 1]);
    lines.push(row.join(","));
  });
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = (r.well ? r.well.toLowerCase().replace(/\s+/g, "_") : "well") + "_facies_predictions.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

/* ---------- Events ---------- */

document.addEventListener("DOMContentLoaded", () => {
  renderLegend();
  $("api-base").textContent = window.location.origin;

  document.querySelectorAll("#demo-buttons button").forEach((b) => {
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => loadDemo(b.dataset.well, true));
  });

  const drop = $("drop"), input = $("file");
  input.addEventListener("change", () => { upload(input.files[0]); input.value = ""; });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

  const svg = $("logplot");
  svg.addEventListener("pointermove", (e) => { if (current && geometry) moveCursor(nearestIndex(depthFromEvent(e))); });
  svg.addEventListener("keydown", (e) => {
    if (!current) return;
    const stepSize = e.shiftKey ? 10 : 1;
    if (e.key === "ArrowDown") { moveCursor(cursor + stepSize); e.preventDefault(); }
    if (e.key === "ArrowUp") { moveCursor(cursor - stepSize); e.preventDefault(); }
  });

  $("download").addEventListener("click", downloadCsv);

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (current) drawLog(); }, 150);
  });

  $("example-link").addEventListener("click", () => loadDemo("synthetic_a", true));

  // Sections below the input panel appear only when asked for
  document.querySelectorAll('a[href="#about"], a[href="#api"]').forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      reveal(a.getAttribute("href").slice(1));
    });
  });
  const hash = window.location.hash.slice(1);
  if (hash === "about" || hash === "api") reveal(hash);
});

function reveal(id) {
  const section = $(id);
  section.hidden = false;
  history.replaceState(null, "", "#" + id);
  section.scrollIntoView({ block: "start" });
  $(id + "-title").focus({ preventScroll: true });
}
