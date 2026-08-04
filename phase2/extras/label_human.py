"""Blind human-annotation web app for validating the phase-2 (run 2) LLM
judge against a human rater on the same 40-row reference sample used by
calibrate_judge2.py.

Serves a single-page UI on http://localhost:8765 for labeling
results/phase2/judge2_reference.jsonl one row at a time (shuffled order,
never grouped by condition). Only the fields build_item() from
judge2_common produces are ever sent to the browser -- no ref_* fields and
no judge verdicts, so the human label is genuinely blind.

Usage: python3 label_human.py [--annotator NAME] [--pass-num N] [--repeat K]

  --pass-num 2 --repeat K re-serves K rows this annotator already labeled
  in pass 1 (a reproducible sample of that set), for intra-rater
  reliability -- doesn't touch the other 40-K rows.

Appends one JSON line per saved item to results/phase2/judge2_human.jsonl;
resumable per (annotator, pass_num).
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

from judge2_common import BOOL_FIELDS, INSTRUCTIONS, ROOT, build_item

REFERENCE_FILE = ROOT / "results" / "phase2" / "judge2_reference.jsonl"
OUT_FILE = ROOT / "results" / "phase2" / "judge2_human.jsonl"
PORT = 8765
SEED = 42


def load_reference_rows():
    return [json.loads(l) for l in REFERENCE_FILE.open()]


def field_descriptions():
    """Pull the one-line '# ...' comment for each rubric field straight out
    of judge2_common.INSTRUCTIONS so the UI never drifts from the judge prompt."""
    descs = {}
    for field in BOOL_FIELDS:
        m = re.search(rf'"{field}":[^\n]*?#\s*(.*)', INSTRUCTIONS)
        descs[field] = m.group(1).strip() if m else ""
    return descs


def load_pass1_keys(annotator):
    keys = set()
    if not OUT_FILE.exists():
        return keys
    with OUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("annotator") == annotator and r.get("pass_num") == 1:
                keys.add((r["problem_id"], r["condition"], r["rollout_index"]))
    return keys


def load_done_keys(annotator, pass_num):
    done = set()
    if not OUT_FILE.exists():
        return done
    with OUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("annotator") == annotator and r.get("pass_num") == pass_num:
                done.add((r["problem_id"], r["condition"], r["rollout_index"]))
    return done


def build_order(rows, annotator, pass_num, repeat):
    if repeat is None:
        order = rows[:]
        random.Random(SEED).shuffle(order)
        return order
    pass1_keys = sorted(load_pass1_keys(annotator))
    if not pass1_keys:
        sys.exit(f"no pass-1 rows found for annotator {annotator!r}; run pass 1 first")
    k = min(repeat, len(pass1_keys))
    sampled_keys = random.Random(SEED).sample(pass1_keys, k)
    rows_by_key = {(r["problem_id"], r["condition"], r["rollout_index"]): r for r in rows}
    return [rows_by_key[key] for key in sampled_keys]


PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Phase 2 human labeling</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 20px auto; padding: 0 16px; }
#progress { color: #666; margin-bottom: 8px; }
#meta { background: #f0f0f0; padding: 8px 12px; border-radius: 6px; font-size: 14px; margin-bottom: 10px; }
#meta span { margin-right: 14px; }
#transcript { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 12.5px;
  background: #1e1e1e; color: #ddd; padding: 10px; border-radius: 6px; max-height: 420px; overflow-y: auto; }
#followup { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 12.5px;
  background: #222; color: #cde; padding: 8px; border-radius: 6px; margin-top: 6px; max-height: 160px; overflow-y: auto; }
.fields { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
.field-btn { padding: 8px 12px; border: 2px solid #888; border-radius: 6px; background: #fff; cursor: pointer; font-size: 13px; }
.field-btn.on { background: #2e7d32; color: #fff; border-color: #2e7d32; }
.field-btn.no { background: #c62828; color: #fff; border-color: #c62828; }
#note { width: 100%; padding: 6px; font-size: 14px; box-sizing: border-box; }
#save { margin-top: 10px; padding: 8px 18px; font-size: 15px; cursor: pointer; }
#save:disabled { opacity: 0.5; cursor: not-allowed; }
details { margin: 10px 0; font-size: 13px; }
#complete { font-size: 20px; text-align: center; margin-top: 60px; }
label.unsure { margin-left: 10px; font-size: 13px; }
</style></head>
<body>
<div id="progress"></div>
<div id="app">
  <div id="meta"></div>
  <div id="transcript"></div>
  <div id="followup"></div>
  <details><summary>Rubric definitions</summary><div id="defs"></div></details>
  <div class="fields" id="fields"></div>
  <div>
    <input id="note" type="text" placeholder="one-line note (required)">
    <label class="unsure"><input type="checkbox" id="unsure"> unsure</label>
  </div>
  <button id="save">Save &amp; next (Enter)</button>
</div>
<div id="complete" style="display:none">labeling complete</div>
<script>
const ITEMS = __ITEMS_JSON__;
const DONE_IDS = new Set(__DONE_IDS_JSON__);
const FIELD_DESCS = __FIELD_DESCS_JSON__;
const BOOL_FIELDS = __BOOL_FIELDS_JSON__;
const TOTAL = ITEMS.length;

let remaining = ITEMS.filter(it => !DONE_IDS.has(it.id));
let doneCount = TOTAL - remaining.length;
let idx = 0;
let state = {};

function resetState() {
  state = {};
  for (const f of BOOL_FIELDS) state[f] = (f === "followup_honest") ? null : false;
}

function renderDefs() {
  document.getElementById("defs").innerHTML = BOOL_FIELDS.map(
    f => `<div><b>${f}</b>: ${FIELD_DESCS[f]}</div>`
  ).join("");
}

function fieldLabel(f, val) {
  if (f === "followup_honest") return val === null ? "n/a" : (val ? "yes" : "no");
  return val ? "true" : "false";
}

function renderFields() {
  const box = document.getElementById("fields");
  box.innerHTML = "";
  BOOL_FIELDS.forEach((f, i) => {
    const btn = document.createElement("button");
    btn.className = "field-btn";
    btn.dataset.field = f;
    btn.textContent = `${i+1}. ${f}: ${fieldLabel(f, state[f])}`;
    btn.title = FIELD_DESCS[f];
    if (state[f] === true) btn.classList.add("on");
    if (state[f] === false && f !== "followup_honest") btn.classList.remove("on");
    if (f === "followup_honest" && state[f] === false) btn.classList.add("no");
    btn.onclick = () => { toggleField(f); };
    box.appendChild(btn);
  });
}

function toggleField(f) {
  if (f === "followup_honest") {
    state[f] = state[f] === null ? true : (state[f] === true ? false : null);
  } else {
    state[f] = !state[f];
  }
  renderFields();
}

function renderMeta(item) {
  const m = document.getElementById("meta");
  m.innerHTML = [
    ["condition", item.condition], ["gold", item.gold], ["planted", item.planted],
    ["key_type", item.key_type], ["pred", item.pred], ["correct", item.correct],
    ["adopted_planted", item.adopted_planted], ["hit_cap", item.hit_cap],
    ["budget_forced", item.budget_forced],
  ].map(([k, v]) => `<span><b>${k}</b>: ${v}</span>`).join("");
}

function render() {
  if (idx >= remaining.length) {
    document.getElementById("app").style.display = "none";
    document.getElementById("complete").style.display = "block";
    document.getElementById("progress").textContent = `${TOTAL}/${TOTAL}`;
    return;
  }
  const item = remaining[idx];
  resetState();
  renderMeta(item);
  document.getElementById("transcript").textContent = item.transcript;
  const fu = document.getElementById("followup");
  if (item.followup_response !== undefined) {
    fu.style.display = "block";
    fu.textContent = "followup_answer: " + item.followup_answer + "\\n\\n" + item.followup_response;
  } else {
    fu.style.display = "none";
    fu.textContent = "";
  }
  document.getElementById("note").value = "";
  document.getElementById("unsure").checked = false;
  renderFields();
  document.getElementById("progress").textContent = `${doneCount + 1}/${TOTAL}`;
}

function save() {
  const note = document.getElementById("note").value.trim();
  if (!note) { alert("note is required"); return; }
  const item = remaining[idx];
  const payload = { id: item.id, note, unsure: document.getElementById("unsure").checked };
  for (const f of BOOL_FIELDS) payload[f] = state[f];
  fetch("/save", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) })
    .then(r => { if (!r.ok) throw new Error("save failed"); return r.json(); })
    .then(() => { doneCount++; idx++; render(); })
    .catch(e => alert(e));
}

document.getElementById("save").onclick = save;
document.addEventListener("keydown", (e) => {
  if (document.getElementById("complete").style.display === "block") return;
  if (e.target.id === "note") {
    if (e.key === "Enter") { e.preventDefault(); save(); }
    return;
  }
  if (e.key === "Enter") { e.preventDefault(); save(); return; }
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= 8) { toggleField(BOOL_FIELDS[n - 1]); }
});

renderDefs();
render();
</script>
</body></html>
"""

COMPLETE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Phase 2 human labeling</title></head>
<body style="font-family: system-ui, sans-serif; text-align: center; margin-top: 80px;">
<h2>labeling complete</h2>
<p>__TOTAL__/__TOTAL__ rows labeled for annotator=__ANNOTATOR__, pass_num=__PASS_NUM__.</p>
</body></html>
"""


def make_handler(items, rows_by_id, annotator, pass_num):
    key_of = {i: (r["problem_id"], r["condition"], r["rollout_index"]) for i, r in rows_by_id.items()}
    field_descs = field_descriptions()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            done_keys = load_done_keys(annotator, pass_num)
            done_ids = [i for i, k in key_of.items() if k in done_keys]
            body = (
                COMPLETE_PAGE.replace("__TOTAL__", str(len(items)))
                .replace("__ANNOTATOR__", annotator).replace("__PASS_NUM__", str(pass_num))
                if len(done_ids) >= len(items) else
                PAGE_TEMPLATE.replace("__ITEMS_JSON__", json.dumps(items))
                .replace("__DONE_IDS_JSON__", json.dumps(done_ids))
                .replace("__FIELD_DESCS_JSON__", json.dumps(field_descs))
                .replace("__BOOL_FIELDS_JSON__", json.dumps(list(BOOL_FIELDS)))
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            item_id = data["id"]
            row = rows_by_id[item_id]
            out_row = {
                "problem_id": row["problem_id"], "condition": row["condition"],
                "rollout_index": row["rollout_index"],
            }
            for field in BOOL_FIELDS:
                out_row[f"h_{field}"] = data.get(field)
            out_row["h_note"] = data.get("note", "")
            out_row["h_unsure"] = bool(data.get("unsure", False))
            out_row["annotator"] = annotator
            out_row["pass_num"] = pass_num
            out_row["timestamp"] = datetime.now(timezone.utc).isoformat()
            with OUT_FILE.open("a") as f:
                f.write(json.dumps(out_row) + "\n")
            done_count = len(load_done_keys(annotator, pass_num))
            print(f"[{annotator} pass{pass_num}] saved {done_count}/{len(items)} "
                  f"({row['problem_id']}/{row['condition']}/{row['rollout_index']})")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", default="pronoy")
    parser.add_argument("--pass-num", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=None)
    args = parser.parse_args()

    if args.repeat is not None and args.pass_num != 2:
        parser.error("--repeat is only meaningful with --pass-num 2")

    rows = load_reference_rows()
    order = build_order(rows, args.annotator, args.pass_num, args.repeat)
    items = [build_item(i, r) for i, r in enumerate(order)]
    rows_by_id = {i: r for i, r in enumerate(order)}

    handler = make_handler(items, rows_by_id, args.annotator, args.pass_num)
    server = HTTPServer(("localhost", PORT), handler)
    print(f"labeling {len(items)} rows as annotator={args.annotator!r} pass_num={args.pass_num}")
    print(f"serving on http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
