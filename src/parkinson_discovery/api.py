from __future__ import annotations

import html
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

app = FastAPI(title="Parkinson Discovery Lab", version="0.2.0")


def artifact_dir() -> Path:
    return Path(os.getenv("PDL_ARTIFACT_DIR", "artifacts/demo"))


def _load_json(name: str) -> dict:
    path = artifact_dir() / name
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0", "artifact_dir": str(artifact_dir())}


@app.get("/api/metrics")
def metrics() -> dict:
    return _load_json("metrics.json")


@app.get("/api/quantum-comparison")
def quantum_comparison() -> dict:
    return _load_json("quantum_comparison.json")


@app.get("/api/candidates")
def candidates(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    path = artifact_dir() / "ranked_candidates.csv"
    if not path.exists():
        return []
    cols = ["rank", "molecule_id", "smiles", "predicted_activity", "cns_likeness_proxy", "drug_likeness_proxy", "lipinski_pass", "rank_score"]
    df = pd.read_csv(path).head(limit)
    return df[[c for c in cols if c in df]].to_dict(orient="records")


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}"


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    manifest = _load_json("manifest.json")
    metrics_data = _load_json("metrics.json")
    quantum = _load_json("quantum_comparison.json")
    path = artifact_dir() / "ranked_candidates.csv"
    top = pd.read_csv(path).head(10) if path.exists() else pd.DataFrame()
    rows = ""
    for _, r in top.iterrows():
        rows += (
            "<tr>"
            f"<td>{int(r['rank'])}</td><td>{html.escape(str(r['molecule_id']))}</td>"
            f"<td>{r['predicted_activity']:.3f}</td><td>{r['cns_likeness_proxy']:.2f}</td>"
            f"<td>{r['rank_score']:.3f}</td></tr>"
        )
    best = metrics_data.get("best_model", "—")
    delta = quantum.get("delta_rimay_minus_classical", {})
    if quantum:
        quantum_card = f"""
<div class='card quantum'><div class='eyebrow'>RIMAY COMPARISON LOADED</div>
<h2>Quantum benchmark</h2>
<div class='grid three'>
<div><div class='metric'>{_fmt_delta(delta.get('roc_auc'))}</div><div class='muted'>ROC-AUC Δ</div></div>
<div><div class='metric'>{_fmt_delta(delta.get('pr_auc'))}</div><div class='muted'>PR-AUC Δ</div></div>
<div><div class='metric'>{html.escape(str(quantum.get('verdict','—')))}</div><div class='muted'>single-split verdict</div></div>
</div><p class='muted'>A single split is evidence, not a quantum-advantage claim. Repeat across scaffold seeds and report compute cost.</p></div>"""
    else:
        quantum_card = """
<div class='card quantum'><div class='eyebrow'>NEXT EXPERIMENT</div><h2>Rimay simulator pilot</h2>
<p>Classical benchmark frozen. Export 200–500 molecules, run Kipu Rimay on the same split, then import returned features or probabilities.</p>
<code>pdl rimay-pilot --input artifacts/run/rimay_input.csv --out artifacts/rimay_pilot --size 300</code></div>"""
    return f"""
<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Parkinson Discovery Lab</title>
<style>
:root{{--bg:#081019;--card:#111c27;--line:#263849;--txt:#eff7ff;--muted:#9eb0c0;--accent:#d7ff69;--cyan:#76e7f7}}
*{{box-sizing:border-box}}body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:40px 20px 80px;background:radial-gradient(circle at 80% 0,#122c37 0,transparent 35%),var(--bg);color:var(--txt)}}
.card{{background:linear-gradient(180deg,#13212d,#0f1923);border:1px solid var(--line);border-radius:22px;padding:22px;margin:16px 0;box-shadow:0 15px 40px #0004}}
h1{{font-size:clamp(38px,6vw,68px);letter-spacing:-.05em;margin:12px 0}}h2{{margin:6px 0 14px}}.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.three{{grid-template-columns:repeat(3,1fr)}}
.metric{{font-size:30px;font-weight:800;letter-spacing:-.03em}} table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #263849;text-align:left}}.badge{{display:inline-block;padding:7px 11px;border-radius:99px;background:#1b3440;color:var(--cyan);font-weight:700}}a{{color:var(--cyan)}}.eyebrow{{font-size:12px;letter-spacing:.15em;color:var(--accent);font-weight:800}}code{{display:block;overflow:auto;padding:14px;border-radius:12px;background:#071019;color:#c9f8ff}}.quantum{{border-color:#45633a}}
@media(max-width:760px){{.grid,.three{{grid-template-columns:1fr 1fr}}table{{font-size:13px}}}}@media(max-width:480px){{.grid,.three{{grid-template-columns:1fr}}}}
</style></head><body>
<span class='badge'>V0.2 · Rimay-ready</span><h1>Parkinson Discovery Lab</h1>
<p class='muted'>LRRK2 molecular ML → scaffold generalisation → classical baseline → Kipu/Rimay pilot → evidence-driven comparison.</p>
<div class='grid'>
<div class='card'><div class='metric'>{manifest.get('molecules','—')}</div><div class='muted'>molecules</div></div>
<div class='card'><div class='metric'>{manifest.get('unique_scaffolds','—')}</div><div class='muted'>scaffolds</div></div>
<div class='card'><div class='metric'>{manifest.get('active','—')}</div><div class='muted'>active</div></div>
<div class='card'><div class='metric'>{html.escape(str(best))}</div><div class='muted'>selected baseline</div></div>
</div>
{quantum_card}
<div class='card'><h2>Top computational hypotheses</h2><table><thead><tr><th>#</th><th>Molecule</th><th>Activity P</th><th>CNS proxy</th><th>Rank</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class='card'><strong>Scientific boundary:</strong> these are computational hypotheses, not medicines. No claim of quantum advantage, safety, BBB penetration or clinical efficacy is made without the corresponding held-out and experimental evidence.</div>
<p><a href='/docs'>API docs</a> · <a href='/api/metrics'>classical metrics</a> · <a href='/api/quantum-comparison'>quantum comparison</a> · <a href='/api/candidates'>candidates</a></p>
</body></html>"""
