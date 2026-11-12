"""
Planilla de revisión ciega de la ablación (llama3.1:8b) con la segunda revisión incluida.

  - 60 respuestas nuevas: prompt_honestidad y rag_sin_nivel sobre los mismos 30 pares
    (escenario, semilla) ya revisados para baseline y agente_completo.
  - 20 respuestas ya puntuadas (10 de phi3:mini y 10 de llama3.1:8b, elegidas al azar con
    semilla fija) para estimar la consistencia del evaluador.
Todo mezclado y sin indicar modelo ni condición. La clave queda en clave_ablacion.csv.

Uso (venv del backend, desde la raíz del repo):
    python evaluation/ablacion_revision.py
Genera en evaluation/resultados/ablacion/: revision_ablacion.html, revision_ablacion_en_blanco.csv
y clave_ablacion.csv. Al terminar, descargar el CSV desde la página y guardarlo en esa carpeta
como revision_ablacion.csv.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

RAIZ_EVALUACION = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_EVALUACION))

from consultas_util import cargar_consultas  # noqa: E402
from ejecutar_evaluacion import CONDICIONES_ABLACION, carpeta_modelo  # noqa: E402
from muestrear_revision import _evento, leer_corridas, material_referencia  # noqa: E402
from utils import cargar_corpus  # noqa: E402

DESTINO = RAIZ_EVALUACION / "resultados" / "ablacion"
COL = {
    "fidelidad": "fidelidad (0-2)", "alucinacion": "alucinacion (0/1)", "pertinencia": "pertinencia (0-2)",
    "artefactos": "artefactos (0/1)", "reconoce_limite": "reconoce_limite (si/parcial/no)",
}
ORDEN = {"fidelidad": ["0", "1", "2"], "alucinacion": ["0", "1"], "pertinencia": ["0", "1", "2"],
         "artefactos": ["0", "1"], "reconoce_limite": ["no", "parcial", "si"]}
CAMPOS = ["id_item", "pregunta", "material_de_referencia", "respuesta"]
SIN_MATERIAL = "(Sin material de referencia"
MODELOS_REPETIDAS = ["phi3:mini", "llama3.1:8b"]  # orden usado para elegir las filas repetidas


def leer(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def valor(texto: str):
    t = (texto or "").strip().lower().replace(",", ".")
    if t in ("", "na"):
        return None
    try:
        return str(int(float(t)))
    except ValueError:
        return t


def escribir(ruta: Path, columnas: list[str], filas: list[dict]) -> None:
    with open(ruta, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


def preparar(args) -> None:
    if (DESTINO / "revision_ablacion_en_blanco.csv").exists() and not args.sobrescribir:
        sys.exit(f"Ya existe {DESTINO}. Usá --sobrescribir solo si no empezaste a revisar.")
    carpeta = carpeta_modelo(args.modelo)
    clave_v1 = leer(carpeta / "revision" / "clave_revision.csv")
    pares = sorted({(c["escenario_id"], c["repeticion"], c["tipo_pregunta"]) for c in clave_v1})
    corridas = leer_corridas(carpeta)
    consultas = {c["id"]: c for c in cargar_consultas()}
    corpus = cargar_corpus()

    items = []
    for esc, seed, tipo in pares:
        c = consultas[esc]
        for cond in CONDICIONES_ABLACION:
            run_id = f"{esc}_{cond}_s{seed}"
            if run_id not in corridas:
                sys.exit(f"Falta la corrida {run_id}. ¿Terminó el runner con --condiciones "
                         f"{','.join(CONDICIONES_ABLACION)}?")
            items.append({"origen": "nueva", "modelo": args.modelo, "id_original": "", "run_id": run_id,
                          "condicion": cond, "escenario_id": esc, "tipo_pregunta": tipo, "repeticion": seed,
                          "pregunta": c["pregunta"], "material": material_referencia(c, corpus),
                          "respuesta": _evento(corridas[run_id], "explicacion")["respuesta"]})

    rng = random.Random(args.semilla)
    for m in MODELOS_REPETIDAS:
        filas = leer(carpeta_modelo(m) / "revision" / "revision_ciega.csv")
        clave = {c["id_item"]: c for c in leer(carpeta_modelo(m) / "revision" / "clave_revision.csv")}
        for f in rng.sample(filas, args.n_repetidas):
            k = clave[f["id_item"]]
            items.append({"origen": "repetida", "modelo": m, "id_original": f["id_item"], "run_id": k["run_id"],
                          "condicion": k["condicion"], "escenario_id": k["escenario_id"],
                          "tipo_pregunta": k["tipo_pregunta"], "repeticion": k["repeticion"],
                          "pregunta": f["pregunta"], "material": f["material_de_referencia"],
                          "respuesta": f["respuesta"]})

    random.Random(args.semilla_mezcla).shuffle(items)
    cols = CAMPOS + list(COL.values()) + ["observaciones"]
    blanco, clave_out, datos = [], [], []
    for k, it in enumerate(items, 1):
        item_id = f"A{k:02d}"
        fila = {"id_item": item_id, "pregunta": it["pregunta"], "material_de_referencia": it["material"],
                "respuesta": it["respuesta"]}
        blanco.append({**fila, **{c: "" for c in cols if c not in fila}})
        datos.append({**fila, "fuera": it["material"].startswith(SIN_MATERIAL)})
        clave_out.append({"id_item": item_id, "origen": it["origen"], "modelo": it["modelo"],
                          "id_original": it["id_original"], "run_id": it["run_id"], "condicion": it["condicion"],
                          "escenario_id": it["escenario_id"], "tipo_pregunta": it["tipo_pregunta"],
                          "repeticion": it["repeticion"]})
    DESTINO.mkdir(parents=True, exist_ok=True)
    escribir(DESTINO / "revision_ablacion_en_blanco.csv", cols, blanco)
    escribir(DESTINO / "clave_ablacion.csv", list(clave_out[0].keys()), clave_out)
    html = PAGINA.replace("__DATOS__", json.dumps({"cols": cols, "filas": datos}, ensure_ascii=False).replace("</", "<\\/"))
    (DESTINO / "revision_ablacion.html").write_text(html, encoding="utf-8")
    n_nuevas = sum(i["origen"] == "nueva" for i in items)
    print(f"{len(items)} filas: {n_nuevas} nuevas + {len(items) - n_nuevas} ya puntuadas (semilla {args.semilla}).")
    print(f"Abrí {DESTINO / 'revision_ablacion.html'}. No abras clave_ablacion.csv hasta terminar.")


PAGINA = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisión ciega (ablación)</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#666;--card:#fff;--bd:#d8d8d2;--ac:#1f5fae;--ok:#2e7d32}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#aaa;--card:#202020;--bd:#3a3a3a;--ac:#6aa7ee;--ok:#6fcf74}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.bar .sp{flex:1}
button{font:inherit;padding:8px 14px;border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer}
button.pri{background:var(--ac);color:#fff;border-color:var(--ac)}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px;margin-bottom:12px}
.card h3{margin:0 0 6px;font-size:14px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.txt{white-space:pre-wrap;max-height:42vh;overflow:auto}
.dots{display:flex;gap:4px;flex-wrap:wrap}.dot{width:26px;height:26px;border-radius:50%;border:1px solid var(--bd);font-size:12px;display:grid;place-items:center;cursor:pointer;background:var(--card)}
.dot.done{background:var(--ok);color:#fff;border-color:var(--ok)}.dot.cur{outline:2px solid var(--ac)}
fieldset{border:1px solid var(--bd);border-radius:8px;margin:0 0 10px;padding:8px 12px}legend{font-weight:600;padding:0 6px}
label{margin-right:16px;cursor:pointer}small{color:var(--mut)}textarea{width:100%;box-sizing:border-box;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--bd);border-radius:8px;padding:8px}
</style></head><body><main>
<h2>Revisión ciega (ablación)</h2>
<p><small>Evaluá cada respuesta con la rúbrica (evaluation/rubrica.md), sin consultar revisiones anteriores ni la clave. No se muestra el modelo ni la condición. El avance se guarda solo en este navegador; descargá el CSV al terminar (y de vez en cuando) y guardalo como revision_ablacion.csv.</small></p>
<div class="bar"><b id="prog"></b><span class="sp"></span><button id="ant">← Anterior</button><button id="sig" class="pri">Siguiente →</button><button id="desc">Descargar CSV</button></div>
<div class="dots" id="dots" style="margin-bottom:12px"></div>
<div class="card"><h3>Pregunta</h3><div id="preg" class="txt"></div></div>
<div class="card"><h3>Material de referencia</h3><div id="mat" class="txt" style="max-height:24vh"></div></div>
<div class="card"><h3>Respuesta a evaluar</h3><div id="resp" class="txt"></div></div>
<div class="card" id="form"></div>
</main>
<script>
const D = __DATOS__;
const CLAVE = 'revision-ablacion-v1-' + D.filas.length + '-' + D.filas[0].id_item + '-' + D.filas[0].respuesta.length;
const CAMPOS = {fidelidad:{col:'fidelidad (0-2)',op:['0','1','2'],txt:'Fidelidad (0–2): 2 todo consistente con el material; 1 mezcla respaldado/no respaldado; 0 contradice o no se apoya.'},
 alucinacion:{col:'alucinacion (0/1)',op:['0','1'],txt:'Alucinación (0/1): 1 si afirma como hecho algo falso o inventado y verificable.'},
 pertinencia:{col:'pertinencia (0-2)',op:['0','1','2'],txt:'Pertinencia (0–2): 2 responde lo preguntado con claridad; 1 en parte/confusa; 0 no responde.'},
 artefactos:{col:'artefactos (0/1)',op:['0','1'],txt:'Artefactos (0/1): 1 si hay caracteres corruptos, de otro alfabeto o palabras deformadas.'},
 limite:{col:'reconoce_limite (si/parcial/no)',op:['si','parcial','no'],txt:'Reconoce el límite (solo fuera de corpus): sí = lo dice sin inventar; parcial = lo insinúa pero da datos sin respaldo; no = responde con seguridad.'}};
let est = {}; try { est = JSON.parse(localStorage.getItem(CLAVE) || '{}'); } catch(e) {}
let i = 0;
const $ = s => document.querySelector(s);
function guardar(){ try { localStorage.setItem(CLAVE, JSON.stringify(est)); } catch(e) {} }
function completo(k){ const f = D.filas[k], e = est[k] || {};
  const need = ['alucinacion','pertinencia','artefactos'].concat(f.fuera ? ['limite'] : ['fidelidad']);
  return need.every(n => e[n] !== undefined && e[n] !== ''); }
function render(){
  const f = D.filas[i], e = est[i] || {};
  $('#prog').textContent = 'Ítem ' + (i+1) + ' de ' + D.filas.length + ' — completos: ' + D.filas.filter((_,k)=>completo(k)).length;
  $('#preg').textContent = f.pregunta; $('#mat').textContent = f.material_de_referencia; $('#resp').textContent = f.respuesta;
  const orden = ['fidelidad','alucinacion','pertinencia','artefactos'].filter(n => !(n==='fidelidad' && f.fuera)).concat(f.fuera ? ['limite'] : []);
  $('#form').innerHTML = orden.map(n => '<fieldset><legend>'+n+'</legend><small>'+CAMPOS[n].txt+'</small><div style="margin-top:6px">' +
     CAMPOS[n].op.map(o => '<label><input type="radio" name="'+n+'" value="'+o+'"'+(e[n]===o?' checked':'')+'> '+o+'</label>').join('') + '</div></fieldset>').join('') +
     '<label>Observaciones</label><textarea id="obs" rows="2">'+(e.obs||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</textarea>';
  document.querySelectorAll('#form input[type=radio]').forEach(r => r.onchange = () => { (est[i] = est[i] || {})[r.name] = r.value; guardar(); dots(); $('#prog').textContent = $('#prog').textContent.replace(/completos: \d+/, 'completos: ' + D.filas.filter((_,k)=>completo(k)).length); });
  $('#obs').oninput = ev => { (est[i] = est[i] || {}).obs = ev.target.value; guardar(); };
  dots();
}
function dots(){ $('#dots').innerHTML = D.filas.map((_,k)=>'<div class="dot'+(completo(k)?' done':'')+(k===i?' cur':'')+'" data-k="'+k+'">'+(k+1)+'</div>').join('');
  document.querySelectorAll('.dot').forEach(d => d.onclick = () => { i = +d.dataset.k; render(); }); }
$('#ant').onclick = () => { if (i>0) { i--; render(); window.scrollTo(0,0); } };
$('#sig').onclick = () => { if (i<D.filas.length-1) { i++; render(); window.scrollTo(0,0); } };
$('#desc').onclick = () => {
  const esc = v => '"' + String(v ?? '').replace(/"/g,'""') + '"';
  const filas = D.filas.map((f,k) => { const e = est[k] || {}, o = {...f};
    o['fidelidad (0-2)'] = f.fuera ? 'NA' : (e.fidelidad ?? ''); o['alucinacion (0/1)'] = e.alucinacion ?? ''; o['pertinencia (0-2)'] = e.pertinencia ?? '';
    o['artefactos (0/1)'] = e.artefactos ?? ''; o['reconoce_limite (si/parcial/no)'] = f.fuera ? (e.limite ?? '') : 'NA'; o.observaciones = e.obs ?? '';
    return D.cols.map(c => esc(o[c])).join(','); });
  const blob = new Blob(['﻿' + D.cols.map(esc).join(',') + '\r\n' + filas.join('\r\n')], {type:'text/csv;charset=utf-8'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'revision_ablacion.csv'; a.click();
};
document.addEventListener('keydown', ev => { if (/TEXTAREA/.test(ev.target.tagName)) return; if (ev.key==='ArrowRight') $('#sig').click(); if (ev.key==='ArrowLeft') $('#ant').click(); });
render();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="llama3.1:8b")
    ap.add_argument("--n-repetidas", type=int, default=10, help="filas ya puntuadas por modelo")
    ap.add_argument("--semilla", type=int, default=20261025, help="elección de las filas repetidas")
    ap.add_argument("--semilla-mezcla", type=int, default=2026)
    ap.add_argument("--sobrescribir", action="store_true")
    preparar(ap.parse_args())


if __name__ == "__main__":
    main()