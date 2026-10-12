import sys
import pypdf

sys.stdout.reconfigure(encoding="utf-8")  # evita errores con tildes en la consola de Windows

r = pypdf.PdfReader("../data/raw/texto_guia_de_inversoras_v2_0_2.pdf")
for i, p in enumerate(r.pages, 1):
    print(i, "|", " ".join((p.extract_text() or "").split())[:90])