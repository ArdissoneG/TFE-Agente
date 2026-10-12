import sys
import pypdf

sys.stdout.reconfigure(encoding="utf-8")
r = pypdf.PdfReader(sys.argv[1])
for i, p in enumerate(r.pages, 1):
    print(i, "|", " ".join((p.extract_text() or "").split())[:90])