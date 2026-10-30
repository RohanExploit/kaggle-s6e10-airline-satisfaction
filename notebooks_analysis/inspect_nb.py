import json
from pathlib import Path

def inspect_notebook(path, out_f):
    out_f.write(f"\n\n=== {path.name} ===\n")
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if any(w in src.lower() for w in ["0.962", "blend", "weights", "read_csv", "sub"]):
            out_f.write(f"\nCell {i} ({c['cell_type']}):\n")
            out_f.write(src[:500].strip() + "\n")
            out_f.write("-" * 50 + "\n")

nb_dir = Path(r"E:\projects\playground-series-s6e10\notebooks_analysis")
out_path = nb_dir / "summary.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for p in nb_dir.glob("*.ipynb"):
        inspect_notebook(p, f)
print("Summary written to", out_path)
