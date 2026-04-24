import json

with open("notebook/output_analysis.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        lines = cell.get("source", [])
        if any("Defining *k* topics number" in line for line in lines) or any("Topic modeling" in line for line in lines) or any("Coherence (C_v)" in line for line in lines) or any("Models Diversity" in line for line in lines) or any("IRBO" in line for line in lines):
            print(f"Cell {i}:", "".join(lines))
