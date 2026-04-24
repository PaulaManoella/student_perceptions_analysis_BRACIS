import json

with open("notebook/output_analysis.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        lines = cell.get("source", [])
        text = "".join(lines)
        if "Qualitative comparative analysis inter-model" in text or "Common topics" in text or "Exclusive topics" in text or "super_topics.csv" in text:
            print(f"Cell {i}:", text)
        elif "Community Detection" in text or "Louvain" in text:
            print(f"Cell {i}:", text)
