import json

with open("notebook/output_analysis.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        lines = cell.get("source", [])
        text = "".join(lines)
        if "Topic modeling" in text or "Topic Modeling" in text or "LSA:" in text or "LDA e NMF:" in text or "the chosen number of topics" in text:
            print(f"Cell {i}:", text)
        elif "LSA topics" in text or "NMF topics" in text or "LDA topics" in text or "BERTopic topics" in text:
            print(f"Cell {i}:", text)
