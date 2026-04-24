import json
with open('notebook/output_analysis.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells'][28:]):
    if cell['cell_type'] == 'markdown':
        print(f'Cell {i+28}:', ''.join(cell.get('source', [])))
