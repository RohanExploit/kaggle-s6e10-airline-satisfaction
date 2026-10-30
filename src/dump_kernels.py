import json, os

with open(r'E:\projects\playground-series-s6e10\kernels_utf8.txt', 'w', encoding='utf-8') as out:
    for root, dirs, files in sorted(os.walk(r'E:\projects\playground-series-s6e10\kernels')):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f == 'kernel-metadata.json':
                with open(p, encoding='utf-8') as fp:
                    d = json.load(fp)
                out.write('====================================\n')
                out.write(f'META: {os.path.basename(root)}\n')
                out.write(f'dataset_sources: {d.get("dataset_sources")}\n')
                out.write(f'kernel_sources: {d.get("kernel_sources")}\n')
            elif f.endswith('.ipynb'):
                with open(p, encoding='utf-8') as fp:
                    nb = json.load(fp)
                out.write(f'NB: {os.path.basename(root)} {f}\n')
                for i, cell in enumerate(nb.get('cells', [])):
                    src = ''.join(cell.get('source', []))
                    if cell.get('cell_type') == 'code' and src.strip():
                        out.write(f'--- Code Cell {i} ---\n')
                        out.write(src + '\n')
