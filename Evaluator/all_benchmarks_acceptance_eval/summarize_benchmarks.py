#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import sys
from report_directional_errors import build_report


def main():
    root=Path(sys.argv[1])
    manifest=json.loads((root/'manifest.json').read_text())
    rows=[]; reports=[]
    for name in manifest['benchmarks']:
        folder=root/name/'evaluation'
        m=json.loads((folder/'metrics.json').read_text())
        rows.append(dict(benchmark=name,prompts=m['evaluated_rows'],blocks=m['evaluated_blocks'],
                         exact_pct=100*m['exact_accuracy'],under_pct=100*m['underprediction_rate'],
                         over_pct=100*m['overprediction_rate'],mae_tokens=m['mean_absolute_error_tokens']))
        reports.append('\nBENCHMARK: '+name.upper()+'\n'+build_report(folder))
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=['Benchmark       Prompts    Blocks    Exact%     Less%     More%   MAE(tokens)']
    for r in rows:
        summary.append(f"{r['benchmark']:<15} {r['prompts']:>7} {r['blocks']:>9} {r['exact_pct']:>9.2f} {r['under_pct']:>9.2f} {r['over_pct']:>9.2f} {r['mae_tokens']:>13.3f}")
    text='\n'.join(summary)+'\n'+''.join(reports)
    (root/'all_directional_errors.txt').write_text(text)
    print(text)
    print('Report to share: '+str(root/'all_directional_errors.txt'))

if __name__=='__main__': main()
