#!/usr/bin/env python3
"""Print and save exact signed-error counts. Denominator: all evaluated blocks."""
import csv
import json
from pathlib import Path
import sys


def build_report(folder):
    folder=Path(folder)
    metrics=json.loads((folder/'metrics.json').read_text())
    errors={}
    with (folder/'confusion_matrix.csv').open(newline='') as f:
        reader=csv.reader(f)
        predictions=list(map(int,next(reader)[1:]))
        if predictions!=list(range(len(predictions))):
            raise ValueError('Expected ordered prediction classes 0..K')
        actuals=[]
        for row in reader:
            actual=int(row[0]); actuals.append(actual)
            counts=list(map(int,row[1:]))
            if len(counts)!=len(predictions) or any(c<0 for c in counts):
                raise ValueError('Invalid confusion matrix row')
            for predicted,count in zip(predictions,counts):
                error=predicted-actual
                errors[error]=errors.get(error,0)+count
        if actuals!=predictions: raise ValueError('Missing/duplicate actual classes')
    total=sum(errors.values())
    if not total or total!=metrics['evaluated_blocks']:
        raise ValueError('Matrix total differs from metrics.json')
    rows=[]
    for direction,distances in [('exact',[0]),('less',range(1,len(predictions))),('more',range(1,len(predictions)))]:
        for distance in distances:
            signed=-distance if direction=='less' else distance
            count=errors.get(signed,0)
            rows.append(dict(direction=direction,token_difference=distance,signed_error=signed,
                             count=count,percentage_of_all_blocks=100*count/total))
    assert sum(row['count'] for row in rows)==total
    lines=['ACCEPTANCE-LENGTH PREDICTION: DIRECTIONAL ERROR REPORT',
           f"Evaluated dataset rows: {metrics.get('evaluated_rows','unknown')}",
           f'Evaluated blocks: {total}',
           'Error = predicted length - actual verified length.',
           'LESS = before actual boundary; MORE = after actual boundary.',
           'All percentages use ALL evaluated blocks as denominator.',
           '',f"{'Prediction':<24} {'Blocks':>9} {'Percentage':>12}",'-'*47]
    for row in rows:
        d=row['direction']; n=row['token_difference']
        if (d=='less' or d=='more') and n==1:
            lines.extend(['', 'UNDERPREDICTION (LESS)' if d=='less' else 'OVERPREDICTION (MORE)'])
        label='Exact' if d=='exact' else f'{n} token(s) {d.upper()}'
        lines.append(f"{label:<24} {row['count']:>9} {row['percentage_of_all_blocks']:>11.2f}%")
    lines.extend(['','TOTALS'])
    for d in ['exact','less','more']:
        count=sum(row['count'] for row in rows if row['direction']==d)
        lines.append(f'{d.upper():<24} {count:>9} {100*count/total:>11.2f}%')
    report='\n'.join(lines)+'\n'
    (folder/'directional_errors.txt').write_text(report)
    with (folder/'directional_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (folder/'directional_errors.json').write_text(json.dumps(dict(
        evaluated_blocks=total,evaluated_rows=metrics.get('evaluated_rows'),
        definition='predicted_minus_actual',denominator='all_evaluated_blocks',rows=rows),indent=2))
    return report

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Usage: python report_directional_errors.py /path/to/evaluation')
    print(build_report(sys.argv[1]))
    print('Report to share: '+str(Path(sys.argv[1])/'directional_errors.txt'))
