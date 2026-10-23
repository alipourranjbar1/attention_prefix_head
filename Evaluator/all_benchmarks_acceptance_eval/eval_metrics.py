"""Block-weighted statistics; confusion rows=true, columns=predicted."""
def summarize(cm):
    n=sum(map(sum,cm))
    if not n: raise ValueError('No evaluated blocks')
    exact=within=under=over=under_tokens=over_tokens=0
    for truth,row in enumerate(cm):
        for pred,count in enumerate(row):
            exact += count*(pred==truth)
            within += count*(abs(pred-truth)<=1)
            under += count*(pred<truth)
            over += count*(pred>truth)
            under_tokens += count*max(truth-pred,0)
            over_tokens += count*max(pred-truth,0)
    return dict(evaluated_blocks=n, exact_accuracy=exact/n, within_one_accuracy=within/n,
                mean_absolute_error_tokens=(under_tokens+over_tokens)/n,
                underprediction_rate=under/n, overprediction_rate=over/n,
                mean_under_tokens=under_tokens/n, mean_over_tokens=over_tokens/n,
                asymmetric_cost_3_under_1_over=(3*under_tokens+over_tokens)/n,
                majority_class_baseline_accuracy=max(map(sum,cm))/n,
                majority_class=max(range(len(cm)),key=lambda i:sum(cm[i])))
