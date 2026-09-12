"""Known-effect Monte Carlo. Constant effect makes population ATE = ATT = 2000."""

import numpy as np

from .estimators import LABELS, aipw_att, cross_fit, mean_difference


SCENARIOS = ("good_overlap","weak_overlap","hidden_confounding")


def generate(n=800,seed=42,scenario="good_overlap"):
    if scenario not in SCENARIOS or n < 20:
        raise ValueError("unknown scenario or n < 20")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n,5))
    hidden = rng.normal(size=n)
    nonlinear = .55*x[:,0]+.6*np.sin(x[:,1])+.35*(x[:,2]**2-1)
    logits = nonlinear*(3.5 if scenario == "weak_overlap" else 1.)
    if scenario == "hidden_confounding":
        logits = logits+1.25*hidden
    probability = np.exp(-np.logaddexp(0,-logits))
    d = rng.binomial(1,probability).astype(float)
    m0 = 4000+1100*x[:,0]+1400*np.sin(x[:,1])+900*x[:,2]**2+600*x[:,3]*x[:,4]
    if scenario == "hidden_confounding":
        m0 = m0+1800*hidden
    y = m0+2000*d+rng.normal(0,1800,n)
    return x,y,d,2000.


def monte_carlo(repetitions=100,n=1000,folds=3,seed=2026,progress=None):
    if repetitions < 2:
        raise ValueError("use at least two Monte Carlo repetitions")
    summary,records = [],[]
    for scenario in SCENARIOS:
        per_method = {method:[] for method in LABELS}
        for repetition in range(repetitions):
            # Same draw and folds across estimators within a repetition.
            sample_seed = seed+repetition
            x,y,d,truth = generate(n,sample_seed,scenario)
            for method in LABELS:
                if method == "naive":
                    result = mean_difference(y,d)
                else:
                    nuisance = cross_fit(x,y,d,method,folds=folds,seed=sample_seed)
                    result,_ = aipw_att(y,d,nuisance.propensity,nuisance.m0)
                row = dict(scenario=scenario,method=method,repetition=repetition,
                           seed=sample_seed,truth=truth,**result)
                per_method[method].append(row)
                records.append(row)
            if progress and (repetition+1)%10 == 0:
                progress(f"Simulation {scenario}: {repetition+1}/{repetitions}")
        for method,rows in per_method.items():
            error = np.array([r["estimate"]-r["truth"] for r in rows])
            covered = np.array([r["ci_low"] <= r["truth"] <= r["ci_high"] for r in rows])
            summary.append(dict(scenario=scenario,method=method,label=LABELS[method],
                                repetitions=repetitions,n=n,bias=float(error.mean()),
                                rmse=float(np.sqrt(np.mean(error**2))),coverage=float(covered.mean()),
                                coverage_mc_se=float(np.sqrt(covered.mean()*(1-covered.mean())/repetitions)),
                                mean_ci_width=float(np.mean([r["ci_high"]-r["ci_low"] for r in rows]))))
    return summary,records
