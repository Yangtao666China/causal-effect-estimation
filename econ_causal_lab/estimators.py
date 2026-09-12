"""Cross-fitted ATT score and diagnostics.

ATT = E[Y(1)-Y(0)|D=1]. Causal interpretation additionally requires
consistency, conditional mean independence for Y(0), and overlap for treated.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


LABELS = {"naive":"Unadjusted difference", "linear":"AIPW · linear", "hgb":"AIPW · boosted trees"}


def validate(y, d, x=None):
    y, d = np.asarray(y, dtype=float), np.asarray(d, dtype=float)
    if y.ndim != 1 or d.ndim != 1 or y.shape != d.shape or len(y) < 4:
        raise ValueError("y and treatment must be matching 1-D arrays, n >= 4")
    if not np.isfinite(y).all() or not np.isfinite(d).all() or not np.isin(d,[0,1]).all():
        raise ValueError("outcomes must be finite and treatment exactly 0/1")
    if min(np.sum(d == 0),np.sum(d == 1)) < 2:
        raise ValueError("at least two treated and two controls are required")
    if x is not None:
        x = np.asarray(x,dtype=float)
        if x.ndim != 2 or len(x) != len(y) or x.shape[1] < 1 or not np.isfinite(x).all():
            raise ValueError("features must be a finite (n, p) matrix with p >= 1")
    return y,d,x


def interval(estimate, se):
    return dict(estimate=float(estimate),se=float(se),
                ci_low=float(estimate-1.96*se),ci_high=float(estimate+1.96*se))


def mean_difference(y,d):
    y,d,_ = validate(y,d)
    a,b = y[d == 1],y[d == 0]
    se = np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    return interval(a.mean()-b.mean(),se)


def aipw_att(y,d,propensity,m0,clip=.01):
    """Unnormalized (canonical) cross-fitted ATT score.

    theta = sum[D*(Y-m0) - (1-D)*e/(1-e)*(Y-m0)] / sum(D).
    Standard errors use the empirical centered influence function, including
    estimation of P(D=1). Intervals are asymptotic, not finite-sample guarantees.
    Clipping is numerical regularization; it can change finite-sample behavior
    and does not establish causal identification or solve poor overlap.
    """
    y,d,_ = validate(y,d)
    e,m0 = np.asarray(propensity,dtype=float),np.asarray(m0,dtype=float)
    if e.shape != y.shape or m0.shape != y.shape or not np.isfinite(e).all() or not np.isfinite(m0).all():
        raise ValueError("propensity and m0 must be finite arrays matching y")
    if (e < 0).any() or (e > 1).any():
        raise ValueError("propensity must lie in [0, 1]")
    if not 0 < clip < .5:
        raise ValueError("clip must be between 0 and 0.5")
    e_clipped = np.clip(e,clip,1-clip)
    odds = e_clipped/(1-e_clipped)
    score = d*(y-m0)-(1-d)*odds*(y-m0)
    theta = score.sum()/d.sum()
    influence = (score-d*theta)/d.mean()
    se = influence.std(ddof=1)/np.sqrt(len(y))
    return interval(theta,se),influence


@dataclass
class NuisancePredictions:
    propensity: np.ndarray
    m0: np.ndarray
    fold: np.ndarray


def learners(method, seed):
    if method == "linear":
        return (make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000)),
                make_pipeline(StandardScaler(),Ridge(alpha=1.)))
    if method == "hgb":
        options = dict(max_iter=100,max_leaf_nodes=8,min_samples_leaf=25,
                       learning_rate=.06,l2_regularization=2.,early_stopping=False,
                       random_state=seed)
        return HistGradientBoostingClassifier(**options),HistGradientBoostingRegressor(**options)
    raise ValueError("method must be linear or hgb")


def cross_fit(x,y,d,method="linear",folds=5,seed=42,custom_learners=None):
    """Every prediction is out of fold; outcome model sees training controls only.

    Feature preprocessing lives inside each cloned model pipeline. No outcome
    or propensity hyperparameter is selected using the experimental benchmark.
    """
    y,d,x = validate(y,d,x)
    if not isinstance(folds,int) or folds < 2 or min(np.bincount(d.astype(int))) < folds:
        raise ValueError("folds >= 2 and each treatment group must contain at least folds rows")
    propensity_model,outcome_model = custom_learners or learners(method,seed)
    propensity,m0 = np.empty(len(y)),np.empty(len(y))
    assignment = np.full(len(y),-1,dtype=int)
    splitter = StratifiedKFold(folds,shuffle=True,random_state=seed)
    # Bound native threads: this is faster than oversubscription for small data.
    with threadpool_limits(limits=2):
        for fold,(train,test) in enumerate(splitter.split(x,d)):
            controls = train[d[train] == 0]
            propensity_fit,outcome_fit = clone(propensity_model),clone(outcome_model)
            propensity_fit.fit(x[train],d[train])
            outcome_fit.fit(x[controls],y[controls])
            propensity[test] = propensity_fit.predict_proba(x[test])[:,1]
            m0[test] = outcome_fit.predict(x[test])
            assignment[test] = fold
    return NuisancePredictions(propensity,m0,assignment)


def diagnostics(x,d,propensity,feature_names,clip=.01):
    """SMD uses the same unweighted pooled SD before and after weighting."""
    _,d,x = validate(np.zeros(len(d)),d,x)
    e = np.asarray(propensity,dtype=float)
    if e.shape != d.shape or not np.isfinite(e).all() or (e < 0).any() or (e > 1).any():
        raise ValueError("invalid propensity")
    if not 0 < clip < .5 or len(feature_names) != x.shape[1]:
        raise ValueError("invalid clip or feature names")
    clipped = np.clip(e,clip,1-clip)
    treated,controls = x[d == 1],x[d == 0]
    weights = clipped[d == 0]/(1-clipped[d == 0])
    denominator = np.sqrt((treated.var(axis=0,ddof=1)+controls.var(axis=0,ddof=1))/2)
    before = treated.mean(axis=0)-controls.mean(axis=0)
    after = treated.mean(axis=0)-np.average(controls,axis=0,weights=weights)
    # Constant features in both groups convey no imbalance; a zero denominator
    # with different means is reported as null rather than an invented value.
    def smd(difference,sd):
        return float(difference/sd) if sd > 0 else (0. if difference == 0 else None)
    edges = np.linspace(0,1,21)
    hist_t,_ = np.histogram(e[d == 1],edges)
    hist_c,_ = np.histogram(e[d == 0],edges)
    return dict(clip=float(clip),ess_controls=float(weights.sum()**2/(weights@weights)),
                clipped_fraction=float(np.mean(e != clipped)),
                clipped_treated_fraction=float(np.mean(e[d == 1] != clipped[d == 1])),
                clipped_control_fraction=float(np.mean(e[d == 0] != clipped[d == 0])),
                max_control_weight=float(weights.max()),control_weight_sum=float(weights.sum()),
                propensity_bins=[dict(left=float(edges[i]),right=float(edges[i+1]),
                                       treated_count=int(hist_t[i]),control_count=int(hist_c[i])) for i in range(20)],
                balance=[dict(feature=name,before=smd(before[i],denominator[i]),
                              after=smd(after[i],denominator[i])) for i,name in enumerate(feature_names)])
