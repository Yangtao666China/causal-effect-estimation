"""Fixed analysis protocol. Split stability is reported, never used to pick a result."""

import csv
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from .data import FEATURES,SOURCES,load_benchmark,load_csv
from .estimators import LABELS,aipw_att,cross_fit,diagnostics,mean_difference
from .simulation import monte_carlo


DATASET_LABELS = {"experimental":"NSW randomized sample", "observational":"NSW treated + CPS controls",
                  "custom":"Your CSV · observational assumptions"}
NOTES = [
    "ATT 的因果解释依赖一致性、给定处理前协变量后的 Y(0) 均值可交换性，以及受训者相关区域的重叠。",
    "观察性数据的未观测混杂不能由机器学习自动消除；截断只是一种数值正则化，不是识别策略。",
    "NSW 随机样本的均值差是有抽样误差的参考，并非已知真值；它与观察性分析共享受训者，结果并不独立。",
    "区间是影响函数近似 95% 区间；偏差、重叠不足、小样本与截断会影响覆盖率。",
    "主结果采用第一个 seed，其余 seed 仅作分折稳定性检查；不挑选最接近实验参考的模型或随机种子。",
    "金额按来源数据的历史美元口径展示，未换算为当前购买力；报告不是当前培训项目收益预测。",
]


def analyze_frames(frames,features,outcome="re78",treatment="treatment",folds=5,seeds=(42,7,2026),
                   clip=.01,progress=print):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not 0 < clip < .5:
        raise ValueError("clip must be between 0 and 0.5")
    report = dict(project="Econ Causal Lab",schema_version=1,benchmark=None,datasets=[],results=[],
                  diagnostics=[],sensitivity=[],simulation=[],split_results=[],sources=[],placebo=[],
                  configuration=dict(folds=folds,seeds=list(seeds),clip=clip,features=list(features),
                                     outcome=outcome,treatment=treatment,
                                     primary_seed=seeds[0],outcome_units="source data units"),
                  notes=list(NOTES))
    for dataset,frame in frames.items():
        x,y,d = frame[features].to_numpy(),frame[outcome].to_numpy(),frame[treatment].to_numpy()
        baseline = mean_difference(y,d)
        if dataset == "experimental":
            report["benchmark"] = baseline
        report["datasets"].append(dict(id=dataset,label=DATASET_LABELS.get(dataset,dataset),n=len(y),
                                       treated=int(d.sum()),controls=int((1-d).sum())))
        report["results"].append(dict(dataset=dataset,method="naive",label=LABELS["naive"],
                                      split_min=baseline["estimate"],split_max=baseline["estimate"],**baseline))
        for method in ("linear","hgb"):
            splits = []
            primary = None
            for seed in seeds:
                if progress:
                    progress(f"Fitting {dataset} / {method} / seed={seed}")
                nuisance = cross_fit(x,y,d,method,folds=folds,seed=seed)
                result,_ = aipw_att(y,d,nuisance.propensity,nuisance.m0,clip=clip)
                splits.append(result)
                report["split_results"].append(dict(dataset=dataset,method=method,seed=seed,**result))
                if seed == seeds[0]:
                    primary = result
                    report["diagnostics"].append(dict(dataset=dataset,method=method,
                        **diagnostics(x,d,nuisance.propensity,features,clip=clip)))
                    for threshold in sorted(set([.001,.005,.01,.025,.05,clip])):
                        estimate,_ = aipw_att(y,d,nuisance.propensity,nuisance.m0,clip=threshold)
                        report["sensitivity"].append(dict(dataset=dataset,method=method,clip=threshold,**estimate))
            report["results"].append(dict(dataset=dataset,method=method,label=LABELS[method],
                split_min=min(r["estimate"] for r in splits),split_max=max(r["estimate"] for r in splits),**primary))
    report["environment"] = {"python":platform.python_version(),
        **{p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","scipy")}}
    return report


def save_outputs(report,output,simulation_records=None):
    from .plots import make_plots
    from .report import render_report
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"results.json").write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    for name,rows in [("estimates",report["results"]),("split_results",report["split_results"]),
                       ("clip_sensitivity",report["sensitivity"]),("placebo",report["placebo"]),
                       ("simulation_summary",report["simulation"]),
                       ("simulation_runs",simulation_records or [])]:
        if rows:
            with (output/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
                writer = csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    render_report(report,output/"report.html")
    make_plots(report,output)


def benchmark_run(cache="data/raw",output="artifacts",offline=False,folds=5,seeds=(42,7,2026),
                  clip=.01,repetitions=100,simulation_n=1000,progress=print):
    frames = load_benchmark(cache,offline)
    report = analyze_frames(frames,FEATURES,folds=folds,seeds=seeds,clip=clip,progress=progress)
    report["sources"] = [*SOURCES,dict(name="Data use & required citations · CC by NC",
                                      url="https://users.nber.org/~rdehejia/nswdata2.html")]
    report["notes"].append("数据引用：LaLonde (1986), American Economic Review 76:604–620；Dehejia & Wahba (1999), JASA 94:1053–1062；Dehejia & Wahba (2002), Review of Economics and Statistics 84:151–161。数据限署名非商业使用；软件 MIT 许可不改变数据条款。")
    report["configuration"]["outcome_units"] = "historical USD, as supplied; no present-value conversion"
    placebo_features = [f for f in FEATURES if f != "re75"]
    report["configuration"]["placebo"] = dict(outcome="re75",features=placebo_features,seed=seeds[0])
    for dataset,frame in frames.items():
        x,y,d = frame[placebo_features].to_numpy(),frame["re75"].to_numpy(),frame["treatment"].to_numpy()
        for method in LABELS:
            if progress:
                progress(f"Pre-treatment outcome check: {dataset} / {method}")
            if method == "naive":
                effect = mean_difference(y,d)
            else:
                nuisance = cross_fit(x,y,d,method,folds=folds,seed=seeds[0])
                effect,_ = aipw_att(y,d,nuisance.propensity,nuisance.m0,clip)
            report["placebo"].append(dict(dataset=dataset,method=method,label=LABELS[method],**effect))
    report["configuration"]["simulation"] = dict(repetitions=repetitions,n=simulation_n,folds=3,seed=2026,clip=.01)
    records = []
    if repetitions:
        report["simulation"],records = monte_carlo(repetitions,simulation_n,3,progress=progress)
    save_outputs(report,output,records)
    return report


def custom_run(path,features,outcome,treatment,output="outputs/custom",folds=5,seeds=(42,7,2026),clip=.01):
    frame = load_csv(path,outcome,treatment,features)
    report = analyze_frames({"custom":frame},features,outcome,treatment,folds,seeds,clip)
    report["notes"] = [n for n in report["notes"] if "NSW" not in n and "历史美元" not in n]
    report["notes"].append("自有 CSV 的处理发生时间、样本独立性、遗漏混杂和协变量有效性由研究者核实；工具无法从列名验证识别假设。")
    save_outputs(report,output)
    return report
