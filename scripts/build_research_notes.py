"""Regenerate the findings tables and executed reading notebook from measured results.

Run from the repository root after a benchmark run targeting artifacts/.
The notebook uses only aggregate benchmark results and fresh synthetic data.
"""

import contextlib
import io
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/"artifacts/results.json").read_text(encoding="utf-8"))
    labels = {row["id"]:row["label"] for row in report["datasets"]}
    text = """# 研究结果：一个接近参考的点估计，仍可能缺乏因果可信度

本页由 `artifacts/results.json` 中的实际运行数值生成。真实数据采用 5 折、主种子 42、额外分折种子 7/2026、propensity 截断 0.01；模型配置固定，未依据实验参考选择算法。复现命令：

```bash
python -m econ_causal_lab benchmark --output artifacts
python scripts/build_research_notes.py
```

## 1. 更换比较组，未调整差异发生反转

两套数据使用同一批 185 名受训者。NSW 随机对照与 CPS 调查对照具有不同的选择机制。下表单位沿用数据源的历史美元，未换算当前购买力。

| 数据 | 方法 | 点估计 | 近似 95% 区间 | 3 次分折的点估计范围 |
|---|---|---:|---|---|
"""
    for row in report["results"]:
        text += f"| {labels[row['dataset']]} | {row['label']} | {row['estimate']:,.1f} | [{row['ci_low']:,.1f}, {row['ci_high']:,.1f}] | [{row['split_min']:,.1f}, {row['split_max']:,.1f}] |\n"
    text += """
随机实验未调整差异约为 +1,794，而更换为 CPS 对照后未调整差异约为 −8,498。线性和树模型的 AIPW 调整明显改变点估计，但不能仅凭调整后的结果接近随机参考就宣布识别成功。两套分析共享受训者，估计相关；这里没有把它们当独立估计进行显著性差异检验。分折范围不是置信区间。

## 2. 对照数量很多，不等于受训者的反事实信息充分

| 数据 | 模型 | 原始对照人数 | 加权对照 ESS | 被截断比例 |
|---|---|---:|---:|---:|
"""
    for row in report["diagnostics"]:
        controls = next(d["controls"] for d in report["datasets"] if d["id"] == row["dataset"])
        text += f"| {labels[row['dataset']]} | {row['method']} | {controls:,} | {row['ess_controls']:,.1f} | {row['clipped_fraction']:.1%} |\n"
    text += """
ESS 描述权重集中程度，并不证明重叠充分或模型正确。这里对 CPS 大量低 propensity 对照进行下端截断；因此截断比例高不等于大量受训者 e 接近 1，需结合组内分布与分组截断比例阅读。双侧截断也可能改变不相关对照的权重，这是当前协议的局限与后续研究问题。

![Balance and clipping sensitivity](../artifacts/diagnostics.png)

## 3. 负对照揭示“数值接近”之外的问题

结果变量换成按数据设计视为处理前的 1975 年收入，协变量中移除 `re75`。其余使用主种子和固定配置。

| 数据 | 方法 | 负对照估计 | 近似 95% 区间 |
|---|---|---:|---|
"""
    for row in report["placebo"]:
        text += f"| {labels[row['dataset']]} | {row['label']} | {row['estimate']:,.1f} | [{row['ci_low']:,.1f}, {row['ci_high']:,.1f}] |\n"
    text += """
NSW 随机样本各方法的区间包含 0，而 CPS 观察性样本两个调整后估计的区间仍不包含 0。培训无法改变过去；这提示选择机制、条件模型或负对照设计存在值得调查的问题。负对照依赖它自己的假设，不能单独证明是哪一种偏差；同样，未拒绝零也不能证明无混杂。

## 4. 模拟中能看到机器学习的帮助与边界

每人的真实处理效应恒为 2,000。每种情景独立生成 100 份、每份 1,000 行数据，共 300 份数据、900 个方法估计；每次 3 折。线性模型不包含模拟中的所有非线性项，因此不是“正确指定的经济计量模型”对照。此设计比较固定学习器在这些机制下的行为，不能概括为机器学习普遍优于传统模型。

| 情景 | 方法 | 偏差 | RMSE | 95% 区间覆盖率 |
|---|---|---:|---:|---:|
"""
    for row in report["simulation"]:
        text += f"| {row['scenario']} | {row['label']} | {row['bias']:,.1f} | {row['rmse']:,.1f} | {row['coverage']:.0%} |\n"
    text += """
良好重叠且无隐藏混杂时，树模型明显减少了线性错设带来的偏差；弱重叠时，其区间覆盖率下降，出现隐藏混杂时仍有很大偏差。**更灵活的条件模型没有替研究者解决识别问题。**

覆盖率是 100 次重复的经验比例，不是精确的总体概率。例如 0/100 只代表本次没有覆盖，不能推断总体覆盖概率严格为零。经验比例存在 Monte Carlo 误差，不能用 94% 与 95% 的细小差别断言区间完全校准。

![Known-effect simulation results](../artifacts/simulation.png)

## 5. 可以继续形成研究问题的地方

1. **ATT 单侧截断**：只限制高 propensity 与双侧截断相比，如何影响估计、ESS 和 bias/coverage？预先规定模拟方案，不按真实参考挑阈值。
2. **更丰富但预先固定的线性基线**：加入二次项、交互项，与树模型比较拟合与因果估计，区分算法能力和特征规格。
3. **不同负对照设计**：核查收入测量期、样本选择与更多处理前变量，不能只删去一个失败检查。
4. **更严格的推断**：检验小处理组样本下的影响函数近似，以及聚类样本应如何改变推断。

## 运行记录与来源

完整配置、Python/依赖版本、各分折结果、每次模拟记录、数据 URL 与 SHA-256 随 [artifacts](../artifacts/) 提供。原始微观数据不在仓库中。

方法与识别假设见 [methodology.md](methodology.md)，数据及必须引用的 LaLonde 1986、Dehejia-Wahba 1999/2002 文献见 [data_card.md](data_card.md)。本项目是 AI 辅助的实现与复现实验，不声称新的因果估计理论或已完成的学术论文。
"""
    (root/"docs/findings.md").write_text(text,encoding="utf-8")

    entries = [
        ("markdown", "# Econ Causal Lab：从结果到研究问题\n\n本笔记只读取仓库中的聚合结果，并运行一个合成数据小例子。无须下载真实微观数据。先在仓库安装 `python -m pip install -e .`。完整方法与研究边界见 `docs/methodology.md`。"),
        ("code", "import json\nfrom pathlib import Path\nimport pandas as pd\nroot = Path.cwd() if (Path.cwd() / 'artifacts').exists() else Path.cwd().parent\nreport = json.loads((root / 'artifacts/results.json').read_text(encoding='utf-8'))\nprint(report['configuration'])"),
        ("markdown", "## 1. 同样的受训者，为什么结论改变？\n\n随机实验参考有抽样误差；观察性结果接近它也不是识别成立的证明。"),
        ("code", "table = pd.DataFrame(report['results'])\nprint(table[['dataset','method','estimate','ci_low','ci_high']].round(2).to_string(index=False))"),
        ("markdown", "## 2. 培训能改变过去吗？\n\n检查 `re75` 负对照。结果变量已从特征中移除。区间含零不证明无混杂，不含零则需要调查。"),
        ("code", "placebo = pd.DataFrame(report['placebo'])\nprint(placebo[['dataset','method','estimate','ci_low','ci_high']].round(2).to_string(index=False))"),
        ("markdown", "## 3. 已知真值时，区间有多可靠？\n\n经验覆盖率来自有限次重复，不能认为 0/100 就是总体概率严格为零。"),
        ("code", "simulation = pd.DataFrame(report['simulation'])\nprint(simulation[['scenario','method','bias','rmse','coverage','repetitions']].round(3).to_string(index=False))"),
        ("markdown", "## 4. 亲手调用估计器\n\n下面是另一份合成样本，真值 2000。一次估计偏离真值是正常现象；不能以挑种子代替重复实验。"),
        ("code", "from econ_causal_lab.simulation import generate\nfrom econ_causal_lab import cross_fit, aipw_att\nx, y, d, truth = generate(n=1000, seed=42)\nfold_predictions = cross_fit(x, y, d, method='hgb', folds=3, seed=42)\nresult, influence = aipw_att(y, d, fold_predictions.propensity, fold_predictions.m0)\nprint('Known synthetic ATT:', truth)\nprint({key: round(value, 2) for key, value in result.items()})\nprint('Mean influence (should be near zero):', round(float(influence.mean()), 10))"),
        ("markdown", "## 自己扩展\n\n先提出假设并固定方案，例如加入二次项的线性基线或单侧 ATT 截断。保存全部结果，解释失败案例；将自己完成的扩展与本仓库提供的 AI 辅助实现区分开。")]
    cells,namespace,count = [],{},0
    # These are actual executed cells; captured stdout is stored in the notebook.
    import os
    previous = Path.cwd()
    os.chdir(root)
    try:
        for i,(kind,source) in enumerate(entries):
            cell = dict(cell_type=kind,id=f"econ-{i:02d}",metadata={},source=source.splitlines(keepends=True))
            if kind == "code":
                count += 1
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exec(compile(source,"<research-notebook>","exec"),namespace)
                cell.update(execution_count=count,outputs=[dict(output_type="stream",name="stdout",text=stdout.getvalue().splitlines(keepends=True))])
            cells.append(cell)
    finally:
        os.chdir(previous)
    notebook = dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python",version=report["environment"]["python"])))
    directory = root/"notebooks"
    directory.mkdir(exist_ok=True)
    (directory/"01_read_the_study.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print("Updated findings and executed notebook.")


if __name__ == "__main__":
    main()
