# 数据卡：NSW / LaLonde–Dehejia–Wahba

本项目用一项历史就业培训研究，检验“用机器学习调整可观测差异后，观察性估计会如何变化”。它不是对今天中国就业培训政策效果的估计。数据来自研究作者发布的 [Dehejia 数据页](https://users.nber.org/~rdehejia/nswdata2.html)，无需账号或 API 密钥。

## 来源与两个分析样本

| 文件 | 行数 | 在项目中的角色 | 原始下载 |
|---|---:|---|---|
| `nswre74_treated.txt` | 185 | 两个分析样本共同的 NSW 处理组 | [TXT](https://users.nber.org/~rdehejia/data/nswre74_treated.txt) |
| `nswre74_control.txt` | 260 | NSW 实验对照组 | [TXT](https://users.nber.org/~rdehejia/data/nswre74_control.txt) |
| `cps_controls.txt` | 15,992 | CPS 非实验比较组 | [TXT](https://users.nber.org/~rdehejia/data/cps_controls.txt) |

- **NSW 实验样本，445 人**：185 名处理者 + 260 名实验对照者。
- **NSW–CPS 观察性样本，16,177 人**：同一批 185 名处理者 + 15,992 名 CPS 比较对象；不混入 NSW 实验对照者。

这些是包含 1974 年收入的 Dehejia–Wahba 男性子样本，不是 LaLonde 原始的 297 名处理者 / 425 名对照者样本，也不是常见的 614 行软件示例数据。样本说明、列顺序和数量均以[作者数据页](https://users.nber.org/~rdehejia/nswdata2.html)为准。

## 字段

原始 TXT 无表头，以空白分隔；每行按以下固定顺序读取。

| 字段 | 含义 | 模型角色 |
|---|---|---|
| `treatment` | 处理标记，1 为 NSW 处理组，0 为相应比较组；此列名由加载器指定 | $D$ |
| `age` | 年龄 | 处理前协变量 |
| `education` | 受教育年数 | 处理前协变量 |
| `black` | 原始数据的 Black 指示变量 | 处理前协变量 |
| `hispanic` | 原始数据的 Hispanic 指示变量 | 处理前协变量 |
| `married` | 已婚指示变量 | 处理前协变量 |
| `nodegree` | 无学位 / 文凭指示变量，保留原始字段定义 | 处理前协变量 |
| `re74` | 1974 年收入 | 处理前协变量 |
| `re75` | 1975 年收入 | 处理前协变量 |
| `re78` | 1978 年收入 | 主要结局 $Y$ |

收入沿用原文件的美元尺度，不转换为今天购买力，不解释为月薪；收入为零是有效观测，不当作缺失值。人口特征字段保留用于历史样本调整，不用于刻画群体的固有能力或制定个人决策。完整字段定义见[原始来源](https://users.nber.org/~rdehejia/nswdata2.html)。

## 研究人群与可比性

NSW 原项目面向就业前景不利的人群；这里分析的只是经过原研究筛选、具有所需收入记录的男性子样本。CPS 对照者不是从同一个实验随机分配而来；年龄、教育、收入历史等差异使其不能直接视作“未接受培训的同类人”。合并样本中的倾向得分描述该合并数据内的处理组归属概率，不能直接解释为全体美国居民参加培训的概率。参见 [Imbens 与 Xu 的研究重分析](https://arxiv.org/html/2406.00827v3)。

实验样本具有比外部比较样本更强的设计依据，但其筛选、历史背景及有限规模仍限制外推。所有经济解释都应和[方法说明](methodology.md)中的目标人群及识别假设一起阅读。

## 使用条件、缓存和审计

截至本项目核查时，作者页面写明 **“attributable non-commercial use (CC by NC)”**，并要求引用以下三篇论文、在网页中链接数据来源。本仓库不自行推定该声明的具体许可版本。[查看原始使用与引用说明](https://users.nber.org/~rdehejia/nswdata2.html)。

代码许可与数据使用条件分别适用。原始数据下载到本地 `data/raw/` 缓存，该目录不纳入 Git；发布的复现材料包括聚合结果、源文件 URL 和 SHA-256 校验值。下载器校验列数、预期行数、有限数值与处理组标签，并检查文件哈希。校验值用于锁定复现输入，不能替代来源引用或数据使用条件。

项目没有增加个人姓名、联系方式、位置标识，也不进行个体识别。公开展示以组级诊断和聚合估计为主。

## 数据作者要求引用的文献

1. LaLonde, R. J. (1986). *Evaluating the Econometric Evaluations of Training Programs with Experimental Data*. American Economic Review, 76(4), 604–620. [作者数据页中的出处](https://users.nber.org/~rdehejia/nswdata2.html)
2. Dehejia, R. H., & Wahba, S. (1999). *Causal Effects in Nonexperimental Studies: Reevaluating the Evaluation of Training Programs*. Journal of the American Statistical Association, 94(448), 1053–1062. [作者提供的论文](https://users.nber.org/~rdehejia/papers/dehejia_wahba_jasa.pdf)
3. Dehejia, R. H., & Wahba, S. (2002). *Propensity Score-Matching Methods for Nonexperimental Causal Studies*. Review of Economics and Statistics, 84(1), 151–161. [作者提供的论文](https://users.nber.org/~rdehejia/papers/matching.pdf)
