# Hierarchical RQ-VAE-Based Clinical Semantic ID Generation for Cross-Disease Transfer in LLM-Driven Discharge Medication Recommendation

基于层级 RQ-VAE 的临床语义 ID 生成与跨疾病迁移研究——面向大语言模型驱动的出院用药推荐

本仓库记录硕士论文的研究设计，并用合成病例演示患者划分、药物编号、约束生成和迁移评估。

## 目录

| 位置 | 内容 |
| --- | --- |
| [goal-objective.md](goal-objective.md) | 原始研究提示词 |
| [docs](docs/) | [研究蓝图](docs/MASTER_THESIS_RESEARCH_BLUEPRINT.md)、[逐章导读](docs/MASTER_THESIS_BLUEPRINT_READING_GUIDE.md)和 [Poster 文案指南](docs/POSTER_CONTENT_GUIDE.md) |
| [references/chip2025](references/chip2025/) | CHIP2025 出院用药推荐任务速读 |
| [references/psp408](references/psp408/) | PSP408 Poster 模板与评分细则 |
| [thesis_starter](thesis_starter/) | Python 教学代码、测试和[代码学习指南](thesis_starter/README.md) |

## 运行教学示例

在仓库根目录使用 Python 3.11 或以上版本运行。示例使用标准库，运行时在内存中构造虚构病例。

~~~powershell
python -X utf8 -B -m thesis_starter
python -X utf8 -B -m unittest discover -s thesis_starter/tests -v
~~~
