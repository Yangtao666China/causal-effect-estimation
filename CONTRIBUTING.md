# 参与改进

欢迎关注经济学识别问题和工程复现问题。请在 Issue 中说明预期行为、最小可复现输入、Python/依赖版本与观察到的结果，不上传私人微观数据。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

方法变更请同时说明估计目标、识别假设、有限样本影响和验证方式。不要仅以“更接近 NSW 参考值”为理由调整模型；参考不是已知真值。新增算法应保留原始基线与预先固定的实验配置。

数据下载测试应 mock 网络或使用合成数据；CI 不下载 NSW/CPS 原始数据。源数据使用条件与 MIT 软件许可不同，详见 `docs/data_card.md`。
