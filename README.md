# Prompt-tuned RoBERTa for ABSA

本项目提供一个最小但完整的实验脚手架，复现使用提示调优（prompt tuning）的 RoBERTa 在 Aspect-Based Sentiment Analysis (ABSA) 任务上的训练与评估流程。目录结构、脚本命名和配置组织均与内部项目保持一致，便于直接迁移。

## 目录结构

```
absa-prompt/
├── README.md
├── requirements.txt
├── config.yaml                 # 训练/评估配置
├── data/
│   ├── absa14_rest_train.jsonl # SemEval-2014 Restaurant（示例）
│   ├── absa14_rest_test.jsonl
│   └── chnsc.csv               # 备用中文句级情感数据
├── src/
│   ├── dataset_absa.py         # 数据加载与 DataCollator
│   ├── prompts.py              # 模板与 verbalizer 定义
│   ├── trainer_prompt.py       # 训练/评估循环
│   ├── metrics.py              # Accuracy / F1 / Macro-F1
│   └── utils.py                # 读写、随机种子、日志
├── scripts/
│   ├── run_absa_roberta.sh     # 一键训练 + 评估（英文）
│   └── run_check_cn_baseline.sh# 中文对照实验
└── paper/
    ├── outline.md
    └── figs/
```

## 快速上手

1. 创建虚拟环境并安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 执行英文 ABSA 提示调优实验：

   ```bash
   bash scripts/run_absa_roberta.sh
   ```

3. 可选：运行中文 ChnSentiCorp 句级对照实验，验证 pipeline 是否正常：

   ```bash
   bash scripts/run_check_cn_baseline.sh
   ```

训练脚本会自动读取 `config.yaml`，下载 Hugging Face 模型权重，构建 prompt tuning PEFT 模型，并在 `outputs/` 下写入 checkpoints、指标与日志。所有示例数据规模极小，仅用于流程验证，实际实验时请替换为完整数据集。
