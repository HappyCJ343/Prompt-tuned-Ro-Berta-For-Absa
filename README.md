# Prompt-tuned RoBERTa for Aspect-Based Sentiment Analysis

This project demonstrates how to prompt tune a RoBERTa encoder for aspect-based
sentiment analysis (ABSA). It provides:

- A light-weight training wrapper built on top of Hugging Face Transformers and
  PEFT.
- Command-line interface for launching prompt-tuning experiments.
- Toy ABSA dataset for quick sanity checks.

## Quick start

1. Install the dependencies (ideally inside a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Launch prompt tuning on the toy dataset:

   ```bash
   python -m src.train \
       --train-file data/sample_train.csv \
       --validation-file data/sample_validation.csv \
       --test-file data/sample_test.csv \
       --text-column text \
       --label-column label \
       --output-dir outputs/sample_run
   ```

   The script downloads a RoBERTa checkpoint, attaches a prompt-tuning head, and
   trains it using the provided CSV files. Checkpoints and logs are written to
   the directory specified via `--output-dir`.

## Project layout

```
├── data/                       # sample ABSA dataset
├── src/
│   ├── prompt_absa/            # reusable prompt tuning utilities
│   └── train.py                # CLI entry point
└── requirements.txt
```

## Notes

- The toy dataset is intentionally small and serves only as a smoke test. For a
  realistic experiment replace the CSV files with a larger ABSA corpus (e.g.
  SemEval-2014 Task 4) keeping the same column names (`text`, `label`).
- Adjust `--num-virtual-tokens`, `--prompt-init-text`, and learning rate to
  explore different prompt-tuning setups.
- The script relies on GPUs for best performance, but it will also run on CPU
  albeit significantly slower.
