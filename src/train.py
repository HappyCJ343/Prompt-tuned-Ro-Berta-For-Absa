"""Command line interface for training a prompt-tuned RoBERTa model on ABSA data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prompt_absa import (
    DatasetPaths,
    PromptTrainingArguments,
    PromptTuningConfigBuilder,
    PromptTuningTrainer,
    load_absa_dataset,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing the input text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing the sentiment label.",
    )
    parser.add_argument(
        "--model-name",
        default="roberta-base",
        help="Base encoder to prompt tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where checkpoints will be written.",
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Tokenizer maximum length."
    )
    parser.add_argument(
        "--num-virtual-tokens",
        type=int,
        default=20,
        help="Number of virtual tokens to use for prompt tuning.",
    )
    parser.add_argument(
        "--prompt-init-text",
        type=str,
        default="Review sentiment classification:",
        help="Natural language initialization of the soft prompt.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for prompt tuning head.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size per device during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_paths = DatasetPaths(
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
    )
    dataset = load_absa_dataset(
        dataset_paths,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    config_builder = PromptTuningConfigBuilder(
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init_text=args.prompt_init_text,
        tokenizer_name=args.model_name,
    )

    trainer = PromptTuningTrainer(
        model_name=args.model_name,
        dataset=dataset,
        label_column=args.label_column,
        config_builder=config_builder,
        text_column=args.text_column,
        max_length=args.max_length,
    )

    training_args = PromptTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )

    LOGGER.info("Starting prompt tuning for %s", args.model_name)
    trainer.train(training_args)


if __name__ == "__main__":
    main()
