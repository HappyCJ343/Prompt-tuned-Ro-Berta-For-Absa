"""Prompt-tuning training/validation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .dataset_absa import (
    PromptDataCollator,
    encode_prompt_dataset,
    load_chnsenticorp_dataset,
    load_restaurant_dataset,
)
from .metrics import build_compute_metrics
from .prompts import DEFAULT_TEMPLATE
from .utils import configure_logging, load_yaml, save_metrics, set_seed

LOGGER = logging.getLogger(__name__)


def _build_prompt_config(model_name: str, cfg: Dict[str, object]) -> PromptTuningConfig:
    return PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=int(cfg.get("num_virtual_tokens", 20)),
        prompt_tuning_init_text=str(cfg.get("prompt_init_text", "ABSA task")),
        tokenizer_name_or_path=model_name,
    )


def train_from_config(config_path: Path, dataset_key: str = "rest") -> Dict[str, float]:
    cfg = load_yaml(config_path)

    experiment_cfg: Dict[str, object] = cfg.get("experiment", {})
    training_cfg: Dict[str, object] = cfg.get("training", {})
    data_cfg: Dict[str, object] = cfg.get("data", {})

    output_dir = Path(experiment_cfg.get("output_dir", "outputs/run"))
    configure_logging(output_dir)
    set_seed(int(experiment_cfg.get("seed", 42)))

    model_name = str(experiment_cfg.get("model_name", "roberta-base"))
    max_length = int(experiment_cfg.get("max_length", 128))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if dataset_key == "rest":
        train_path = Path(data_cfg.get("train_file", "data/absa14_rest_train.jsonl"))
        validation_value = data_cfg.get("validation_file", "")
        validation_path = Path(validation_value) if validation_value else None
        test_value = data_cfg.get("test_file", "data/absa14_rest_test.jsonl")
        test_path = Path(test_value) if test_value else None
        dataset = load_restaurant_dataset(
            train_path,
            validation_path,
            test_path,
            text_field=str(data_cfg.get("text_field", "sentence")),
            aspect_field=str(data_cfg.get("aspect_field", "aspect")),
            label_field=str(data_cfg.get("label_field", "label")),
        )
    elif dataset_key == "chnsc":
        dataset = load_chnsenticorp_dataset(Path("data/chnsc.csv"))
    else:
        raise ValueError(f"Unsupported dataset '{dataset_key}'")

    prompt_template = DEFAULT_TEMPLATE

    encoded_dataset = encode_prompt_dataset(
        dataset,
        tokenizer,
        prompt_template.render,
        max_length=max_length,
    )

    collator = PromptDataCollator(tokenizer)

    label_feature = dataset["train"].features["label"]
    num_labels = label_feature.num_classes

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    peft_model = get_peft_model(base_model, _build_prompt_config(model_name, experiment_cfg))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 5)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 8)),
        learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        evaluation_strategy=str(training_cfg.get("evaluation_strategy", "epoch")),
        save_strategy=str(training_cfg.get("save_strategy", "epoch")),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    compute_metrics = build_compute_metrics(num_labels)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_dataset = encoded_dataset.get("test") or encoded_dataset.get("validation")
    metrics = trainer.evaluate(test_dataset)

    save_metrics(output_dir / f"metrics_{dataset_key}.json", metrics)
    LOGGER.info("Evaluation metrics: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-tuned RoBERTa ABSA trainer")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("rest", "chnsc"),
        default="rest",
        help="Which dataset split to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_config(args.config, dataset_key=args.dataset)


if __name__ == "__main__":
    main()
