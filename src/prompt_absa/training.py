"""High-level training utilities for prompt tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import evaluate
import numpy as np
from datasets import DatasetDict
from peft import get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import PromptTuningConfigBuilder

LOGGER = logging.getLogger(__name__)


@dataclass
class PromptTrainingArguments:
    """Semantic wrapper around :class:`transformers.TrainingArguments`."""

    output_dir: Path
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"

    def build(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
        )


class PromptTuningTrainer:
    """Convenience wrapper orchestrating model, tokenizer, and trainer."""

    def __init__(
        self,
        model_name: str,
        dataset: DatasetDict,
        label_column: str,
        config_builder: Optional[PromptTuningConfigBuilder] = None,
        *,
        text_column: str = "text",
        max_length: int = 128,
    ) -> None:
        self.dataset = dataset
        self.label_column = label_column
        self.text_column = text_column
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if config_builder is None:
            config_builder = PromptTuningConfigBuilder(tokenizer_name=model_name)
        else:
            config_builder.tokenizer_name = (
                config_builder.tokenizer_name or model_name
            )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=dataset["train"].features[label_column].num_classes,
        )
        self.model = get_peft_model(base_model, config_builder.build())

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.metric = evaluate.load("f1")

    def _tokenize_dataset(self) -> DatasetDict:
        text_column = self.text_column
        label_column = self.label_column

        def _tokenize_batch(batch: Dict[str, list]):
            tokens = self.tokenizer(
                batch[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            tokens["labels"] = batch[label_column]
            return tokens

        return self.dataset.map(_tokenize_batch, batched=True)

    def train(self, args: PromptTrainingArguments) -> Trainer:
        encoded = self._tokenize_dataset()

        def _compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if predictions.ndim == 3:  # logits of prompt tuned model
                predictions = predictions.mean(axis=1)
            preds = np.argmax(predictions, axis=-1)
            return {
                "macro_f1": self.metric.compute(
                    predictions=preds, references=labels, average="macro"
                )["f1"]
            }

        trainer = Trainer(
            model=self.model,
            args=args.build(),
            train_dataset=encoded["train"],
            eval_dataset=encoded.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=_compute_metrics,
        )

        trainer.train()
        return trainer
