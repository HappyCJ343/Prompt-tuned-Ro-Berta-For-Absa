"""ABSA dataset helpers and prompt-aware data collators."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence

import torch
from datasets import ClassLabel, Dataset, DatasetDict
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from .utils import load_jsonl


@dataclass
class ABSASample:
    """Light-weight representation of a single ABSA record."""

    sentence: str
    aspect: str
    label: str


def _normalize_records(
    records: Iterable[MutableMapping[str, str]],
    *,
    text_field: str,
    aspect_field: str,
    label_field: str,
) -> List[ABSASample]:
    samples: List[ABSASample] = []
    for row in records:
        sentence = str(row[text_field]).strip()
        aspect = str(row.get(aspect_field, "overall")).strip() or "overall"
        label = str(row[label_field]).strip()
        if not sentence or not label:
            continue
        samples.append(ABSASample(sentence=sentence, aspect=aspect, label=label))
    return samples


def load_restaurant_dataset(
    train_file: Path,
    validation_file: Optional[Path],
    test_file: Optional[Path],
    *,
    text_field: str = "sentence",
    aspect_field: str = "aspect",
    label_field: str = "label",
) -> DatasetDict:
    """Load SemEval-style restaurant ABSA splits from JSONL files."""

    def _load(path: Path) -> List[ABSASample]:
        if not path.exists():
            raise FileNotFoundError(path)
        return _normalize_records(
            load_jsonl(path),
            text_field=text_field,
            aspect_field=aspect_field,
            label_field=label_field,
        )

    train_data = _load(train_file)
    validation_data = _load(validation_file) if validation_file else train_data
    test_data = _load(test_file) if test_file else []

    all_labels = sorted({sample.label for sample in train_data})
    label_feature = ClassLabel(names=all_labels)

    def _to_dataset(samples: Sequence[ABSASample]) -> Dataset:
        payload: Dict[str, List[str]] = {
            "sentence": [s.sentence for s in samples],
            "aspect": [s.aspect for s in samples],
            "label": [s.label for s in samples],
        }
        dataset = Dataset.from_dict(payload)
        return dataset.cast_column("label", label_feature)

    dataset_dict = DatasetDict(
        {
            "train": _to_dataset(train_data),
            "validation": _to_dataset(validation_data),
        }
    )
    if test_data:
        dataset_dict["test"] = _to_dataset(test_data)
    return dataset_dict


def load_chnsenticorp_dataset(
    csv_file: Path,
    *,
    text_field: str = "text",
    label_field: str = "label",
    default_aspect: str = "整体体验",
) -> DatasetDict:
    """Load a sentence-level Chinese sentiment dataset as ABSA-style splits."""

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    with csv_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)

    samples = _normalize_records(
        records,
        text_field=text_field,
        aspect_field="__missing__",
        label_field=label_field,
    )
    for sample in samples:
        sample.aspect = default_aspect

    label_feature = ClassLabel(names=sorted({s.label for s in samples}))
    dataset = Dataset.from_dict(
        {
            "sentence": [s.sentence for s in samples],
            "aspect": [s.aspect for s in samples],
            "label": [s.label for s in samples],
        }
    ).cast_column("label", label_feature)

    # Reuse the same split for train/validation/test in this tiny baseline.
    return DatasetDict({"train": dataset, "validation": dataset, "test": dataset})


class PromptDataCollator(DataCollatorWithPadding):
    """Pad batch elements and convert labels to tensors."""

    def __call__(self, features: List[Dict[str, object]]):  # type: ignore[override]
        labels = [feature.pop("labels") for feature in features]
        for feature in features:
            feature.pop("prompt_text", None)
        batch = super().__call__(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def encode_prompt_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder: Callable[[str, str], str],
    *,
    text_field: str = "sentence",
    aspect_field: str = "aspect",
    label_field: str = "label",
    max_length: int = 128,
) -> DatasetDict:
    """Tokenize dataset rows after applying the prompt template."""

    def _prepare(batch: Dict[str, List[str]]) -> Dict[str, List[object]]:
        prompts = [
            prompt_builder(sentence, aspect)
            for sentence, aspect in zip(batch[text_field], batch[aspect_field])
        ]
        encoded = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        encoded["labels"] = list(batch[label_field])
        encoded["prompt_text"] = prompts
        return encoded

    return dataset.map(_prepare, batched=True)
