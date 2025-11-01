"""Dataset utilities for ABSA prompt tuning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional

from datasets import ClassLabel, DatasetDict, load_dataset


@dataclass
class DatasetPaths:
    """Container holding dataset file locations."""

    train_file: Path
    validation_file: Path
    test_file: Optional[Path] = None

    def to_dict(self) -> Dict[str, str]:
        data_files: MutableMapping[str, str] = {
            "train": str(self.train_file),
            "validation": str(self.validation_file),
        }
        if self.test_file is not None:
            data_files["test"] = str(self.test_file)
        return dict(data_files)


def load_absa_dataset(
    paths: DatasetPaths,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> DatasetDict:
    """Load an aspect-based sentiment dataset stored in CSV files."""

    if not paths.train_file.exists():
        raise FileNotFoundError(paths.train_file)
    if not paths.validation_file.exists():
        raise FileNotFoundError(paths.validation_file)

    dataset = load_dataset("csv", data_files=paths.to_dict())

    def _strip(example: MutableMapping[str, str]) -> MutableMapping[str, str]:
        example[text_column] = example[text_column].strip()
        example[label_column] = example[label_column].strip()
        return example

    dataset = dataset.map(_strip)
    dataset = dataset.filter(lambda e: e[text_column] != "")

    label_values: Iterable[str] = dataset["train"].unique(label_column)
    sorted_labels: List[str] = sorted(label_values)
    label_feature = ClassLabel(names=sorted_labels)
    dataset = dataset.cast_column(label_column, label_feature)

    return dataset
