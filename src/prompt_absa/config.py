"""Configuration helpers for prompt tuning experiments."""

from dataclasses import dataclass
from typing import Optional

from peft import PromptTuningConfig, PromptTuningInit, TaskType


@dataclass
class PromptTuningConfigBuilder:
    """Builder for :class:`~peft.PromptTuningConfig` with sensible defaults."""

    task_type: TaskType = TaskType.SEQ_CLS
    num_virtual_tokens: int = 20
    prompt_tuning_init: PromptTuningInit = PromptTuningInit.TEXT
    prompt_tuning_init_text: Optional[str] = (
        "Review sentiment classification:"
    )
    tokenizer_name: Optional[str] = None

    def build(self) -> PromptTuningConfig:
        """Create the underlying :class:`PromptTuningConfig` instance."""

        return PromptTuningConfig(
            task_type=self.task_type,
            num_virtual_tokens=self.num_virtual_tokens,
            prompt_tuning_init=self.prompt_tuning_init,
            prompt_tuning_init_text=self.prompt_tuning_init_text,
            tokenizer_name_or_path=self.tokenizer_name,
        )
