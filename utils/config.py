"""Configuration dataclasses for CLA experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar("T")


@dataclass
class DataConfig:
    """Dataset and offline anchor generation parameters."""

    dataset_name: str = "gsm8k"
    dataset_path: Optional[str] = None
    dataset_subset: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "test"
    cache_dir: Optional[str] = None
    anchor_dir: str = "artifacts/anchors"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_length: int = 768
    generation_max_new_tokens: int = 192
    anchor_shard_size: int = 256
    negative_strategy: str = "hard_semantic"


@dataclass
class TeacherConfig:
    """Teacher and anchor encoder parameters."""

    teacher_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    teacher_backend: str = "local"
    anchor_encoder_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_dtype: str = "bfloat16"
    encoder_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    teacher_device: str = "auto"
    encoder_device: str = "auto"
    temperature: float = 0.2
    top_p: float = 0.9
    teacher_timeout_seconds: float = 120.0
    teacher_max_retries: int = 4
    use_dataset_rationale_fallback: bool = True


@dataclass
class ModelConfig:
    """Continuous CoT student model parameters."""

    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ponder_steps: int = 5
    projection_dropout: float = 0.0
    freeze_backbone: bool = True
    use_step_residual: bool = True


@dataclass
class TrainingConfig:
    """Optimization hyperparameters."""

    output_dir: str = "artifacts/checkpoints"
    batch_size: int = 2
    eval_batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.06
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    lambda_cla: float = 1.0
    cla_temperature: float = 0.1
    amp_dtype: str = "bfloat16"
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 250
    seed: int = 42
    num_workers: int = 0
    use_wandb: bool = False
    wandb_project: str = "cla-project"
    wandb_run_name: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Evaluation and profiling parameters."""

    max_new_tokens: int = 64
    num_beams: int = 1
    profile_repeats: int = 10
    run_drift_metrics: bool = True
    run_hardware_profile: bool = True
    run_stepwise_instability: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config tree into a JSON-friendly dictionary."""

        return asdict(self)

    def save_json(self, path: str) -> None:
        """Persist the configuration to disk."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load a configuration tree from a JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data=_coerce_dataclass(DataConfig, payload.get("data", {})),
            teacher=_coerce_dataclass(TeacherConfig, payload.get("teacher", {})),
            model=_coerce_dataclass(ModelConfig, payload.get("model", {})),
            training=_coerce_dataclass(TrainingConfig, payload.get("training", {})),
            evaluation=_coerce_dataclass(EvaluationConfig, payload.get("evaluation", {})),
        )


def _coerce_dataclass(cls: Type[T], payload: Dict[str, Any]) -> T:
    """Filter a dictionary down to the fields of a target dataclass."""

    allowed = {item.name for item in fields(cls)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return cls(**filtered)
