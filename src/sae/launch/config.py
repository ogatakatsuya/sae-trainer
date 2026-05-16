from dataclasses import dataclass
from typing import Optional

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_path: str = "./data/examples.parquet"
    split: Optional[str] = "train"
    subset: Optional[str] = None
    text_key: Optional[str] = "text"
    image_key: Optional[str] = "images"
    video_key: Optional[str] = "videos"
    audio_key: Optional[str] = "audios"
    is_video: bool = False
    video_root: Optional[str] = None
    aux_alpha: Optional[float] = 0.1
    # Knowledge distillation from a teacher SAE
    distillation_sae_path: Optional[str] = None
    distillation_alpha: float = 0.1


@dataclass
class ModelArguments:
    model_path: str
    attn_implementation: str = "sdpa"
    # Optional: fix the image resolution passed to the processor (min_pixels = max_pixels = image_pixels).
    image_pixels: Optional[int] = None
    video_total_pixels: Optional[int] = 20480 * 32 * 32
    video_min_pixels: Optional[int] = 64 * 32 * 32
    video_max_frames: Optional[int] = 2048
    video_sample_fps: Optional[float] = 1.0


@dataclass
class SaeConfig:
    sae_type: str = "TOPK_SAE"
    num_latents: int = 4096
    k: Optional[int] = 32
    dead_tokens_threshold: Optional[int] = 100000
    target_modules: Optional[str] = "model.layers.24.o_proj"
