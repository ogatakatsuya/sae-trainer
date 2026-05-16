import os

import datasets
import torch
import transformers
import wandb

from sae import get_peft_sae_model
from sae.launch.config import ModelArguments, SaeConfig, TrainingArguments
from sae.trainer import SaeTrainer
from sae.utils import hf_processor, hf_tokenizer
from sae.utils.datasets import CacheDataset
from sae.utils.factory import ModelFactory, SaeFactory

try:
    if not os.environ.get("WANDB_API_KEY", None):
        wandb.login(key=os.environ.get("WANDB_API_KEY", None))
except Exception as e:
    pass


def main():
    parser = transformers.HfArgumentParser(
        [TrainingArguments, SaeConfig, ModelArguments]
    )
    trainer_args, sae_config, model_args = parser.parse_args_into_dataclasses()
    sae_type = sae_config.sae_type

    sae_config = SaeFactory.create_sae_config(
        sae_type=sae_type,
        sae_config=sae_config.__dict__,
    )

    processor_kwargs = {}
    if model_args.image_pixels is not None:
        processor_kwargs["min_pixels"] = model_args.image_pixels
        processor_kwargs["max_pixels"] = model_args.image_pixels
    processor = hf_processor(model_args.model_path, **processor_kwargs)
    tokenizer = hf_tokenizer(model_args.model_path)

    model_kwargs = {
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch.bfloat16 if trainer_args.bf16 else torch.float32,
        "model_name": model_args.model_path,
    }

    model = ModelFactory.create_model(**model_kwargs)
    model_config = model.config
    model = get_peft_sae_model(model, sae_config)
    model.config = model_config
    model.print_trainable_parameters()

    if os.path.isdir(trainer_args.dataset_path):
        import glob as _glob
        split_prefix = trainer_args.split if trainer_args.split else "*"
        parquet_files = sorted(_glob.glob(os.path.join(trainer_args.dataset_path, f"{split_prefix}-*.parquet")))
        dataset = datasets.Dataset.from_parquet(parquet_files)
    else:
        dataset = datasets.load_dataset(
            trainer_args.dataset_path, split=trainer_args.split, name=trainer_args.subset
        )

    sae_dataset = CacheDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        processor=processor,
        text_key=trainer_args.text_key,
        image_key=trainer_args.image_key,
        video_key=trainer_args.video_key,
        audio_key=trainer_args.audio_key,
        is_video=trainer_args.is_video,
        video_root=trainer_args.video_root,
        video_total_pixels=model_args.video_total_pixels,
        video_min_pixels=model_args.video_min_pixels,
        video_max_frames=model_args.video_max_frames,
        video_sample_fps=model_args.video_sample_fps,
    )

    trainer = SaeTrainer(
        model=model,
        args=trainer_args,
        data_collator=sae_dataset.get_collator(),
        train_dataset=sae_dataset,
    )
    trainer.train(resume_from_checkpoint=trainer_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
