import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

# ShareGPT "from" values → OpenAI "role" values
_ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "user": "user",
    "system": "system",
}


def _text_to_multimodal_content(
    text: str, image_iter
) -> Union[str, List[Dict[str, Any]]]:
    """If *text* contains <image> placeholders, replace each one with an image
    content block and return a multimodal content list.  Otherwise return the
    original string so that text-only turns stay compact."""
    if "<image>" not in text:
        return text
    parts = text.split("<image>")
    content: List[Dict[str, Any]] = []
    for i, chunk in enumerate(parts):
        if i > 0:
            try:
                content.append({"type": "image", "image": next(image_iter)})
            except StopIteration:
                # More placeholders than images — insert a blank marker so
                # apply_chat_template still generates a vision token slot.
                content.append({"type": "image"})
        if chunk:
            content.append({"type": "text", "text": chunk})
    return content


def normalize_conversations(
    conversations: List[Dict[str, Any]],
    images: Optional[List[Image.Image]] = None,
) -> List[Dict[str, Any]]:
    """Convert conversations to OpenAI role/content format and embed images.

    Handles:
    - ShareGPT format (from/value keys) → OpenAI format (role/content keys)
    - <image> placeholders in text → multimodal content list with image blocks

    When *images* is provided the PIL Images are embedded directly into the
    content blocks so that apply_chat_template can compute the correct number
    of vision tokens (required for dynamic-resolution models like Qwen-VL).
    """
    image_iter = iter(images or [])
    normalized = []
    for turn in conversations:
        if "role" in turn:
            role = turn["role"]
            content = turn.get("content", "")
        elif "from" in turn:
            role = _ROLE_MAP.get(turn["from"], turn["from"])
            content = turn.get("value", "")
        else:
            normalized.append(turn)
            continue

        if isinstance(content, str):
            content = _text_to_multimodal_content(content, image_iter)

        normalized.append({"role": role, "content": content})
    return normalized


def to_image_list(value: Any) -> List[Image.Image]:
    """Ensure image field is always a list of PIL Images."""
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value]
    # list / sequence of PIL Images (or already a list)
    return list(value)


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    processor: Optional[ProcessorMixin] = None

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor is not None:
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.tokenizer
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        input_ids = inputs.pop("input_ids")
        input_ids = [input_id.squeeze(0) for input_id in input_ids]
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        inputs.pop("attention_mask")

        # mm_token_type_ids (Qwen3-VL) has the same (1, seq_len) shape as input_ids
        # and must be padded to the same length. Pad with 0 (= text token type).
        mm_token_type_ids = None
        if "mm_token_type_ids" in inputs:
            mm_token_type_ids_list = [v.squeeze(0) for v in inputs.pop("mm_token_type_ids")]
            mm_token_type_ids = self.pad_sequence(
                mm_token_type_ids_list, batch_first=True, padding_value=0
            )

        batched_inputs = {}
        for key, values in inputs.items():
            batched_inputs[key] = torch.concatenate(values, dim=0)
        batched_inputs["input_ids"] = input_ids
        batched_inputs["attention_mask"] = attention_mask
        if mm_token_type_ids is not None:
            batched_inputs["mm_token_type_ids"] = mm_token_type_ids

        return batched_inputs


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Union[HFDataset, str],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        text_key: str,
        image_key: Optional[str] = None,
        video_key: Optional[str] = None,
        audio_key: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(dataset, str):
            dataset = HFDataset.from_parquet(dataset)

        self.tokenizer = tokenizer
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
        self.audio_key = audio_key
        self.text_key = text_key
        self.dataframe = dataset

    def __getitem__(self, index):
        row = self.dataframe[index]

        if self.processor is not None:
            images = to_image_list(row[self.image_key]) if self.image_key in row else []
            # Pass images so <image> placeholders are converted to multimodal
            # content blocks — apply_chat_template uses them to compute the
            # correct vision token count for dynamic-resolution models.
            conversations = normalize_conversations(row[self.text_key], images)
            text = self.processor.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
            multi_modal_inputs = {}
            if images:
                multi_modal_inputs["images"] = images

            # TODO
            # Implement the load logic for video and audios later
            if self.video_key in row:
                videos = [video for video in row[self.video_key]]
                multi_modal_inputs["videos"] = videos

            if self.audio_key in row:
                audios = [audio for audio in row[self.audio_key]]
                multi_modal_inputs["audios"] = audios

            model_inputs = self.processor(
                text=[text], return_tensors="pt", **multi_modal_inputs
            )

        else:
            text = self.tokenizer.apply_chat_template(
                row[self.text_key], tokenize=False, add_generation_prompt=False
            )
            model_inputs = self.tokenizer([text], return_tensors="pt")

        return model_inputs

    def get_collator(self):
        return DataCollator(self.tokenizer, self.processor)

    def __len__(self):
        return len(self.dataframe)
