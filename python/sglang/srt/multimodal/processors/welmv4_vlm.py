import time
from typing import Any, List, Union

import torch

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.welmv4_vlm import WeLMV4VLMForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.utils import logger


class WeLMV4VLMImageProcessor(SGLangBaseProcessor):
    models = [WeLMV4VLMForConditionalGeneration]
    # WeLMV4 image preprocessing intentionally patchifies on CPU.
    gpu_image_decode = False
    prompt_input_type = "token_ids"

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        tokenizer = _processor.tokenizer
        self.image_token = getattr(hf_config, "image_token", "<|image_pad|>")
        self.image_token_id = getattr(hf_config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        self.video_token = getattr(hf_config, "video_token", "<|video_pad|>")
        self.video_token_id = getattr(hf_config, "video_token_id", None)
        if self.video_token_id is None:
            self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(
            self.vision_start_token
        )
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(
            self.vision_end_token
        )

        # Required by base class for language_only EPD mode (encode_receiver
        # uses these to reconstruct MultimodalInputs from embeddings).
        self.IM_START_TOKEN_ID = self.vision_start_token_id
        self.IM_END_TOKEN_ID = self.vision_end_token_id
        self.IM_TOKEN_ID = self.image_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.image_token_id,
            video_token=self.video_token,
            video_token_id=self.video_token_id,
        ).build(_processor)

    def _move_features_to_cpu(self, ret):
        if self.server_args.keep_mm_feature_on_device:
            return
        for feature_name in self.FEATURE_NAMES:
            if feature_name in ret and isinstance(ret[feature_name], torch.Tensor):
                ret[feature_name] = ret[feature_name].to("cpu")

    def _build_output_from_processor_result(self, ret):
        self._move_features_to_cpu(ret)
        input_ids = ret["input_ids"].flatten()
        mm_items = self.collect_mm_items_from_processor_output(ret)
        for mm_item in mm_items:
            if mm_item.modality == Modality.IMAGE:
                token_id = self.image_token_id
            elif mm_item.modality == Modality.VIDEO:
                # WeLMV4 renders videos as timestamped image-token runs.
                token_id = self.image_token_id
            else:
                token_id = None
            if token_id is not None:
                mm_item.offsets = self.get_mm_items_offset(input_ids, token_id)
        token_type_ids = ret.get("mm_token_type_ids")
        if token_type_ids is None:
            token_type_ids = ret.get("token_type_ids")
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=get_new_expanded_mm_items(mm_items),
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            token_type_ids=token_type_ids,
        )

    @staticmethod
    def _is_token_id_input(input_text) -> bool:
        return isinstance(input_text, list) and (
            not input_text or isinstance(input_text[0], int)
        )

    @staticmethod
    def _is_precomputed_item(item: Any) -> bool:
        return isinstance(item, dict) and str(item.get("format", "")).lower() in {
            "processor_output",
            "precomputed_embedding",
        }

    @classmethod
    def _has_precomputed_item(cls, items) -> bool:
        return any(cls._is_precomputed_item(item) for item in (items or []))

    def _processor_kwargs(self):
        kwargs = {}
        if self.image_config:
            kwargs["images_kwargs"] = dict(self.image_config)
        if self.video_config:
            kwargs["videos_kwargs"] = dict(self.video_config)
        return kwargs

    def _load_image_item(self, image_item):
        return self.__class__._load_single_item(
            image_item,
            Modality.IMAGE,
            None,  # frame_count_limit
            None,  # audio_sample_rate
            True,  # discard_alpha_channel
        )

    def _load_images(self, image_data):
        loaded_images = [None] * len(image_data)
        pending = []
        for idx, image_item in enumerate(image_data):
            if isinstance(image_item, (str, bytes)) or hasattr(image_item, "url"):
                pending.append((idx, image_item))
            else:
                loaded_images[idx] = image_item
        futures = [
            self.io_executor.submit(self._load_image_item, image_item)
            for _, image_item in pending
        ]
        for (idx, _), future in zip(pending, futures):
            loaded_images[idx] = future.result()
        return loaded_images

    def _normalize_videos(self, video_data, processor_kwargs):
        if not video_data:
            return None
        video_kwargs = dict((processor_kwargs or {}).get("videos_kwargs") or {})
        futures = [
            self.io_executor.submit(
                self._processor.normalize_video_item,
                video_item,
                processor_kwargs={"videos_kwargs": video_kwargs},
            )
            for video_item in video_data
        ]
        return [future.result() for future in futures]

    def _process_token_ids(self, input_ids, image_data, request_obj):
        if not self._is_token_id_input(input_ids):
            return None

        video_data = getattr(request_obj, "video_data", None)
        if self._has_precomputed_item(image_data) or self._has_precomputed_item(
            video_data
        ):
            raise ValueError(
                "WeLMV4 token-id multimodal path does not support precomputed "
                "multimodal inputs."
            )

        processor_kwargs = self._processor_kwargs()
        normalized_videos = self._normalize_videos(video_data, processor_kwargs)
        resolved = self._processor.resolve_tokenized_multimodal_inputs(
            input_ids,
            images=image_data,
            normalized_videos=normalized_videos,
            load_images=False,
        )
        resolved_images = resolved["images"]
        loaded_images = self._load_images(resolved_images) if resolved_images else None
        return self._processor.process_resolved_tokenized_multimodal_prompt(
            input_ids,
            images=loaded_images,
            video_timestamp_groups=resolved["video_timestamp_groups"],
            image_resize_specs=resolved["image_resize_specs"],
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )

    def _fallback_text_processor(self, input_text, image_data, request_obj):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=getattr(request_obj, "video_data", None),
            audio_data=getattr(request_obj, "audio_data", None),
            multimodal_tokens=self.mm_tokens,
        )
        _, input_ids, ret = self.process_and_combine_mm_data(base_output, self.mm_tokens)
        ret["input_ids"] = input_ids.unsqueeze(0)
        return ret

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        del audio_data, args, kwargs
        entry_time = time.perf_counter()
        ret = self._process_token_ids(input_text, image_data, request_obj)
        used_token_id_path = ret is not None
        if ret is None:
            ret = self._fallback_text_processor(input_text, image_data, request_obj)
        process_time = time.perf_counter()
        rid = getattr(request_obj, "rid", "anonymous_rid")
        logger.debug(
            f"[WeLMV4VLMProcessor Perf] {rid=}, "
            f"path: {'token_ids' if used_token_id_path else 'text_fallback'}, "
            f"total_time: {(process_time - entry_time) * 1000:.2f} ms"
        )

        return self._build_output_from_processor_result(ret)
