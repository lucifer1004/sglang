import logging
from typing import List, Optional

from sglang.srt.function_call.welm_v4_detector import (
    _IncrementalDecoder,
    _resolve_token_id,
    filter_id_based_stream_stop,
    trim_matched_stop_for_id_parser,
)
from sglang.srt.parser.reasoning_types import (
    BaseReasoningFormatDetector,
    StreamingParseResult,
)

logger = logging.getLogger(__name__)


class WelmV4ReasoningDetector(BaseReasoningFormatDetector):
    """Token-id based reasoning detector for WeLM-v4 models.

    WeLM-v4 reasoning markers are added tokens. When ids are available, locate
    ``<think>`` / ``</think>`` by those ids so quoted marker strings in content
    cannot accidentally open or close reasoning. If ids are unavailable, fall
    back to the text-based ``BaseReasoningFormatDetector`` path.
    """

    accepts_token_ids = True
    filter_id_based_stream_stop = staticmethod(filter_id_based_stream_stop)

    def __init__(
        self,
        tokenizer=None,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ):
        # WeLM-v4 follows the DeepSeek-R1 parsing contract: generation is
        # considered reasoning until the first generated </think> token. Its
        # HF chat template owns any <think>/</think> prompt prefix, so mark it
        # as thinks_internally to prevent the conversation-template fallback
        # from appending another <think>.
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="always",
        )
        self.tokenizer = None
        self.skip_special_tokens = True
        self.spaces_between_special_tokens = True
        self.think_start_id = None
        self.think_end_id = None
        self.id_capable = False

        self.remaining_token_ids: List[int] = []
        self._stripped_reasoning_start = False
        self._reasoning_ids: List[int] = []
        self._answer_ids: List[int] = []
        self._reasoning_decoder: Optional[_IncrementalDecoder] = None
        self._answer_decoder: Optional[_IncrementalDecoder] = None
        self.configure_tokenizer(
            tokenizer,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    def configure_tokenizer(
        self,
        tokenizer,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.think_start_id = _resolve_token_id(tokenizer, self.think_start_token)
        self.think_end_id = _resolve_token_id(tokenizer, self.think_end_token)
        self.id_capable = (
            self.think_start_id is not None and self.think_end_id is not None
        )
        self._reset_streaming_state()

    @staticmethod
    def trim_matched_stop(output, finished_reason: Optional[dict], no_stop_trim: bool):
        return trim_matched_stop_for_id_parser(output, finished_reason, no_stop_trim)

    def _new_decoder(self) -> _IncrementalDecoder:
        return _IncrementalDecoder(
            self.tokenizer,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

    def _decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

    def _reasoning_start(self, ids: List[int]) -> int:
        """Skip a leading ``<think>`` token if the model emitted one."""
        if (
            ids
            and not self._stripped_reasoning_start
            and self.think_start_id is not None
            and ids[0] == self.think_start_id
        ):
            self._in_reasoning = True
            self._stripped_reasoning_start = True
            return 1
        return 0

    def _reset_streaming_state(self, keep_reasoning_state: bool = False) -> None:
        if not keep_reasoning_state:
            self._in_reasoning = getattr(self, "force_reasoning", True)
            self._stripped_reasoning_start = False
        self.remaining_token_ids = []
        self._reasoning_ids = []
        self._answer_ids = []
        self._reasoning_decoder = None
        self._answer_decoder = None

    def _ensure_stream_decoders(self) -> None:
        if self._reasoning_decoder is None:
            self._reasoning_decoder = self._new_decoder()
        if self._answer_decoder is None:
            self._answer_decoder = self._new_decoder()

    def _streaming_error_result(
        self, new_text: str, token_ids: Optional[List[int]]
    ) -> StreamingParseResult:
        logger.exception("Error in WeLM-v4 token-id streaming reasoning parser")
        in_reasoning = self._in_reasoning
        self._reset_streaming_state(keep_reasoning_state=True)
        self._in_reasoning = in_reasoning
        if in_reasoning:
            return StreamingParseResult(reasoning_text=new_text)
        self.remaining_token_ids = list(token_ids or [])
        return StreamingParseResult(normal_text=new_text)

    def detect_and_parse(
        self, text: str, token_ids: Optional[List[int]] = None
    ) -> StreamingParseResult:
        if not self.id_capable or token_ids is None:
            self.remaining_token_ids = []
            return super().detect_and_parse(text)

        ids = list(token_ids or [])
        start = self._reasoning_start(ids)
        try:
            end = ids.index(self.think_end_id, start)
        except ValueError:
            end = -1

        if end == -1:
            self.remaining_token_ids = []
            return StreamingParseResult(reasoning_text=self._decode(ids[start:]))

        reasoning_ids = ids[start:end]
        answer_ids = ids[end + 1 :]
        self.remaining_token_ids = answer_ids
        return StreamingParseResult(
            normal_text=self._decode(answer_ids),
            reasoning_text=self._decode(reasoning_ids),
        )

    def parse_streaming_increment(
        self, new_text: str, token_ids: Optional[List[int]] = None
    ) -> StreamingParseResult:
        if not self.id_capable or token_ids is None:
            self.remaining_token_ids = []
            return super().parse_streaming_increment(new_text)

        self._ensure_stream_decoders()
        reasoning_out = ""
        normal_out = ""
        self.remaining_token_ids = []

        try:
            for tid in token_ids:
                if self._in_reasoning:
                    if (
                        not self._stripped_reasoning_start
                        and tid == self.think_start_id
                    ):
                        self._stripped_reasoning_start = True
                        continue
                    self._stripped_reasoning_start = True

                    if tid == self.think_end_id:
                        reasoning_out += self._reasoning_decoder.flush(
                            self._reasoning_ids
                        )
                        self._in_reasoning = False
                        continue

                    self._reasoning_ids.append(tid)
                    if self.stream_reasoning:
                        reasoning_out += self._reasoning_decoder.step(
                            self._reasoning_ids
                        )
                else:
                    self._answer_ids.append(tid)
                    normal_out += self._answer_decoder.step(self._answer_ids)
                    self.remaining_token_ids.append(tid)
        except Exception:
            return self._streaming_error_result(new_text, token_ids)

        return StreamingParseResult(
            normal_text=normal_out, reasoning_text=reasoning_out
        )
