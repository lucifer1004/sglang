from typing import List, Optional


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        self.normal_text = normal_text or ""
        self.reasoning_text = reasoning_text or ""


class BaseReasoningFormatDetector:
    """Base class providing one-time and streaming reasoning parsing."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        think_excluded_tokens: Optional[List[str]] = None,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
        thinks_internally: bool = False,
        reasoning_default: str = "always",
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.think_excluded_tokens = think_excluded_tokens
        self.tool_start_token = tool_start_token
        self.force_reasoning = force_reasoning
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.thinks_internally = thinks_internally
        self.reasoning_default = reasoning_default

        self._buffer = ""
        self.stripped_think_start = False
        self.think_start_self_label = ""

        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(
            self.think_start_token + self.think_start_self_label, ""
        ).strip()

        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):
            # Check for tool_start_token interruption
            if (
                in_reasoning
                and self.tool_start_token is not None
                and self.tool_start_token in processed_text
            ):
                # Find the first occurrence of tool_start_token and split there
                tool_idx = processed_text.find(self.tool_start_token)
                reasoning_text = processed_text[:tool_idx].strip()
                # Preserve tool_start_token in normal text
                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
            # Assume reasoning was truncated before end token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1].strip()

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )
        else:
            # think_end_token is in self.previous_content for continue_final_message=True case
            return StreamingParseResult(normal_text=processed_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        think_start_text = self.think_start_token + self.think_start_self_label

        # If the current text is a prefix of the think token, keep buffering
        tokens_to_check = [think_start_text, self.think_end_token]
        if self.tool_start_token:
            tokens_to_check.append(self.tool_start_token)
        if any(
            token.startswith(current_text) and token != current_text
            for token in tokens_to_check
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and think_start_text in current_text:
            current_text = current_text.replace(think_start_text, "", 1)
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        # Continue with reasoning content
        if self._in_reasoning:
            # Check for tool_start_token interruption
            if self.tool_start_token and self.tool_start_token in current_text:
                tool_idx = current_text.find(self.tool_start_token)
                reasoning_text = current_text[:tool_idx]
                # Preserve tool_start_token in normal text
                normal_text = current_text[tool_idx:]
                self._buffer = ""
                self._in_reasoning = False
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()
