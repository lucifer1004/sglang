"""
Unit-tests for OpenAIServingChat -- rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import hashlib
import json
import unittest
import uuid
from http import HTTPStatus
from typing import Optional
from unittest.mock import Mock, PropertyMock, call, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
    normalize_tool_content,
)
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import FINISHED_MATCHED_REGEX
from sglang.srt.managers.template_detection import ReasoningToggleConfig
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="stage-a-test-cpu")


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            stream_response_default_include_usage=False,
        )
        # Mock hf_config for _resolve_chat_encoding_spec check
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        self.model_config.hf_config = mock_hf_config

        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None
        self.reasoning_config = None
        self.force_reasoning = False


class _FakeWelmTokenizer:
    CONTROL = {
        "<think>": 1001,
        "</think>": 1002,
        "<tool_call>": 1003,
        "</tool_call>": 1004,
        "<arg_key>": 1005,
        "</arg_key>": 1006,
        "<arg_value>": 1007,
        "</arg_value>": 1008,
        "<|im_end|>": 1009,
    }
    eos_token_id = CONTROL["<|im_end|>"]
    additional_stop_token_ids = []
    unk_token_id = 0

    def __init__(self):
        self._id_to_token = {
            token_id: token for token, token_id in self.CONTROL.items()
        }
        self.chat_template = None
        self.bos_token_id = 1

    def convert_tokens_to_ids(self, token):
        return self.CONTROL.get(token, self.unk_token_id)

    def decode(
        self, token_ids, skip_special_tokens=True, spaces_between_special_tokens=True
    ):
        del spaces_between_special_tokens
        parts = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            parts.append(self._id_to_token.get(token_id, chr(token_id)))
        return "".join(parts)


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    def _welm_chat(self, *, with_tools=False):
        self.tm.server_args.incremental_streaming_output = True
        self.tm.server_args.reasoning_parser = "welm-v4"
        self.tm.server_args.tool_call_parser = "welm-v4" if with_tools else None
        self.tm.tokenizer = _FakeWelmTokenizer()
        self.tm.model_config.hf_eos_token_id = {self.tm.tokenizer.eos_token_id}
        return OpenAIServingChat(self.tm, self.template_manager)

    def _run_chat_stream(self, chat, request):
        adapted_request = Mock(spec=GenerateReqInput)
        adapted_request.sampling_params = {
            "stop": request.stop,
            "stop_regex": request.stop_regex,
            "no_stop_trim": request.no_stop_trim,
        }

        async def collect():
            return [
                chunk
                async for chunk in chat._generate_chat_stream(
                    adapted_request, request, self.fastapi_request
                )
            ]

        return get_or_create_event_loop().run_until_complete(collect())

    @staticmethod
    def _stream_choices(output):
        for chunk in output:
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                yield json.loads(chunk[len("data: ") :])["choices"][0]

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with (
            patch(
                "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
            ) as conv_mock,
            patch.object(self.chat, "_process_messages") as proc_mock,
        ):
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = MessageProcessingResult(
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    def test_jinja_uses_openai_tool_schema_first(self):
        """Ensure Jinja chat templates receive OpenAI-shaped tools by default."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.chat._process_messages(req, is_multimodal=False)

        expected_tools = [tool.model_dump(exclude_unset=True) for tool in req.tools]
        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs["tools"], expected_tools)

    def test_jinja_tool_schema_fallback_to_flat_function(self):
        """Fallback to function-only schema when template rejects OpenAI wrapper."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.tm.tokenizer.apply_chat_template.side_effect = [
            RuntimeError("template expects flat tools format"),
            [1, 2, 3],
        ]

        self.chat._process_messages(req, is_multimodal=False)

        first_tools = self.tm.tokenizer.apply_chat_template.call_args_list[0].kwargs[
            "tools"
        ]
        second_tools = self.tm.tokenizer.apply_chat_template.call_args_list[1].kwargs[
            "tools"
        ]
        self.assertEqual(
            first_tools,
            [tool.model_dump(exclude_unset=True) for tool in req.tools],
        )
        self.assertEqual(
            second_tools,
            [tool.function.model_dump(exclude_unset=True) for tool in req.tools],
        )

    def test_tool_constraint_reasoning_ownership(self):
        named = {"type": "function", "function": {"name": "get_weather"}}
        cases = [
            ("required", True, "welm-v4", False),
            (named, True, None, True),
            ("required", False, None, False),
            (named, False, "welm-v4", False),
        ]
        for tool_choice, thinking_mode, reasoning_parser, expected in cases:
            with self.subTest(
                tool_choice=tool_choice,
                thinking_mode=thinking_mode,
                reasoning_parser=reasoning_parser,
            ):
                request = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "Hi?"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "get_weather"},
                        }
                    ],
                    tool_choice=tool_choice,
                )
                parser = Mock()
                parser.get_structure_constraint.return_value = (
                    "structural_tag",
                    "constraint",
                )
                self.tm.server_args.reasoning_parser = reasoning_parser

                with (
                    patch.object(
                        self.chat,
                        "_get_reasoning_from_request",
                        return_value=thinking_mode,
                    ),
                    patch.object(
                        self.chat,
                        "_build_function_call_parser",
                        return_value=parser,
                    ),
                    patch.object(
                        self.chat,
                        "_apply_conversation_template",
                        return_value=Mock(),
                    ),
                ):
                    self.chat._process_messages(request, is_multimodal=False)

                self.assertEqual(
                    parser.get_structure_constraint.call_args.kwargs["thinking_mode"],
                    expected,
                )

    def test_stop_str_isolation_between_requests(self):
        """Test that stop strings from one request don't affect subsequent requests.

        This tests the fix for the bug where conv.stop_str was being mutated globally,
        causing stop strings from one request to persist in subsequent requests.
        """
        # Mock conversation template with initial stop_str
        initial_stop_str = ["\n"]

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object that will be returned by generate_chat_conv
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = None
            conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = (
                initial_stop_str.copy()
            )  # Template's default stop strings
            conv_mock.return_value = conv_ins

            # First request with additional stop string
            req1 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "First request"}],
                stop=["CUSTOM_STOP"],
            )

            # Call the actual _apply_conversation_template method (not mocked)
            result1 = self.chat._apply_conversation_template(req1, is_multimodal=False)

            # Verify first request has both stop strings
            expected_stop1 = initial_stop_str + ["CUSTOM_STOP"]
            self.assertEqual(result1.stop, expected_stop1)

            # Verify the original template's stop_str wasn't mutated after first request
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

            # Second request without additional stop string
            req2 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Second request"}],
                # No custom stop strings
            )
            result2 = self.chat._apply_conversation_template(req2, is_multimodal=False)

            # Verify second request only has original stop strings (no CUSTOM_STOP from req1)
            self.assertEqual(result2.stop, initial_stop_str)
            self.assertNotIn("CUSTOM_STOP", result2.stop)
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

    def test_unstreamed_tool_args_completion(self):
        """Test that remaining tool call arguments are sent when generation finishes."""

        # Mock FunctionCallParser with detector that has partial tool call data
        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.skip_unstreamed_arg_backfill = False

        # Simulate a tool call that was partially streamed
        mock_detector.prev_tool_call_arr = [
            {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            }
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"'  # Partial arguments streamed so far
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return a chunk with remaining arguments
        self.assertIsNotNone(result, "Should return chunk with remaining arguments")

        # Parse the result to verify content
        self.assertTrue(result.startswith("data: "))
        chunk = json.loads(result[6:])
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        arguments = tool_calls[0]["function"]["arguments"]
        self.assertIn(', "unit": "celsius"}', arguments)

        self.assertIn(
            '"finish_reason":null',
            result,
            "Should not include finish_reason in completion chunk",
        )

    def test_unstreamed_tool_args_no_completion_needed(self):
        """Test that no completion chunk is sent when all arguments were already streamed."""

        # Mock FunctionCallParser with detector that has complete tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was completely streamed
        mock_detector.prev_tool_call_arr = [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"}'  # All arguments already streamed
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since no completion is needed
        self.assertIsNone(result, "Should return None when no completion is needed")

    def test_unstreamed_tool_args_no_parser_data(self):
        """Test that no completion chunk is sent when parser has no tool call data."""

        # Mock FunctionCallParser with empty detector
        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = []
        mock_detector.streamed_args_for_tool = []
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since there's no parser data
        self.assertIsNone(
            result, "Should return None when parser has no tool call data"
        )

    # ------------- kimi_k2 tool_call_id formatting -------------
    def test_kimi_k2_non_streaming_tool_call_id_format(self):
        """Ensure non-streaming tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Paris"}'
            call_info.tool_index = 0

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = ("", [call_info])

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            tool_calls, remaining_text, finish_reason = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
            )

            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:0")
            self.assertEqual(tool_calls[0].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_format(self):
        """Ensure streaming first chunk tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tools
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:0")

    def test_kimi_k2_non_streaming_tool_call_id_with_history(self):
        """Ensure non-streaming tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA and Tokyo?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=False,
        )

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Loa Angeles"}'
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            call_info.tool_index = 0

            call_info2 = Mock()
            call_info2.name = "get_weather"
            call_info2.parameters = '{"city":"Tokyo"}'
            call_info2.tool_index = 1

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = (
                "",
                [call_info, call_info2],
            )

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            history_tool_calls_cnt = self.chat._get_history_tool_calls_cnt(req)
            tool_calls, remaining_text, _ = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
                history_tool_calls_cnt=history_tool_calls_cnt,
            )

            self.assertEqual(history_tool_calls_cnt, 1)
            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 2)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:1")
            self.assertEqual(tool_calls[0].function.name, "get_weather")
            self.assertEqual(tool_calls[1].id, "functions.get_weather:2")
            self.assertEqual(tool_calls[1].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_with_history(self):
        """Ensure streaming first chunk tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:1")

    def test_dpsk_v32_encoding_path(self):
        """Test DeepSeek V3.2 encoding path detection and application."""
        from sglang.srt.managers.template_manager import TemplateManager

        # Only mock the fields that _use_dpsk_v32_encoding() actually reads:
        # tokenizer.chat_template and hf_config.architectures
        tm = _MockTokenizerManager()

        mock_hf_config = Mock()
        mock_hf_config.architectures = ["DeepseekV32ForCausalLM"]
        tm.model_config.hf_config = mock_hf_config

        # Case 1: No chat template + DeepSeek V3.2 arch -> should use dsv32 encoding
        tm.tokenizer.chat_template = None
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertEqual(serving_chat.chat_encoding_spec, "dsv32")

        # Case 2: Chat template exists -> should NOT use dsv32 encoding
        tm.tokenizer.chat_template = "some template"
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertIsNone(serving_chat.chat_encoding_spec)

        # Case 3: Not DeepSeek V3.2 architecture -> should NOT use dsv32 encoding
        tm.tokenizer.chat_template = None
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertIsNone(serving_chat.chat_encoding_spec)

        # Case 4: DeepseekV4 arch -> always dsv4, even with chat_template
        # (release ships a stale V3 jinja we deliberately override).
        mock_hf_config.architectures = ["DeepseekV4ForCausalLM"]
        tm.tokenizer.chat_template = "stale v3 jinja"
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertEqual(serving_chat.chat_encoding_spec, "dsv4")

        tm.tokenizer.chat_template = None
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertEqual(serving_chat.chat_encoding_spec, "dsv4")

    # ------------- dsv4 task + latest_reminder -------------
    def test_dsv4_task_field_schema(self):
        """Top-level `task` accepts the 6 DS task tokens and rejects others."""
        for valid in ("action", "query", "authority", "domain", "title", "read_url"):
            req = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "hi"}],
                task=valid,
            )
            self.assertEqual(req.task, valid)

        # None / unset is fine
        self.assertIsNone(self.basic_req.task)

        # Bogus value rejected at validation time
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "hi"}],
                task="bogus",
            )

    def test_latest_reminder_role_accepted(self):
        """`latest_reminder` is a first-class message role on generic param."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionMessageGenericParam,
        )

        msg = ChatCompletionMessageGenericParam(
            role="latest_reminder", content="Be terse."
        )
        self.assertEqual(msg.role, "latest_reminder")

        # Full request with reminder before user parses cleanly.
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "latest_reminder", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ],
        )
        self.assertEqual(req.messages[0].role, "latest_reminder")
        self.assertEqual(req.messages[1].role, "user")

    def test_attach_task_to_last_user_message(self):
        """Helper attaches task to the nearest user/developer message."""
        from sglang.srt.entrypoints.openai import encoding_dsv4

        messages = [{"role": "user", "content": "Hi"}]
        encoding_dsv4.attach_task_to_last_user_message(messages, "domain")
        self.assertEqual(messages[0]["task"], "domain")

        # Prefers the LAST user message across a multi-turn conversation.
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
        encoding_dsv4.attach_task_to_last_user_message(messages, "query")
        self.assertNotIn("task", messages[0])
        self.assertEqual(messages[2]["task"], "query")

        # `developer` role is treated like `user` (matches encoder semantics).
        messages = [{"role": "developer", "content": "dev"}]
        encoding_dsv4.attach_task_to_last_user_message(messages, "authority")
        self.assertEqual(messages[0]["task"], "authority")

        # No user/developer present -> raises.
        with self.assertRaises(ValueError):
            encoding_dsv4.attach_task_to_last_user_message(
                [{"role": "system", "content": "s"}], "domain"
            )

    def test_dsv4_content_parts_list_normalized(self):
        """OpenAI list-of-parts content flattens to text before reaching the encoder."""
        from sglang.srt.entrypoints.openai import encoding_dsv4
        from sglang.srt.parser.jinja_template_utils import (
            process_content_for_template_format,
        )

        req = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "say hi"}],
                }
            ],
        )
        messages = [m.model_dump() for m in req.messages]
        # Mirror the boundary normalization _process_messages does for any
        # non-None chat_encoding_spec.
        for i, msg in enumerate(messages):
            if isinstance(msg.get("content"), list):
                messages[i] = process_content_for_template_format(
                    msg, "string", [], [], [], []
                )
        out = encoding_dsv4.encode_messages(messages, thinking_mode="chat")
        self.assertIn("<｜User｜>say hi", out)

        # Multiple text parts concat with single space; non-text parts dropped.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            }
        ]
        for i, msg in enumerate(messages):
            if isinstance(msg.get("content"), list):
                messages[i] = process_content_for_template_format(
                    msg, "string", [], [], [], []
                )
        out = encoding_dsv4.encode_messages(messages, thinking_mode="chat")
        self.assertIn("<｜User｜>describe", out)
        self.assertNotIn("image_url", out)

    def test_dsv4_task_and_reminder_encode_end_to_end(self):
        """Task + latest_reminder plumb through to the dsv4 encoder correctly."""
        from sglang.srt.entrypoints.openai import encoding_dsv4

        # 1) task='domain' in chat mode -> `<｜domain｜>` appended, no Assistant
        #    prefix (this is a single-shot classification, not a chat turn).
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is SGLang?"}],
            task="domain",
        )
        messages = [m.model_dump() for m in req.messages]
        encoding_dsv4.attach_task_to_last_user_message(messages, req.task)
        out = encoding_dsv4.encode_messages(messages, thinking_mode="chat")
        self.assertIn("<｜domain｜>", out)
        self.assertTrue(out.rstrip().endswith("<｜domain｜>"))
        self.assertNotIn("<｜Assistant｜>", out)

        # 2) task='action' in thinking mode -> Assistant + <think> + <｜action｜>
        #    (action is the one task that still runs a reasoning pass).
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            task="action",
        )
        messages = [m.model_dump() for m in req.messages]
        encoding_dsv4.attach_task_to_last_user_message(messages, req.task)
        out = encoding_dsv4.encode_messages(messages, thinking_mode="thinking")
        self.assertIn("<｜Assistant｜>", out)
        self.assertIn("<think>", out)
        self.assertTrue(out.rstrip().endswith("<｜action｜>"))

        # 3) latest_reminder preceding user -> reminder renders before user,
        #    Assistant prefix still comes after user.
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "latest_reminder", "content": "Be terse."},
                {"role": "user", "content": "Hello"},
            ],
        )
        messages = [m.model_dump() for m in req.messages]
        out = encoding_dsv4.encode_messages(messages, thinking_mode="chat")
        self.assertIn("<｜latest_reminder｜>Be terse.", out)
        self.assertIn("<｜User｜>Hello", out)
        self.assertLess(
            out.index("<｜latest_reminder｜>"),
            out.index("<｜User｜>"),
        )
        self.assertIn("<｜Assistant｜>", out)

    def test_streaming_parser_value_error_yields_error_and_done(self):
        error_message = "malformed committed tool call"

        async def _raise_parser_error(*args, **kwargs):
            del args, kwargs
            raise ValueError(error_message)
            yield  # pragma: no cover - make this an async generator

        self.chat._process_tool_call_stream = _raise_parser_error
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Use the tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            stream=True,
        )

        chunks = self._run_chat_stream(self.chat, req)

        self.assertEqual(len(chunks), 3)
        role_chunk = json.loads(chunks[0][len("data: ") :])
        self.assertEqual(role_chunk["choices"][0]["delta"]["role"], "assistant")
        error_chunk = json.loads(chunks[1][len("data: ") :])
        self.assertEqual(error_chunk["error"]["message"], error_message)
        self.assertEqual(chunks[2], "data: [DONE]\n\n")

    def test_streaming_abort_yields_error(self):
        """Test that an abort finish reason during streaming correctly yields an error and stops."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort():
            yield {
                "text": "Partial ",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = _mock_generate_abort()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                try:
                    async for chunk in self.chat._generate_chat_stream(
                        adapted_request, req, self.fastapi_request
                    ):
                        chunks.append(chunk)
                except Exception as e:
                    print(f"Error during stream iteration: {e}")
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        error_chunk_data = None
        for c in chunks:
            if "error" in c:
                error_chunk_data = json.loads(c[len("data: ") :])
                break
        self.assertIsNotNone(error_chunk_data, "Error chunk not found in stream")
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

        # Ensure the stream stops after the abort error
        # The last chunk should be "data: [DONE]\n\n"
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        # Check that there is an error chunk and a DONE chunk
        self.assertEqual(len(chunks), 2)
        self.assertIn("error", chunks[0])

    def test_non_streaming_cached_tokens_details_emits_sglext(self):
        """Test that non-streaming chat responses emit cached token details in sglext."""

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            return_cached_tokens_details=True,
        )
        ret = [
            {
                "text": "Cached response",
                "meta_info": {
                    "id": "chatcmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
        ]

        response = self.chat._build_chat_response(req, ret, 1234567890)

        self.assertIsNotNone(response.sglext)
        self.assertEqual(
            response.sglext.cached_tokens_details.model_dump(exclude_none=True),
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )

    def test_streaming_cached_tokens_details_emits_sglext(self):
        """Test that streaming chat responses emit cached token details in sglext."""

        async def _mock_generate_with_cached_tokens_details():
            yield {
                "text": "Cached response",
                "meta_info": {
                    "id": "chatcmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = (
            _mock_generate_with_cached_tokens_details()
        )

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            stream=True,
            return_cached_tokens_details=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                async for chunk in self.chat._generate_chat_stream(
                    adapted_request, req, self.fastapi_request
                ):
                    chunks.append(chunk)
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        sglext_chunks = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk[len("data: ") :])
            if "sglext" in data:
                sglext_chunks.append(data)

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["choices"], [])
        self.assertEqual(
            sglext_chunks[0]["sglext"]["cached_tokens_details"],
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )

    # ------------- incremental streaming output tests -------------
    def test_incremental_streaming_output_delta(self):
        """Test that streaming with incremental_streaming_output produces correct deltas.

        When incremental_streaming_output is enabled, content["text"] is already the
        incremental delta (not the full accumulated text). The delta computation must
        use content["text"] directly instead of slicing by the accumulated buffer length.

        Regression test for https://github.com/sgl-project/sglang/issues/22510.
        """
        # Enable incremental_streaming_output on the mock
        self.tm.server_args.incremental_streaming_output = True

        # Simulate incremental streaming: each yield has ONLY the new text (delta),
        # NOT the full accumulated text.
        incremental_chunks = [
            ("I am", None),
            (" a large", None),
            (" language model", None),
            (".", {"type": "stop", "matched": None}),
        ]

        async def _mock_generate_incremental():
            for text, finish_reason in incremental_chunks:
                yield {
                    "text": text,
                    "meta_info": {
                        "id": "chatcmpl-incr-test",
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "cached_tokens": 0,
                        "finish_reason": finish_reason,
                        "output_token_logprobs": None,
                        "output_top_logprobs": None,
                    },
                    "index": 0,
                }

        self.tm.generate_request.return_value = _mock_generate_incremental()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                async for chunk in self.chat._generate_chat_stream(
                    adapted_request, req, self.fastapi_request
                ):
                    chunks.append(chunk)
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        # Extract content deltas from SSE chunks
        deltas = []
        for c in chunks:
            if not c.startswith("data: ") or c.strip() == "data: [DONE]":
                continue
            data = json.loads(c[len("data: ") :])
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["delta"].get("content")
                if content:
                    deltas.append(content)

        joined = "".join(deltas)
        self.assertEqual(
            joined,
            "I am a large language model.",
            f"Streaming deltas produced broken text: {deltas!r}",
        )

    def test_welm_incremental_ids_and_logprobs_split(self):
        chat = self._welm_chat(with_tools=True)
        eos_id = self.tm.tokenizer.eos_token_id

        output_ids = [
            self.tm.tokenizer.CONTROL["<think>"],
            ord("r"),
            self.tm.tokenizer.CONTROL["</think>"],
            ord("a"),
            eos_id,
        ]

        async def _mock_generate_incremental():
            frames = [
                (
                    "<think>",
                    output_ids[:1],
                    [(-0.1, output_ids[0], "<think>")],
                    None,
                ),
                (
                    "r</think>a<|im_end|>",
                    output_ids[1:],
                    [
                        (-0.2, output_ids[1], "r"),
                        (-0.3, output_ids[2], "</think>"),
                        (-0.4, output_ids[3], "a"),
                        (-0.5, output_ids[4], "<|im_end|>"),
                    ],
                    {"type": "stop", "matched": eos_id},
                ),
            ]
            for frame_index, (text, frame_ids, logprobs, finish_reason) in enumerate(
                frames
            ):
                yield {
                    "text": text,
                    "output_ids": frame_ids,
                    "meta_info": {
                        "id": "chatcmpl-welm-logprobs",
                        "prompt_tokens": 1,
                        "completion_tokens": 5,
                        "reasoning_tokens": 3,
                        "cached_tokens": 0,
                        "finish_reason": finish_reason,
                        "output_token_logprobs": logprobs,
                        "output_token_logprobs_length": 1 if frame_index == 0 else 5,
                        "output_top_logprobs": [],
                    },
                    "index": 0,
                }

        self.tm.generate_request.return_value = _mock_generate_incremental()
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stream=True,
            stream_reasoning=True,
            logprobs=True,
        )

        reasoning_parts, content_parts = [], []
        reasoning_logprobs, content_logprobs = [], []
        for choice in self._stream_choices(self._run_chat_stream(chat, req)):
            delta = choice["delta"]
            tokens = [
                item["token"]
                for item in (choice.get("logprobs") or {}).get("content", [])
            ]
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
                reasoning_logprobs.extend(tokens)
            if delta.get("content"):
                content_parts.append(delta["content"])
                content_logprobs.extend(tokens)

        self.assertEqual("".join(reasoning_parts), "r")
        self.assertEqual("".join(content_parts), "a")
        self.assertEqual(reasoning_logprobs, ["<think>", "r", "</think>"])
        self.assertEqual(content_logprobs, ["a", "<|im_end|>"])

    def test_welm_streaming_handoff_preserves_content_and_done(self):
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        output_ids = [
            tokenizer.CONTROL["<think>"],
            ord("r"),
            tokenizer.CONTROL["</think>"],
            ord("a"),
            tokenizer.eos_token_id,
        ]

        async def generate():
            frames = [
                ("<think>r", output_ids[:2], None),
                (
                    "</think>a<|im_end|>",
                    output_ids[2:],
                    {
                        "type": "stop",
                        "matched": tokenizer.eos_token_id,
                    },
                ),
            ]
            for text, frame_ids, finish_reason in frames:
                yield {
                    "text": text,
                    "output_ids": frame_ids,
                    "meta_info": {
                        "id": "chatcmpl-welm-handoff",
                        "prompt_tokens": 1,
                        "completion_tokens": len(output_ids),
                        "reasoning_tokens": 3,
                        "cached_tokens": 0,
                        "finish_reason": finish_reason,
                        "output_token_logprobs": None,
                        "output_top_logprobs": None,
                    },
                    "index": 0,
                }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stream=True,
            stream_reasoning=True,
            stop="STOP",
            no_stop_trim=True,
        )

        reasoning_parser = chat._build_reasoning_parser(
            chat.reasoning_parser, True, True, request
        )
        tool_parser = chat._build_function_call_parser(request.tools, request)
        with (
            patch.object(
                type(reasoning_parser.detector),
                "handoff_content_ids",
                new_callable=PropertyMock,
                create=True,
            ) as handoff_flag,
            patch.object(
                type(tool_parser.detector),
                "reparse_content_ids",
                new_callable=PropertyMock,
                create=True,
            ) as reparse_flag,
            patch.object(
                chat, "_build_reasoning_parser", return_value=reasoning_parser
            ),
            patch.object(
                chat, "_build_function_call_parser", return_value=tool_parser
            ) as build_tool_parser,
        ):
            output = self._run_chat_stream(chat, request)
        self.assertEqual(build_tool_parser.call_count, 1)
        self.assertEqual(
            [entry for entry in handoff_flag.call_args_list if entry.args],
            [call(True)],
        )
        self.assertEqual(
            [entry for entry in reparse_flag.call_args_list if entry.args],
            [call(True)],
        )
        reasoning, content = [], []
        for choice in self._stream_choices(output):
            delta = choice["delta"]
            reasoning.append(delta.get("reasoning_content") or "")
            content.append(delta.get("content") or "")
        self.assertEqual("".join(reasoning), "r")
        self.assertEqual("".join(content), "a")
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_streaming_text_stop_uses_text_parsers_from_first_chunk(self):
        for stop_kwargs in ({"stop": "STOP"}, {"stop_regex": "ST.*"}):
            with self.subTest(stop_kwargs=stop_kwargs):
                chat = self._welm_chat(with_tools=True)
                tokenizer = self.tm.tokenizer
                output_ids = [
                    tokenizer.CONTROL["<think>"],
                    ord("r"),
                    tokenizer.CONTROL["</think>"],
                    tokenizer.CONTROL["<tool_call>"],
                    *map(ord, "get_weather\n"),
                    tokenizer.CONTROL["</tool_call>"],
                    *map(ord, "STOP"),
                ]

                async def generate():
                    frames = [
                        ("<think>r", output_ids[:2], None),
                        (
                            "</think><tool_call>get_weather\n",
                            output_ids[2:-5],
                            None,
                        ),
                        (
                            "</tool_call>",
                            output_ids[-5:],
                            {"type": "stop", "matched": "STOP"},
                        ),
                    ]
                    for text, frame_ids, finish_reason in frames:
                        yield {
                            "text": text,
                            "output_ids": frame_ids,
                            "meta_info": {
                                "id": "chatcmpl-welm-text-stop",
                                "prompt_tokens": 1,
                                "completion_tokens": len(output_ids),
                                "reasoning_tokens": 3,
                                "cached_tokens": 0,
                                "finish_reason": finish_reason,
                                "output_token_logprobs": None,
                                "output_top_logprobs": None,
                            },
                            "index": 0,
                        }

                self.tm.generate_request.return_value = generate()
                request = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "Hi?"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "get_weather"},
                        }
                    ],
                    stream=True,
                    stream_reasoning=True,
                    **stop_kwargs,
                )
                reasoning_parser = chat._build_reasoning_parser(
                    chat.reasoning_parser, True, True, request
                )
                tool_parser = chat._build_function_call_parser(request.tools, request)
                with (
                    patch.object(
                        chat,
                        "_build_reasoning_parser",
                        return_value=reasoning_parser,
                    ),
                    patch.object(
                        chat,
                        "_build_function_call_parser",
                        return_value=tool_parser,
                    ),
                    patch.object(
                        chat,
                        "_filter_id_based_stream_stop",
                        wraps=chat._filter_id_based_stream_stop,
                    ) as stop_filter,
                ):
                    output = self._run_chat_stream(chat, request)

                reasoning, names = [], []
                for choice in self._stream_choices(output):
                    delta = choice["delta"]
                    reasoning.append(delta.get("reasoning_content") or "")
                    names.extend(
                        tool_call["function"].get("name")
                        for tool_call in delta.get("tool_calls") or []
                        if tool_call["function"].get("name")
                    )
                self.assertEqual(stop_filter.call_count, 0)
                self.assertFalse(reasoning_parser.detector.handoff_content_ids)
                self.assertFalse(tool_parser.detector.reparse_content_ids)
                self.assertEqual("".join(reasoning), "r")
                self.assertEqual(names, ["get_weather"])
                self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_streaming_text_stop_configured_but_not_hit_still_text_routes(self):
        """The streaming gate is request-level: a configured string stop routes
        the whole request through text parsers even when generation finishes
        via a stop token (EOS) and the string stop never matches."""
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        output_ids = [
            tokenizer.CONTROL["<think>"],
            ord("r"),
            tokenizer.CONTROL["</think>"],
            tokenizer.CONTROL["<tool_call>"],
            *map(ord, "get_weather\n"),
            tokenizer.CONTROL["</tool_call>"],
            tokenizer.eos_token_id,
        ]

        async def generate():
            frames = [
                ("<think>r", output_ids[:2], None),
                (
                    "</think><tool_call>get_weather\n",
                    output_ids[2:-2],
                    None,
                ),
                (
                    "</tool_call>",
                    output_ids[-2:],
                    {"type": "stop", "matched": tokenizer.eos_token_id},
                ),
            ]
            for text, frame_ids, finish_reason in frames:
                yield {
                    "text": text,
                    "output_ids": frame_ids,
                    "meta_info": {
                        "id": "chatcmpl-welm-stop-not-hit",
                        "prompt_tokens": 1,
                        "completion_tokens": len(output_ids),
                        "reasoning_tokens": 3,
                        "cached_tokens": 0,
                        "finish_reason": finish_reason,
                        "output_token_logprobs": None,
                        "output_top_logprobs": None,
                    },
                    "index": 0,
                }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather"},
                }
            ],
            stream=True,
            stream_reasoning=True,
            stop="STOP",
        )
        reasoning_parser = chat._build_reasoning_parser(
            chat.reasoning_parser, True, True, request
        )
        tool_parser = chat._build_function_call_parser(request.tools, request)
        with (
            patch.object(
                chat,
                "_build_reasoning_parser",
                return_value=reasoning_parser,
            ),
            patch.object(
                chat,
                "_build_function_call_parser",
                return_value=tool_parser,
            ),
            patch.object(
                chat,
                "_filter_id_based_stream_stop",
                wraps=chat._filter_id_based_stream_stop,
            ) as stop_filter,
        ):
            output = self._run_chat_stream(chat, request)

        reasoning, names = [], []
        for choice in self._stream_choices(output):
            delta = choice["delta"]
            reasoning.append(delta.get("reasoning_content") or "")
            names.extend(
                tool_call["function"].get("name")
                for tool_call in delta.get("tool_calls") or []
                if tool_call["function"].get("name")
            )
        self.assertEqual(stop_filter.call_count, 0)
        self.assertFalse(reasoning_parser.detector.handoff_content_ids)
        self.assertFalse(tool_parser.detector.reparse_content_ids)
        self.assertEqual("".join(reasoning), "r")
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_streaming_partial_token_text_stop_keeps_authoritative_text(self):
        chat = self._welm_chat()
        tokenizer = self.tm.tokenizer

        async def generate():
            yield {
                "text": "r</think>a",
                "output_ids": [
                    ord("r"),
                    tokenizer.CONTROL["</think>"],
                    *map(ord, "apple"),
                ],
                "meta_info": {
                    "id": "chatcmpl-welm-partial-token-stop",
                    "prompt_tokens": 1,
                    "completion_tokens": 7,
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": "p"},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
            stop="p",
        )
        with patch.object(
            chat,
            "_filter_id_based_stream_stop",
            wraps=chat._filter_id_based_stream_stop,
        ) as stop_filter:
            output = self._run_chat_stream(chat, request)
        visible_content = "".join(
            choice["delta"].get("content") or ""
            for choice in self._stream_choices(output)
        )
        self.assertEqual(stop_filter.call_count, 0)
        self.assertEqual(visible_content, "a")

    def test_welm_streaming_string_finish_with_terminal_eos_stays_successful(self):
        chat = self._welm_chat()
        tokenizer = self.tm.tokenizer

        async def generate():
            yield {
                "text": "r</think>a<|im_end|>",
                "output_ids": [
                    ord("r"),
                    tokenizer.CONTROL["</think>"],
                    ord("a"),
                    tokenizer.eos_token_id,
                ],
                "meta_info": {
                    "id": "chatcmpl-welm-vocab-boundary",
                    "prompt_tokens": 1,
                    "completion_tokens": 4,
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": "NaN happened"},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
            ignore_eos=True,
        )
        output = self._run_chat_stream(chat, request)
        visible_content = "".join(
            choice["delta"].get("content") or ""
            for choice in self._stream_choices(output)
        )
        self.assertEqual(visible_content, "a")
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_streaming_unexpected_string_finish_is_structured_error(self):
        chat = self._welm_chat()
        tokenizer = self.tm.tokenizer

        async def generate():
            yield {
                "text": "r</think>a",
                "output_ids": [
                    ord("r"),
                    tokenizer.CONTROL["</think>"],
                    ord("a"),
                ],
                "meta_info": {
                    "id": "chatcmpl-welm-invalid-string-finish",
                    "prompt_tokens": 1,
                    "completion_tokens": 3,
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": "unexpected"},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
        )
        output = self._run_chat_stream(chat, request)
        error = json.loads(output[-2][len("data: ") :])
        self.assertIn("terminal stop token ID", error["error"]["message"])
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_handoff_splits_reasoning_and_tool_logprobs(self):
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        tool_ids = (
            [tokenizer.CONTROL["<tool_call>"]]
            + list(map(ord, "get_weather"))
            + [tokenizer.CONTROL["</tool_call>"]]
        )
        output_ids = [
            tokenizer.CONTROL["<think>"],
            ord("r"),
            tokenizer.CONTROL["</think>"],
            *tool_ids,
        ]
        token_text = [
            "<think>",
            "r",
            "</think>",
            "<tool_call>",
            *list("get_weather"),
            "</tool_call>",
        ]

        async def generate():
            yield {
                "text": "<think>r</think><tool_call>get_weather</tool_call>",
                "output_ids": output_ids,
                "meta_info": {
                    "id": "chatcmpl-welm-tool-logprobs",
                    "prompt_tokens": 1,
                    "completion_tokens": len(output_ids),
                    "reasoning_tokens": 3,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [
                        (-0.1, token_id, text)
                        for token_id, text in zip(output_ids, token_text)
                    ],
                    "output_token_logprobs_length": len(output_ids),
                    "output_top_logprobs": [],
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stream=True,
            stream_reasoning=True,
            logprobs=True,
        )

        reasoning_logprobs = []
        tool_logprobs = []
        tool_names = []
        for choice in self._stream_choices(self._run_chat_stream(chat, request)):
            tokens = [
                item["token"]
                for item in (choice.get("logprobs") or {}).get("content", [])
            ]
            delta = choice["delta"]
            if delta.get("reasoning_content"):
                reasoning_logprobs.extend(tokens)
            if delta.get("tool_calls"):
                tool_logprobs.extend(tokens)
                name = delta["tool_calls"][0]["function"].get("name")
                if name:
                    tool_names.append(name)

        self.assertEqual(reasoning_logprobs, token_text[:3])
        self.assertEqual(tool_logprobs, token_text[3:])
        self.assertEqual(tool_names, ["get_weather"])

    def test_welm_handoff_keeps_choice_state_isolated(self):
        chat = self._welm_chat(with_tools=True)
        think_end = self.tm.tokenizer.CONTROL["</think>"]

        async def generate():
            frames = [
                (0, "r0", list(map(ord, "r0")), None),
                (
                    1,
                    "r1</think>b",
                    list(map(ord, "r1")) + [think_end, ord("b")],
                    {"type": "stop", "matched": None},
                ),
                (
                    0,
                    "</think>a",
                    [think_end, ord("a")],
                    {"type": "stop", "matched": None},
                ),
            ]
            for index, text, output_ids, finish_reason in frames:
                yield {
                    "text": text,
                    "output_ids": output_ids,
                    "meta_info": {
                        "id": "chatcmpl-welm-choice-state",
                        "prompt_tokens": 1,
                        "completion_tokens": len(output_ids),
                        "reasoning_tokens": 0,
                        "cached_tokens": 0,
                        "finish_reason": finish_reason,
                        "output_token_logprobs": None,
                        "output_top_logprobs": None,
                    },
                    "index": index,
                }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            n=2,
            stream=True,
            stream_reasoning=True,
        )

        outputs = {
            0: {"reasoning": "", "content": ""},
            1: {"reasoning": "", "content": ""},
        }
        for chunk in self._run_chat_stream(chat, request):
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            payload = json.loads(chunk[len("data: ") :])
            for choice in payload.get("choices", []):
                delta = choice["delta"]
                state = outputs[choice["index"]]
                state["reasoning"] += delta.get("reasoning_content") or ""
                state["content"] += delta.get("content") or ""

        self.assertEqual(outputs[0], {"reasoning": "r0", "content": "a"})
        self.assertEqual(outputs[1], {"reasoning": "r1", "content": "b"})

    def test_welm_streaming_handoff_missing_ids_is_structured_error(self):
        chat = self._welm_chat(with_tools=True)

        async def generate():
            yield {
                "text": "r</think>a",
                "output_ids": [
                    ord("r"),
                    self.tm.tokenizer.CONTROL["</think>"],
                    ord("a"),
                ],
                "meta_info": {
                    "id": "chatcmpl-welm-handoff-error",
                    "prompt_tokens": 1,
                    "completion_tokens": 3,
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": None,
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        chat._filter_id_based_stream_stop = Mock(
            return_value=("r</think>a", None, None)
        )
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stream=True,
        )

        output = self._run_chat_stream(chat, request)
        error = json.loads(output[-2][len("data: ") :])
        self.assertIn("requires token ids", error["error"]["message"])
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_streaming_incomplete_before_name_recovers_actual_content(self):
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        incomplete = [tokenizer.CONTROL["<tool_call>"]] + list(map(ord, "unknown_tool"))
        output_ids = [
            ord("r"),
            tokenizer.CONTROL["</think>"],
            *incomplete,
        ]

        async def generate():
            yield {
                "text": "r</think><tool_call>unknown_tool",
                "output_ids": output_ids,
                "meta_info": {
                    "id": "chatcmpl-welm-incomplete-content",
                    "prompt_tokens": 1,
                    "completion_tokens": len(output_ids),
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        output = self._run_chat_stream(chat, request)
        visible_content = "".join(
            choice["delta"].get("content") or ""
            for choice in self._stream_choices(output)
        )
        self.assertEqual(visible_content, "<tool_call>unknown_tool")
        self.assertNotIn("</tool_call>", visible_content)
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_streaming_incomplete_after_name_is_structured_error(self):
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        incomplete = (
            [tokenizer.CONTROL["<tool_call>"]]
            + list(map(ord, "get_weather"))
            + [tokenizer.CONTROL["<arg_key>"]]
            + list(map(ord, "city"))
        )
        output_ids = [
            ord("r"),
            tokenizer.CONTROL["</think>"],
            *incomplete,
        ]

        async def generate():
            yield {
                "text": "r</think><tool_call>get_weather<arg_key>city",
                "output_ids": output_ids,
                "meta_info": {
                    "id": "chatcmpl-welm-incomplete-tool",
                    "prompt_tokens": 1,
                    "completion_tokens": len(output_ids),
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            stream=True,
        )

        output = self._run_chat_stream(chat, request)
        error = json.loads(output[-2][len("data: ") :])
        self.assertIn("irreversible output", error["error"]["message"])
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_streaming_valid_tool_call_matches_sse_golden(self):
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        output_ids = [
            tokenizer.CONTROL["<think>"],
            ord("r"),
            tokenizer.CONTROL["</think>"],
            tokenizer.CONTROL["<tool_call>"],
            *map(ord, "get_weather"),
            tokenizer.CONTROL["</tool_call>"],
        ]

        async def generate():
            yield {
                "text": "<think>r</think><tool_call>get_weather</tool_call>",
                "output_ids": output_ids,
                "meta_info": {
                    "id": "chatcmpl-welm-golden",
                    "prompt_tokens": 1,
                    "completion_tokens": len(output_ids),
                    "reasoning_tokens": 3,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = generate()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
            stream_reasoning=True,
        )

        with (
            patch(
                "sglang.srt.entrypoints.openai.serving_chat.time.time",
                return_value=123,
            ),
            patch(
                "sglang.srt.entrypoints.openai.serving_chat.uuid.uuid4",
                return_value=uuid.UUID(int=1),
            ),
        ):
            output = self._run_chat_stream(chat, request)
        wire = "".join(output).encode()
        self.assertEqual(len(output), 6)
        self.assertEqual(
            hashlib.sha256(wire).hexdigest(),
            "ee9af56312954feba68cbb55cac9d9297c3c729d69dfc014d73af906c9557097",
        )
        self.assertNotIn(b"<tool_call>", wire)
        self.assertNotIn(b"</tool_call>", wire)

    def test_welm_required_and_named_streaming_keep_native_xml(self):
        tool_choices = [
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ]
        for tool_choice in tool_choices:
            for parallel_tool_calls in (False, True):
                with self.subTest(
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                ):
                    self.tm = _MockTokenizerManager()
                    chat = self._welm_chat(with_tools=True)
                    tokenizer = self.tm.tokenizer
                    output_ids = [
                        tokenizer.CONTROL["<think>"],
                        ord("r"),
                        tokenizer.CONTROL["</think>"],
                        tokenizer.CONTROL["<tool_call>"],
                        *map(ord, "get_weather"),
                        tokenizer.CONTROL["</tool_call>"],
                    ]

                    async def generate():
                        yield {
                            "text": (
                                "<think>r</think>" "<tool_call>get_weather</tool_call>"
                            ),
                            "output_ids": output_ids,
                            "meta_info": {
                                "id": "chatcmpl-welm-native-xml",
                                "prompt_tokens": 1,
                                "completion_tokens": len(output_ids),
                                "reasoning_tokens": 3,
                                "cached_tokens": 0,
                                "finish_reason": {
                                    "type": "stop",
                                    "matched": None,
                                },
                                "output_token_logprobs": None,
                                "output_top_logprobs": None,
                            },
                            "index": 0,
                        }

                    self.tm.generate_request.return_value = generate()
                    request = ChatCompletionRequest(
                        model="x",
                        messages=[{"role": "user", "content": "Hi?"}],
                        tools=[
                            {
                                "type": "function",
                                "function": {"name": "get_weather"},
                            }
                        ],
                        tool_choice=tool_choice,
                        parallel_tool_calls=parallel_tool_calls,
                        stream=True,
                        stream_reasoning=True,
                    )

                    output = self._run_chat_stream(chat, request)
                    names = []
                    visible_content = ""
                    for choice in self._stream_choices(output):
                        delta = choice["delta"]
                        visible_content += delta.get("content") or ""
                        for tool_call in delta.get("tool_calls") or []:
                            name = tool_call["function"].get("name")
                            if name:
                                names.append(name)

                    self.assertEqual(names, ["get_weather"])
                    self.assertNotIn("<tool_call>", visible_content)
                    self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_welm_nonstream_handoff_preserves_plain_content(self):
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        output_ids = [ord("r"), tokenizer.CONTROL["</think>"], ord("a")]
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        ret = [
            {
                "text": "r</think>a",
                "output_ids": output_ids,
                "meta_info": {
                    "id": "chatcmpl-welm-nonstream",
                    "prompt_tokens": 1,
                    "completion_tokens": len(output_ids),
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "weight_version": "default",
                    "finish_reason": {"type": "stop", "matched": None},
                },
                "index": 0,
            }
        ]

        reasoning_parser = chat._build_reasoning_parser(
            chat.reasoning_parser, False, True, request
        )
        tool_parser = chat._build_function_call_parser(request.tools, request)
        with (
            patch.object(
                chat, "_build_reasoning_parser", return_value=reasoning_parser
            ),
            patch.object(
                chat, "_build_function_call_parser", return_value=tool_parser
            ) as build_tool_parser,
        ):
            response = chat._build_chat_response(request, ret, created=0)
        self.assertEqual(build_tool_parser.call_count, 1)
        self.assertFalse(reasoning_parser.detector.handoff_content_ids)
        self.assertFalse(tool_parser.detector.reparse_content_ids)
        self.assertEqual(response.choices[0].message.reasoning_content, "r")
        self.assertEqual(response.choices[0].message.content, "a")
        self.assertIsNone(response.choices[0].message.tool_calls)
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_welm_nonstream_text_stop_ignores_all_post_stop_ids(self):
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        output_ids = [
            ord("r"),
            tokenizer.CONTROL["</think>"],
            ord("a"),
            *map(ord, "STOP"),
            tokenizer.CONTROL["<tool_call>"],
            *map(ord, "get_weather"),
            tokenizer.CONTROL["</tool_call>"],
        ]
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stop="STOP",
        )
        ret = [
            {
                "text": "r</think>a",
                "output_ids": output_ids,
                "meta_info": {
                    "id": "chatcmpl-welm-text-stop",
                    "prompt_tokens": 1,
                    "completion_tokens": len(output_ids),
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "weight_version": "default",
                    "finish_reason": {"type": "stop", "matched": "STOP"},
                },
                "index": 0,
            }
        ]

        reasoning_parser = chat._build_reasoning_parser(
            chat.reasoning_parser, False, True, request
        )
        tool_parser = chat._build_function_call_parser(request.tools, request)
        with (
            patch.object(
                chat, "_build_reasoning_parser", return_value=reasoning_parser
            ),
            patch.object(chat, "_build_function_call_parser", return_value=tool_parser),
        ):
            response = chat._build_chat_response(request, ret, created=0)

        self.assertIsNone(reasoning_parser.remaining_token_ids)
        self.assertEqual(response.choices[0].message.reasoning_content, "r")
        self.assertEqual(response.choices[0].message.content, "a")
        self.assertIsNone(response.choices[0].message.tool_calls)
        self.assertEqual(response.choices[0].matched_stop, "STOP")

    def test_welm_nonstream_stop_configured_but_eos_finish_keeps_id_parser(self):
        """A configured string stop only forces text fallback when it actually
        matches. Finishing via an int stop-token (EOS) must keep the token-ID
        parser: the trailing stop ID is trimmed and the tool call parses from
        the remaining IDs."""
        chat = self._welm_chat(with_tools=True)
        tokenizer = self.tm.tokenizer
        output_ids = [
            ord("r"),
            tokenizer.CONTROL["</think>"],
            ord("a"),
            tokenizer.CONTROL["<tool_call>"],
            *map(ord, "get_weather"),
            tokenizer.CONTROL["<arg_key>"],
            *map(ord, "city"),
            tokenizer.CONTROL["</arg_key>"],
            tokenizer.CONTROL["<arg_value>"],
            *map(ord, "Paris"),
            tokenizer.CONTROL["</arg_value>"],
            tokenizer.CONTROL["</tool_call>"],
            tokenizer.eos_token_id,
        ]
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            stop="STOP",
        )
        ret = [
            {
                "text": "r</think>a<tool_call>get_weather...",
                "output_ids": output_ids,
                "meta_info": {
                    "id": "chatcmpl-welm-eos-stop",
                    "prompt_tokens": 1,
                    "completion_tokens": len(output_ids),
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "weight_version": "default",
                    "finish_reason": {
                        "type": "stop",
                        "matched": tokenizer.eos_token_id,
                    },
                },
                "index": 0,
            }
        ]

        reasoning_parser = chat._build_reasoning_parser(
            chat.reasoning_parser, False, True, request
        )
        tool_parser = chat._build_function_call_parser(request.tools, request)
        with (
            patch.object(
                chat, "_build_reasoning_parser", return_value=reasoning_parser
            ),
            patch.object(chat, "_build_function_call_parser", return_value=tool_parser),
        ):
            response = chat._build_chat_response(request, ret, created=0)

        self.assertIsNotNone(reasoning_parser.remaining_token_ids)
        choice = response.choices[0]
        self.assertEqual(choice.message.reasoning_content, "r")
        self.assertEqual(choice.message.content, "a")
        self.assertIsNotNone(choice.message.tool_calls)
        self.assertEqual(choice.message.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(choice.message.tool_calls[0].function.arguments),
            {"city": "Paris"},
        )
        self.assertEqual(choice.finish_reason, "tool_calls")

    def test_welm_nonstream_tool_decode_error_preserves_content(self):
        class FailingTokenizer(_FakeWelmTokenizer):
            fail_id = 424247

            def decode(
                self,
                token_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            ):
                if self.fail_id in token_ids:
                    raise RuntimeError("injected decode failure")
                return super().decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        self.tm.server_args.reasoning_parser = "welm-v4"
        self.tm.server_args.tool_call_parser = "welm-v4"
        self.tm.tokenizer = FailingTokenizer()
        chat = OpenAIServingChat(self.tm, self.template_manager)
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        tool_ids = [
            self.tm.tokenizer.fail_id,
            self.tm.tokenizer.CONTROL["<tool_call>"],
            *map(ord, "get_weather"),
            self.tm.tokenizer.CONTROL["<arg_key>"],
            *map(ord, "city"),
            self.tm.tokenizer.CONTROL["</arg_key>"],
            self.tm.tokenizer.CONTROL["<arg_value>"],
            *map(ord, "Paris"),
            self.tm.tokenizer.CONTROL["</arg_value>"],
            self.tm.tokenizer.CONTROL["</tool_call>"],
        ]
        ret = [
            {
                "text": "r</think>bad<tool_call>get_weather...",
                "output_ids": [
                    ord("r"),
                    self.tm.tokenizer.CONTROL["</think>"],
                    *tool_ids,
                ],
                "meta_info": {
                    "id": "chatcmpl-welm-nonstream-error",
                    "prompt_tokens": 1,
                    "completion_tokens": 3,
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "weight_version": "default",
                    "finish_reason": {"type": "stop", "matched": None},
                },
                "index": 0,
            }
        ]

        response = chat._build_chat_response(request, ret, created=0)
        self.assertEqual(
            response.choices[0].message.content,
            "bad<tool_call>get_weather...",
        )
        self.assertIsNone(response.choices[0].message.tool_calls)

    def test_regex_finish_reason_trims_actual_match_not_pattern(self):
        finish_reason = FINISHED_MATCHED_REGEX(r"\d+", "123").to_json()
        self.assertEqual(finish_reason["matched"], r"\d+")
        self.assertEqual(finish_reason["matched_text"], "123")

        legacy_finish_reason = FINISHED_MATCHED_REGEX(r"\d+").to_json()
        self.assertEqual(legacy_finish_reason["matched"], r"\d+")
        self.assertIsNone(legacy_finish_reason["matched_text"])

        manager = DetokenizerManager.__new__(DetokenizerManager)
        self.assertEqual(
            manager.trim_matched_stop("hello123tail", finish_reason, False),
            "hello",
        )

    # ------------- X-Data-Parallel-Rank header tests -------------
    def test_extract_routed_dp_rank_from_header_no_header(self):
        """Test that None is returned when no header is present."""
        self.fastapi_request.headers = {}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=None
        )
        self.assertIsNone(result)

    def test_extract_routed_dp_rank_from_header_with_header(self):
        """Test that header value is extracted correctly."""
        self.fastapi_request.headers = {"x-data-parallel-rank": "2"}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=None
        )
        self.assertEqual(result, 2)

    def test_extract_routed_dp_rank_header_overrides_body(self):
        """Test that header value has higher priority than body."""
        self.fastapi_request.headers = {"x-data-parallel-rank": "3"}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=1
        )
        self.assertEqual(result, 3)  # header wins

    def test_extract_routed_dp_rank_from_header_invalid(self):
        """Test that invalid header value raises HTTPException."""
        from fastapi import HTTPException

        self.fastapi_request.headers = {"x-data-parallel-rank": "abc"}
        with self.assertRaises(HTTPException) as context:
            self.chat.extract_routed_dp_rank_from_header(
                self.fastapi_request, body_routed_dp_rank=None
            )
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("must be an integer", context.exception.detail)

    def test_hunyuan_reasoning_effort_dispatch(self):
        tm = _MockTokenizerManager()
        tm.server_args.reasoning_parser = "hunyuan"
        chat = OpenAIServingChat(tm, _MockTemplateManager())
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
        cases = [
            ("no_think", False),
            ("none", False),
            (None, False),
            ("high", True),
            ("low", True),
        ]
        for effort, expected in cases:
            with self.subTest(effort=effort):
                req.reasoning_effort = effort
                self.assertEqual(chat._get_reasoning_from_request(req), expected)

    # ------------- reasoning config tests -------------
    def test_get_reasoning_from_request_default_true_toggle(self):
        self.tm.server_args.reasoning_parser = "qwen3"
        self.chat.reasoning_parser = "qwen3"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        )

        enabled_by_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        disabled_explicitly = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertTrue(self.chat._get_reasoning_from_request(enabled_by_default))
        self.assertFalse(self.chat._get_reasoning_from_request(disabled_explicitly))

    def test_get_reasoning_from_request_default_false_toggle(self):
        self.tm.server_args.reasoning_parser = "deepseek-v3"
        self.chat.reasoning_parser = "deepseek-v3"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=False
        )

        disabled_by_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        enabled_explicitly = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"thinking": True},
        )

        self.assertFalse(self.chat._get_reasoning_from_request(disabled_by_default))
        self.assertTrue(self.chat._get_reasoning_from_request(enabled_explicitly))

    def test_get_reasoning_from_request_special_cases(self):
        self.tm.server_args.reasoning_parser = "mistral"
        self.chat.reasoning_parser = "mistral"
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )

        self.template_manager.reasoning_config = ReasoningToggleConfig(
            special_case="always"
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req))

        self.template_manager.reasoning_config = ReasoningToggleConfig(
            special_case="mistral"
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req))
        req.reasoning_effort = "medium"
        self.assertTrue(self.chat._get_reasoning_from_request(req))

    # --- fallback path tests (config=None, uses reasoning_default) ---

    def _setup_fallback(self, parser_name):
        """Set up reasoning with config=None to exercise the fallback path."""
        self.tm.server_args.reasoning_parser = parser_name
        self.chat = OpenAIServingChat(self.tm, self.template_manager)
        self.chat.reasoning_parser = parser_name
        self.template_manager.reasoning_config = None

    def test_fallback_always_mode(self):
        self._setup_fallback("deepseek-r1")
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req))

    def test_fallback_mistral_mode(self):
        self._setup_fallback("mistral")
        req_no_effort = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_no_effort))

        req_with_effort = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            reasoning_effort="high",
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_with_effort))

    def test_fallback_enable_thinking_mode_default_on(self):
        self._setup_fallback("qwen3")
        req_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_default))

        req_disabled = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_disabled))

    def test_fallback_explicit_thinking_mode_default_off(self):
        self._setup_fallback("deepseek-v3")
        req_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_default))

        req_enabled = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"thinking": True},
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_enabled))

    def test_fallback_explicit_enable_thinking_mode_default_off(self):
        self._setup_fallback("mimo")
        req_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_default))

        req_enabled = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_enabled))

    def test_fallback_no_detector_returns_false(self):
        self.chat.reasoning_parser = "qwen3"
        self.chat._reasoning_detector = None
        self.template_manager.reasoning_config = None
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req))

    def test_build_chat_response_qwen3_thinking_forces_reasoning(self):
        self.tm.server_args.reasoning_parser = "qwen3-thinking"
        self.chat.reasoning_parser = "qwen3-thinking"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        )

        req = ChatCompletionRequest(
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "Hi?"}],
            separate_reasoning=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        ret_item = {
            "text": "42",
            "meta_info": {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "weight_version": "default",
                "finish_reason": {"type": "stop", "matched": None},
            },
            "index": 0,
        }

        response = self.chat._build_chat_response(req, [ret_item], created=0)
        msg = response.choices[0].message
        self.assertIsNone(msg.content)
        self.assertEqual(msg.reasoning_content, "42")

    # --- poolside_v1 (Laguna-XS.2) regression tests ---

    def test_poolside_v1_enable_thinking_dispatch(self):
        """Laguna chat template defaults `enable_thinking=false`. Parser must
        follow that default — must NOT return True via the generic fallback.
        After the reasoning-config refactor, this is driven by
        `_PoolsideV1Detector.reasoning_default = "explicit_enable_thinking"`."""
        self._setup_fallback("poolside_v1")
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
        cases = [
            (None, False),  # no chat_template_kwargs → non-thinking (default)
            ({}, False),  # empty kwargs → non-thinking
            ({"enable_thinking": False}, False),  # explicit off
            ({"enable_thinking": True}, True),  # explicit on
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                req.chat_template_kwargs = kwargs
                self.assertEqual(self.chat._get_reasoning_from_request(req), expected)

    def test_poolside_v1_does_not_double_prepend_think(self):
        """When `enable_thinking=True` for poolside_v1, the HF chat template
        already emits `<think>` via add_generation_prompt — server must NOT
        append a second `<think>`. After the refactor this is guarded by
        `_PoolsideV1Detector.thinks_internally = True` (inherited from Qwen3Detector).
        """
        self._setup_fallback("poolside_v1")
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "BASE_PROMPT"
            conv_ins.image_data = conv_ins.audio_data = conv_ins.video_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = []
            conv_mock.return_value = conv_ins
            result = self.chat._apply_conversation_template(req, is_multimodal=False)
        self.assertEqual(result.prompt, "BASE_PROMPT")


class TestProcessToolCallsWithRequiredToolChoice(unittest.TestCase):
    """Test _process_tool_calls with tool_choice='required' uses model-specific parser."""

    def setUp(self):
        tm = _MockTokenizerManager()
        tm.server_args.tool_call_parser = "kimi_k2"
        self.chat = OpenAIServingChat(tm, _MockTemplateManager())

    def test_required_with_parser_uses_function_call_parser(self):
        """tool_choice='required' should use FunctionCallParser when tool_call_parser is set."""
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"location":"Tokyo"}'
            call_info.tool_index = 0

            parser_instance = ParserMock.return_value
            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = ("", [call_info])

            finish_reason = {"type": "stop", "matched": None}
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            tool_calls, text, fr = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...<|tool_calls_section_end|>",
                tools=tools,
                finish_reason=finish_reason,
                tool_choice="required",
            )

            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].function.name, "get_weather")
            self.assertEqual(fr["type"], "tool_calls")

    def test_required_without_parser_falls_back_to_json(self):
        """tool_choice='required' without parser should parse as JSON array."""
        self.chat.tool_call_parser = None

        finish_reason = {"type": "stop", "matched": None}
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        tool_calls, text, fr = self.chat._process_tool_calls(
            text='[{"name":"get_weather","parameters":{"location":"Tokyo"}}]',
            tools=tools,
            finish_reason=finish_reason,
            tool_choice="required",
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "get_weather")

    def test_required_without_parser_invalid_json_returns_none(self):
        """tool_choice='required' without parser and invalid JSON returns tool_calls=None."""
        self.chat.tool_call_parser = None

        finish_reason = {"type": "stop", "matched": None}
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        tool_calls, text, fr = self.chat._process_tool_calls(
            text="<|tool_calls_section_begin|>not json",
            tools=tools,
            finish_reason=finish_reason,
            tool_choice="required",
        )

        self.assertIsNone(tool_calls)


class TestNormalizeToolContent(unittest.TestCase):
    """Unit tests for normalize_tool_content()."""

    def test_openai_text_parts_flattened(self):
        result = normalize_tool_content("tool", [{"type": "text", "text": "10525"}])
        self.assertEqual(result, "10525")

    def test_multiple_text_parts_joined(self):
        result = normalize_tool_content(
            "tool",
            [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
        )
        self.assertEqual(result, "hello world")

    def test_non_text_part_list_preserved(self):
        content = [{"name": "func", "output": "result"}]
        result = normalize_tool_content("tool", content)
        self.assertIs(result, content)

    def test_string_content_unchanged(self):
        self.assertEqual(normalize_tool_content("tool", "hello"), "hello")

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(normalize_tool_content("tool", []), "")

    def test_non_tool_role_unchanged(self):
        content = [{"type": "text", "text": "hi"}]
        result = normalize_tool_content("user", content)
        self.assertIs(result, content)

    def test_mixed_str_and_dict_parts(self):
        result = normalize_tool_content(
            "tool", ["plain", {"type": "text", "text": "rich"}]
        )
        self.assertEqual(result, "plain rich")


if __name__ == "__main__":
    unittest.main(verbosity=2)
