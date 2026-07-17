import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.welm_v4_detector import (
    WelmV4StreamingParseError,
    WelmV4ToolDetector,
    filter_id_based_stream_stop,
    trim_matched_stop_for_id_parser,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.welm_v4_detector import WelmV4ReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")


class FakeWelmTokenizer:
    """Tokenizer stub with dedicated ids for WeLM control tokens."""

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
    # Match real ChatML-style tokenizers where only <|im_end|> is special:true.
    SPECIAL_TRUE_IDS = {1009}
    unk_token_id = 0
    eos_token_id = 1009

    def __init__(self):
        self._id2tok = {v: k for k, v in self.CONTROL.items()}

    def convert_tokens_to_ids(self, token):
        return self.CONTROL.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        # Keep ordinary text distinct from added-token ids.
        return [ord(c) for c in text]

    def decode(
        self, ids, skip_special_tokens=False, spaces_between_special_tokens=True
    ):
        out = []
        for i in ids:
            if skip_special_tokens and i in self.SPECIAL_TRUE_IDS:
                continue
            if i in self._id2tok:
                out.append(self._id2tok[i])
            else:
                out.append(chr(i))
        return "".join(out)


def enc(text):
    return [ord(c) for c in text]


C = FakeWelmTokenizer.CONTROL


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather",
                parameters={
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "number"},
                    },
                    "required": ["city"],
                },
            ),
        ),
    ]


class TestWelmV4Reasoning(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()

    def _det(self, **kw):
        return WelmV4ReasoningDetector(tokenizer=self.tok, **kw)

    def test_non_stream_basic(self):
        det = self._det(force_reasoning=True)
        ids = enc("let me think") + [C["</think>"]] + enc("the answer")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "let me think")
        self.assertEqual(res.normal_text, "the answer")
        self.assertEqual(det.remaining_token_ids, enc("the answer"))

    def test_non_stream_preserves_answer_whitespace(self):
        det = self._det(force_reasoning=True)
        ids = enc("reason") + [C["</think>"]] + enc(" answer ")

        res = det.detect_and_parse("", ids)

        self.assertEqual(res.normal_text, " answer ")

    def test_non_stream_strips_leading_think(self):
        det = self._det(force_reasoning=True)
        ids = [C["<think>"]] + enc("hmm") + [C["</think>"]] + enc("done")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "hmm")
        self.assertEqual(res.normal_text, "done")

    def test_non_stream_lookalike_in_content_is_ignored(self):
        det = self._det(force_reasoning=True)
        ids = (
            enc("discussing the </think> tag here")
            + [C["</think>"]]
            + enc("real answer")
        )
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "discussing the </think> tag here")
        self.assertEqual(res.normal_text, "real answer")

    def test_non_stream_no_end_token_force_reasoning(self):
        det = self._det(force_reasoning=True)
        ids = enc("still thinking")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "still thinking")
        self.assertEqual(res.normal_text, "")

    def test_thinking_true_false_and_adaptive_outputs(self):
        """Test generated output splits for thinking on/off/adaptive modes."""
        det = self._det(force_reasoning=False)

        res = det.detect_and_parse("", enc("reason") + [C["</think>"]] + enc("answer"))
        self.assertEqual(res.reasoning_text, "reason")
        self.assertEqual(res.normal_text, "answer")

        det = self._det(force_reasoning=False)
        res = det.detect_and_parse("", [C["</think>"]] + enc("answer"))
        self.assertEqual(res.reasoning_text, "")
        self.assertEqual(res.normal_text, "answer")

        det = self._det(force_reasoning=False)
        res = det.detect_and_parse(
            "", [C["<think>"]] + enc("maybe") + [C["</think>"]] + enc("ok")
        )
        self.assertEqual(res.reasoning_text, "maybe")
        self.assertEqual(res.normal_text, "ok")

        det = self._det(force_reasoning=False)
        res = det.detect_and_parse("", [C["</think>"]] + enc("direct"))
        self.assertEqual(res.reasoning_text, "")
        self.assertEqual(res.normal_text, "direct")

    def test_stream_token_by_token(self):
        det = self._det(force_reasoning=True, stream_reasoning=True)
        seq = enc("reason") + [C["</think>"]] + enc("ans")
        reasoning, normal, remaining = "", "", []
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
            remaining += det.remaining_token_ids
        self.assertEqual(reasoning, "reason")
        self.assertEqual(normal, "ans")
        self.assertEqual(remaining, enc("ans"))

    def test_stream_lookalike_in_content(self):
        det = self._det(force_reasoning=True, stream_reasoning=True)
        seq = enc("a </think> b") + [C["</think>"]] + enc("final")
        reasoning, normal = "", ""
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
        self.assertEqual(reasoning, "a </think> b")
        self.assertEqual(normal, "final")

    def test_non_stream_trailing_eos_not_in_output(self):
        det = self._det(force_reasoning=True)
        ids = enc("reason") + [C["</think>"]] + enc("answer") + [C["<|im_end|>"]]
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "reason")
        self.assertEqual(res.normal_text, "answer")

    def test_stream_trailing_eos_not_in_output(self):
        det = self._det(force_reasoning=True, stream_reasoning=True)
        seq = enc("r") + [C["</think>"]] + enc("a") + [C["<|im_end|>"]]
        reasoning, normal = "", ""
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
        self.assertEqual(reasoning, "r")
        self.assertEqual(normal, "a")

    def test_stream_no_stream_reasoning_emits_at_end(self):
        det = self._det(force_reasoning=True, stream_reasoning=False)
        seq = enc("reason") + [C["</think>"]] + enc("ans")
        reasoning, normal = "", ""
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
        self.assertEqual(reasoning, "reason")
        self.assertEqual(normal, "ans")

    def test_stream_answer_leading_think_token_does_not_reopen_reasoning(self):
        det = self._det(
            force_reasoning=True, stream_reasoning=True, skip_special_tokens=False
        )
        first = [C["<think>"]] + enc("reason") + [C["</think>"]] + enc("answer")
        second = [C["<think>"]] + enc(" text")
        reasoning, normal = "", ""
        for chunk in [first, second]:
            res = det.parse_streaming_increment("", chunk)
            reasoning += res.reasoning_text
            normal += res.normal_text

        self.assertEqual(reasoning, "reason")
        self.assertEqual(normal, "answer<think> text")

    def test_stream_reasoning_decode_error_falls_back_and_continues(self):
        class FailingTokenizer(FakeWelmTokenizer):
            FAIL_ID = 424242

            def decode(
                self,
                ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            ):
                if self.FAIL_ID in ids:
                    raise ValueError("injected decode failure")
                return super().decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        tok = FailingTokenizer()
        det = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)

        first = det.parse_streaming_increment("bad", [tok.FAIL_ID])
        second = det.parse_streaming_increment("", enc(" ok") + [C["</think>"]])
        third = det.parse_streaming_increment("", enc("answer"))

        self.assertEqual(first.reasoning_text, "bad")
        self.assertEqual(second.reasoning_text, " ok")
        self.assertEqual(third.normal_text, "answer")

    def test_stream_answer_decode_error_falls_back_to_normal_text(self):
        class FailingTokenizer(FakeWelmTokenizer):
            FAIL_ID = 424243

            def decode(
                self,
                ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            ):
                if self.FAIL_ID in ids:
                    raise ValueError("injected decode failure")
                return super().decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        tok = FailingTokenizer()
        det = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)

        first = det.parse_streaming_increment("", enc("r") + [C["</think>"]])
        second = det.parse_streaming_increment("bad", [tok.FAIL_ID])

        self.assertEqual(first.reasoning_text, "r")
        self.assertEqual(second.normal_text, "bad")
        self.assertEqual(det.remaining_token_ids, [tok.FAIL_ID])


class TestWelmV4Tool(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.tools = _tools()

    def _det(self):
        return WelmV4ToolDetector(tokenizer=self.tok)

    def _call_ids(self, name, args):
        ids = [C["<tool_call>"]] + enc(name)
        for k, v in args:
            ids += enc("\n") + [C["<arg_key>"]] + enc(k) + [C["</arg_key>"]]
            ids += enc("\n") + [C["<arg_value>"]] + enc(v) + [C["</arg_value>"]]
        ids += enc("\n") + [C["</tool_call>"]]
        return ids

    def _stream_ids(self, det, tools, seq, chunk_size=1):
        normal, calls = "", []
        for i in range(0, len(seq), chunk_size):
            res = det.parse_streaming_increment("", tools, seq[i : i + chunk_size])
            normal += res.normal_text
            calls += res.calls
        return normal, calls

    def _merged_args_by_index(self, calls):
        args_by_index = {}
        for call in calls:
            if call.name is None:
                args_by_index.setdefault(call.tool_index, "")
                args_by_index[call.tool_index] += call.parameters
        return args_by_index

    def test_non_stream_basic(self):
        det = self._det()
        ids = enc("Here you go:") + self._call_ids(
            "get_weather", [("city", "Beijing"), ("days", "3")]
        )
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "Here you go:")
        self.assertEqual(len(res.calls), 1)
        self.assertEqual(res.calls[0].name, "get_weather")
        args = json.loads(res.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(args["days"], 3)

    def test_non_stream_lookalike_in_content_is_ignored(self):
        det = self._det()
        ids = enc("To call a tool you write <tool_call> like this.")
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.calls, [])
        self.assertEqual(
            res.normal_text, "To call a tool you write <tool_call> like this."
        )

    def test_non_stream_multiple_calls(self):
        det = self._det()
        ids = (
            self._call_ids("get_weather", [("city", "A")])
            + enc("\n")
            + self._call_ids("get_weather", [("city", "B")])
        )
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(len(res.calls), 2)
        self.assertEqual(json.loads(res.calls[0].parameters)["city"], "A")
        self.assertEqual(json.loads(res.calls[1].parameters)["city"], "B")

    def test_non_stream_preserves_text_around_tool_calls(self):
        det = self._det()
        ids = (
            enc(" before ")
            + self._call_ids("get_weather", [("city", "A")])
            + enc(" between ")
            + self._call_ids("get_weather", [("city", "B")])
            + enc(" after ")
        )

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.normal_text, " before  between  after ")
        self.assertEqual(len(res.calls), 2)

    def test_non_stream_incomplete_value_falls_back_to_content(self):
        det = self._det()
        ids = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Beijing")
            + [C["</tool_call>"]]
        )

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_non_stream_does_not_parse_tool_without_tool_end(self):
        det = self._det()
        ids = self._call_ids("get_weather", [("city", "Paris")])
        ids.pop()

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_non_stream_incomplete_value_before_next_arg_falls_back_to_content(self):
        det = self._det()
        ids = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Beijing")
            + [C["<arg_key>"]]
            + enc("days")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("3")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_argument_key_whitespace_is_preserved(self):
        det = self._det()
        ids = self._call_ids("get_weather", [(" city ", "Beijing")])

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(json.loads(res.calls[0].parameters), {" city ": "Beijing"})

    def test_non_stream_string_value_is_not_rewritten(self):
        det = self._det()
        ids = self._call_ids("get_weather", [("city", "true")])

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(json.loads(res.calls[0].parameters), {"city": "true"})

    def test_non_stream_unknown_tool_falls_back_to_content(self):
        det = self._det()
        ids = enc("prefix") + self._call_ids("python", [("code", "1")]) + enc("suffix")

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_stream_token_by_token(self):
        det = self._det()
        seq = enc("ok") + self._call_ids("get_weather", [("city", "Paris")])
        normal, calls = "", []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            normal += res.normal_text
            calls += res.calls
        self.assertEqual(normal, "ok")
        names = [c.name for c in calls if c.name]
        self.assertIn("get_weather", names)
        merged = "".join(c.parameters for c in calls if c.parameters)
        self.assertIn("Paris", merged)

    def test_stream_parser_error_falls_back_to_exact_raw_block(self):
        det = self._det()
        original_dispatch = det._dispatch_stream_token

        def dispatch_or_fail(tid, tools, calls, normal_chunks):
            if tid == ord("!"):
                raise RuntimeError("injected parser failure")
            return original_dispatch(tid, tools, calls, normal_chunks)

        det._dispatch_stream_token = dispatch_or_fail
        seq = [C["<tool_call>"]] + enc("get_weather!")

        res = det.parse_streaming_increment("", self.tools, seq)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(seq))
        self.assertNotIn("</tool_call>", res.normal_text)
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_parser_and_fallback_decode_error_do_not_duplicate_text(self):
        class FailingTokenizer(FakeWelmTokenizer):
            FAIL_ID = 424244

            def decode(
                self,
                ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            ):
                if self.FAIL_ID in ids:
                    raise ValueError("injected decode failure")
                return super().decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        tok = FailingTokenizer()
        det = WelmV4ToolDetector(tokenizer=tok)
        original_dispatch = det._dispatch_stream_token

        def dispatch_or_fail(tid, tools, calls, normal_chunks):
            if tid == tok.FAIL_ID:
                raise RuntimeError("injected parser failure")
            return original_dispatch(tid, tools, calls, normal_chunks)

        det._dispatch_stream_token = dispatch_or_fail
        res = det.parse_streaming_increment("A?", self.tools, enc("A") + [tok.FAIL_ID])

        self.assertEqual(res.normal_text, "A?")
        self.assertEqual(res.calls, [])

    def test_stream_parser_error_after_second_call_committed_aborts_chunk(self):
        det = self._det()
        original_dispatch = det._dispatch_stream_token

        def dispatch_or_fail(tid, tools, calls, normal_chunks):
            if tid == ord("!"):
                raise RuntimeError("injected parser failure")
            return original_dispatch(tid, tools, calls, normal_chunks)

        det._dispatch_stream_token = dispatch_or_fail
        good_call = self._call_ids("get_weather", [("city", "Paris")])
        bad_call = [C["<tool_call>"]] + enc("get_weather") + [C["<arg_key>"]] + enc("!")

        with self.assertRaises(WelmV4StreamingParseError):
            det.parse_streaming_increment("", self.tools, good_call + bad_call)
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_parser_error_after_partial_arg_raises(self):
        det = self._det()
        original_dispatch = det._dispatch_stream_token

        def dispatch_or_fail(tid, tools, calls, normal_chunks):
            if tid == ord("!"):
                raise RuntimeError("injected parser failure")
            return original_dispatch(tid, tools, calls, normal_chunks)

        det._dispatch_stream_token = dispatch_or_fail
        first_chunk = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Par")
        )
        first = det.parse_streaming_increment("", self.tools, first_chunk)

        self.assertIn('{"city": "Par', "".join(c.parameters for c in first.calls))
        with self.assertRaises(WelmV4StreamingParseError):
            det.parse_streaming_increment("", self.tools, enc("!"))
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_parser_error_keeps_later_tool_end_visible_as_text(self):
        class SpecialControlTokenizer(FakeWelmTokenizer):
            SPECIAL_TRUE_IDS = set(FakeWelmTokenizer.CONTROL.values())

        tok = SpecialControlTokenizer()
        det = WelmV4ToolDetector(tokenizer=tok)
        original_dispatch = det._dispatch_stream_token

        def dispatch_or_fail(tid, tools, calls, normal_chunks):
            if tid == ord("!"):
                raise RuntimeError("injected parser failure")
            return original_dispatch(tid, tools, calls, normal_chunks)

        det._dispatch_stream_token = dispatch_or_fail
        first_chunk = [C["<tool_call>"]] + enc("get_weather") + enc("!")

        first = det.parse_streaming_increment("", self.tools, first_chunk)
        second = det.parse_streaming_increment("", self.tools, [C["</tool_call>"]])

        self.assertIn("<tool_call>get_weather!", first.normal_text)
        self.assertEqual(second.normal_text, "</tool_call>")

    def test_non_stream_trailing_eos_not_in_output(self):
        det = self._det()
        ids = (
            enc("Here:")
            + self._call_ids("get_weather", [("city", "X")])
            + [C["<|im_end|>"]]
        )
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "Here:")
        self.assertEqual(len(res.calls), 1)

    def test_stream_lookalike_not_triggered(self):
        det = self._det()
        seq = enc("write <tool_call> in prose")
        normal, calls = "", []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            normal += res.normal_text
            calls += res.calls
        self.assertEqual(calls, [])
        self.assertEqual(normal, "write <tool_call> in prose")

    def test_stream_separator_accepts_exact_non_stream_whitespace_rule(self):
        det = self._det()
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + enc(" \t\r\n")
            + [C["<arg_value>"]]
            + enc("Paris")
            + [C["</arg_value>"]]
            + enc("\n\t ")
            + [C["</tool_call>"]]
        )

        non_stream = det.detect_and_parse("", self.tools, seq)
        self.assertEqual(json.loads(non_stream.calls[0].parameters), {"city": "Paris"})

        stream_det = self._det()
        normal, calls = self._stream_ids(stream_det, self.tools, seq, chunk_size=3)
        self.assertEqual(normal, "")
        self.assertEqual(
            json.loads("".join(c.parameters for c in calls if c.name is None)),
            {"city": "Paris"},
        )

    def test_stream_malformed_before_name_falls_back_for_all_chunk_sizes(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_value>"]]
            + enc("junk")
            + [C["</tool_call>"]]
        )

        for chunk_size in range(1, len(seq) + 1):
            with self.subTest(chunk_size=chunk_size):
                det = self._det()
                normal, calls = self._stream_ids(
                    det, self.tools, seq, chunk_size=chunk_size
                )
                self.assertEqual(calls, [])
                self.assertEqual(normal, self.tok.decode(seq))

    def test_stream_malformed_after_name_raises_for_all_chunk_sizes(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + enc("JUNK")
            + [C["<arg_value>"]]
            + enc("Paris")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        non_stream = self._det().detect_and_parse("", self.tools, seq)
        self.assertEqual(non_stream.calls, [])
        self.assertEqual(non_stream.normal_text, self.tok.decode(seq))

        for chunk_size in range(1, len(seq) + 1):
            with self.subTest(chunk_size=chunk_size):
                det = self._det()
                raised = False
                for i in range(0, len(seq), chunk_size):
                    try:
                        det.parse_streaming_increment(
                            "", self.tools, seq[i : i + chunk_size]
                        )
                    except WelmV4StreamingParseError:
                        raised = True
                        break
                self.assertTrue(raised)
                self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_non_whitespace_between_arguments_raises(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Paris")
            + [C["</arg_value>"]]
            + enc("JUNK")
            + [C["<arg_key>"]]
            + enc("days")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("3")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        det = self._det()
        with self.assertRaises(WelmV4StreamingParseError):
            det.parse_streaming_increment("", self.tools, seq)

    def test_actual_tool_start_inside_value_is_malformed(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("A")
            + [C["<tool_call>"]]
            + enc("B")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        non_stream = self._det().detect_and_parse("", self.tools, seq)
        self.assertEqual(non_stream.calls, [])
        self.assertEqual(non_stream.normal_text, self.tok.decode(seq))

        stream_det = self._det()
        with self.assertRaises(WelmV4StreamingParseError):
            stream_det.parse_streaming_increment("", self.tools, seq)

    def test_stream_empty_args(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]
        seq = self._call_ids("get_time", [])
        calls = []
        for tid in seq:
            res = det.parse_streaming_increment("", tools, [tid])
            calls += res.calls
        self.assertEqual([c.name for c in calls if c.name], ["get_time"])
        arg_deltas = [c.parameters for c in calls if c.name is None]
        self.assertEqual(arg_deltas, ["{}"])

    def test_stream_arg_value_lookalike_not_triggered(self):
        det = self._det()
        value = "literal <arg_key>x</arg_key><arg_value>y</arg_value>"
        seq = self._call_ids("get_weather", [("city", value)])
        calls = []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            calls += res.calls
        merged = "".join(c.parameters for c in calls if c.name is None)
        args = json.loads(merged)
        self.assertEqual(args["city"], value)

    def test_stream_multiple_calls_token_id_fsm(self):
        det = self._det()
        seq = (
            self._call_ids("get_weather", [("city", "A")])
            + enc("\n")
            + self._call_ids("get_weather", [("city", "B")])
        )
        calls = []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            calls += res.calls
        self.assertEqual(
            [c.name for c in calls if c.name], ["get_weather", "get_weather"]
        )
        args_by_index = self._merged_args_by_index(calls)
        arg_deltas = [json.loads(args_by_index[i]) for i in sorted(args_by_index)]
        self.assertEqual([args["city"] for args in arg_deltas], ["A", "B"])

    def test_stream_long_string_arguments_incrementally(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="write_file",
                    parameters={
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        }
                    },
                ),
            )
        ]
        content = "\n".join(f"line {i}: print('hello')" for i in range(200))
        seq = self._call_ids(
            "write_file", [("path", "/tmp/a.py"), ("content", content)]
        )

        normal, calls = self._stream_ids(det, tools, seq, chunk_size=17)
        self.assertEqual(normal, "")
        self.assertEqual([c.name for c in calls if c.name], ["write_file"])
        arg_deltas = [c.parameters for c in calls if c.name is None]
        self.assertGreater(len(arg_deltas), 10)
        merged = "".join(arg_deltas)
        self.assertIn("line 10", merged)
        args = json.loads(merged)
        self.assertEqual(args["content"], content)

    def test_stream_string_escapes_are_stable(self):
        det = self._det()
        value = 'quote " backslash \\ newline\nunicode 尾'
        _, calls = self._stream_ids(
            det, self.tools, self._call_ids("get_weather", [("city", value)])
        )
        args = json.loads("".join(c.parameters for c in calls if c.name is None))
        self.assertEqual(args["city"], value)

    def test_stream_multiple_argument_types(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="submit",
                    parameters={
                        "properties": {
                            "title": {"type": "string"},
                            "count": {"type": "number"},
                            "enabled": {"type": "boolean"},
                            "meta": {"type": "object"},
                            "items": {"type": "array"},
                        }
                    },
                ),
            )
        ]
        seq = self._call_ids(
            "submit",
            [
                ("title", "job"),
                ("count", "3"),
                ("enabled", "true"),
                ("meta", '{"a":1}'),
                ("items", '["x","y"]'),
            ],
        )
        _, calls = self._stream_ids(det, tools, seq, chunk_size=9)
        args = json.loads("".join(c.parameters for c in calls if c.name is None))
        self.assertEqual(args["title"], "job")
        self.assertEqual(args["count"], 3)
        self.assertEqual(args["enabled"], True)
        self.assertEqual(args["meta"], {"a": 1})
        self.assertEqual(args["items"], ["x", "y"])

    def test_stream_optional_string_schema_stays_string(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="label",
                    parameters={
                        "properties": {
                            "value": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                        }
                    },
                ),
            )
        ]
        _, calls = self._stream_ids(
            det, tools, self._call_ids("label", [("value", "true")]), chunk_size=2
        )
        args = json.loads("".join(c.parameters for c in calls if c.name is None))
        self.assertEqual(args["value"], "true")

    def test_stream_mtp_chunk_matches_token_by_token(self):
        seq = self._call_ids("get_weather", [("city", "A"), ("days", "5")])

        det_token = self._det()
        _, calls_token = self._stream_ids(det_token, self.tools, seq, chunk_size=1)

        det_chunk = self._det()
        _, calls_chunk = self._stream_ids(det_chunk, self.tools, seq, chunk_size=11)

        self.assertEqual(
            [c.name for c in calls_token if c.name],
            [c.name for c in calls_chunk if c.name],
        )
        self.assertEqual(
            json.loads("".join(c.parameters for c in calls_token if c.name is None)),
            json.loads("".join(c.parameters for c in calls_chunk if c.name is None)),
        )

    def test_stream_result_is_chunk_size_invariant(self):
        seq = self._call_ids(
            "get_weather",
            [("city", "Paris"), ("days", "7")],
        )

        expected_names = None
        expected_args = None
        for chunk_size in range(1, len(seq) + 1):
            det = self._det()
            normal, calls = self._stream_ids(
                det, self.tools, seq, chunk_size=chunk_size
            )
            names = [c.name for c in calls if c.name]
            args = json.loads("".join(c.parameters for c in calls if c.name is None))
            self.assertEqual(normal, "")
            if expected_names is None:
                expected_names = names
                expected_args = args
            self.assertEqual(names, expected_names)
            self.assertEqual(args, expected_args)

    def test_stream_unknown_tool_falls_back_to_content(self):
        det = self._det()
        seq = (
            enc("prefix")
            + self._call_ids("python", [("code", "print(1)")])
            + enc("suffix")
        )

        normal, calls = self._stream_ids(det, self.tools, seq)

        self.assertEqual(calls, [])
        self.assertEqual(normal, self.tok.decode(seq))

    def test_stream_unknown_tool_fallback_preserves_special_token_text(self):
        class SpecialControlTokenizer(FakeWelmTokenizer):
            SPECIAL_TRUE_IDS = set(FakeWelmTokenizer.CONTROL.values())

        tok = SpecialControlTokenizer()
        det = WelmV4ToolDetector(tokenizer=tok)
        seq = enc("prefix") + self._call_ids("python", [("code", "print(1)")])

        normal, calls = self._stream_ids(det, self.tools, seq)

        self.assertEqual(calls, [])
        self.assertEqual(normal, tok.decode(seq, skip_special_tokens=False))
        self.assertIn("<tool_call>", normal)
        self.assertIn("</tool_call>", normal)

    def test_stream_missing_value_end_raises_after_name_was_sent(self):
        det = self._det()
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Beijing")
            + [C["</tool_call>"]]
        )

        with self.assertRaises(WelmV4StreamingParseError):
            self._stream_ids(det, self.tools, seq)
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_without_tool_end_does_not_finalize_arguments(self):
        det = self._det()
        seq = self._call_ids("get_weather", [("city", "Paris")])
        seq.pop()

        _, calls = self._stream_ids(det, self.tools, seq)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(merged, '{"city": "Paris"')
        self.assertEqual(det.prev_tool_call_arr[0]["arguments"], {})

    def test_stream_truncated_value_is_not_completed_or_rewritten(self):
        det = self._det()
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Par")
        )

        _, calls = self._stream_ids(det, self.tools, seq)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(merged, '{"city": "Par')
        self.assertEqual(det.prev_tool_call_arr[0]["arguments"], {})

    def test_stream_duplicate_key_uses_last_value(self):
        det = self._det()
        seq = self._call_ids("get_weather", [("city", "A"), ("city", "B")])

        _, calls = self._stream_ids(det, self.tools, seq, chunk_size=4)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(json.loads(merged), {"city": "B"})

    def test_stream_schema_extra_key_is_preserved(self):
        det = self._det()
        seq = self._call_ids("get_weather", [("city", "A"), ("unit", "celsius")])

        _, calls = self._stream_ids(det, self.tools, seq, chunk_size=5)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(json.loads(merged), {"city": "A", "unit": "celsius"})

    def test_stream_raw_object_no_corrupt_suffix(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="submit",
                    parameters={"properties": {"meta": {"type": "object"}}},
                ),
            )
        ]
        _, calls = self._stream_ids(
            det, tools, self._call_ids("submit", [("meta", '{"a":1}')]), chunk_size=3
        )
        merged = "".join(c.parameters for c in calls if c.name is None)
        self.assertEqual(json.loads(merged), {"meta": {"a": 1}})
        self.assertNotIn("}}}", merged)


class TestTrimMatchedStopForIdParser(CustomTestCase):
    def test_int_stop_trims_last_id(self):
        out = trim_matched_stop_for_id_parser([1, 2, 3], {"matched": 3}, False)
        self.assertEqual(out, [1, 2])

    def test_int_stop_no_stop_trim_keeps(self):
        out = trim_matched_stop_for_id_parser([1, 2, 3], {"matched": 3}, True)
        self.assertEqual(out, [1, 2, 3])

    def test_str_stop_truncates_text(self):
        out = trim_matched_stop_for_id_parser(
            "hello<stop>x", {"matched": "<stop>"}, False
        )
        self.assertEqual(out, "hello")

    def test_str_stop_no_stop_trim_keeps(self):
        out = trim_matched_stop_for_id_parser(
            "hello<stop>", {"matched": "<stop>"}, True
        )
        self.assertEqual(out, "hello<stop>")

    def test_str_stop_not_present_noop(self):
        out = trim_matched_stop_for_id_parser("hello", {"matched": "<stop>"}, False)
        self.assertEqual(out, "hello")

    def test_regex_stop_uses_actual_matched_text(self):
        out = trim_matched_stop_for_id_parser(
            "hello123tail",
            {"matched": r"\d+", "matched_text": "123"},
            False,
        )
        self.assertEqual(out, "hello")

    def test_no_finished_reason_noop(self):
        self.assertEqual(trim_matched_stop_for_id_parser([1, 2], None, False), [1, 2])
        self.assertEqual(trim_matched_stop_for_id_parser("ab", {}, False), "ab")

    def test_no_matched_noop(self):
        out = trim_matched_stop_for_id_parser([1, 2], {"matched": None}, False)
        self.assertEqual(out, [1, 2])

    def test_type_mismatch_noop(self):
        self.assertEqual(
            trim_matched_stop_for_id_parser("abc", {"matched": 3}, False), "abc"
        )
        self.assertEqual(
            trim_matched_stop_for_id_parser([1, 2], {"matched": "x"}, False), [1, 2]
        )

    def test_empty_list_int_stop_noop(self):
        self.assertEqual(trim_matched_stop_for_id_parser([], {"matched": 3}, False), [])


class TestWelmV4ToolDecodeFlags(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.tools = _tools()

    def test_skip_true_drops_special_in_content(self):
        det = WelmV4ToolDetector(tokenizer=self.tok, skip_special_tokens=True)
        ids = enc("ab") + [C["<|im_end|>"]] + enc("cd")
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "abcd")

    def test_skip_false_keeps_special_in_content(self):
        det = WelmV4ToolDetector(tokenizer=self.tok, skip_special_tokens=False)
        ids = enc("ab") + [C["<|im_end|>"]] + enc("cd")
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "ab<|im_end|>cd")

    def test_stream_skip_false_keeps_special(self):
        det = WelmV4ToolDetector(tokenizer=self.tok, skip_special_tokens=False)
        seq = enc("ab") + [C["<|im_end|>"]] + enc("cd")
        normal = ""
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            normal += res.normal_text
        self.assertEqual(normal, "ab<|im_end|>cd")

    def test_detector_decode_flags_default_and_settable(self):
        parser = FunctionCallParser(self.tools, "welm-v4")
        parser.configure_tokenizer(self.tok)
        self.assertTrue(parser.detector.skip_special_tokens)
        self.assertTrue(parser.detector.spaces_between_special_tokens)
        parser.detector.skip_special_tokens = False
        parser.detector.spaces_between_special_tokens = False
        self.assertFalse(parser.detector.skip_special_tokens)
        self.assertFalse(parser.detector.spaces_between_special_tokens)


class TestWelmV4StreamStrStopHoldback(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.tools = _tools()

    def _scheduler_holds(self, sp, output_ids):
        if not sp.stop_strs:
            return False
        max_len_tail_str = max(sp.stop_str_max_len + 1, sp.stop_regex_max_len + 1)
        tail_len = min((max_len_tail_str + 1), len(output_ids))
        tail_str = self.tok.decode(output_ids[-tail_len:])
        if not tail_str:
            return False
        for stop_str in sp.stop_strs:
            if not stop_str:
                continue
            if stop_str in tail_str:
                return True
            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True
        return False

    def _simulate_stream(self, answer, stop, no_stop_trim):
        from sglang.srt.sampling.sampling_params import SamplingParams

        output_ids = enc(answer + stop)
        n = len(output_ids)

        sp = SamplingParams(stop=[stop])
        sp.normalize(tokenizer=None)

        det = WelmV4ToolDetector(tokenizer=self.tok)  # no tool call -> all content
        streamed = ""
        held_partial = False  # observed at least one withheld partial-stop chunk
        sent_offset = 0
        for step in range(1, n + 1):
            cur = output_ids[:step]
            finished = step == n
            if not finished and self._scheduler_holds(sp, cur):
                # Scheduler withholds this chunk: nothing is streamed, the new
                # tokens stay buffered until a later chunk is released.
                held_partial = True
                continue

            delta_ids = output_ids[sent_offset:step]
            sent_offset = step
            res = det.parse_streaming_increment("", self.tools, delta_ids)
            finish_reason = {"matched": stop} if finished else None
            delta = trim_matched_stop_for_id_parser(
                res.normal_text, finish_reason, no_stop_trim
            )
            streamed += delta
        return streamed, held_partial

    def test_cross_chunk_str_stop_trimmed_no_leak(self):
        answer = "The answer is 42"
        streamed, held_partial = self._simulate_stream(answer, "STOP", False)
        self.assertTrue(held_partial)
        self.assertEqual(streamed, answer)
        self.assertNotIn("STOP", streamed)

    def test_cross_chunk_str_stop_kept_when_no_stop_trim(self):
        answer = "The answer is 42"
        streamed, held_partial = self._simulate_stream(answer, "STOP", True)
        self.assertTrue(held_partial)
        self.assertEqual(streamed, answer + "STOP")


class TestWelmV4ReasoningDecodeFlags(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()

    def test_skip_false_keeps_special_in_answer(self):
        det = WelmV4ReasoningDetector(
            tokenizer=self.tok, force_reasoning=True, skip_special_tokens=False
        )
        ids = enc("re") + [C["</think>"]] + enc("an") + [C["<|im_end|>"]] + enc("d")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "re")
        self.assertEqual(res.normal_text, "an<|im_end|>d")

    def test_skip_true_drops_special_in_answer(self):
        det = WelmV4ReasoningDetector(
            tokenizer=self.tok, force_reasoning=True, skip_special_tokens=True
        )
        ids = enc("re") + [C["</think>"]] + enc("an") + [C["<|im_end|>"]] + enc("d")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "re")
        self.assertEqual(res.normal_text, "and")


class TestWelmV4ReasoningRegionStrStop(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()

    def _reason(self, no_stop_trim):
        det = WelmV4ReasoningDetector(tokenizer=self.tok, force_reasoning=True)
        ids = enc("thinking STOP")
        res = det.detect_and_parse("", ids)
        return trim_matched_stop_for_id_parser(
            res.reasoning_text, {"matched": "STOP"}, no_stop_trim
        )

    def test_reasoning_region_str_stop_trimmed(self):
        self.assertEqual(self._reason(False), "thinking ")

    def test_reasoning_region_str_stop_kept_when_no_stop_trim(self):
        self.assertEqual(self._reason(True), "thinking STOP")


class TestWelmV4Wrappers(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.tools = _tools()

    def test_reasoning_parser_wrapper(self):
        parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=True,
        )
        parser.configure_tokenizer(self.tok)
        ids = enc("think") + [C["</think>"]] + enc("answer")
        reasoning, normal = parser.parse_non_stream("", token_ids=ids)
        self.assertEqual(reasoning, "think")
        self.assertEqual(normal, "answer")
        self.assertEqual(parser.remaining_token_ids, enc("answer"))

    def test_reasoning_parser_wrapper_forwards_decode_flags(self):
        parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=True,
        )
        parser.configure_tokenizer(self.tok, skip_special_tokens=False)
        ids = enc("think") + [C["</think>"]] + enc("answer") + [C["<|im_end|>"]]
        reasoning, normal = parser.parse_non_stream("", token_ids=ids)
        self.assertEqual(reasoning, "think")
        self.assertEqual(normal, "answer<|im_end|>")

    def test_reasoning_parser_default_matches_deepseek_r1_contract(self):
        parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=False,
        )
        parser.configure_tokenizer(self.tok)
        self.assertEqual(parser.detector.reasoning_default, "always")
        self.assertTrue(parser.detector.thinks_internally)
        self.assertEqual(
            parser.detector.think_excluded_tokens,
            ["<tool_call>", "</tool_call>", "<|im_end|>", "<|endoftext|>"],
        )
        reasoning, normal = parser.parse_non_stream("", token_ids=enc("reasoning"))
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(normal, "")

    def test_function_call_parser_wrapper(self):
        parser = FunctionCallParser(self.tools, "welm-v4")
        parser.configure_tokenizer(self.tok)
        ids = [C["<tool_call>"]] + enc("get_weather")
        ids += enc("\n") + [C["<arg_key>"]] + enc("city") + [C["</arg_key>"]]
        ids += enc("\n") + [C["<arg_value>"]] + enc("Tokyo") + [C["</arg_value>"]]
        ids += enc("\n") + [C["</tool_call>"]]
        self.assertTrue(parser.has_tool_call("", token_ids=ids))
        normal, calls = parser.parse_non_stream("", token_ids=ids)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters)["city"], "Tokyo")

    def test_registered_in_maps(self):
        self.assertIn("welm-v4", ReasoningParser.DetectorMap)
        self.assertIn("welm-v4", FunctionCallParser.ToolCallParserEnum)

    def test_token_id_capability_api(self):
        self.assertTrue(ReasoningParser.accepts_token_ids_for("welm-v4"))
        self.assertFalse(ReasoningParser.accepts_token_ids_for("deepseek-r1"))
        self.assertFalse(ReasoningParser.accepts_token_ids_for(None))

        reasoning_parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=True,
        )
        self.assertTrue(reasoning_parser.accepts_token_ids)

        self.assertTrue(FunctionCallParser.accepts_token_ids_for("welm-v4"))
        self.assertFalse(FunctionCallParser.accepts_token_ids_for("glm"))
        self.assertFalse(FunctionCallParser.accepts_token_ids_for(None))

        tool_parser = FunctionCallParser(self.tools, "welm-v4")
        self.assertTrue(tool_parser.accepts_token_ids)


class TestWelmV4IdBasedStreamStopFilter(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.model_config = SimpleNamespace(hf_eos_token_id={C["<|im_end|>"]})

    def _request(self, **kwargs):
        defaults = dict(
            no_stop_trim=False,
            ignore_eos=False,
            stop_token_ids=None,
            skip_special_tokens=False,
            chat_template_kwargs={},
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    @staticmethod
    def _logprobs(tokens):
        return {
            "content": [
                {"token": token, "bytes": [], "logprob": 0.0, "top_logprobs": []}
                for token in tokens
            ]
        }

    def _filter(self, req, delta, delta_ids, tokens, finish_reason=None):
        return filter_id_based_stream_stop(
            req,
            self.tok,
            self.model_config,
            delta,
            delta_ids,
            self._logprobs(tokens) if tokens is not None else None,
            finish_reason,
            req.skip_special_tokens,
            req.chat_template_kwargs.get("spaces_between_special_tokens", True),
        )

    def test_filters_token_stops_before_id_parser_but_preserves_raw_logprobs(self):
        cases = [
            (
                self._request(),
                "A<|im_end|>",
                enc("A") + [C["<|im_end|>"]],
                ["A", "<|im_end|>"],
                None,
                "A",
                enc("A"),
                ["A", "<|im_end|>"],
            ),
            (
                self._request(),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "",
                [],
                ["<|im_end|>"],
            ),
            (
                self._request(stop_token_ids=[ord("|")]),
                "A|B",
                enc("A|B"),
                ["A", "|", "B"],
                None,
                "A",
                enc("A"),
                ["A", "|", "B"],
            ),
            (
                self._request(),
                "A<|im_end|>B",
                enc("A") + [C["<|im_end|>"]] + enc("B"),
                ["A", "<|im_end|>", "B"],
                None,
                "A",
                enc("A"),
                ["A", "<|im_end|>", "B"],
            ),
            (
                self._request(no_stop_trim=True),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
            ),
            (
                self._request(ignore_eos=True, stop_token_ids=[C["<|im_end|>"]]),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
            ),
            (
                self._request(ignore_eos=True),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                {"type": "stop", "matched": C["<|im_end|>"]},
                "",
                [],
                ["<|im_end|>"],
            ),
            (
                self._request(skip_special_tokens=True),
                "",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "",
                [],
                ["<|im_end|>"],
            ),
        ]

        for req, delta, ids, tokens, finish, exp_delta, exp_ids, exp_tokens in cases:
            with self.subTest(
                no_stop_trim=req.no_stop_trim,
                ignore_eos=req.ignore_eos,
                finish_reason=finish,
            ):
                delta, delta_ids, logprobs = self._filter(
                    req, delta, ids, tokens, finish
                )
                self.assertEqual(delta, exp_delta)
                self.assertEqual(delta_ids, exp_ids)
                if exp_tokens is None:
                    self.assertIsNone(logprobs)
                else:
                    self.assertEqual(
                        [item["token"] for item in logprobs["content"]], exp_tokens
                    )

    def test_additional_stop_token_is_hidden_from_parser_but_kept_in_logprobs(self):
        additional_stop_id = ord("#")
        self.tok.additional_stop_token_ids = [additional_stop_id]

        delta, delta_ids, logprobs = self._filter(
            self._request(),
            "A#",
            enc("A#"),
            ["A", "#"],
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "#"],
        )

    def test_hf_eos_token_is_hidden_from_parser_but_kept_in_logprobs(self):
        hf_eos_id = ord("~")
        self.model_config.hf_eos_token_id = {hf_eos_id}

        delta, delta_ids, logprobs = self._filter(
            self._request(),
            "A~",
            enc("A~"),
            ["A", "~"],
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "~"],
        )

    def test_non_stop_special_token_keeps_ids_and_logprobs(self):
        special_id = 1010
        self.tok.SPECIAL_TRUE_IDS = {*self.tok.SPECIAL_TRUE_IDS, special_id}
        self.tok._id2tok[special_id] = "<special>"
        req = self._request(skip_special_tokens=True)
        delta = self.tok.decode([special_id], skip_special_tokens=True)

        delta, delta_ids, logprobs = self._filter(
            req,
            delta,
            [special_id],
            ["<special>"],
        )

        self.assertEqual(delta, "")
        self.assertEqual(delta_ids, [special_id])
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["<special>"],
        )

    def test_string_stop_aligns_delta_ids_to_trimmed_delta(self):
        stop = "|STOP"
        self.assertGreater(len(enc(stop)), 1)
        delta, delta_ids, logprobs = self._filter(
            self._request(),
            "A",
            enc("A" + stop),
            ["A", "|", "S", "T", "O", "P"],
            {"type": "stop", "matched": stop},
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "|", "S", "T", "O", "P"],
        )

    def test_string_stop_with_multi_token_prefix_is_not_treated_as_token_stop(self):
        stop = "XYZ"
        self.assertGreater(len(enc(stop)), 1)
        delta, delta_ids, logprobs = self._filter(
            self._request(stop_token_ids=[ord("X")]),
            "A",
            enc("A" + stop),
            ["A", "X", "Y", "Z"],
            {"type": "stop", "matched": stop},
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "X", "Y", "Z"],
        )

    def test_string_stop_drops_uncertain_ids_but_preserves_raw_logprobs(self):
        delta, delta_ids, logprobs = self._filter(
            self._request(),
            "unmatched",
            enc("A|STOP"),
            ["A", "|", "S", "T", "O", "P"],
            {"type": "stop", "matched": "|STOP"},
        )

        self.assertEqual(delta, "unmatched")
        self.assertIsNone(delta_ids)
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "|", "S", "T", "O", "P"],
        )

    def test_string_stop_uses_longest_decoding_prefix(self):
        delta, delta_ids, logprobs = self._filter(
            self._request(skip_special_tokens=True),
            "A",
            enc("A") + [C["<|im_end|>"]],
            ["A", "<|im_end|>"],
            {"type": "stop", "matched": "STOP"},
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A") + [C["<|im_end|>"]])
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "<|im_end|>"],
        )

    def test_reasoning_parser_dispatches_stream_stop_filter(self):
        delta, delta_ids, logprobs = ReasoningParser.filter_id_based_stream_stop(
            "welm-v4",
            self._request(),
            self.tok,
            self.model_config,
            "A<|im_end|>",
            enc("A") + [C["<|im_end|>"]],
            self._logprobs(["A", "<|im_end|>"]),
            None,
            False,
            True,
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "<|im_end|>"],
        )


class MissingControlTokenizer(FakeWelmTokenizer):
    def convert_tokens_to_ids(self, token):
        if token in {
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        }:
            return self.unk_token_id
        return super().convert_tokens_to_ids(token)


class TestWelmV4Fallback(CustomTestCase):
    def test_reasoning_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = ReasoningParser("welm-v4", stream_reasoning=False)
        parser.configure_tokenizer(tok)
        reasoning, normal = parser.parse_non_stream(
            "<think>why</think>answer", token_ids=enc("whyanswer")
        )
        self.assertEqual(reasoning, "why")
        self.assertEqual(normal, "answer")
        self.assertEqual(parser.remaining_token_ids, [])

    def test_reasoning_stream_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = ReasoningParser("welm-v4", stream_reasoning=True)
        parser.configure_tokenizer(tok)
        reasoning, normal = "", ""
        for chunk in ["<think>", "why", "</think>", "answer"]:
            reasoning_delta, normal_delta = parser.parse_stream_chunk(
                chunk, token_ids=enc("not-used")
            )
            reasoning += reasoning_delta
            normal += normal_delta
        self.assertEqual(reasoning, "why")
        self.assertEqual(normal, "answer")
        self.assertEqual(parser.remaining_token_ids, [])

    def test_tool_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = FunctionCallParser(_tools(), "welm-v4")
        parser.configure_tokenizer(tok)
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n"
            "<arg_value>Tokyo</arg_value>\n"
            "</tool_call>"
        )
        normal, calls = parser.parse_non_stream(text, token_ids=enc("not-used"))
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters)["city"], "Tokyo")

    def test_tool_stream_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = FunctionCallParser(_tools(), "welm-v4")
        parser.configure_tokenizer(tok)
        normal, calls = "", []
        for chunk in [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n",
            "<arg_value>Tokyo</arg_value>\n",
            "</tool_call>",
        ]:
            normal_delta, call_delta = parser.parse_stream_chunk(
                chunk, token_ids=enc("not-used")
            )
            normal += normal_delta
            calls += call_delta
        self.assertEqual(normal, "")
        self.assertTrue(any(call.name == "get_weather" for call in calls))
        self.assertIn("Tokyo", "".join(call.parameters for call in calls))


if __name__ == "__main__":
    unittest.main()
