import gzip
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils.request_trace import (
    RequestTraceState,
    add_generation_output_ids,
    add_generation_prompt_ids,
    configure_request_trace_recording,
    write_request_trace,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="stage-a-test-cpu", nightly=True)


class TestRequestTrace(CustomTestCase):
    def tearDown(self):
        configure_request_trace_recording(
            record_dir=None,
            max_bytes=1024,
            backup_count=1,
            model_path=None,
            tokenizer_path=None,
        )

    def test_generation_token_ids_support_delta_and_cumulative(self):
        trace = _make_trace()

        add_generation_prompt_ids(
            trace=trace,
            generation_rid="rid-1",
            prompt_token_ids=[1, 2],
        )
        add_generation_output_ids(
            trace=trace,
            generation_rid="rid-1",
            output_ids=[10, 11],
            meta_info={"finish_reason": None},
            is_delta=False,
            finished=False,
        )
        add_generation_output_ids(
            trace=trace,
            generation_rid="rid-1",
            output_ids=[10, 11, 12],
            meta_info={"finish_reason": None},
            is_delta=False,
            finished=False,
        )
        add_generation_output_ids(
            trace=trace,
            generation_rid="rid-1",
            output_ids=[13],
            meta_info={"finish_reason": {"type": "stop"}},
            is_delta=True,
            finished=True,
        )

        generation = trace.generations["rid-1"]
        self.assertEqual(generation.prompt_token_ids, [1, 2])
        self.assertEqual(generation.output_token_ids, [10, 11, 12, 13])
        self.assertTrue(generation.finished)

    def test_write_jsonl_gzip_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_request_trace_recording(
                record_dir=tmpdir,
                max_bytes=1 << 20,
                backup_count=1,
                model_path="model",
                tokenizer_path="tokenizer",
            )
            trace = _make_trace()
            trace.http_response = {"text": "hello"}
            add_generation_prompt_ids(
                trace=trace,
                generation_rid="rid-1",
                prompt_token_ids=[1],
            )
            add_generation_output_ids(
                trace=trace,
                generation_rid="rid-1",
                output_ids=[2],
                meta_info={"finish_reason": {"type": "stop"}},
                is_delta=True,
                finished=True,
            )
            write_request_trace(trace)

            lines = _read_record_lines(tmpdir)
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["schema_version"], 1)
            self.assertEqual(lines[0]["request_id"], "req-1")
            self.assertEqual(lines[0]["http_response"], {"text": "hello"})
            self.assertEqual(lines[0]["generations"][0]["output_token_ids"], [2])
            self.assertEqual(lines[0]["model_path"], "model")
            self.assertEqual(lines[0]["tokenizer_path"], "tokenizer")
            self.assertTrue(_record_paths(tmpdir)[0].name.endswith(".jsonl.gz"))

    def test_rotate_jsonl_gzip_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_request_trace_recording(
                record_dir=tmpdir,
                max_bytes=1,
                backup_count=2,
                model_path="model",
                tokenizer_path="tokenizer",
            )

            for index in range(5):
                trace = _make_trace(request_id=f"req-{index}")
                trace.http_response = {"text": f"hello-{index}"}
                add_generation_prompt_ids(
                    trace=trace,
                    generation_rid=f"rid-{index}",
                    prompt_token_ids=[index],
                )
                add_generation_output_ids(
                    trace=trace,
                    generation_rid=f"rid-{index}",
                    output_ids=[index + 10],
                    meta_info={"finish_reason": {"type": "stop"}},
                    is_delta=True,
                    finished=True,
                )
                write_request_trace(trace)

            paths = _record_paths(tmpdir)
            self.assertEqual(len(paths), 3)
            self.assertTrue(any(path.name.endswith(".jsonl.gz.1") for path in paths))
            lines = _read_record_lines(tmpdir)
            request_ids = {line["request_id"] for line in lines}
            self.assertEqual(request_ids, {"req-2", "req-3", "req-4"})


def _make_trace(request_id: str = "req-1"):
    return RequestTraceState(
        request_id=request_id,
        endpoint="/generate",
        stream=False,
        http_request={
            "method": "POST",
            "path": "/generate",
            "headers": {},
            "query": {},
            "body": {"text": "hi"},
        },
        created_at=1.0,
        finished_at=2.0,
    )


def _read_record_lines(record_dir: str):
    lines = []
    for path in _record_paths(record_dir):
        with gzip.open(path, "rt", encoding="utf-8") as file:
            file_lines = file.read().splitlines()
        for line in file_lines:
            if line:
                lines.append(json.loads(line))
    return lines


def _record_paths(record_dir: str):
    return sorted(Path(record_dir).glob("request_trace_*.jsonl.gz*"))


if __name__ == "__main__":
    unittest.main()
