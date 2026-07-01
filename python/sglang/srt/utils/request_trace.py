from __future__ import annotations

import atexit
import dataclasses
import gzip
import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import numpy as np
import orjson
import torch
import torch.distributed as dist
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key"}
NO_LOGS_BODY_ENDPOINTS = {"/generate"}
DEFAULT_ASYNC_WORKER_NUM = 8
DEFAULT_ASYNC_WRITER_QUEUE_MAXSIZE_PER_WORKER = 2
DEFAULT_PROCESS_MAX_INFLIGHT_PER_WORKER = 4
DEFAULT_PROCESS_START_METHOD = "spawn"
DEFAULT_GZIP_COMPRESSLEVEL = 1
JSONL_TERMINATOR = b"\n"


@dataclasses.dataclass
class _QueuedTrace:
    sequence: int
    trace: "RequestTraceState"


@dataclasses.dataclass
class _SerializedTrace:
    sequence: int
    compressed: Optional[bytes]


@dataclasses.dataclass(frozen=True)
class _WriterConfig:
    filename: str
    max_bytes: int
    backup_count: int
    compresslevel: int


@dataclasses.dataclass(frozen=True)
class _JsonArrayField:
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class _DeferredJsonBody:
    body: bytes


@dataclasses.dataclass(frozen=True)
class _DeferredJsonFragment:
    body: bytes
    start: int
    end: int


@dataclasses.dataclass
class GenerationTrace:
    rid: str
    prompt_token_ids: Optional[List[int]] = None
    output_token_ids: List[int] = dataclasses.field(default_factory=list)
    meta_info: Dict[str, Any] = dataclasses.field(default_factory=dict)
    finished: bool = False


@dataclasses.dataclass
class RequestTraceState:
    request_id: str
    endpoint: str
    stream: bool
    http_request: Dict[str, Any]
    created_at: float
    http_response: Any = None
    chunks: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    generations: Dict[str, GenerationTrace] = dataclasses.field(default_factory=dict)
    status: str = "ok"
    error: Optional[Dict[str, Any]] = None
    finished_at: Optional[float] = None
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    _written: bool = False

    def get_generation(self, rid: str) -> GenerationTrace:
        generation = self.generations.get(rid)
        if generation is None:
            generation = GenerationTrace(rid=rid)
            self.generations[rid] = generation
        return generation


def _compress_jsonl_payload(payload: bytes, compresslevel: int) -> bytes:
    if not payload.endswith(JSONL_TERMINATOR):
        payload += JSONL_TERMINATOR
    return gzip.compress(payload, compresslevel=compresslevel)


def _serialize_trace_for_worker(
    item: _QueuedTrace, compresslevel: int
) -> _SerializedTrace:
    try:
        payload = orjson.dumps(
            _to_record(item.trace), option=orjson.OPT_APPEND_NEWLINE
        )
        compressed = _compress_jsonl_payload(payload, compresslevel)
        return _SerializedTrace(item.sequence, compressed)
    except Exception:
        logger.exception("Failed to write request trace")
        return _SerializedTrace(item.sequence, None)


_process_writer_queue: Optional[Any] = None


def _ignore_sigint_in_child_process() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _init_process_writer_worker(writer_queue: Any) -> None:
    global _process_writer_queue
    _ignore_sigint_in_child_process()
    _process_writer_queue = writer_queue


def _serialize_trace_to_writer_for_worker(
    item: _QueuedTrace, compresslevel: int
) -> int:
    if _process_writer_queue is None:
        raise RuntimeError("request trace process writer queue is not configured")
    _process_writer_queue.put(_serialize_trace_for_worker(item, compresslevel))
    return item.sequence


current_request_trace: ContextVar[Optional[RequestTraceState]] = ContextVar(
    "current_request_trace", default=None
)


class GzipRotatingJsonlWriter:
    def __init__(
        self,
        filename: str,
        *,
        max_bytes: int,
        backup_count: int,
    ):
        self.baseFilename = os.path.abspath(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._file = None

    def write_compressed(self, compressed: bytes) -> None:
        if self._should_rollover(len(compressed)):
            self._do_rollover()
        self._open_file()
        self._file.write(compressed)
        self._file.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _open_file(self) -> None:
        if self._file is None:
            self._file = open(self.baseFilename, "ab")

    def _should_rollover(self, compressed_len: int) -> bool:
        if self.max_bytes <= 0 or self.backup_count <= 0:
            return False
        current_size = self._current_size()
        if current_size == 0:
            return False
        return current_size + compressed_len >= self.max_bytes

    def _current_size(self) -> int:
        if self._file is not None:
            return self._file.tell()
        if os.path.exists(self.baseFilename):
            return os.path.getsize(self.baseFilename)
        return 0

    def _do_rollover(self) -> None:
        self.close()
        if not os.path.exists(self.baseFilename):
            return
        for index in range(self.backup_count - 1, 0, -1):
            src = f"{self.baseFilename}.{index}"
            dst = f"{self.baseFilename}.{index + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)

        dst = f"{self.baseFilename}.1"
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(self.baseFilename, dst)


def _write_ready_serialized_records(
    writer: GzipRotatingJsonlWriter,
    pending: Dict[int, Optional[bytes]],
    next_sequence: int,
) -> int:
    while next_sequence in pending:
        compressed = pending.pop(next_sequence)
        if compressed is not None:
            writer.write_compressed(compressed)
        next_sequence += 1
    return next_sequence


def _writer_process_loop(
    filename: str,
    max_bytes: int,
    backup_count: int,
    writer_queue: Any,
) -> None:
    _ignore_sigint_in_child_process()
    writer = GzipRotatingJsonlWriter(
        filename,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    next_sequence = 0
    pending: Dict[int, Optional[bytes]] = {}
    try:
        while True:
            item = writer_queue.get()
            try:
                if item is None:
                    _write_ready_serialized_records(writer, pending, next_sequence)
                    writer.flush()
                    return
                pending[item.sequence] = item.compressed
                next_sequence = _write_ready_serialized_records(
                    writer, pending, next_sequence
                )
            finally:
                writer_queue.task_done()
    finally:
        writer.close()


class RequestTraceWriter:
    def __init__(self):
        self.enabled = False
        self.model_path: Optional[str] = None
        self.tokenizer_path: Optional[str] = None
        self._writer_config: Optional[_WriterConfig] = None
        self._writer_queue: Optional[Any] = None
        self._process_pool: Optional[Any] = None
        self._writer_process: Optional[Any] = None
        self._process_inflight_slots: Optional[threading.BoundedSemaphore] = None
        self._process_inflight = 0
        self._process_inflight_condition = threading.Condition()
        self._process_context: Optional[Any] = None
        self._sequence_lock = threading.Lock()
        self._next_sequence = 0

    def configure(
        self,
        *,
        record_dir: Optional[str],
        max_bytes: int,
        backup_count: int,
        model_path: Optional[str],
        tokenizer_path: Optional[str],
    ) -> None:
        self.close()
        self.enabled = bool(record_dir)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._next_sequence = 0
        if not self.enabled:
            return

        os.makedirs(record_dir, exist_ok=True)
        hostname = socket.gethostname()
        rank = dist.get_rank() if dist.is_initialized() else 0
        filename = os.path.join(
            record_dir, f"request_trace_{hostname}_{rank}_{os.getpid()}.jsonl.gz"
        )
        worker_num = _get_env_int(
            "SGLANG_REQUEST_TRACE_ASYNC_WORKER_NUM",
            DEFAULT_ASYNC_WORKER_NUM,
            minimum=1,
        )
        writer_queue_maxsize = _get_env_int(
            "SGLANG_REQUEST_TRACE_ASYNC_WRITER_QUEUE_MAXSIZE",
            worker_num * DEFAULT_ASYNC_WRITER_QUEUE_MAXSIZE_PER_WORKER,
            minimum=1,
        )
        compresslevel = _get_env_int(
            "SGLANG_REQUEST_TRACE_GZIP_COMPRESSLEVEL",
            DEFAULT_GZIP_COMPRESSLEVEL,
            minimum=0,
            maximum=9,
        )
        self._writer_config = _WriterConfig(
            filename=filename,
            max_bytes=max_bytes,
            backup_count=backup_count,
            compresslevel=compresslevel,
        )
        process_max_inflight = _get_env_int(
            "SGLANG_REQUEST_TRACE_PROCESS_MAX_INFLIGHT",
            worker_num * DEFAULT_PROCESS_MAX_INFLIGHT_PER_WORKER,
            minimum=worker_num,
        )
        self._start_process_writer(
            worker_num, process_max_inflight, writer_queue_maxsize
        )

    def close(self) -> None:
        self._close_process_pool()
        self._close_process_writer()
        self._writer_config = None
        self._writer_queue = None
        self._process_pool = None
        self._writer_process = None
        self._process_inflight_slots = None
        self._process_inflight = 0
        self._process_context = None

    def flush(self) -> None:
        self._wait_process_pool()
        if self._writer_queue is not None:
            self._writer_queue.join()

    def write(self, trace: RequestTraceState) -> None:
        if (
            not self.enabled
            or self._writer_config is None
            or self._writer_queue is None
            or self._process_pool is None
            or self._writer_process is None
            or trace._written
        ):
            return

        if trace.finished_at is None:
            trace.finished_at = datetime.now().timestamp()
        trace.model_path = self.model_path
        trace.tokenizer_path = self.tokenizer_path
        trace._written = True
        with self._sequence_lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        item = _QueuedTrace(sequence, trace)
        self._submit_process_writer_task(item)

    def _start_process_writer(
        self, worker_num: int, max_inflight: int, writer_queue_maxsize: int
    ) -> None:
        assert self._writer_config is not None
        context = self._get_process_context()
        self._writer_queue = context.JoinableQueue(maxsize=writer_queue_maxsize)
        self._writer_process = context.Process(
            target=_writer_process_loop,
            args=(
                self._writer_config.filename,
                self._writer_config.max_bytes,
                self._writer_config.backup_count,
                self._writer_queue,
            ),
            name="request-trace-file-writer-process",
            daemon=True,
        )
        self._writer_process.start()
        self._process_pool = context.Pool(
            processes=worker_num,
            initializer=_init_process_writer_worker,
            initargs=(self._writer_queue,),
        )
        self._process_inflight_slots = threading.BoundedSemaphore(max_inflight)
        self._process_inflight = 0

    def _get_process_context(self) -> Any:
        if self._process_context is not None:
            return self._process_context
        start_method = _get_env_choice(
            "SGLANG_REQUEST_TRACE_PROCESS_START_METHOD",
            DEFAULT_PROCESS_START_METHOD,
            set(mp.get_all_start_methods()),
        )
        self._process_context = mp.get_context(start_method)
        return self._process_context

    def _close_process_pool(self) -> None:
        if self._process_pool is None:
            return
        self._wait_process_pool()
        self._process_pool.close()
        self._process_pool.join()
        self._process_pool = None
        self._process_inflight_slots = None

    def _close_process_writer(self) -> None:
        if self._writer_process is None:
            return
        assert self._writer_queue is not None
        self._writer_queue.join()
        self._writer_queue.put(None)
        self._writer_queue.join()
        self._writer_process.join()
        self._writer_process = None

    def _wait_process_pool(self) -> None:
        with self._process_inflight_condition:
            while self._process_inflight > 0:
                self._process_inflight_condition.wait()

    def _submit_process_writer_task(self, item: _QueuedTrace) -> None:
        assert self._writer_config is not None
        assert self._process_pool is not None
        assert self._process_inflight_slots is not None
        self._process_inflight_slots.acquire()
        with self._process_inflight_condition:
            self._process_inflight += 1
        try:
            self._process_pool.apply_async(
                _serialize_trace_to_writer_for_worker,
                args=(item, self._writer_config.compresslevel),
                callback=self._on_process_writer_task_done,
                error_callback=lambda exc, sequence=item.sequence: (
                    self._on_process_task_error(sequence, exc)
                ),
            )
        except BaseException:
            self._finish_process_task()
            raise

    def _on_process_writer_task_done(self, _sequence: int) -> None:
        self._finish_process_task()

    def _on_process_task_error(self, sequence: int, exc: BaseException) -> None:
        logger.error(
            "Request trace process worker failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        assert self._writer_queue is not None
        try:
            self._writer_queue.put(_SerializedTrace(sequence, None))
        finally:
            self._finish_process_task()

    def _finish_process_task(self) -> None:
        assert self._process_inflight_slots is not None
        self._process_inflight_slots.release()
        with self._process_inflight_condition:
            self._process_inflight -= 1
            self._process_inflight_condition.notify_all()


_writer = RequestTraceWriter()


def configure_request_trace_recording(
    *,
    record_dir: Optional[str],
    max_bytes: int,
    backup_count: int,
    model_path: Optional[str],
    tokenizer_path: Optional[str],
) -> None:
    _writer.configure(
        record_dir=record_dir,
        max_bytes=max_bytes,
        backup_count=backup_count,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )


def request_trace_enabled() -> bool:
    return _writer.enabled


def write_request_trace(trace: RequestTraceState) -> None:
    _writer.write(trace)


def flush_request_trace() -> None:
    _writer.flush()


def close_request_trace_recording() -> None:
    _writer.close()


def _get_env_int(
    name: str, default: int, *, minimum: int, maximum: Optional[int] = None
) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer value for %s: %r", name, value)
        return default
    if parsed < minimum:
        logger.warning(
            "Using minimum value %d for %s because configured value is %d",
            minimum,
            name,
            parsed,
        )
        return minimum
    if maximum is not None and parsed > maximum:
        logger.warning(
            "Using maximum value %d for %s because configured value is %d",
            maximum,
            name,
            parsed,
        )
        return maximum
    return parsed


def _get_env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in choices:
        return normalized
    logger.warning(
        "Ignoring invalid value for %s: %r. Supported values are: %s",
        name,
        value,
        ", ".join(sorted(choices)),
    )
    return default


def trace_http_request(endpoint: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not _writer.enabled:
                return await func(*args, **kwargs)

            raw_request = _find_request(args, kwargs)
            if raw_request is None or raw_request.url.path != endpoint:
                return await func(*args, **kwargs)

            trace = await _create_trace_state(raw_request, endpoint)
            if _request_body_disables_logs(endpoint, trace.http_request.get("body")):
                return await func(*args, **kwargs)

            token = current_request_trace.set(trace)
            try:
                response = await func(*args, **kwargs)
                if isinstance(response, StreamingResponse):
                    response.body_iterator = _wrap_streaming_body(
                        trace, response.body_iterator
                    )
                    return response

                trace.http_response = _serialize_response(response)
                trace.finished_at = datetime.now().timestamp()
                _writer.write(trace)
                return response
            except BaseException as exc:
                trace.status = "error"
                trace.error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                trace.finished_at = datetime.now().timestamp()
                _writer.write(trace)
                raise
            finally:
                current_request_trace.reset(token)

        return wrapper

    return decorator


async def _create_trace_state(
    raw_request: Request, endpoint: str
) -> RequestTraceState:
    return RequestTraceState(
        request_id=uuid.uuid4().hex,
        endpoint=endpoint,
        stream=False,
        http_request=await _serialize_http_request(raw_request),
        created_at=datetime.now().timestamp(),
    )


def _find_request(args: tuple, kwargs: Dict[str, Any]) -> Optional[Request]:
    for value in kwargs.values():
        if isinstance(value, Request):
            return value
    for value in args:
        if isinstance(value, Request):
            return value
    return None


async def _serialize_http_request(raw_request: Request) -> Dict[str, Any]:
    body: Any = None
    body_bytes = await raw_request.body()
    if body_bytes:
        body = _DeferredJsonBody(body_bytes)

    return {
        "method": raw_request.method,
        "path": raw_request.url.path,
        "headers": _sanitize_headers(dict(raw_request.headers)),
        "query": dict(raw_request.query_params),
        "body": body,
    }


def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {
        key: ("<redacted>" if key.lower() in SENSITIVE_HEADERS else value)
        for key, value in headers.items()
    }


def _parse_body(body: bytes) -> Any:
    try:
        return orjson.loads(body)
    except orjson.JSONDecodeError:
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes len={len(body)}>"


def _request_body_disables_logs(endpoint: str, body: Any) -> bool:
    if endpoint not in NO_LOGS_BODY_ENDPOINTS:
        return False
    if isinstance(body, _DeferredJsonBody):
        if b"no_logs" not in body.body:
            return False
        body = _parse_body(body.body)
    return isinstance(body, dict) and bool(body.get("no_logs"))


async def _wrap_streaming_body(
    trace: RequestTraceState, body_iterator: AsyncIterator[Any]
) -> AsyncIterator[Any]:
    token = current_request_trace.set(trace)
    trace.stream = True
    try:
        async for chunk in body_iterator:
            trace.chunks.append(
                {
                    "chunk_index": len(trace.chunks),
                    "chunk": _serialize_chunk(chunk),
                }
            )
            yield chunk
        trace.http_response = None
        trace.finished_at = datetime.now().timestamp()
        _writer.write(trace)
    except BaseException as exc:
        trace.status = "error"
        trace.error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        trace.finished_at = datetime.now().timestamp()
        _writer.write(trace)
        raise
    finally:
        current_request_trace.reset(token)


def _serialize_chunk(chunk: Any) -> Any:
    if isinstance(chunk, bytes):
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes len={len(chunk)}>"
    return _to_jsonable(chunk)


def _serialize_response(response: Any) -> Any:
    if isinstance(response, Response) and hasattr(response, "body"):
        body = response.body
        if isinstance(body, bytes):
            return _DeferredJsonBody(body)
    return _to_jsonable(response)


def add_generation_prompt_ids(
    *,
    trace: RequestTraceState,
    generation_rid: str,
    prompt_token_ids: Optional[List[int]],
    prompt_token_ids_from_request_body: bool = False,
) -> None:
    generation = trace.get_generation(generation_rid)
    generation.prompt_token_ids = (
        None if prompt_token_ids_from_request_body else prompt_token_ids
    )


def add_generation_output_ids(
    *,
    trace: RequestTraceState,
    generation_rid: str,
    output_ids: Optional[List[int]],
    meta_info: Optional[Dict[str, Any]],
    is_delta: bool,
    finished: bool,
) -> None:
    generation = trace.get_generation(generation_rid)
    if output_ids:
        new_output_ids = list(output_ids)
        if is_delta:
            generation.output_token_ids.extend(new_output_ids)
        else:
            prefix_len = len(generation.output_token_ids)
            if new_output_ids[:prefix_len] == generation.output_token_ids:
                generation.output_token_ids.extend(new_output_ids[prefix_len:])
            else:
                generation.output_token_ids = new_output_ids
    if meta_info is not None:
        generation.meta_info = meta_info
    if finished:
        generation.finished = True


def _to_record(trace: RequestTraceState) -> Dict[str, Any]:
    request_body_prompt_token_ids = None
    request_body_prompt_token_ids_loaded = False
    generations = []
    for generation in trace.generations.values():
        prompt_token_ids = generation.prompt_token_ids
        if prompt_token_ids is None:
            if not request_body_prompt_token_ids_loaded:
                request_body_prompt_token_ids = (
                    _extract_prompt_token_ids_from_request_body(trace)
                )
                request_body_prompt_token_ids_loaded = True
            prompt_token_ids = request_body_prompt_token_ids

        generations.append(
            {
                "rid": generation.rid,
                "prompt_token_ids": prompt_token_ids,
                "output_token_ids": generation.output_token_ids,
                "meta_info": generation.meta_info,
                "finished": generation.finished,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": trace.request_id,
        "endpoint": trace.endpoint,
        "stream": trace.stream,
        "status": trace.status,
        "created_at": trace.created_at,
        "finished_at": trace.finished_at,
        "http_request": _to_jsonable(trace.http_request),
        "http_response": _to_jsonable(trace.http_response),
        "chunks": _to_jsonable(trace.chunks),
        "generations": _to_jsonable(generations),
        "error": _to_jsonable(trace.error),
        "model_path": trace.model_path,
        "tokenizer_path": trace.tokenizer_path,
    }


def _to_jsonable(data: Any) -> Any:
    if isinstance(data, _DeferredJsonBody):
        return _deferred_body_to_jsonable(data.body)
    if isinstance(data, _DeferredJsonFragment):
        return _deferred_fragment_to_jsonable(data)
    if dataclasses.is_dataclass(data):
        return {
            field.name: _to_jsonable(getattr(data, field.name))
            for field in dataclasses.fields(data)
            if not field.name.startswith("_")
        }
    if hasattr(data, "model_dump"):
        return _to_jsonable(data.model_dump(exclude_none=True))
    if isinstance(data, dict):
        return {str(k): _to_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_jsonable(v) for v in data]
    if isinstance(data, bytes):
        return _serialize_chunk(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    return str(data)


def _deferred_body_to_jsonable(body: bytes) -> Any:
    stripped = body.lstrip()
    if stripped[:1] in (b"{", b"["):
        return orjson.Fragment(body)
    return _parse_body(body)


def _deferred_fragment_to_jsonable(fragment: _DeferredJsonFragment) -> Any:
    return orjson.Fragment(fragment.body[fragment.start : fragment.end])


def _extract_prompt_token_ids_from_request_body(trace: RequestTraceState) -> Any:
    body = trace.http_request.get("body")
    if not isinstance(body, _DeferredJsonBody):
        return None

    field = _find_numeric_json_array_field(
        body.body, _prompt_token_id_field_names(trace.endpoint)
    )
    if field is None:
        return None
    return _DeferredJsonFragment(body.body, field.start, field.end)


def _prompt_token_id_field_names(endpoint: str) -> tuple[str, ...]:
    if endpoint == "/v1/completions":
        return ("prompt", "input_ids")
    return ("input_ids",)


def _find_numeric_json_array_field(
    body: bytes, field_names: tuple[str, ...]
) -> Optional[_JsonArrayField]:
    for field_name in field_names:
        key = orjson.dumps(field_name)
        start = 0
        while True:
            key_index = body.find(key, start)
            if key_index < 0:
                break
            value_start = _find_json_field_value_start(body, key_index + len(key))
            if value_start is None:
                start = key_index + len(key)
                continue
            if body[value_start : value_start + 1] != b"[":
                start = key_index + len(key)
                continue
            if not _json_array_looks_numeric(body, value_start):
                start = key_index + len(key)
                continue
            value_end = _find_json_array_end(body, value_start)
            if value_end is not None:
                return _JsonArrayField(value_start, value_end)
            start = key_index + len(key)
    return None


def _find_json_field_value_start(body: bytes, start: int) -> Optional[int]:
    index = _skip_json_ws(body, start)
    if index >= len(body) or body[index] != ord(":"):
        return None
    return _skip_json_ws(body, index + 1)


def _skip_json_ws(body: bytes, start: int) -> int:
    index = start
    while index < len(body) and body[index] in b" \t\r\n":
        index += 1
    return index


def _json_array_looks_numeric(body: bytes, array_start: int) -> bool:
    index = _skip_json_ws(body, array_start + 1)
    return index < len(body) and body[index] in b"-0123456789]"


def _find_json_array_end(body: bytes, array_start: int) -> Optional[int]:
    depth = 0
    in_string = False
    escaped = False
    for index in range(array_start, len(body)):
        byte = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue

        if byte == ord('"'):
            in_string = True
        elif byte == ord("["):
            depth += 1
        elif byte == ord("]"):
            depth -= 1
            if depth == 0:
                return index + 1
    return None


atexit.register(_writer.close)
