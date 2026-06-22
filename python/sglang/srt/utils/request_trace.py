from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import os
import socket
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


current_request_trace: ContextVar[Optional[RequestTraceState]] = ContextVar(
    "current_request_trace", default=None
)


class GzipRotatingFileHandler(logging.Handler):
    terminator = "\n"

    def __init__(
        self, filename: str, *, max_bytes: int, backup_count: int, encoding="utf-8"
    ):
        super().__init__()
        self.baseFilename = os.path.abspath(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            compressed = gzip.compress(
                (msg + self.terminator).encode(self.encoding)
            )
            if self._should_rollover(len(compressed)):
                self._do_rollover()
            with open(self.baseFilename, "ab") as file:
                file.write(compressed)
        except Exception:
            self.handleError(record)

    def _should_rollover(self, compressed_len: int) -> bool:
        if self.max_bytes <= 0 or self.backup_count <= 0:
            return False
        if not os.path.exists(self.baseFilename):
            return False
        return os.path.getsize(self.baseFilename) + compressed_len >= self.max_bytes

    def _do_rollover(self) -> None:
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


class RequestTraceWriter:
    def __init__(self):
        self.enabled = False
        self.model_path: Optional[str] = None
        self.tokenizer_path: Optional[str] = None
        self._logger: Optional[logging.Logger] = None

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
        if not self.enabled:
            return

        os.makedirs(record_dir, exist_ok=True)
        hostname = socket.gethostname()
        rank = dist.get_rank() if dist.is_initialized() else 0
        filename = os.path.join(
            record_dir, f"request_trace_{hostname}_{rank}_{os.getpid()}.jsonl.gz"
        )
        handler = GzipRotatingFileHandler(
            filename, max_bytes=max_bytes, backup_count=backup_count
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        trace_logger = logging.getLogger(
            f"{__name__}.{hostname}.{rank}.{os.getpid()}"
        )
        trace_logger.setLevel(logging.INFO)
        trace_logger.propagate = False
        trace_logger.handlers.clear()
        trace_logger.addHandler(handler)
        self._logger = trace_logger

    def close(self) -> None:
        if self._logger is None:
            return
        for handler in self._logger.handlers:
            handler.close()
        self._logger.handlers.clear()
        self._logger = None

    def write(self, trace: RequestTraceState) -> None:
        if not self.enabled or self._logger is None or trace._written:
            return

        if trace.finished_at is None:
            trace.finished_at = datetime.now().timestamp()
        trace.model_path = self.model_path
        trace.tokenizer_path = self.tokenizer_path
        try:
            self._logger.info(json.dumps(_to_record(trace), ensure_ascii=False))
            trace._written = True
        except Exception:
            logger.exception("Failed to write request trace")


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
            if _request_body_disables_logs(trace.http_request.get("body")):
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
        body = _parse_body(body_bytes)

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


def _request_body_disables_logs(body: Any) -> bool:
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
            return _parse_body(body)
    return _to_jsonable(response)


def add_generation_prompt_ids(
    *,
    trace: RequestTraceState,
    generation_rid: str,
    prompt_token_ids: Optional[List[int]],
) -> None:
    generation = trace.get_generation(generation_rid)
    generation.prompt_token_ids = (
        list(prompt_token_ids) if prompt_token_ids is not None else None
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
        generation.meta_info = _to_jsonable(meta_info)
    if finished:
        generation.finished = True


def _to_record(trace: RequestTraceState) -> Dict[str, Any]:
    generations = [
        {
            "rid": generation.rid,
            "prompt_token_ids": generation.prompt_token_ids,
            "output_token_ids": generation.output_token_ids,
            "meta_info": generation.meta_info,
            "finished": generation.finished,
        }
        for generation in trace.generations.values()
    ]
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
