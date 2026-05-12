# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse


def summarize_routed_experts_value(value) -> Dict[str, Any]:
    if value is None:
        return {"shape": None, "dtype": None, "byte_size": 0}
    if hasattr(value, "shape") and hasattr(value, "numel"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "byte_size": value.numel() * value.element_size(),
        }
    if isinstance(value, (list, tuple)):
        parts = [summarize_routed_experts_value(item) for item in value]
        return {
            "shape": [part["shape"] for part in parts],
            "dtype": next((part["dtype"] for part in parts if part["dtype"]), None),
            "byte_size": sum(part["byte_size"] for part in parts),
            "num_parts": len(parts),
        }
    return {"shape": None, "dtype": type(value).__name__, "byte_size": 0}


class RoutedExpertsStore(ABC):
    @abstractmethod
    def put(self, value) -> Dict[str, Any]:
        raise NotImplementedError()


class DummyRoutedExpertsStore(RoutedExpertsStore):
    def put(self, value) -> Dict[str, Any]:
        return {
            "format": "dummy",
            "backend": "dummy",
            "dropped": True,
            **summarize_routed_experts_value(value),
        }


class _RedisRespClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        db: int,
        username: Optional[str],
        password: Optional[str],
    ):
        self.sock = socket.create_connection((host, port), timeout=5)
        if password:
            if username:
                self._execute("AUTH", username, password)
            else:
                self._execute("AUTH", password)
        if db:
            self._execute("SELECT", str(db))

    def set(self, key: str, value: bytes):
        self._execute("SET", key, value)

    def setex(self, key: str, ttl_sec: int, value: bytes):
        self._execute("SETEX", key, str(ttl_sec), value)

    def close(self):
        self.sock.close()

    def _execute(self, *items):
        self.sock.sendall(self._encode_command(items))
        return self._read_response()

    def _encode_command(self, items) -> bytes:
        out = [f"*{len(items)}\r\n".encode("ascii")]
        for item in items:
            data = item if isinstance(item, bytes) else str(item).encode("utf-8")
            out.append(f"${len(data)}\r\n".encode("ascii"))
            out.append(data)
            out.append(b"\r\n")
        return b"".join(out)

    def _read_response(self):
        prefix = self._read_exact(1)
        if prefix in (b"+", b"-", b":"):
            line = self._read_line()
            if prefix == b"-":
                raise RuntimeError(f"Redis error: {line.decode('utf-8', 'replace')}")
            if prefix == b":":
                return int(line)
            return line.decode("utf-8")
        if prefix == b"$":
            length = int(self._read_line())
            if length == -1:
                return None
            data = self._read_exact(length)
            self._read_exact(2)
            return data
        raise RuntimeError(f"Unsupported Redis response prefix: {prefix!r}")

    def _read_line(self) -> bytes:
        chunks = []
        while True:
            ch = self._read_exact(1)
            if ch == b"\r":
                self._read_exact(1)
                return b"".join(chunks)
            chunks.append(ch)

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Redis connection closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class RedisRoutedExpertsStore(RoutedExpertsStore):
    def __init__(self, dsn: str):
        parsed = urlparse(dsn)
        try:
            import redis
        except ImportError:
            redis = None
            if parsed.scheme == "rediss":
                raise ImportError(
                    "rediss:// routed experts backend requires the optional "
                    "'redis' Python package."
                )

        query = parse_qs(parsed.query)
        self.prefix = query.get("prefix", ["sglang:routed_experts"])[0].strip(":")
        ttl_values = query.get("ttl") or query.get("ttl_sec")
        self.ttl_sec = int(ttl_values[0]) if ttl_values else None

        db = 0
        if parsed.path and parsed.path != "/":
            db = int(parsed.path.strip("/"))

        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        if redis is None:
            self.client = _RedisRespClient(
                host=host,
                port=port,
                db=db,
                username=parsed.username,
                password=parsed.password,
            )
        else:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                username=parsed.username,
                password=parsed.password,
                ssl=parsed.scheme == "rediss",
            )

    def put(self, value) -> Dict[str, Any]:
        if value is None:
            return {
                "format": "remote",
                "backend": "redis",
                "key": None,
                "encoding": None,
                **summarize_routed_experts_value(value),
            }

        if not (hasattr(value, "detach") and hasattr(value, "numel")):
            raise TypeError(
                "Redis routed experts backend currently expects a tensor value, "
                f"got {type(value).__name__}."
            )

        tensor = value.detach().cpu().contiguous()
        payload = tensor.numpy().tobytes()
        key = f"{self.prefix}:{uuid.uuid4().hex}"
        if self.ttl_sec is None:
            self.client.set(key, payload)
        else:
            self.client.setex(key, self.ttl_sec, payload)

        response = {
            "format": "remote",
            "backend": "redis",
            "key": key,
            "encoding": "raw_tensor_bytes",
            "schema_version": 1,
            **summarize_routed_experts_value(tensor),
        }
        if self.ttl_sec is not None:
            response["ttl_sec"] = self.ttl_sec
        return response


def create_routed_experts_store(dsn: Optional[str]) -> Optional[RoutedExpertsStore]:
    if not dsn:
        return None

    scheme = urlparse(dsn).scheme
    if scheme == "dummy":
        return DummyRoutedExpertsStore()
    if scheme in ("redis", "rediss"):
        return RedisRoutedExpertsStore(dsn)

    raise ValueError(
        f"Unsupported routed experts store DSN scheme: {scheme!r}. "
        "Supported schemes are dummy://, redis:// and rediss://."
    )
