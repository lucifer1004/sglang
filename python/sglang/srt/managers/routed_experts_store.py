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

import os
import socket
import uuid
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
    if isinstance(value, dict):
        parts = {
            key: summarize_routed_experts_value(item) for key, item in value.items()
        }
        return {
            "shape": parts.get("values", {}).get("shape"),
            "dtype": parts.get("values", {}).get("dtype"),
            "byte_size": sum(part["byte_size"] for part in parts.values()),
            "parts": parts,
        }
    return {"shape": None, "dtype": type(value).__name__, "byte_size": 0}


class RoutedExpertsStore(ABC):
    @abstractmethod
    def put(self, value) -> Dict[str, Any]:
        raise NotImplementedError()

    def get(self, ref: Dict[str, Any]) -> bytes:
        raise NotImplementedError(
            f"{type(self).__name__} does not support reading routed experts."
        )


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

    def get(self, key: str):
        return self._execute("GET", key)

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

        if isinstance(value, (list, tuple)):
            summary = summarize_routed_experts_value(value)
            return {
                "format": "remote_list",
                "backend": "redis",
                "schema_version": 3,
                "items": [self.put(item) for item in value],
                "remote_summary": summary,
            }

        if isinstance(value, dict):
            if "values" not in value:
                raise TypeError(
                    "Redis routed experts backend expects sparse payloads to "
                    "contain a 'values' tensor."
                )
            summary = summarize_routed_experts_value(value)
            values_ref = self.put(value["values"])
            response = {
                **value,
                "values": values_ref,
                "format": "remote_masked_dense",
                "backend": "redis",
                "schema_version": 2,
                "remote_summary": summary,
            }
            return response

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

    def get(self, ref: Dict[str, Any]) -> bytes:
        key = ref.get("key")
        if not key:
            raise ValueError(f"Redis routed experts reference is missing key: {ref}")
        payload = self.client.get(key)
        if payload is None:
            raise KeyError(f"Redis routed experts key was not found: {key}")
        return payload


def _query_value(query, names, default=None):
    for name in names:
        values = query.get(name)
        if values:
            return values[0]
    return default


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_size(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, multiplier in sorted(
        multipliers.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if text.endswith(suffix):
            return int(text[: -len(suffix)]) * multiplier
    return int(text)


class MooncakeRoutedExpertsStore(RoutedExpertsStore):
    def __init__(self, dsn: str):
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as exc:
            raise ImportError(
                "mooncake:// routed experts backend requires the Mooncake "
                "Python package."
            ) from exc

        parsed = urlparse(dsn)
        query = parse_qs(parsed.query)
        self.prefix = _query_value(query, ("prefix",), "sglang:routed_experts").strip(
            ":"
        )
        self.replica_num = int(_query_value(query, ("replica_num",), 1))
        self._replicate_config_cls = ReplicateConfig

        local_hostname = _query_value(
            query,
            ("local_hostname", "client_hostname"),
            parsed.netloc
            or os.getenv("MOONCAKE_LOCAL_HOSTNAME")
            or os.getenv("LOCAL_HOSTNAME")
            or "localhost",
        )
        metadata_server = _query_value(
            query,
            ("metadata_server", "metadata"),
            os.getenv("MOONCAKE_TE_META_DATA_SERVER") or "P2PHANDSHAKE",
        )
        global_segment_size = _parse_size(
            _query_value(
                query,
                ("global_segment_size",),
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE") or "4gb",
            )
        )
        local_buffer_size = _parse_size(
            _query_value(query, ("local_buffer_size",), 16 * 1024 * 1024)
        )
        protocol = _query_value(
            query, ("protocol",), os.getenv("MOONCAKE_PROTOCOL") or "tcp"
        )
        device_name = _query_value(
            query,
            ("device", "device_name", "rdma_devices"),
            os.getenv("MOONCAKE_DEVICE") or "",
        )
        master_server_addr = _query_value(
            query,
            ("master_server", "master_server_addr", "master_server_address"),
            os.getenv("MOONCAKE_MASTER"),
        )
        if not master_server_addr:
            raise ValueError(
                "mooncake:// routed experts backend requires a master server. "
                "Set MOONCAKE_MASTER or pass master_server=<host:port> in the DSN."
            )

        enable_ssd_offload = _parse_bool(
            _query_value(query, ("enable_ssd_offload",), False)
        )
        ssd_offload_path = _query_value(query, ("ssd_offload_path",), "")

        self.store = MooncakeDistributedStore()
        ret_code = self.store.setup(
            local_hostname,
            metadata_server,
            global_segment_size,
            local_buffer_size,
            protocol,
            device_name,
            master_server_addr,
            None,
            enable_ssd_offload,
            ssd_offload_path,
        )
        if ret_code != 0:
            raise RuntimeError(
                f"Failed to setup Mooncake routed experts store: {ret_code}"
            )

    def put(self, value) -> Dict[str, Any]:
        if value is None:
            return {
                "format": "remote",
                "backend": "mooncake",
                "key": None,
                "encoding": None,
                **summarize_routed_experts_value(value),
            }

        if isinstance(value, (list, tuple)):
            summary = summarize_routed_experts_value(value)
            return {
                "format": "remote_list",
                "backend": "mooncake",
                "schema_version": 3,
                "items": [self.put(item) for item in value],
                "remote_summary": summary,
            }

        if isinstance(value, dict):
            if "values" not in value:
                raise TypeError(
                    "Mooncake routed experts backend expects sparse payloads to "
                    "contain a 'values' tensor."
                )
            summary = summarize_routed_experts_value(value)
            response = {
                **value,
                "values": self.put(value["values"]),
                "format": "remote_masked_dense",
                "backend": "mooncake",
                "schema_version": 2,
                "remote_summary": summary,
            }
            valid_mask = value.get("valid_mask")
            if hasattr(valid_mask, "detach") and hasattr(valid_mask, "numel"):
                response["valid_mask"] = self.put(valid_mask)
            return response

        if not (hasattr(value, "detach") and hasattr(value, "numel")):
            raise TypeError(
                "Mooncake routed experts backend currently expects a tensor value, "
                f"got {type(value).__name__}."
            )

        tensor = value.detach().cpu().contiguous()
        payload = tensor.numpy().tobytes()
        key = f"{self.prefix}:{uuid.uuid4().hex}"
        config = self._replicate_config_cls()
        config.replica_num = self.replica_num
        ret_code = self.store.put(key, payload, config)
        if ret_code != 0:
            raise RuntimeError(
                f"Failed to put routed experts into Mooncake: {ret_code}"
            )

        return {
            "format": "remote",
            "backend": "mooncake",
            "key": key,
            "encoding": "raw_tensor_bytes",
            "schema_version": 1,
            "replica_num": self.replica_num,
            **summarize_routed_experts_value(tensor),
        }

    def get(self, ref: Dict[str, Any]) -> bytes:
        key = ref.get("key")
        if not key:
            raise ValueError(f"Mooncake routed experts reference is missing key: {ref}")
        payload = self.store.get(key)
        if payload == b"":
            raise KeyError(f"Mooncake routed experts key was not found: {key}")
        return payload


def create_routed_experts_store(dsn: Optional[str]) -> Optional[RoutedExpertsStore]:
    if not dsn:
        return None

    scheme = urlparse(dsn).scheme
    if scheme == "dummy":
        return DummyRoutedExpertsStore()
    if scheme in ("redis", "rediss"):
        return RedisRoutedExpertsStore(dsn)
    if scheme == "mooncake":
        return MooncakeRoutedExpertsStore(dsn)

    raise ValueError(
        f"Unsupported routed experts store DSN scheme: {scheme!r}. "
        "Supported schemes are dummy://, redis://, rediss:// and mooncake://."
    )


def is_remote_routed_experts_ref(value) -> bool:
    """Return True if `value` is a remote-fetch ref dict (rather than a tensor
    or a Python list of expert ids)."""
    return isinstance(value, dict) and value.get("format") == "remote"


def _dtype_from_routed_experts_ref_dict(ref: Dict[str, Any]):
    import numpy as np
    dtype = str(ref.get("dtype", "int32")).removeprefix("torch.")
    if dtype == "bool":
        return np.dtype(np.bool_)
    return np.dtype(dtype)


def decode_remote_routed_experts_tensor(
    store: Optional[RoutedExpertsStore], ref: Dict[str, Any]
):
    """Fetch and reshape a `remote` routed-experts ref into a CPU torch.Tensor.
    Used by tokenizer_manager (when fetching at HTTP-admission time) AND by
    scheduler (when the tokenizer_manager opts to defer the fetch — passes the
    ref through ZMQ instead of the full 64 MB tensor — to avoid pickling the
    raw bytes across the IPC boundary)."""
    import numpy as np
    import torch

    if store is None:
        raise ValueError(
            "Remote routed_experts replay requires --routed-experts-store-dsn."
        )
    if ref.get("format") != "remote":
        raise ValueError(f"Unsupported routed_experts remote reference: {ref}")
    payload = store.get(ref)
    dtype = _dtype_from_routed_experts_ref_dict(ref)
    tensor = np.frombuffer(payload, dtype=dtype).copy()
    shape = ref.get("shape")
    if shape is not None:
        tensor = tensor.reshape(shape)
    return torch.from_numpy(tensor)
