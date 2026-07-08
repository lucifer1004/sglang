import json
import logging
import os
from typing import List, Optional

from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress, get_free_port

logger = logging.getLogger(__name__)

# Module-level shared engine instance, set by init_mooncake_transfer_engine().
_mooncake_transfer_engine: Optional["MooncakeTransferEngine"] = None


_ROLE_ALIASES = {
    "encoder": ("encoder", "vision_encoder", "mm_encoder"),
    "encoder_receiver": (
        "encoder_receiver",
        "receiver",
        "mm_receiver",
        "vlm_receiver",
    ),
    "worker": ("worker", "model_worker", "llm_worker", "scheduler"),
    "elastic_ep": ("elastic_ep", "expert_backup"),
}


def _role_candidates(role: Optional[str]) -> List[str]:
    if role is None:
        return []
    role = role.strip()
    if not role:
        return []

    candidates = [role]
    for canonical_role, aliases in _ROLE_ALIASES.items():
        if role == canonical_role or role in aliases:
            candidates.extend([canonical_role, *aliases])

    return list(dict.fromkeys(candidates))


def _is_gpu_mapping_key(key) -> bool:
    return isinstance(key, int) or (
        isinstance(key, str) and (key.isdigit() or key in ("default", "*"))
    )


def _is_legacy_gpu_mapping(mapping: dict) -> bool:
    return all(
        _is_gpu_mapping_key(key) and isinstance(value, str)
        for key, value in mapping.items()
    )


def _resolve_gpu_mapping(mapping: dict, gpu_id: int, context: str) -> str:
    gpu_mapping = {}
    default_value = None
    for gpu_key, ib_devices in mapping.items():
        if not isinstance(ib_devices, str):
            raise ValueError(f"Invalid {context}: mapping values must be strings")

        if isinstance(gpu_key, str) and gpu_key.isdigit():
            gpu_mapping[int(gpu_key)] = ib_devices.strip()
        elif isinstance(gpu_key, int):
            gpu_mapping[gpu_key] = ib_devices.strip()
        elif isinstance(gpu_key, str) and gpu_key in ("default", "*"):
            default_value = ib_devices.strip()
        else:
            raise ValueError(
                f"Invalid {context}: keys must be integers, string integer "
                "keys, 'default', or '*'"
            )

    if not gpu_mapping and default_value is None:
        raise ValueError(f"No valid GPU mappings found in {context}")

    if gpu_id in gpu_mapping:
        return gpu_mapping[gpu_id]
    if default_value is not None:
        return default_value

    raise ValueError(
        f"No IB devices configured for GPU {gpu_id} in {context}. "
        f"Available GPUs: {list(gpu_mapping.keys())}"
    )


def _resolve_role_value(value, gpu_id: int, context: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _resolve_gpu_mapping(value, gpu_id, context)
    raise ValueError(f"Invalid {context}: role values must be strings or mappings")


def get_ib_devices_for_gpu(
    ib_device_str: Optional[str],
    gpu_id: int,
    role: Optional[str] = None,
) -> Optional[str]:
    """
    Parse IB device string and get IB devices for a specific GPU ID.

    Supports all the following formats:
    1. Old format: "ib0, ib1, ib2"
    2. New format: {0: "ib0, ib1", 1: "ib2, ib3", 2: "ib4"}
    3. JSON file: path to a JSON file containing the mapping
    4. Role-aware JSON:
       {
         "encoder": {"0": "ib0", "1": "ib1"},
         "encoder_receiver": "ib4,ib5",
         "worker": {"4": "ib4", "5": "ib5"},
         "default": "ib0,ib1"
       }

    Args:
        ib_device_str: The original IB device string or path to JSON file
        gpu_id: The GPU ID to get devices for
        role: Optional process role, e.g. "encoder", "encoder_receiver",
            "worker". Role-specific entries are preferred when present.

    Returns:
        IB devices string for the GPU, or None if not available
    """
    if ib_device_str is None or not ib_device_str.strip():
        return None

    ib_device_str = ib_device_str.strip()
    original_ib_device_str = ib_device_str

    # Check if it's a JSON file first and load its content
    is_json_file = ib_device_str.endswith(".json")
    if is_json_file:
        try:
            if os.path.isfile(ib_device_str):
                with open(ib_device_str, "r", encoding="utf-8") as f:
                    ib_device_str = f.read()
            else:
                # File doesn't exist, treat as old format
                raise RuntimeError(f"File {ib_device_str} does not exist.")
        except (IOError, OSError) as e:
            # File reading failed, raise exception
            raise RuntimeError(f"Failed to read JSON file {ib_device_str}: {e}") from e

    # Check if it's JSON format (new format)
    try:
        parsed_json = json.loads(ib_device_str)
        if isinstance(parsed_json, dict):
            # Prefer role-specific mappings when the caller supplies a role.
            for role_key in _role_candidates(role):
                if role_key in parsed_json:
                    return _resolve_role_value(
                        parsed_json[role_key],
                        gpu_id,
                        f"role '{role_key}' IB mapping",
                    )

            for default_key in ("default", "*"):
                if default_key in parsed_json and not _is_legacy_gpu_mapping(
                    parsed_json
                ):
                    return _resolve_role_value(
                        parsed_json[default_key],
                        gpu_id,
                        f"'{default_key}' IB mapping",
                    )

            if _is_legacy_gpu_mapping(parsed_json):
                return _resolve_gpu_mapping(parsed_json, gpu_id, "IB mapping")

            raise ValueError(
                f"No IB devices configured for role {role!r}. "
                f"Available role keys: {list(parsed_json.keys())}"
            )

    except json.JSONDecodeError:
        if is_json_file:
            # It was supposed to be a JSON file but failed to parse
            raise RuntimeError(
                f"Failed to parse JSON content from file {original_ib_device_str}"
            )
        # Not JSON format, treat as old format - return same devices for all GPUs
        return ib_device_str


class MooncakeTransferEngine:
    """Shared Mooncake transfer engine for RDMA/transfer operations."""

    def __init__(
        self,
        hostname: str,
        gpu_id: Optional[int] = None,
        ib_device: Optional[str] = None,
        role: Optional[str] = None,
    ):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id if gpu_id is not None else 0
        self.role = role
        self.ib_device = get_ib_devices_for_gpu(ib_device, self.gpu_id, role=role)
        logger.info(
            "Mooncake Transfer Engine IB selection: role=%s gpu_id=%s ib_device=%s",
            self.role,
            self.gpu_id,
            self.ib_device,
        )

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        self.session_id = NetworkAddress(
            self.hostname, self.engine.get_rpc_port()
        ).to_host_port_str()

    def register(self, ptr, length):
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory registration %s failed.", ptr)

    def deregister(self, ptr):
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            # Mark deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory deregistration %s failed.", ptr)

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        """Batch register multiple memory regions."""
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark batch register as failed
            ret_value = -1
            if not hasattr(self.engine, "batch_register_memory"):
                raise RuntimeError(
                    "Mooncake's batch register requires a newer version of "
                    "mooncake-transfer-engine. Please upgrade Mooncake."
                )

        if ret_value != 0:
            logger.debug("Mooncake batch memory registration failed.")
        return ret_value

    def batch_deregister(self, ptrs: List[int]) -> int:
        """Batch deregister multiple memory regions."""
        try:
            ret_value = self.engine.batch_unregister_memory(ptrs)
        except Exception:
            # Mark batch deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake batch memory deregistration failed.")
        return ret_value

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        if envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.get():
            npu_phy_id = envs.ASCEND_NPU_PHY_ID.get()
            if npu_phy_id == -1:
                hostname += f":{get_free_port()}:npu_{self.gpu_id}"
            else:
                hostname += f":{get_free_port()}:npu_{npu_phy_id}"
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "ascend",
                device_name if device_name is not None else "",
            )
        else:
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "rdma",
                device_name if device_name is not None else "",
            )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        try:
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            ret = -1

        if ret < 0:
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                buffer,
                session_id,
                peer_buffer_address,
            )

        return ret

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        """Synchronously transfer data to the specified addresses in batches."""
        try:
            ret = self.engine.batch_transfer_sync_write(
                session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1
            if not hasattr(self.engine, "batch_transfer_sync_write"):
                raise RuntimeError(
                    "Mooncake's batch transfer requires mooncake-transfer-engine "
                    ">= 0.3.4.post2. Please upgrade Mooncake by "
                    "'pip install mooncake-transfer-engine --upgrade'"
                )

        if ret < 0:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, "
                "Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
        return ret

    def get_session_id(self):
        return self.session_id

    def get_engine(self):
        return self.engine.get_engine()

    def get_ib_device(self):
        return self.ib_device


def init_mooncake_transfer_engine(
    hostname: str,
    gpu_id: Optional[int] = None,
    ib_device: Optional[str] = None,
    role: Optional[str] = None,
) -> MooncakeTransferEngine:
    """
    Initialize the shared MooncakeTransferEngine. Note: if already
    initialized with the same (hostname, gpu_id, ib_device), returns existing
    instance. Call from parallel_state when model parallel is set up and
    mooncake transfer is needed.
    """
    global _mooncake_transfer_engine
    if _mooncake_transfer_engine is not None:
        if ib_device is not None:
            requested_gpu_id = (
                gpu_id if gpu_id is not None else _mooncake_transfer_engine.gpu_id
            )
            requested_ib_device = get_ib_devices_for_gpu(
                ib_device, requested_gpu_id, role=role
            )
            existing_ib_device = _mooncake_transfer_engine.get_ib_device()
            if requested_ib_device != existing_ib_device:
                raise RuntimeError(
                    "Mooncake Transfer Engine is already initialized with "
                    f"ib_device={existing_ib_device!r}, but requested "
                    f"ib_device={requested_ib_device!r} for role={role!r} "
                    f"gpu_id={requested_gpu_id}. A single process cannot reuse "
                    "one Mooncake Transfer Engine with different IB bindings."
                )
        return _mooncake_transfer_engine
    _mooncake_transfer_engine = MooncakeTransferEngine(
        hostname=hostname, gpu_id=gpu_id, ib_device=ib_device, role=role
    )
    return _mooncake_transfer_engine


def get_mooncake_transfer_engine() -> Optional[MooncakeTransferEngine]:
    """Return the shared MooncakeTransferEngine if initialized, else None."""
    return _mooncake_transfer_engine
