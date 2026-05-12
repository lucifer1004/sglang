import sys
import types

import pytest
import torch

from sglang.srt.managers.routed_experts_store import create_routed_experts_store


class FakeReplicateConfig:
    def __init__(self):
        self.replica_num = 0


class FakeMooncakeDistributedStore:
    instances = []

    def __init__(self):
        self.setup_args = None
        self.puts = []
        FakeMooncakeDistributedStore.instances.append(self)

    def setup(self, *args):
        self.setup_args = args
        return 0

    def put(self, key, value, config):
        self.puts.append((key, value, config.replica_num))
        return 0


@pytest.fixture(autouse=True)
def fake_mooncake(monkeypatch):
    mooncake_mod = types.ModuleType("mooncake")
    store_mod = types.ModuleType("mooncake.store")
    store_mod.MooncakeDistributedStore = FakeMooncakeDistributedStore
    store_mod.ReplicateConfig = FakeReplicateConfig
    FakeMooncakeDistributedStore.instances = []

    monkeypatch.setitem(sys.modules, "mooncake", mooncake_mod)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_mod)


def test_mooncake_routed_experts_store_puts_raw_tensor_bytes():
    store = create_routed_experts_store(
        "mooncake://localhost:17913"
        "?metadata_server=http://127.0.0.1:18080/metadata"
        "&master_server=127.0.0.1:15051"
        "&global_segment_size=32mb"
        "&local_buffer_size=1mb"
        "&protocol=tcp"
        "&device=mlx5_0"
        "&prefix=test:routed"
        "&replica_num=2"
        "&enable_ssd_offload=1"
        "&ssd_offload_path=/tmp/mooncake-ssd"
    )

    fake_store = FakeMooncakeDistributedStore.instances[-1]
    assert fake_store.setup_args == (
        "localhost:17913",
        "http://127.0.0.1:18080/metadata",
        32 * 1024 * 1024,
        1024 * 1024,
        "tcp",
        "mlx5_0",
        "127.0.0.1:15051",
        None,
        True,
        "/tmp/mooncake-ssd",
    )

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    metadata = store.put(tensor)

    assert metadata["format"] == "remote"
    assert metadata["backend"] == "mooncake"
    assert metadata["encoding"] == "raw_tensor_bytes"
    assert metadata["schema_version"] == 1
    assert metadata["replica_num"] == 2
    assert metadata["shape"] == [2, 3]
    assert metadata["dtype"] == "int32"
    assert metadata["byte_size"] == tensor.numel() * tensor.element_size()
    assert metadata["key"].startswith("test:routed:")

    key, payload, replica_num = fake_store.puts[-1]
    assert key == metadata["key"]
    assert payload == tensor.contiguous().numpy().tobytes()
    assert replica_num == 2


def test_mooncake_routed_experts_store_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "env-host:1234")
    monkeypatch.setenv("MOONCAKE_TE_META_DATA_SERVER", "http://metadata/metadata")
    monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")
    monkeypatch.setenv("MOONCAKE_PROTOCOL", "tcp")
    monkeypatch.setenv("MOONCAKE_DEVICE", "")
    monkeypatch.setenv("MOONCAKE_MASTER", "master:50051")

    create_routed_experts_store("mooncake://?prefix=envtest")

    fake_store = FakeMooncakeDistributedStore.instances[-1]
    assert fake_store.setup_args == (
        "env-host:1234",
        "http://metadata/metadata",
        64 * 1024 * 1024,
        16 * 1024 * 1024,
        "tcp",
        "",
        "master:50051",
        None,
        False,
        "",
    )


def test_mooncake_routed_experts_store_requires_master(monkeypatch):
    monkeypatch.delenv("MOONCAKE_MASTER", raising=False)

    with pytest.raises(ValueError, match="requires a master server"):
        create_routed_experts_store("mooncake://localhost:17913")
