import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from importlib import util
from pathlib import Path


class _FakeEnvValue:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakeNetworkAddress:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def to_host_port_str(self):
        return f"{self.host}:{self.port}"


@contextmanager
def _stub_mooncake_transfer_engine_imports():
    module_names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.environ",
        "sglang.srt.utils",
        "sglang.srt.utils.network",
        "mooncake",
        "mooncake.engine",
    ]
    sentinel = object()
    saved_modules = {name: sys.modules.get(name, sentinel) for name in module_names}

    try:
        sglang_module = types.ModuleType("sglang")
        srt_module = types.ModuleType("sglang.srt")
        environ_module = types.ModuleType("sglang.srt.environ")
        utils_module = types.ModuleType("sglang.srt.utils")
        network_module = types.ModuleType("sglang.srt.utils.network")
        mooncake_module = types.ModuleType("mooncake")
        mooncake_engine_module = types.ModuleType("mooncake.engine")

        environ_module.envs = types.SimpleNamespace(
            ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=_FakeEnvValue(False),
            ASCEND_NPU_PHY_ID=_FakeEnvValue(-1),
        )
        network_module.NetworkAddress = _FakeNetworkAddress
        network_module.get_free_port = lambda: 0

        class _FakeTransferEngine:
            def initialize(self, *args):
                self.initialize_args = args
                return 0

            def get_rpc_port(self):
                return 12345

            def get_engine(self):
                return self

        mooncake_engine_module.TransferEngine = _FakeTransferEngine
        mooncake_module.engine = mooncake_engine_module

        sys.modules.update(
            {
                "sglang": sglang_module,
                "sglang.srt": srt_module,
                "sglang.srt.environ": environ_module,
                "sglang.srt.utils": utils_module,
                "sglang.srt.utils.network": network_module,
                "mooncake": mooncake_module,
                "mooncake.engine": mooncake_engine_module,
            }
        )
        yield
    finally:
        for name, module in saved_modules.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_mooncake_transfer_engine_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "distributed"
        / "device_communicators"
        / "mooncake_transfer_engine.py"
    )
    spec = util.spec_from_file_location(
        "mooncake_transfer_engine_under_test", module_path
    )
    module = util.module_from_spec(spec)
    with _stub_mooncake_transfer_engine_imports():
        spec.loader.exec_module(module)
    return module


def _load_get_ib_devices_for_gpu():
    return _load_mooncake_transfer_engine_module().get_ib_devices_for_gpu


get_ib_devices_for_gpu = _load_get_ib_devices_for_gpu()


class TestMooncakeIBDeviceSelection(unittest.TestCase):
    def test_comma_separated_devices_apply_to_all_gpus(self):
        self.assertEqual(
            get_ib_devices_for_gpu("mlx5_bond_5,mlx5_bond_6", gpu_id=3),
            "mlx5_bond_5,mlx5_bond_6",
        )

    def test_legacy_per_gpu_mapping(self):
        mapping = json.dumps(
            {
                "0": "mlx5_bond_1",
                "1": "mlx5_bond_2",
                "default": "mlx5_bond_8",
            }
        )

        self.assertEqual(get_ib_devices_for_gpu(mapping, gpu_id=0), "mlx5_bond_1")
        self.assertEqual(get_ib_devices_for_gpu(mapping, gpu_id=1), "mlx5_bond_2")
        self.assertEqual(get_ib_devices_for_gpu(mapping, gpu_id=7), "mlx5_bond_8")

    def test_role_aware_mapping(self):
        mapping = json.dumps(
            {
                "encoder": {
                    "0": "mlx5_bond_1",
                    "1": "mlx5_bond_2",
                },
                "worker": {
                    "4": "mlx5_bond_5",
                    "5": "mlx5_bond_6",
                },
                "encoder_receiver": ("mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8"),
                "default": "mlx5_bond_8",
            }
        )

        self.assertEqual(
            get_ib_devices_for_gpu(mapping, gpu_id=1, role="encoder"),
            "mlx5_bond_2",
        )
        self.assertEqual(
            get_ib_devices_for_gpu(mapping, gpu_id=4, role="worker"),
            "mlx5_bond_5",
        )
        self.assertEqual(
            get_ib_devices_for_gpu(mapping, gpu_id=5, role="llm_worker"),
            "mlx5_bond_6",
        )
        self.assertEqual(
            get_ib_devices_for_gpu(mapping, gpu_id=0, role="encoder_receiver"),
            "mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8",
        )
        self.assertEqual(
            get_ib_devices_for_gpu(mapping, gpu_id=99, role="unknown"),
            "mlx5_bond_8",
        )

    def test_role_default_per_gpu_mapping(self):
        mapping = json.dumps(
            {
                "default": {
                    "0": "mlx5_bond_1",
                    "4": "mlx5_bond_5",
                },
            }
        )

        self.assertEqual(
            get_ib_devices_for_gpu(mapping, gpu_id=4, role="missing"),
            "mlx5_bond_5",
        )

    def test_json_file_mapping(self):
        mapping = {
            "encoder": {
                "0": "mlx5_bond_1",
            },
            "encoder_receiver": "mlx5_bond_5,mlx5_bond_6",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ib_mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")

            self.assertEqual(
                get_ib_devices_for_gpu(str(path), gpu_id=0, role="encoder"),
                "mlx5_bond_1",
            )
            self.assertEqual(
                get_ib_devices_for_gpu(str(path), gpu_id=0, role="encoder_receiver"),
                "mlx5_bond_5,mlx5_bond_6",
            )

    def test_missing_role_mapping_fails_loudly(self):
        mapping = json.dumps({"encoder": {"0": "mlx5_bond_1"}})

        with self.assertRaisesRegex(ValueError, "No IB devices configured"):
            get_ib_devices_for_gpu(mapping, gpu_id=0, role="worker")

    def test_missing_gpu_mapping_fails_loudly(self):
        mapping = json.dumps({"encoder": {"0": "mlx5_bond_1"}})

        with self.assertRaisesRegex(ValueError, "No IB devices configured for GPU 1"):
            get_ib_devices_for_gpu(mapping, gpu_id=1, role="encoder")

    def test_reinit_with_different_ib_device_fails_loudly(self):
        module = _load_mooncake_transfer_engine_module()
        mapping = json.dumps(
            {
                "encoder_receiver": {
                    "4": "mlx5_bond_5",
                    "5": "mlx5_bond_6",
                },
            }
        )

        with _stub_mooncake_transfer_engine_imports():
            engine = module.init_mooncake_transfer_engine(
                "127.0.0.1",
                gpu_id=4,
                ib_device=mapping,
                role="encoder_receiver",
            )
            self.assertEqual(engine.get_ib_device(), "mlx5_bond_5")

            self.assertIs(
                module.init_mooncake_transfer_engine(
                    "127.0.0.1",
                    gpu_id=4,
                    ib_device=mapping,
                    role="encoder_receiver",
                ),
                engine,
            )
            with self.assertRaisesRegex(RuntimeError, "already initialized"):
                module.init_mooncake_transfer_engine(
                    "127.0.0.1",
                    gpu_id=5,
                    ib_device=mapping,
                    role="encoder_receiver",
                )


if __name__ == "__main__":
    unittest.main()
