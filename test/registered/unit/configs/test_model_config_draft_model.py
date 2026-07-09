import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig, get_hybrid_layer_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestModelConfigDraftModel(CustomTestCase):
    def test_welmv4_draft_uses_nextn_architecture(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.is_draft_model = True
        model_config.hf_config = SimpleNamespace(
            architectures=["WeLMV4MoeForCausalLM"],
            num_nextn_predict_layers=1,
        )

        model_config._config_draft_model()

        self.assertEqual(
            model_config.hf_config.architectures[0],
            "WeLMV4MoeForCausalLMNextN",
        )
        self.assertEqual(model_config.hf_config.num_nextn_predict_layers, 1)

    def test_welmv4_target_hybrid_layer_ids_use_layerwise_windows(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_target_hidden_layers=4,
            sliding_window=1024,
            max_position_embeddings=1024,
            sliding_window_size_layerwise=[1024, 512, 1024, 512],
        )

        swa_layer_ids, full_layer_ids = get_hybrid_layer_ids(
            ["WeLMV4MoeForCausalLM"], hf_config, context_len=1024
        )

        self.assertEqual(swa_layer_ids, [1, 3])
        self.assertEqual(full_layer_ids, [0, 2])

    def test_welmv4_nextn_hybrid_layer_ids_use_target_layer_offset(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_target_hidden_layers=4,
            num_nextn_predict_layers=2,
            sliding_window=1024,
            max_position_embeddings=1024,
            sliding_window_size_layerwise=[1024, 1024, 1024, 1024, 512, 512],
        )

        swa_layer_ids, full_layer_ids = get_hybrid_layer_ids(
            ["WeLMV4MoeForCausalLMNextN"], hf_config, context_len=1024
        )

        self.assertEqual(swa_layer_ids, [0, 1])
        self.assertEqual(full_layer_ids, [])


if __name__ == "__main__":
    unittest.main()
