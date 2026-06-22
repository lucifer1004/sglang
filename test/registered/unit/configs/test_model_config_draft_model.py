import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig
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


if __name__ == "__main__":
    unittest.main()
