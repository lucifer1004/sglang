import unittest

from transformers import PretrainedConfig

from sglang.srt.models.welm_mtp_version import WelmMTPVersion, get_welm_mtp_version


class TestWelmMTPVersion(unittest.TestCase):
    def test_cached_v2_still_validates_draft_steps(self):
        config = PretrainedConfig()
        config.num_nextn_predict_layers = 3

        self.assertEqual(get_welm_mtp_version(config), WelmMTPVersion.V2)

        with self.assertRaisesRegex(ValueError, "draft_steps=2"):
            get_welm_mtp_version(config, draft_steps=2)

        self.assertEqual(
            get_welm_mtp_version(config, draft_steps=3), WelmMTPVersion.V2
        )

    def test_cached_v1_allows_any_draft_steps(self):
        config = PretrainedConfig()
        config.num_nextn_predict_layers = 1

        self.assertEqual(get_welm_mtp_version(config), WelmMTPVersion.V1)
        self.assertEqual(
            get_welm_mtp_version(config, draft_steps=3), WelmMTPVersion.V1
        )


if __name__ == "__main__":
    unittest.main()
