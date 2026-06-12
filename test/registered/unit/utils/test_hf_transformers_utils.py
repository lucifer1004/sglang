from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
import pytest

from sglang.srt.utils.hf_transformers import tokenizer as hf_tokenizer


class FakeFastTokenizer:
    def __init__(self, backend):
        self._tokenizer = backend


@pytest.mark.parametrize(
    "tokenizer_class",
    [
        "Qwen2Tokenizer",
        "Qwen2TokenizerFast",
        "WeLMV3Tokenizer",
        "WeLMV3TokenizerFast",
    ],
)
def test_fix_v5_tokenizer_components_restores_bpe_backend_from_tokenizer_json(
    tmp_path,
    tokenizer_class,
):
    (tmp_path / "tokenizer_config.json").write_text(
        f'{{"tokenizer_class": "{tokenizer_class}"}}'
    )
    vocab = {"a": 0, "b": 1, "ab": 2}
    good_backend = Tokenizer(BPE(vocab, merges=[("a", "b")], fuse_unk=False))
    good_backend.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    good_backend.decoder = ByteLevelDecoder()
    good_backend.save(str(tmp_path / "tokenizer.json"))

    broken_backend = Tokenizer(BPE(vocab, merges=[], fuse_unk=True))
    assert broken_backend.encode("ab").ids == [0, 1]

    tokenizer = FakeFastTokenizer(broken_backend)
    hf_tokenizer._fix_v5_tokenizer_components(tokenizer, str(tmp_path))

    assert tokenizer._tokenizer.encode("ab").ids == [2]
    assert repr(tokenizer._tokenizer.pre_tokenizer) == repr(good_backend.pre_tokenizer)
    assert repr(tokenizer._tokenizer.decoder) == repr(good_backend.decoder)
