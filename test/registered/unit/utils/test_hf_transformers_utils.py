from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer

from sglang.srt.utils import hf_transformers_utils


class FakeFastTokenizer:
    def __init__(self, backend):
        self._tokenizer = backend


def test_fix_v5_tokenizer_components_restores_bpe_model_from_tokenizer_json(
    tmp_path,
):
    vocab = {"a": 0, "b": 1, "ab": 2}
    good_backend = Tokenizer(BPE(vocab, merges=[("a", "b")], fuse_unk=False))
    good_backend.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    good_backend.decoder = ByteLevelDecoder()
    good_backend.save(str(tmp_path / "tokenizer.json"))

    broken_backend = Tokenizer(BPE(vocab, merges=[], fuse_unk=True))
    assert broken_backend.encode("ab").ids == [0, 1]

    tokenizer = FakeFastTokenizer(broken_backend)
    hf_transformers_utils._fix_v5_tokenizer_components(tokenizer, str(tmp_path))

    assert tokenizer._tokenizer.encode("ab").ids == [2]
    assert repr(tokenizer._tokenizer.pre_tokenizer) == repr(good_backend.pre_tokenizer)
    assert repr(tokenizer._tokenizer.decoder) == repr(good_backend.decoder)
