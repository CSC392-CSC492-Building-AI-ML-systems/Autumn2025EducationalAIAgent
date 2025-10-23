from transformers import LogitsProcessor
from typing import List, Dict, Set, Optional

class AsciiOnlyProcessor(LogitsProcessor):
    """
    Masks any token whose decoded text contains non-ASCII characters.
    Optionally disallow control chars except \t, \n, \r and space.
    """
    def __init__(self, tokenizer, allow_controls: bool = False, always_allow: Optional[Set[int]] = None):
        import torch
        self.torch = torch
        self.tok = tokenizer
        self.allowed: Set[int] = set()

        # Precompute allowed token ids once
        vocab_size = getattr(getattr(tokenizer, "vocab", None), "__len__", lambda: None)()
        if vocab_size is None:
            vocab_size = tokenizer.vocab_size  # fallback

        for tid in range(vocab_size):
            s = tokenizer.decode([tid], skip_special_tokens=False)
            if all(ord(ch) < 128 for ch in s):
                if allow_controls:
                    self.allowed.add(tid)
                else:
                    # Printable ASCII + common whitespace
                    if all(ch in "\t\n\r" or ord(ch) >= 0x20 for ch in s):
                        self.allowed.add(tid)

        if always_allow:
            self.allowed.update(always_allow)

    def __call__(self, input_ids, scores):
        mask = self.torch.full_like(scores, float("-inf"))
        if self.allowed:
            idxs = list(self.allowed)
            mask[:, idxs] = scores[:, idxs]
            return mask
        return scores

