# formatConstraints.py
from typing import Dict, Callable
from transformers import PreTrainedTokenizerBase
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
    build_token_enforcer_tokenizer_data,
)

def _allowed_depths(curr_depth: int) -> list[int]:
    # Enforce: depth >= -1 and currDepth + depth <= 0  → depth <= -currDepth
    max_depth = max(0, -curr_depth)
    return [-1] + list(range(0, max_depth + 1))

def depth_schema(curr_depth: int) -> Dict:
    """
    Use enum instead of minimum/maximum because some lm-format-enforcer versions
    don't apply numeric bounds. Enum is strictly enforced.
    """
    return {
        "type": "object",
        "properties": {
            "annotation": {"type": "string", "minLength": 1},
            "depth": {"type": "integer", "enum": _allowed_depths(curr_depth)},
        },
        "required": ["annotation", "depth"],
        "additionalProperties": False,
    }

def make_prefix_allowed_tokens_fn(
    tokenizer: PreTrainedTokenizerBase, curr_depth: int
) -> Callable[[int, "torch.Tensor"], list[int]]:
    """
    Returns a Transformers-compatible prefix_allowed_tokens_fn that strictly guides
    decoding to the JSON schema (dynamic maximum derived from curr_depth).
    """
    parser = JsonSchemaParser(depth_schema(curr_depth))
    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer,
        parser
    )
