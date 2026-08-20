"""Compatibility facade for the focused REVE head modules.

Keep this module as the stable import path used by experiments and tests. The
implementation lives in reve_contract, reve_head_math, and reve_last_tuned so
each file has one clear responsibility.
"""

from __future__ import annotations

from typing import Any

try:
    from .reve_contract import *  # noqa: F401,F403
    from .reve_contract import _MISSING, _get_path, _last_tuned_group_mismatches, _path_mismatches
    from .reve_head_math import *  # noqa: F401,F403
    from .reve_head_math import (
        _infer_embed_dim,
        _reject_mask_kwargs,
        _unwrap_reve_module,
        _validate_final_tokens,
        _validate_query_shape,
        _validate_query_token,
    )
    from .reve_last_tuned import *  # noqa: F401,F403
    from .reve_last_tuned import (
        _NEURALBENCH_TRAIN_DUMMY_CONTEXT,
        _resolve_last_tuned_model,
        _tensor_sha256,
    )
    from . import reve_head_math as _head_math
except ImportError:
    from reve_contract import *  # noqa: F401,F403
    from reve_contract import _MISSING, _get_path, _last_tuned_group_mismatches, _path_mismatches
    from reve_head_math import *  # noqa: F401,F403
    from reve_head_math import (
        _infer_embed_dim,
        _reject_mask_kwargs,
        _unwrap_reve_module,
        _validate_final_tokens,
        _validate_query_shape,
        _validate_query_token,
    )
    from reve_last_tuned import *  # noqa: F401,F403
    from reve_last_tuned import (
        _NEURALBENCH_TRAIN_DUMMY_CONTEXT,
        _resolve_last_tuned_model,
        _tensor_sha256,
    )
    import reve_head_math as _head_math


def make_upstream_reve_wrapper(*, variant: str, dropout: float = DEFAULT_UPSTREAM_DROPOUT) -> Any:
    """Build the adapter while preserving the historical monkeypatch seam."""

    _head_math.initialize_last_tuned_query = initialize_last_tuned_query
    return _head_math.make_upstream_reve_wrapper(variant=variant, dropout=dropout)


__all__ = [
    "AdapterContractError",
    "AllLayerReveEncoder",
    "DEFAULT_UPSTREAM_DROPOUT",
    "HEAD_VARIANTS",
    "LOCAL_HEAD_VARIANTS",
    "LAST_TUNED_HEAD_ARCHITECTURE",
    "LAST_TUNED_HEAD_SOURCE",
    "LAST_TUNED_INITIAL_ALPHA",
    "LAST_TUNED_BASE_LR",
    "LAST_TUNED_QUERY_LR",
    "LAST_TUNED_SCHEDULER_DIV_FACTOR",
    "LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR",
    "LAST_TUNED_SCHEDULER_MAX_LR",
    "LAST_TUNED_SCHEDULER_PCT_START",
    "LAST_TUNED_WEIGHT_DECAY",
    "LAST_TUNED_PROTOCOL_CLASS",
    "LAST_TUNED_PROTOCOL_VARIANTS",
    "OFFICIAL_HEAD_VARIANTS",
    "PROTOCOL_CONTRACT",
    "ProtocolMismatchError",
    "RMSNorm",
    "MeanLinearCopyHead",
    "UPSTREAM_HEAD_HIDDEN_SIZE",
    "UPSTREAM_HEAD_INIT_CUTOFF",
    "UPSTREAM_HEAD_INIT_STD",
    "UPSTREAM_REVE_COMMIT",
    "UPSTREAM_REVE_FILE_HASHES",
    "UPSTREAM_REVE_REPOSITORY",
    "UPSTREAM_HEAD_VARIANTS",
    "UpstreamReveHead",
    "UpstreamReveHeadModel",
    "build_last_tuned_optimizer_config",
    "concatenate_all_layers",
    "initialize_last_tuned_query",
    "last_tuned_optimizer_metadata",
    "make_upstream_reve_wrapper",
    "runtime_metadata",
    "source_lock_metadata",
    "validate_head_variant",
    "validate_local_head_variant",
    "validate_last_tuned_protocol",
    "validate_official_head_variant",
    "validate_official_protocol",
    "validate_upstream_head_variant",
    "verify_upstream_source_hashes",
]
