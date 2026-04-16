###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Paged Stashing Patches

Patches for enabling paged stashing of MoE expert activations in Megatron-LM.

Paged stashing decouples oversized static buffers (needed for CUDA graph
compatibility with token-dropless MoE) from the backward-pass activation
storage. Forward activations are "stashed" into a compact paged buffer
and "restored" during backward, substantially reducing memory fragmentation.

Reference: NVIDIA/Megatron-LM PR #2690 (merged into dev branch).

The patch applies a unified diff stored alongside this file
(``paged_stashing.patch``) to the Megatron-LM source tree at import time.
Files touched:
    - megatron/core/transformer/moe/paged_stash.py         (new)
    - megatron/core/transformer/moe/experts.py
    - megatron/core/transformer/moe/token_dispatcher.py
    - megatron/core/transformer/transformer_config.py
    - megatron/core/full_cuda_graph.py
    - megatron/core/models/gpt/gpt_model.py
    - megatron/core/pipeline_parallel/schedules.py
    - megatron/core/transformer/multi_token_prediction.py
    - megatron/training/training.py
    - megatron/training/utils.py
    - megatron/core/extensions/transformer_engine_spec_provider.py
    - megatron/core/models/backends.py
    - megatron/core/models/common/model_chunk_schedule_plan.py
"""

import logging
import os
import subprocess
from pathlib import Path

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

logger = logging.getLogger(__name__)

PATCH_FILE = Path(__file__).parent / "paged_stashing.patch"


def _find_megatron_root() -> Path:
    """Locate the Megatron-LM source root (the directory containing ``megatron/``)."""
    import megatron

    megatron_pkg = Path(megatron.__file__).resolve().parent
    return megatron_pkg.parent


def _patch_already_applied(megatron_root: Path) -> bool:
    """Check whether the paged stashing module is already present."""
    paged_stash_path = megatron_root / "megatron" / "core" / "transformer" / "moe" / "paged_stash.py"
    return paged_stash_path.exists()


@register_patch(
    "megatron.moe.paged_stashing",
    backend="megatron",
    phase="setup",
    description="Apply paged stashing patch to Megatron-LM source tree",
    condition=lambda ctx: getattr(get_args(ctx), "moe_paged_stash", False),
    priority=10,
)
def patch_paged_stashing(ctx: PatchContext):
    """
    Apply the paged stashing git diff patch to the Megatron-LM source tree.

    The patch is only applied when ``--moe-paged-stash`` is enabled. It is
    idempotent: if ``paged_stash.py`` already exists the patch is skipped.

    The patch file ``paged_stashing.patch`` lives next to this module and
    contains the unified diff from the base Megatron v0.16.0 commit (d3528a2)
    to the fully integrated paged stashing feature (synced with NVIDIA PR #2690).
    """
    if not PATCH_FILE.exists():
        log_rank_0(
            f"[Patch:megatron.moe.paged_stashing]   WARNING: patch file not found at {PATCH_FILE}"
        )
        return

    megatron_root = _find_megatron_root()

    if _patch_already_applied(megatron_root):
        log_rank_0(
            "[Patch:megatron.moe.paged_stashing]   Paged stashing already present in Megatron-LM, skipping patch"
        )
        return

    log_rank_0(
        f"[Patch:megatron.moe.paged_stashing]   Applying paged stashing patch to {megatron_root}"
    )

    try:
        result = subprocess.run(
            ["git", "apply", "--stat", str(PATCH_FILE)],
            cwd=str(megatron_root),
            capture_output=True,
            text=True,
        )
        log_rank_0(
            f"[Patch:megatron.moe.paged_stashing]   Patch stat:\n{result.stdout}"
        )

        result = subprocess.run(
            ["git", "apply", "--check", str(PATCH_FILE)],
            cwd=str(megatron_root),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log_rank_0(
                f"[Patch:megatron.moe.paged_stashing]   WARNING: patch check failed: {result.stderr}"
            )
            return

        result = subprocess.run(
            ["git", "apply", str(PATCH_FILE)],
            cwd=str(megatron_root),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log_rank_0(
                "[Patch:megatron.moe.paged_stashing]   Paged stashing patch applied successfully"
            )
        else:
            log_rank_0(
                f"[Patch:megatron.moe.paged_stashing]   WARNING: patch apply failed: {result.stderr}"
            )
    except FileNotFoundError:
        log_rank_0(
            "[Patch:megatron.moe.paged_stashing]   WARNING: git not available, cannot apply patch"
        )
