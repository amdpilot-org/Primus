###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Patches

This package groups patches for Megatron's Mixture-of-Experts (MoE) components:
    - Deprecated MoE layer implementations
    - Primus TopKRouter
    - MoE permutation fusion with Transformer Engine
    - Paged stashing for MoE expert activations (from NVIDIA PR #2690)
    - Fused All-to-All dispatch/combine with DeepEP

Patch modules are discovered and imported automatically by
``primus.backends.megatron.patches``; no explicit imports are required here.
"""
