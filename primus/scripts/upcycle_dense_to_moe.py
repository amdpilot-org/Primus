#!/usr/bin/env python3
"""Upcycle a dense checkpoint into an MoE checkpoint.

This script takes a dense model checkpoint (e.g. Qwen/Qwen3-32B) and
creates an MoE variant by splitting each FFN layer into num_experts
expert MLPs, keeping top-k routing.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def split_dense_ffn_to_experts(
    state_dict: Dict[str, torch.Tensor],
    num_experts: int = 32,
    top_k: int = 8,
    router_noise: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Split dense FFN weights into expert MLP weights.

    Each dense FFN layer (gate_proj, up_proj, down_proj) is split
    along the output dimension into ``num_experts`` shards.  The
    resulting state dict contains keys compatible with Megatron-LM
    MoE checkpoints.
    """
    moe_state: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if "mlp.gate_proj" in key or "mlp.up_proj" in key or "mlp.down_proj" in key:
            # Split the dense weight into expert shards
            hidden = tensor.shape[0] if "down_proj" not in key else tensor.shape[1]
            shard_size = hidden // num_experts
            for e in range(num_experts):
                expert_key = key.replace("mlp.", f"mlp.experts.{e}.")
                if "down_proj" in key:
                    moe_state[expert_key] = tensor[:, e * shard_size : (e + 1) * shard_size].clone()
                else:
                    moe_state[expert_key] = tensor[e * shard_size : (e + 1) * shard_size, :].clone()
        else:
            moe_state[key] = tensor

    # Add router gate weights if not present
    first_layer_key = next((k for k in moe_state if "layers.0" in k), None)
    if first_layer_key is not None:
        prefix = first_layer_key.split("layers.")[0]
        device = next(iter(moe_state.values())).device
        dtype = next(iter(moe_state.values())).dtype
        # Infer hidden size from an up_proj weight
        up_key = next(k for k in moe_state if "experts.0.up_proj" in k)
        ffn_hidden = moe_state[up_key].shape[0]
        for layer_idx in range(64):
            router_key = f"{prefix}layers.{layer_idx}.mlp.router.gate_weight"
            if router_key not in moe_state:
                # Initialize router gate with small random values
                moe_state[router_key] = torch.randn(
                    num_experts, ffn_hidden, device=device, dtype=dtype
                ) * 0.02

    logger.info("Upcycled dense checkpoint to %d experts (top-%d)", num_experts, top_k)
    return moe_state


def upcycle_checkpoint(
    input_path: str,
    output_path: str,
    num_experts: int = 32,
    top_k: int = 8,
    expected_gate_entropy: float = math.log(8),
) -> None:
    """Load a dense checkpoint, upcycle to MoE, and save."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (safetensors or pytorch bin)
    state_dict: Dict[str, Any] = {}
    safetensor_files = sorted(input_path.glob("*.safetensors"))
    if safetensor_files:
        for st_file in safetensor_files:
            logger.info("Loading %s", st_file)
            state_dict.update(load_file(st_file))
    else:
        bin_files = sorted(input_path.glob("pytorch_model*.bin"))
        for bf in bin_files:
            logger.info("Loading %s", bf)
            state_dict.update(torch.load(bf, map_location="cpu"))

    moe_state = split_dense_ffn_to_experts(state_dict, num_experts=num_experts, top_k=top_k)

    # Save upcycled checkpoint
    index: Dict[str, Any] = {"metadata": {"total_size": 0}, "weight_map": {}}
    shard_size = 5e9  # bytes per shard (~5GB)
    current_shard: Dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0

    for key, tensor in sorted(moe_state.items()):
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > shard_size and current_shard:
            shard_name = f"model-{shard_idx:05d}-of-?????.safetensors"
            save_file(current_shard, output_path / shard_name.replace("?????", f"{shard_idx:05d}"))
            current_shard = {}
            current_size = 0
            shard_idx += 1
        current_shard[key] = tensor
        index["weight_map"][key] = shard_name if current_size == 0 else shard_name
        current_size += tensor_size

    if current_shard:
        final_shard_name = f"model-{shard_idx:05d}-of-{shard_idx:05d}.safetensors"
        save_file(current_shard, output_path / final_shard_name)

    # Update index with actual shard names
    for key in index["weight_map"]:
        for i in range(shard_idx + 1):
            shard_file = output_path / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
            if shard_file.exists():
                shard_state = load_file(shard_file)
                if key in shard_state:
                    index["weight_map"][key] = f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
                    break

    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_experts": num_experts,
        "num_layers": 64,
        "hidden_size": 8192,
        "moe_router_topk": top_k,
        "expected_gate_entropy": expected_gate_entropy,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved upcycled MoE checkpoint to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upcycle dense checkpoint to MoE")
    parser.add_argument("--input-path", required=True, help="Path to dense checkpoint")
    parser.add_argument("--output-path", required=True, help="Path to save MoE checkpoint")
    parser.add_argument("--num-experts", type=int, default=32, help="Number of MoE experts")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k routing")
    parser.add_argument(
        "--expected-gate-entropy",
        type=float,
        default=math.log(8),
        help="Expected gate entropy (default ln(8))",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    upcycle_checkpoint(
        args.input_path,
        args.output_path,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expected_gate_entropy=args.expected_gate_entropy,
    )


if __name__ == "__main__":
    main()
