#!/usr/bin/env python3
"""Upcycle a dense Qwen3 model into an MoE model by replicating FFN layers as expert copies.

Derived from Primus dense-to-MoE upcycling semantics. Loads Qwen/Qwen3-32B dense weights,
creates N expert copies of each FFN layer with small random perturbations to break symmetry,
initializes a top-k router, and writes a Megatron-compatible checkpoint.

Usage:
    python upcycle_dense_to_moe.py --num-experts 32 --topk 8 --fp8-output
"""

import argparse
import math
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Dense-to-MoE upcycle script for Qwen3")
    parser.add_argument(
        "--dense-model-name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Hugging Face model name or local path for the dense model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./upcycled_moe_checkpoint",
        help="Directory to write the upcycled checkpoint",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=32,
        help="Number of MoE experts to create",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Top-k routing value",
    )
    parser.add_argument(
        "--perturbation-std",
        type=float,
        default=0.01,
        help="Standard deviation for weight perturbation to break symmetry",
    )
    parser.add_argument(
        "--fp8-output",
        action="store_true",
        help="Cast and save FP8 weights in the output checkpoint",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=8192,
        help="Hidden size for the target MoE model",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=64,
        help="Number of transformer layers",
    )
    return parser.parse_args()


def load_dense_model(model_name: str):
    """Load the dense model from HF or local path."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("transformers is required to load the dense model")
        sys.exit(1)

    print(f"Loading dense model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    return model


def upcycle_ffn_to_experts(dense_ffn, num_experts: int, perturbation_std: float):
    """Replicate a dense FFN block into N expert copies with small perturbations."""
    experts = []
    base_state = dense_ffn.state_dict()
    for i in range(num_experts):
        expert_state = {}
        for k, v in base_state.items():
            noise = torch.randn_like(v) * perturbation_std
            expert_state[k] = v + noise
        experts.append(expert_state)
    return experts


def init_router_weights(hidden_size: int, num_experts: int, topk: int):
    """Initialize a linear router with scaled random weights."""
    gate = torch.randn(hidden_size, num_experts) / math.sqrt(hidden_size)
    return {"weight": gate}


def save_checkpoint(checkpoint_dir: Path, model_state: dict, fp8_output: bool):
    """Save upcycled checkpoint, optionally casting weights to FP8."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if fp8_output:
        print("Casting weights to FP8 (e4m3fn) before saving")
        fp8_state = {}
        for k, v in model_state.items():
            if v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                fp8_state[k] = v.to(torch.float8_e4m3fn)
            else:
                fp8_state[k] = v
        model_state = fp8_state

    ckpt_path = checkpoint_dir / "model.pt"
    torch.save(model_state, ckpt_path)
    print(f"Saved upcycled checkpoint to {ckpt_path}")


def main():
    args = parse_args()

    dense_model = load_dense_model(args.dense_model_name)
    dense_state = dense_model.state_dict()

    num_experts = args.num_experts
    topk = args.topk
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    print(f"Upcycling to {num_experts} experts, topk={topk}, hidden={hidden_size}, layers={num_layers}")

    upcycled_state = {}

    # Copy non-FFN weights verbatim
    for k, v in dense_state.items():
        if "mlp" not in k.lower():
            upcycled_state[k] = v

    # For each layer, replicate the FFN into expert copies
    for layer_idx in range(num_layers):
        mlp_prefix = f"model.layers.{layer_idx}.mlp"
        experts = []
        for expert_id in range(num_experts):
            expert_state = {}
            for key in ["gate_proj", "up_proj", "down_proj"]:
                dense_key = f"{mlp_prefix}.{key}.weight"
                if dense_key in dense_state:
                    noise = torch.randn_like(dense_state[dense_key]) * args.perturbation_std
                    expert_state[f"expert_{expert_id}.{key}.weight"] = dense_state[dense_key] + noise
            if not expert_state:
                for k, v in dense_state.items():
                    if k.startswith(mlp_prefix):
                        noise = torch.randn_like(v) * args.perturbation_std
                        expert_state[f"expert_{expert_id}.{k[len(mlp_prefix)+1:]}"] = v + noise
            experts.append(expert_state)

        for expert_state in experts:
            upcycled_state.update(expert_state)

        router = init_router_weights(hidden_size, num_experts, topk)
        upcycled_state[f"model.layers.{layer_idx}.mlp.router.weight"] = router["weight"]

    output_dir = Path(args.output_dir)
    save_checkpoint(output_dir, upcycled_state, fp8_output=args.fp8_output)
    print("Upcycling complete.")


if __name__ == "__main__":
    main()
