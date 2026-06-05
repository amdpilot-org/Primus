#!/usr/bin/env python3
"""Standalone helper to upcycle a dense checkpoint into an MoE checkpoint.

Orchestrator integration:
- Accepts --dense-dir and --moe-dir.
- Copies dense FFN weights into every expert slice (naïve warm-start).
- Saves a new MoE-compatible checkpoint that Megatron can load with
  moe_use_upcycling=true.
"""
import argparse
import gc
import json
import os
import shutil
import sys

import torch


def _copy_ffn_to_experts(dense_state, num_experts, moe_ffn_hidden_size):
    """Split/copy dense FFN weights into `num_experts` expert MLPs."""
    upcycled = {}
    for k, v in dense_state.items():
        if "mlp" in k or "feed_forward" in k:
            # heuristic: replicate dense MLP for every expert slot
            for eidx in range(num_experts):
                ek = k.replace("mlp", f"experts.expert_{eidx}").replace(
                    "feed_forward", f"experts.expert_{eidx}"
                )
                upcycled[ek] = v.clone()
        else:
            upcycled[k] = v
    return upcycled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-dir", required=True, help="Path to dense checkpoint dir")
    ap.add_argument("--moe-dir", required=True, help="Path to output MoE checkpoint dir")
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--moe-ffn-hidden-size", type=int, default=768)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    dense_dir = args.dense_dir
    moe_dir = args.moe_dir
    os.makedirs(moe_dir, exist_ok=True)

    # Discover latest iteration checkpoint
    ckpt_files = [f for f in os.listdir(dense_dir) if f.startswith("mp_rank") and f.endswith(".pt")]
    if not ckpt_files:
        raise FileNotFoundError(f"No Megatron checkpoint files found in {dense_dir}")

    for fname in sorted(ckpt_files):
        src_path = os.path.join(dense_dir, fname)
        dst_path = os.path.join(moe_dir, fname)
        obj = torch.load(src_path, map_location="cpu")
        model_state = obj.get("model", obj)
        upcycled_state = _copy_ffn_to_experts(
            model_state, args.num_experts, args.moe_ffn_hidden_size
        )
        out = {"model": upcycled_state}
        if "iteration" in obj:
            out["iteration"] = 1  # reset iteration after upcycling
        torch.save(out, dst_path)
        print(f"Upcycled {fname} -> {dst_path}")

    # Copy metadata / args if present
    for meta in ("latest_checkpointed_iteration.txt", "iter_0000001"):
        src_meta = os.path.join(dense_dir, meta)
        if os.path.exists(src_meta):
            dst_meta = os.path.join(moe_dir, meta)
            if os.path.isdir(src_meta):
                shutil.copytree(src_meta, dst_meta, dirs_exist_ok=True)
            else:
                shutil.copy2(src_meta, dst_meta)

    # Write a small manifest so the harness can assert topology
    manifest = {
        "source_dense": dense_dir,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
    }
    with open(os.path.join(moe_dir, "upcycle_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("Upcycling complete. Manifest written.")


if __name__ == "__main__":
    main()
