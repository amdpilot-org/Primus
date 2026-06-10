#!/usr/bin/env python3
"""Dense Qwen3 checkpoint to 32E-A8 MoE upcycling scaffold.

This entrypoint intentionally keeps the conversion logic explicit and inspectable:
non-FFN tensors are copied once, FFN tensors are copied to each expert slot, and
router/gate tensors are initialized separately by the training job.  Stage0 uses
this script as a validated starting point; full checkpoint materialization is left
to the executor because it requires the large cached Qwen3-32B state dict.

State-dict conversion flow (executor stage):
  dense_state = torch.load(dense_checkpoint_path, weights_only=False)["state_dict"]
  moe_state_dict = {}
  for name, param in dense_state.items():
      for moe_name in moe_tensor_names(name):
          moe_state_dict[moe_name] = param.clone()
  torch.save(moe_state_dict, output_path / "upcycled_moe_state_dict.pt")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

DEFAULT_NUM_EXPERTS = 32
DEFAULT_TOP_K = 8
DEFAULT_EXPECTED_GATE_ENTROPY = 2.08
FFN_MARKERS = ("mlp.", "feed_forward", "w1", "w2", "w3")


def is_ffn_tensor(name: str) -> bool:
    return any(marker in name for marker in FFN_MARKERS)


def moe_tensor_names(dense_name: str, num_experts: int = DEFAULT_NUM_EXPERTS) -> Iterable[str]:
    if not is_ffn_tensor(dense_name):
        yield dense_name
        return
    for expert_id in range(num_experts):
        yield f"experts.{expert_id}.{dense_name}"


def build_plan(tensor_names: Iterable[str], num_experts: int = DEFAULT_NUM_EXPERTS) -> Dict[str, list[str]]:
    return {name: list(moe_tensor_names(name, num_experts)) for name in tensor_names}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan Qwen3 dense-to-MoE upcycling")
    parser.add_argument("--dense-checkpoint", default="Qwen/Qwen3-32B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dry-run", action="store_true", help="write only conversion metadata")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dense_checkpoint": args.dense_checkpoint,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "expected_gate_entropy": DEFAULT_EXPECTED_GATE_ENTROPY,
        "expert_init": "copy_dense_ffn",
        "gate_init": "scratch",
        "dry_run": bool(args.dry_run),
    }
    (output / "upcycle_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(metadata, sort_keys=True))
        return 0
    raise SystemExit("Full tensor materialization must be run by the executor with the mounted Qwen3-32B cache")


if __name__ == "__main__":
    raise SystemExit(main())
