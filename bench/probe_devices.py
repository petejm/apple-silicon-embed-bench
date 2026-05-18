#!/usr/bin/env python3
"""Probe actual device placement for a CoreML mlpackage.

`MLComputeUnits.cpuAndNeuralEngine` is a HINT, not a guarantee. Apple's
runtime can silently fall back to GPU/CPU for unsupported ops. Without
verification, an "ANE is slow" claim could be "CPU-via-ANE-fallback is slow"
— a totally different finding.

This script uses `MLComputePlan.loadModelAsset` (Apple's introspection API,
macOS 14+) to enumerate which device each op was placed on, then writes a
summary JSON the bench scripts read.

Runs in a separate process to isolate from the destructor crash class.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "bge-small-en-v1.5.mlpackage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", choices=["ane", "gpu", "cpu", "all"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import coremltools as ct

    cu_map = {
        "ane": ct.ComputeUnit.CPU_AND_NE,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "all": ct.ComputeUnit.ALL,
    }
    cu = cu_map[args.compute]

    try:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=MODEL,
            compute_units=cu,
        )
    except Exception as e:
        result = {
            "compute_unit_request": args.compute,
            "error": f"{type(e).__name__}: {e}",
            "device_placement_available": False,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"WARN: device probe failed ({type(e).__name__}); wrote stub to {args.out}",
              flush=True)
        sys.exit(0)

    counts = {}
    try:
        for op in plan.model_structure.program.functions["main"].block.operations:
            device = plan.get_compute_device_usage_for_mlprogram_operation(op)
            name = type(device).__name__ if device else "Unknown"
            counts[name] = counts.get(name, 0) + 1
    except Exception as e:
        counts = {"error": f"{type(e).__name__}: {e}"}

    result = {
        "compute_unit_request": args.compute,
        "device_placement_available": True,
        "op_count_by_device": counts,
        "total_ops": sum(v for k, v in counts.items() if isinstance(v, int)),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe/{args.compute}] {counts}  → {args.out}", flush=True)

    # See bench_coreml.py for rationale; CoreML destructor race on Py3.13+.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
