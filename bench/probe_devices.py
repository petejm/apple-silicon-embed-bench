#!/usr/bin/env python3
"""Probe actual device placement for a CoreML mlpackage.

`MLComputeUnits.cpuAndNeuralEngine` is a HINT, not a guarantee. Apple's
runtime can silently fall back to GPU/CPU for unsupported ops. Without
verification, an "ANE is slow" claim could be "CPU-via-ANE-fallback is slow"
— a totally different finding.

Uses coremltools 9's MLComputePlan introspection: load the compiled model
asset, walk the MIL program operations, ask MLComputePlan which device the
runtime preferred for each. Writes a summary JSON the bench reads.
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
    from coremltools.models.compute_plan import MLComputePlan

    cu_map = {
        "ane": ct.ComputeUnit.CPU_AND_NE,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "all": ct.ComputeUnit.ALL,
    }
    cu = cu_map[args.compute]

    result = {"compute_unit_request": args.compute,
              "device_placement_available": False}

    try:
        # Step 1: load the model so coremltools materializes the compiled
        # .mlmodelc directory we need for MLComputePlan.
        model = ct.models.MLModel(MODEL, compute_units=cu)
        compiled_path = model.get_compiled_model_path()
        # Step 2: load compute plan against the compiled model.
        plan = MLComputePlan.load_from_path(compiled_path, compute_units=cu)
        # Step 3: walk operations, count device assignments.
        program = plan.model_structure.program
        if program is None:
            raise RuntimeError("mlpackage is not an MIL program; cannot introspect")
        counts = {}
        op_total = 0
        # The program has functions[name].block.operations
        for func_name, func in program.functions.items():
            for op in func.block.operations:
                op_total += 1
                # const ops are not dispatched; skip
                if op.operator_name == "const":
                    continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                device_name = type(usage.preferred_compute_device).__name__ if usage else "Unknown"
                counts[device_name] = counts.get(device_name, 0) + 1
        result["device_placement_available"] = True
        result["op_count_by_device"] = counts
        result["op_total"] = op_total
        result["op_const_skipped"] = op_total - sum(counts.values())
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"WARN: device probe ({args.compute}) failed: {result['error']}",
              file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe/{args.compute}] {result.get('op_count_by_device', result.get('error'))}",
          flush=True)

    # CoreML destructor race on Py3.13+; results already on disk.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
