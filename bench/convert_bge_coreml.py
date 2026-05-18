#!/usr/bin/env python3
"""Convert bge-small-en-v1.5 to CoreML (.mlpackage) with FP16 + ANE target.

Fixed shape: seq_len=512. Mean pooling baked into the graph for parity with
sentence-transformers and llama.cpp --pooling mean.
"""
import os
import time
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import coremltools as ct
import numpy as np

MODEL_ID = "BAAI/bge-small-en-v1.5"
SEQ_LEN = 512
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-en-v1.5.mlpackage")


class BgeEmbedder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # torchscript=True returns a tuple: (last_hidden_state, pooler_output, ...)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        norm = pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        return pooled / norm


def main():
    print(f"[convert] loading {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID, return_dict=False)
    base.train(False)
    wrapped = BgeEmbedder(base)
    wrapped.train(False)

    sample = tok(
        "Sample text for tracing",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
    )
    print("[convert] tracing", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (sample["input_ids"], sample["attention_mask"]))

    print("[convert] CoreML conversion (FP16, CPU+NE target)", flush=True)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=sample["input_ids"].shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=sample["attention_mask"].shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    print(f"[convert] converted in {time.time()-t0:.1f}s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.isdir(OUT):
        import shutil
        shutil.rmtree(OUT)
    mlmodel.save(OUT)
    print(f"[convert] saved to {OUT}", flush=True)

    loaded = ct.models.MLModel(OUT, compute_units=ct.ComputeUnit.CPU_AND_NE)
    pred = loaded.predict({
        "input_ids": sample["input_ids"].to(torch.int32).numpy(),
        "attention_mask": sample["attention_mask"].to(torch.int32).numpy(),
    })
    out_key = list(pred.keys())[0]
    vec = pred[out_key].squeeze()
    print(f"[convert] sanity vec[:5]={vec[:5]} norm={np.linalg.norm(vec):.4f}", flush=True)


if __name__ == "__main__":
    main()
