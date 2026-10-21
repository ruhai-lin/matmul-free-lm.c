"""
Run simply with
$ pytest
"""
import os
import subprocess

import pytest
import torch

from model import ModelArgs, MatMulFreeLM
from export import export, load_checkpoint

window = 128

def nll_of(model, tokens):
    # mean next-token NLL of the PyTorch model, state reset per window
    with torch.no_grad():
        model(tokens[:, :-1], tokens[:, 1:])
    return model.last_loss.item()

def nll_of_c(model, tokens, tmp_path):
    # same, but computed by run.c through test.c
    export(model, str(tmp_path / "model.bin"))
    tokens.to(torch.int32).numpy().tofile(tmp_path / "tokens.bin")
    subprocess.check_call(["gcc", "-O3", "-o", str(tmp_path / "test"), "test.c", "-lm"])
    out = subprocess.check_output([str(tmp_path / "test"), str(tmp_path / "model.bin"), str(tmp_path / "tokens.bin"), str(window)], text=True)
    return float(out.split()[-1])

def test_runc_random(tmp_path):
    # a randomly initialized model needs no data or checkpoint
    torch.manual_seed(42)
    model = MatMulFreeLM(ModelArgs()).eval()
    tokens = torch.randint(0, 512, (8, window + 1))
    assert abs(nll_of(model, tokens) - nll_of_c(model, tokens, tmp_path)) < 1e-3

@pytest.mark.skipif(not os.path.exists("out/ckpt.pt"), reason="no trained checkpoint in out/")
def test_runc_trained(tmp_path):
    import numpy as np
    model = load_checkpoint("out/ckpt.pt").eval()
    val = np.memmap("data/tok512/data00.bin", dtype=np.uint16, mode="r")
    tokens = torch.from_numpy(val[:8 * (window + 1)].astype(np.int64)).view(8, window + 1)
    assert abs(nll_of(model, tokens) - nll_of_c(model, tokens, tmp_path)) < 1e-3
