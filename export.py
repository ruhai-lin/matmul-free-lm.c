"""
Export a model to the .bin format read by run.c.

The file is a header, then all fp32 tensors, then the ternary weights as int8.
The int8 section is exactly what gets programmed into crossbar arrays.

Usage:
    python export.py out/model.bin --checkpoint out/ckpt.pt
    python export.py mmfreelm_370m.bin --hf ridger/MMfreeLM-370M
    python export.py out/model.bin --checkpoint out/ckpt.pt --hf-dir out/hf   # also write a Hugging Face model
"""
import os
import json
import struct
import argparse

import numpy as np
import torch

from model import ModelArgs, MatMulFreeLM, BitLinear, from_upstream, to_upstream, weight_quant

MAGIC = 0x4c464d4d # "MMFL"
VERSION = 1


def ternary(w):
    # the integer codes and the dequant scale of model.weight_quant
    wq = weight_quant(w.detach().float())
    return torch.sign(wq).to(torch.int8), wq.abs().max().item()


@torch.no_grad()
def export(model, filepath):
    p = model.params
    layers = model.layers
    if model.lower_bounds is not None:
        lower_bounds = model.lower_bounds.softmax(0)
        lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]
    else:
        lower_bounds = torch.zeros(p.n_layers, p.dim)

    floats = [model.tok_embeddings.weight, lower_bounds]
    floats += [l.mixer_norm.weight for l in layers]
    floats += [l.mixer.g_norm.weight for l in layers]
    floats += [l.ffn_norm.weight for l in layers]
    floats += [model.norm.weight]
    bitlinears = []
    for l in layers:
        bitlinears += [l.mixer.i_proj, l.mixer.f_proj, l.mixer.g_proj, l.mixer.o_proj,
                       l.feed_forward.w13, l.feed_forward.w2]
    bitlinears += [model.output]
    codes = []
    for b in bitlinears:
        c, scale = ternary(b.weight)
        floats += [b.norm.weight, torch.tensor([scale])]
        codes.append(c)

    floats = torch.cat([t.detach().float().cpu().view(-1) for t in floats]).numpy()
    codes = torch.cat([c.cpu().view(-1) for c in codes]).numpy()
    with open(filepath, 'wb') as f:
        f.write(struct.pack('8i', MAGIC, VERSION, p.dim, p.hidden_dim, p.n_layers,
                            p.vocab_size, p.max_seq_len, len(floats)))
        f.write(floats.astype('<f4').tobytes())
        f.write(codes.astype(np.int8).tobytes())
    print(f"wrote {filepath}: {len(floats):,} fp32 values, {len(codes):,} ternary weights")


@torch.no_grad()
def snap(w, margin=1e-3):
    # ~30% of the latent weights of a trained model sit within float error of a rounding
    # threshold, so another implementation (e.g. upstream, with a float32 mean) would round
    # some of them differently. Push those weights `margin` away from the threshold, on the
    # side of the code they have here; the codes and the scale stay the same.
    s = 1.0 / w.double().abs().mean().clamp(min=1e-5)
    codes = torch.sign(weight_quant(w))
    v = (w.double() * s).abs()
    near = (v - 0.5).abs() < margin
    target = torch.where(codes != 0, 0.5 + margin, 0.5 - margin) / s
    w = torch.where(near, torch.sign(w) * target, w.double()).float()
    assert torch.equal(torch.sign(weight_quant(w)), codes)
    return w


def export_hf(model, out_dir, tokenizer_model):
    """Write the model as an HGRNBitForCausalLM Hugging Face directory, loadable with the
    upstream code (ridgerchu/matmulfreellm) and with --hf above."""
    import shutil
    from safetensors.torch import save_file
    p = model.params
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "architectures": ["HGRNBitForCausalLM"], "model_type": "hgrn_bit",
        "attn_mode": "fused_recurrent", "num_heads": 1, "expand_ratio": 1,
        "use_short_conv": False, "conv_size": 4, "share_conv_kernel": True,
        "hidden_size": p.dim, "num_hidden_layers": p.n_layers, "intermediate_size": p.hidden_dim,
        "hidden_ratio": 4, "hidden_act": "swish", "vocab_size": p.vocab_size,
        "max_position_embeddings": p.max_seq_len, "rms_norm_eps": p.norm_eps,
        "use_lower_bound": p.use_lower_bound, "tie_word_embeddings": p.tie_embeddings,
        "bos_token_id": 1, "eos_token_id": 2, "initializer_range": 0.02,
        "use_cache": True, "fuse_cross_entropy": False, "torch_dtype": "float32",
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, 'generation_config.json'), 'w') as f:
        json.dump({"bos_token_id": 1, "eos_token_id": 2}, f, indent=2)
    bitlinear_weights = {n + '.weight' for n, m in model.named_modules() if isinstance(m, BitLinear)}
    if p.tie_embeddings:
        bitlinear_weights.add('tok_embeddings.weight')  # it is also the classifier
    state_dict = {k: (snap(v) if k in bitlinear_weights else v).detach().cpu().contiguous()
                  for k, v in model.state_dict().items()}
    state_dict = to_upstream(state_dict)
    if p.tie_embeddings:
        del state_dict['lm_head.weight']  # shared with model.embeddings.weight
    save_file(state_dict, os.path.join(out_dir, 'model.safetensors'), metadata={"format": "pt"})
    shutil.copy(tokenizer_model, os.path.join(out_dir, 'tokenizer.model'))
    with open(os.path.join(out_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump({"tokenizer_class": "LlamaTokenizer", "add_bos_token": True, "add_eos_token": False,
                   "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "legacy": True}, f, indent=2)
    with open(os.path.join(out_dir, 'special_tokens_map.json'), 'w') as f:
        json.dump({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, f, indent=2)
    print(f"wrote {out_dir}")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    model = MatMulFreeLM(ModelArgs(**checkpoint['model_args']))
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    return model


def load_hf(model_path):
    """HGRNBitForCausalLM in Hugging Face format (config.json + model.safetensors), e.g. Ridger's
    released models: a local directory, or a hub id such as ridger/MMfreeLM-370M."""
    from safetensors.torch import load_file
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model_path, allow_patterns=["config.json", "model.safetensors", "tokenizer.model"])
    with open(os.path.join(model_path, 'config.json')) as f:
        c = json.load(f)
    assert c['expand_ratio'] == 1 and not c['use_short_conv'], "only the configuration of the released models is supported"
    hidden_dim = c['intermediate_size'] or 256 * ((int(c['hidden_size'] * c['hidden_ratio'] * 2 / 3) + 255) // 256)
    args = ModelArgs(dim=c['hidden_size'], n_layers=c['num_hidden_layers'], vocab_size=c['vocab_size'],
                     hidden_dim=hidden_dim, norm_eps=c['rms_norm_eps'], max_seq_len=c['max_position_embeddings'],
                     use_lower_bound=c['use_lower_bound'], tie_embeddings=c.get('tie_word_embeddings', False))
    model = MatMulFreeLM(args)
    state_dict = {k: v.float() for k, v in load_file(os.path.join(model_path, 'model.safetensors')).items()}
    model.load_state_dict(from_upstream(state_dict), strict=False if args.tie_embeddings else True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file written by train.py")
    group.add_argument("--hf", type=str, help="Hugging Face HGRNBit model: a directory or a hub id, e.g. ridger/MMfreeLM-370M")
    parser.add_argument("--hf-dir", type=str, help="also write the model as a Hugging Face directory here")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint) if args.checkpoint else load_hf(args.hf)
    model.eval()
    export(model, args.filepath)
    if args.hf_dir:
        from tinystories import get_tokenizer_model_path
        export_hf(model, args.hf_dir, get_tokenizer_model_path(model.params.vocab_size))
