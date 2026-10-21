import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    # default hyperparameters for the 257K TinyStories model
    dim: int = 64
    n_layers: int = 4
    vocab_size: int = 512
    hidden_dim: int = 160
    norm_eps: float = 1e-6
    max_seq_len: int = 512
    use_lower_bound: bool = True
    tie_embeddings: bool = False  # share the token embedding with the (ternarized) classifier


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def activation_quant(x):
    # per-token absmax to int8, round half away from zero (as the upstream kernel does)
    scale = 127.0 / x.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    y = x * scale
    y = torch.sign(y) * torch.floor(y.abs() + 0.5)
    return y.clamp(-128, 127) / scale


def weight_quant(w):
    # per-tensor absmean to ternary {-1, 0, +1}.
    # By the end of training ~30% of the latent weights sit within float32 error of a rounding
    # threshold, so a float32 mean (whose value depends on reduction order) makes the codes
    # differ between eager, torch.compile and export. A float64 mean makes them agree.
    scale = (1.0 / w.double().abs().mean().clamp(min=1e-5)).float()
    return (w * scale).round().clamp(-1, 1) / scale


class BitLinear(nn.Linear):
    """RMSNorm -> int8 activations -> ternary weights. Quantization uses a
    straight-through estimator, so the stored weights are the latent fp32 ones."""

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6):
        super().__init__(in_features, out_features, bias=False)
        self.norm = RMSNorm(in_features, eps)

    def forward(self, x):
        x = self.norm(x)
        x = x + (activation_quant(x) - x).detach()
        w = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x, w)


try:
    # optional: the fused Triton scan from flash-linear-attention, much faster for training
    from fla.ops.hgrn import chunk_hgrn
except ImportError:
    chunk_hgrn = None


def hgrn_scan(x, log_f, h):
    # h_t = f_t * h_{t-1} + x_t, elementwise; x, log_f: (B, T, D), h: (B, D)
    if chunk_hgrn is not None and x.is_cuda and x.shape[1] > 1:
        return chunk_hgrn(x, log_f, initial_state=h, output_final_state=True)
    f = log_f.exp()
    out = torch.empty_like(x)
    for t in range(x.shape[1]):
        h = f[:, t] * h + x[:, t]
        out[:, t] = h
    return out, h


class Mixer(nn.Module):
    """HGRN token mixer: a gated elementwise linear recurrence, no attention."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.i_proj = BitLinear(args.dim, args.dim)
        self.f_proj = BitLinear(args.dim, args.dim)
        self.g_proj = BitLinear(args.dim, args.dim)
        self.g_norm = RMSNorm(args.dim, args.norm_eps)
        self.o_proj = BitLinear(args.dim, args.dim)

    def forward(self, x, lower_bound, h=None):
        # forget gate in log space: f = lower_bound + (1 - lower_bound) * sigmoid(f_proj(x))
        log_f = F.logsigmoid(self.f_proj(x).float())
        if lower_bound is not None:
            log_f = torch.logaddexp(lower_bound.log(), torch.log1p(-lower_bound) + log_f)
        i = F.silu(self.i_proj(x).float()) * (1 - log_f.exp())
        if h is None:
            h = x.new_zeros(x.shape[0], x.shape[2], dtype=torch.float32)
        o, h = hgrn_scan(i, log_f, h)
        o = self.g_norm(self.g_proj(x)) * F.silu(o.type_as(x))
        return self.o_proj(o), h


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = BitLinear(dim, 2 * hidden_dim)
        self.w2 = BitLinear(hidden_dim, dim)

    def forward(self, x):
        gate, up = self.w13(x).chunk(2, -1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.mixer_norm = RMSNorm(args.dim, args.norm_eps)
        self.mixer = Mixer(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim)

    def forward(self, x, lower_bound, h=None):
        dx, h = self.mixer(self.mixer_norm(x), lower_bound, h)
        x = x + dx
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, h


class MatMulFreeLM(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(Block(layer_id, params))
        # learned per-layer lower bound on the forget gate, increasing with depth
        self.lower_bounds = nn.Parameter(torch.zeros(params.n_layers, params.dim)) if params.use_lower_bound else None
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = BitLinear(params.dim, params.vocab_size)
        if params.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # init all weights
        self.apply(self._init_weights)
        for layer in self.layers:
            for p in (layer.mixer.i_proj, layer.mixer.f_proj, layer.mixer.g_proj, layer.mixer.o_proj):
                torch.nn.init.xavier_uniform_(p.weight, gain=2 ** -2.5)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, state: Optional[list] = None) -> torch.Tensor:
        # state: optional list of per-layer recurrent states, updated in place (for generation)
        h = self.tok_embeddings(tokens)
        if self.lower_bounds is not None:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]
        for i, layer in enumerate(self.layers):
            # the first layer has no lower bound (it would be zero anyway)
            lower_bound = lower_bounds[i] if self.lower_bounds is not None and i > 0 else None
            h, s = layer(h, lower_bound, state[i] if state is not None else None)
            if state is not None:
                state[i] = s
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings + lower bounds decay, all norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        The recurrent state is carried forward, so each new token costs O(1).
        """
        state = [None] * self.n_layers
        logits = self(idx, state=state)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            logits = self(idx_next, state=state)

        return idx


def from_upstream(state_dict):
    """Rename an upstream HGRNBitForCausalLM state dict (ridgerchu/matmulfreellm) to ours."""
    rename = [('model.embeddings.', 'tok_embeddings.'), ('model.lower_bounds', 'lower_bounds'),
              ('model.norm.', 'norm.'), ('model.layers.', 'layers.'), ('lm_head.', 'output.'),
              ('.attn_norm.', '.mixer_norm.'), ('.attn.', '.mixer.'), ('.mlp_norm.', '.ffn_norm.'),
              ('.mlp.gate_proj.', '.feed_forward.w13.'), ('.mlp.down_proj.', '.feed_forward.w2.')]
    out = {}
    for k, v in state_dict.items():
        for a, b in rename:
            k = k.replace(a, b)
        out[k] = v
    return out


def to_upstream(state_dict):
    """Inverse of from_upstream: our state dict in HGRNBitForCausalLM names."""
    out = {}
    for k, v in state_dict.items():
        for a, b in [('.mixer_norm.', '.attn_norm.'), ('.mixer.', '.attn.'), ('.ffn_norm.', '.mlp_norm.'),
                     ('.feed_forward.w13.', '.mlp.gate_proj.'), ('.feed_forward.w2.', '.mlp.down_proj.')]:
            k = k.replace(a, b)
        if k.startswith('tok_embeddings.'):
            k = 'model.embeddings.' + k[len('tok_embeddings.'):]
        elif k.startswith('output.'):
            k = 'lm_head.' + k[len('output.'):]
        elif k.startswith(('layers.', 'norm.', 'lower_bounds')):
            k = 'model.' + k
        out[k] = v
    return out
