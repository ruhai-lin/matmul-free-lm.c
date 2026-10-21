## matmul-free-lm.c

Train a ternary [MatMul-free language model](https://arxiv.org/abs/2406.02528) in PyTorch, then inference it with one simple C file ([run.c](run.c)). This is [llama2.c](https://github.com/karpathy/llama2.c) with the Llama 2 architecture swapped for HGRN-Bit, the architecture of [ridgerchu/matmulfreellm](https://github.com/ridgerchu/matmulfreellm):

- every linear layer is a `BitLinear`: RMSNorm, then int8 activations times ternary weights in {-1, 0, +1}. A matrix-vector product is only additions and subtractions, accumulated as integers.
- token mixing is an elementwise gated linear recurrence, `h = f * h + (1 - f) * silu(i)`, instead of attention. The state is one vector per layer: there is no KV cache and every token costs the same.

As the architecture is identical, you can also inference the MatMul-free models Ridger released (370M, 1.3B and 2.7B). The weights are ternary to begin with, so there is no separate quantized path: `run.c` already runs integer arithmetic in every linear layer, in `bitlinear()`. That function is also the one place where weights meet activations, which makes it the natural place to model compute-in-memory hardware.

## feel the magic

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/ruhai-lin/matmul-free-lm.c.git
```

Then, open the repository folder:

```bash
cd matmul-free-lm.c
```

Now, let's just run a baby MatMul-free model in C. You need a model checkpoint. Download this 15M parameter model trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:

```bash
wget https://huggingface.co/ruhai-lin/matmul-free-lm.c/resolve/main/stories15M/stories15M.bin
```

Compile and run the C code:

```bash
make run
./run stories15M.bin
```

You'll see the text stream a sample. On one core of a desktop CPU this runs at ~700 tokens/s:

> Once upon a time, there was a little boy named Timmy. Timmy loved to play outside and explore. One day, he saw a big pile of coal on the ground. He thought it looked fun to play with, so he started to bounce over it. As Timmy was bouncing, he saw something interesting on the ground. It was a toy dinosaur. He picked it up and showed it to his friend, Sally. "Look, Sally! I found an oak!" he said. Sally thought it was a cool toy and started to play with it too. She pretended it was a spaceship and flew through the air. Timmy and Sally had so much fun playing with the steel toy that they decided to bury it in the backyard. They worked together until it was time to put it back in the backyard and play with it again. Timmy and Sally had so much fun playing together that they forgot all about the chimney.

You can also prompt the model with a prefix or a number of additional command line arguments, e.g. to sample at temperature 0.8 for 256 steps and with a prompt:

```bash
./run stories15M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
```

> One day, Lily met a Shoggoth Fairy. She was very happy and fair. The kind fairy had a magic wand that could make things happen. Lily liked to imagine that she could talk to people. "Hello, I am Shelly," said the fairy. "I am Mr. Shelly. I can do many things. Do you want to play with me?" Lily was very excited. She said, "Yes, I want to play with you, Shelly!" So, Lily and Shelly played together. They made funny noises and laughed. They had a lot of fun. But then, something unexpected happened. Shelly turned into a magic fairy! The fairy said, "Thank you, Lily, for being my friend. I was a fairy, but now I am a fairy. I will grant you one wish." Lily thought for a moment. She wished for more fun. The fairy waved her wand, and all the toys came to life. Lily and Shelly played with all the toys and had a great time. They were very happy and had a great day together.

There are also smaller models that fit compute-in-memory hardware, see [models](#models).

Quick note on sampling, the recommendation for ~best results is to sample with `-t 1.0 -p 0.9`, i.e. temperature 1.0 (default) but also top-p sampling at 0.9 (default). Intuitively, top-p ensures that tokens with tiny probabilities do not get sampled, so we can't get "unlucky" during sampling, and we are less likely to go "off the rails" afterwards. More generally, to control the diversity of samples use either the temperature (i.e. vary `-t` between 0 and 1 and keep top-p off with `-p 0`) or the top-p value (i.e. vary `-p` between 0 and 1 and keep `-t 1`), but not both.

## Ridger's MatMul-free models

Ridger released three MatMul-free models: [MMfreeLM-370M](https://huggingface.co/ridger/MMfreeLM-370M), [MMfreeLM-1.3B](https://huggingface.co/ridger/MMfreeLM-1.3B) and [MMfreeLM-2.7B](https://huggingface.co/ridger/MMfreeLM-2.7B). As the architecture is identical, we can export them to the `.bin` format with `export.py` (`pip install -r requirements.txt` first) and inference them in C. They use their own tokenizer, so export that too and pass it with `-z`:

```bash
python export.py mmfreelm_370m.bin --hf ridger/MMfreeLM-370M
python tokenizer.py --tokenizer-model=path/to/MMfreeLM-370M/tokenizer.model
./run mmfreelm_370m.bin -z path/to/MMfreeLM-370M/tokenizer.bin -i "The capital of France is"
```

`--hf` takes a Hugging Face hub id or a local directory with `config.json` and `model.safetensors`; the tokenizer model is in the same place. The same works for the 1.3B and 2.7B models, with OpenMP (see [performance](#performance)) for reasonable speed.

These are base models, but `run.c` also has the chat loop of llama2.c, which renders the Llama 2 chat template. Use it with a chat-tuned HGRN-Bit model:

```bash
./run model_chat.bin -m chat
```

## huggingface models

We can load any Hugging Face model that uses the HGRN-Bit architecture (`HGRNBitForCausalLM`). See the script [export.py](export.py) and the `--hf` flag to export the model .bin file.

## models

All models are trained on TinyStories. For reference, these are the Llama 2 models of llama2.c, hosted at [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas):

| model | dim | n_layers | n_heads | n_kv_heads | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 260K | 64 | 5 | 8 | 4 | 512 | 260K | 1.297 | [stories260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)
| OG | 288 | 6 | 6 | 6 | 256 | 15M | 1.072 | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M| 512 | 8 | 8 | 8 | 1024 | 42M | 0.847 | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M| 768 | 12 | 12 | 12 | 1024 | 110M | 0.760 | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |

The MatMul-free models are trained on the same data with the same number of tokens and the same context length, and are hosted at [ruhai-lin/matmul-free-lm.c](https://huggingface.co/ruhai-lin/matmul-free-lm.c). Each folder has the model in Hugging Face format (loadable with the upstream code, or with `export.py --hf`), its evaluation in `eval/`, and, like tinyllamas, the PyTorch `.pt` and the `run.c` `.bin`. The 260K and 1M models use the 512-token vocabulary of stories260K (`-z tok512.bin`, shipped with them), the others the Llama 2 tokenizer. 1M has no llama2.c counterpart: it is an extra size, small enough to target compute-in-memory hardware.

| model | dim | n_layers | hidden_dim | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- |
| 260K | 64 | 4 | 160 | 512 | 257K | 1.609 | [stories260K](https://huggingface.co/ruhai-lin/matmul-free-lm.c/tree/main/stories260K) |
| 1M | 128 | 5 | 272 | 512 | 988K | 1.199 | [stories1M](https://huggingface.co/ruhai-lin/matmul-free-lm.c/tree/main/stories1M) |
| 15M | 288 | 6 | 768 | 256 | 15M | 1.252 | [stories15M](https://huggingface.co/ruhai-lin/matmul-free-lm.c/tree/main/stories15M) |
| 42M | 512 | 8 | 1376 | 1024 | 42M | | |
| 110M | 768 | 12 | 2048 | 1024 | 110M | | |

The val loss is measured on 1,048,576 tokens of the TinyStories validation shard in fp32. On the same tokens the official stories15M scores 1.068.

The token mixer has no heads to count: the recurrence is elementwise, with one state of size dim per layer. Like the llama2.c models, the 15M, 42M and 110M models share the token embedding with the classifier (`tie_embeddings`), so that the parameter counts match.

## training

Let's see how we can train a baby MatMul-free model from scratch using the code in this repo. Install the dependencies with `pip install -r requirements.txt`, then let's download and pretokenize the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, with the Llama 2 tokenizer (`tokenizer.model`) and with the 512-token one (`tok512.model`):

```bash
python tinystories.py download
python tinystories.py pretokenize --vocab_size=32000
python tinystories.py pretokenize --vocab_size=512
```

Then train our model:

```bash
python train.py
```

The defaults in `train.py` are the 260K model. The other models only override the model and data shape, e.g. for 15M on 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --vocab_size=32000 --tie_embeddings=True \
  --dim=288 --n_layers=6 --hidden_dim=768 --max_seq_len=256 --batch_size=128 --gradient_accumulation_steps=4
```

`model.py` is plain PyTorch. On a GPU the recurrence uses the Triton `chunk_hgrn` kernel of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), which `requirements.txt` installs; without it, a Python loop computes the same thing (fine for inference, slow for training).

Do not stop training early. Most of the gain of a ternary model comes in the last ~1% of the schedule: the loss drops sharply once the learning rate gets very small, as the latent weights settle onto the quantization thresholds. About 30% of them end within 1e-4 of a threshold, which is why `weight_quant` reduces its mean in float64: otherwise eager PyTorch, `torch.compile` and the export would round some of them differently.

Totally understand if you want to skip model training, for simple demo just download one of the pretrained models (see [models](#models) section), e.g.:

```bash
wget https://huggingface.co/ruhai-lin/matmul-free-lm.c/resolve/main/stories15M/stories15M.bin
```

Once we have the model.bin file, we can inference in C. Compile the C code first:

```bash
make run
```

You can now run it simply as

```bash
./run stories15M.bin
```

Watch the tokens stream by, fun! We can also run the PyTorch inference script for a comparison. Download one of the models again from huggingface hub and point the `sample.py` script at it:

```bash
wget https://huggingface.co/ruhai-lin/matmul-free-lm.c/resolve/main/stories15M/stories15M.pt -P out15M
python sample.py --checkpoint=out15M/stories15M.pt
```

Which gives the same results.

To export a model you trained yourself, use the `--checkpoint` flag:

```bash
python export.py out/model.bin --checkpoint out/ckpt.pt
```

Add `--hf-dir out/hf` to also write it in Hugging Face format. The latent weights that sit within float error of a rounding threshold (see above) are moved 1e-3 away from it there, on the side of their code, so that other implementations round them the same way.

## model.bin

All little-endian.

| | |
| --- | --- |
| header | 8 x int32: magic `MMFL`, version, dim, hidden_dim, n_layers, vocab_size, seq_len, number of fp32 values |
| fp32 | embeddings, forget gate lower bounds, rmsnorm weights, then for each `BitLinear` its input rmsnorm weight and weight scale |
| int8 | the ternary weights of every `BitLinear`, row-major, in the same order |

The int8 section is exactly what would be programmed into compute-in-memory arrays. A layer's output is `(W @ x_int8) * scale / x_scale`, where `scale = mean(|W_latent|)` and `x_scale = 127 / max(|x|)`.

## performance

The `make run` command uses the `-O3` optimization by default, i.e.:

```bash
gcc -O3 -o run run.c -lm
```

**OpenMP**. Bigger models (e.g. Ridger's 1.3B and 2.7B) benefit from compiling with OpenMP, which "activates" the `#pragma omp parallel for` inside `bitlinear()`, splitting the rows of each layer over multiple processors. Compile with `make runomp`, which does:

```bash
gcc -O3 -fopenmp -march=native run.c -lm -o run
```

and set the number of threads when you run it, e.g.:

```bash
OMP_NUM_THREADS=4 ./run out/model.bin
```

There is no `-Ofast` target: fast-math may change the int8 rounding that `run.c` must share with `model.py`.

## tests

You can run tests simply with pytest:

```bash
$ pip install pytest
$ pytest
```

This checks that `run.c` and `model.py` compute the same loss, on a randomly initialized model and, if `out/ckpt.pt` exists, on a trained one.

## acknowledgements

This repo is a port of [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy: `train.py`, `tinystories.py`, `configurator.py`, `sample.py`, `tokenizer.py`, the tokenizers, and the tokenizer, sampler, generate and chat code of `run.c` come from it, mostly unchanged. The model follows [ridgerchu/matmulfreellm](https://github.com/ridgerchu/matmulfreellm) ([paper](https://arxiv.org/abs/2406.02528)), and the fused recurrence kernel used in training comes from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

## License

MIT
