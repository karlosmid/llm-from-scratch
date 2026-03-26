# LLM Scratch

An Elixir/Nx playground for implementing language-model building blocks from scratch.

This repository is not a generic OTP example. It is a learning and experimentation project focused on reproducing pieces of modern LLM pipelines in Elixir, largely following the progression from tokenization through attention mechanisms.

## What This Repo Covers

- Tokenization with both a simple custom tokenizer and `tiktoken`
- Dataset preparation for next-token prediction
- Embedding experiments, including a PyTorch-backed embedding wrapper via `Pythonx`
- Self-attention implementations
- Causal attention for autoregressive decoding
- Multi-head attention in Nx/Axon
- Tests that mirror chapter-style exercises and expected tensor values

## Main Modules

- `LlmScratch.SimpleTokenizerV1` builds a small vocabulary from text and supports encode/decode.
- `LlmScratch.Tokenizer` demonstrates `tiktoken` usage from Elixir.
- `LlmScratch.GptDatasetV1` converts raw text into overlapping input/target token windows.
- `LlmScratch.DataLoader` batches and streams dataset items.
- `LlmScratch.Embedding` wraps PyTorch embeddings through `Pythonx` for compatibility experiments.
- `LlmScratch.SelfAttentionV1` and `LlmScratch.SelfAttentionV2` implement progressively more realistic self-attention layers.
- `LlmScratch.CausalAttention` applies masked self-attention for decoder-style models.
- `LlmScratch.MultiheadAttention` implements batched multi-head causal attention.
- `LlmScratch.MultiheadAttentionWrapper` composes multiple causal-attention heads in a simpler wrapper form.

## Repo Layout

```text
lib/
  llm_scratch/
    tokenizer.ex
    simple_tokenizer_v1.ex
    gpt_dataset_v1.ex
    data_loader.ex
    embedding.ex
    self_attention_v1.ex
    self_attention_v2.ex
    causal_attention.ex
    multihead_attention.ex
    multihead_attention_wrapper.ex
test/
  llm-from-scratch-2_test.exs
  llm-from-scratch_3_test.exs
the-verdict.txt
```

## Dependencies

This project uses:

- `Nx`, `EXLA`, and `Axon` for tensor operations and neural-network primitives
- `tiktoken` for OpenAI-compatible tokenization
- `Pythonx` for interop with Python/PyTorch where exact behavior comparison is useful
- `Req` for fetching example data used in tests

Because of the Python interop, some experiments may require a working Python environment with PyTorch available in addition to Elixir.

## Getting Started

### Prerequisites

- Elixir 1.15+
- Erlang/OTP compatible with your Elixir version
- A working Python installation for modules/tests that use `Pythonx`

### Install dependencies

```bash
mix deps.get
```

### Compile

```bash
mix compile
```

### Run tests

```bash
mix test
```

## Example Areas To Explore

### Tokenization

Use the custom tokenizer or `tiktoken` helpers to inspect how text becomes token IDs.

### Dataset creation

`LlmScratch.GptDatasetV1` creates overlapping `(input, target)` windows suitable for next-token prediction training.

### Attention internals

The attention modules are written to make the tensor shapes and math explicit, which makes them useful as reference implementations while learning.

## Purpose

The purpose of this repo is to understand LLM internals by implementing them directly in Elixir instead of treating the model stack as a black box. It is best read as an executable notebook in code form: small modules, shape-aware tests, and experiments that build toward GPT-style components.
