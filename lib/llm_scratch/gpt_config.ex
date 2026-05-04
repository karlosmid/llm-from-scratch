defmodule LlmScratch.GPTConfig do
  @moduledoc """
  GPT model configuration.

  This struct mirrors the config keys used by the Python examples.

  ## Fields

    * `:vocab_size` - number of tokens in the tokenizer vocabulary. This
      determines the number of rows in the token embedding table and the final
      logits dimension produced by the output head. For GPT-2 this is commonly
      `50_257`.

    * `:context_length` - maximum number of token positions the model can
      process in one forward pass. This determines the size of the positional
      embedding table. Inputs longer than this cannot be represented by the
      configured positional embeddings.

    * `:emb_dim` - width of each token and positional embedding vector. This is
      also the model's hidden size: tensors flowing through transformer blocks
      have shape `{batch_size, seq_len, emb_dim}`.

    * `:n_heads` - number of attention heads used by real transformer blocks.
      The dummy GPT model stores this value for config compatibility, but its
      placeholder transformer blocks do not use it yet.

    * `:n_layers` - number of transformer blocks in the model. In
      `LlmScratch.DummyGPTModel`, this controls how many dummy identity blocks
      are created.

    * `:drop_rate` - dropout probability used after adding token and positional
      embeddings. A value of `0.1` means ten percent of activations are dropped
      during training mode. Dropout is skipped in inference mode.

    * `:qkv_bias` - whether query, key, and value projections include bias
      terms in real attention layers. The dummy GPT model keeps this field for
      compatibility with the full GPT config shape, but does not use it because
      its transformer blocks are placeholders.
  """

  @enforce_keys [
    :vocab_size,
    :context_length,
    :emb_dim,
    :n_heads,
    :n_layers,
    :drop_rate,
    :qkv_bias
  ]

  defstruct [
    :vocab_size,
    :context_length,
    :emb_dim,
    :n_heads,
    :n_layers,
    :drop_rate,
    :qkv_bias
  ]

  @type t :: %__MODULE__{
          vocab_size: pos_integer(),
          context_length: pos_integer(),
          emb_dim: pos_integer(),
          n_heads: pos_integer(),
          n_layers: non_neg_integer(),
          drop_rate: float(),
          qkv_bias: boolean()
        }
end
