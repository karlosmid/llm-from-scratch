defmodule LlmScratch.DummyGPTModel do
  @moduledoc """
  Minimal GPT-shaped model implemented with Nx tensors.

  This mirrors the Python `DummyGPTModel` from the book:

    * token embedding
    * positional embedding
    * embedding dropout
    * a list of placeholder transformer blocks
    * placeholder layer norm
    * vocabulary projection without bias

  The transformer blocks and layer norm are intentionally no-ops.
  """

  alias LlmScratch.{
    DummyLayerNorm,
    DummyTransformerBlock,
    EmbeddingNative,
    GPTConfig,
    SelfAttentionV2
  }

  defstruct [:cfg, :tok_emb, :pos_emb, :drop_emb, :trf_blocks, :final_norm, :out_head]

  @type linear_no_bias :: %{kernel: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          cfg: GPTConfig.t(),
          tok_emb: EmbeddingNative.t(),
          pos_emb: EmbeddingNative.t(),
          drop_emb: float(),
          trf_blocks: [DummyTransformerBlock.t()],
          final_norm: DummyLayerNorm.t(),
          out_head: linear_no_bias()
        }

  @spec new(GPTConfig.t(), keyword()) :: t()
  @doc """
  Creates a dummy GPT model.

  `cfg` is a `LlmScratch.GPTConfig` struct.

  ## Options

    * `:seed` - deterministic seed for native Nx/Axon initialization.

  ## Returns

  A `%LlmScratch.DummyGPTModel{}` struct with these fields:

    * `:cfg` - the `LlmScratch.GPTConfig` used to build the model. It records
      the vocabulary size, context length, embedding dimension, number of
      layers, dropout rate, and compatibility fields from the GPT config.

    * `:tok_emb` - token embedding table with stored weight shape
      `{vocab_size, emb_dim}`.

    * `:pos_emb` - positional embedding table with stored weight shape
      `{context_length, emb_dim}`.

    * `:drop_emb` - embedding dropout rate from `cfg.emb_drop_rate`, falling
      back to `cfg.drop_rate`.

    * `:trf_blocks` - list of dummy transformer blocks. Its length is
      `cfg.n_layers`. In this dummy model each block returns its input
      unchanged.

    * `:final_norm` - placeholder final layer normalization module. It currently
      returns its input unchanged.

    * `:out_head` - final vocabulary projection without bias. It contains a
      kernel shaped `{emb_dim, vocab_size}`.
  """
  def new(%GPTConfig{} = cfg, opts \\ []) when is_list(opts) do
    seed = normalize_seed(Keyword.get(opts, :seed))

    tok_emb = EmbeddingNative.new(cfg.vocab_size, cfg.emb_dim, seed: seed)
    pos_emb = EmbeddingNative.new(cfg.context_length, cfg.emb_dim, seed: seed + 1)
    out_head = init_out_head(cfg, seed + 2)

    %__MODULE__{
      cfg: cfg,
      tok_emb: tok_emb,
      pos_emb: pos_emb,
      drop_emb: GPTConfig.embedding_dropout(cfg),
      trf_blocks: for(_ <- 1..cfg.n_layers, do: DummyTransformerBlock.new(cfg)),
      final_norm: DummyLayerNorm.new(cfg.emb_dim),
      out_head: out_head
    }
  end

  @spec forward(t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  @doc """
  Runs a forward pass over token ids shaped `{batch_size, seq_len}`.

  ## Steps

    * Read `seq_len` from the input tensor shape.
    * Look up token embeddings for every token id in `in_idx`, producing a
      tensor shaped `{batch_size, seq_len, emb_dim}`.
    * Build positional indices `0..seq_len-1`.
    * Look up positional embeddings for those positions, producing a tensor
      shaped `{seq_len, emb_dim}`.
    * Add token embeddings and positional embeddings. Nx broadcasts the
      positional embeddings across the batch dimension.
    * Apply embedding dropout when `mode: :train`; leave the tensor unchanged
      in inference mode.
    * Pass the tensor through each dummy transformer block. These placeholder
      blocks currently return their input unchanged.
    * Apply the placeholder final layer norm. It currently returns its input
      unchanged.
    * Project the final hidden states through the output head kernel.

  Options:

    * `:mode` - `:inference` (default) or `:train`.
    * `:key` - optional `Nx.Random` key for dropout in train mode.

  ## Returns

  A logits tensor shaped `{batch_size, seq_len, vocab_size}`.

  For each token position, the last dimension contains one score per vocabulary
  token.
  """
  def forward(%__MODULE__{} = model, %Nx.Tensor{} = in_idx, opts \\ []) do
    {_batch_size, seq_len} = Nx.shape(in_idx)
    tok_embeds = EmbeddingNative.forward(model.tok_emb, in_idx)

    pos_indices = Nx.iota({seq_len}, type: {:s, 64})
    pos_embeds = EmbeddingNative.forward(model.pos_emb, pos_indices)

    x =
      tok_embeds
      |> Nx.add(pos_embeds)
      |> maybe_dropout(model.drop_emb, opts)

    x = Enum.reduce(model.trf_blocks, x, &DummyTransformerBlock.forward(&1, &2))
    x = DummyLayerNorm.forward(model.final_norm, x)

    # {batch_size, seq_len, emb_dim} dot {emb_dim, vocab_size} ->
    # {batch_size, seq_len, vocab_size}
    linear(x, model.out_head)
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def call(model, in_idx), do: forward(model, in_idx)

  defp linear(x, %{kernel: kernel}), do: Nx.dot(x, [2], kernel, [0])

  defp maybe_dropout(x, drop_rate, opts) when drop_rate > 0.0 do
    case Keyword.get(opts, :mode, :inference) do
      :train ->
        key = Keyword.get(opts, :key, Nx.Random.key(System.unique_integer([:positive])))

        %Axon.StatefulOutput{output: dropped} =
          Axon.Layers.dropout(x, key, rate: drop_rate, mode: :train)

        dropped

      :inference ->
        x

      mode ->
        raise ArgumentError, "mode must be :train or :inference, got: #{inspect(mode)}"
    end
  end

  defp maybe_dropout(x, _drop_rate, _opts), do: x

  @spec init_out_head(GPTConfig.t(), integer()) :: linear_no_bias()
  defp init_out_head(cfg, seed) do
    %{kernel: kernel} =
      SelfAttentionV2.init_dense_weights(
        cfg.emb_dim,
        cfg.vocab_size,
        seed,
        false,
        "out_head"
      )

    %{kernel: kernel}
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
  end
end
