defmodule LlmScratch.GPTModel do
  @moduledoc """
  GPT model implemented with Nx tensors and the real transformer block.

  This mirrors the book's PyTorch `GPTModel`:

      tok_emb = nn.Embedding(vocab_size, emb_dim)
      pos_emb = nn.Embedding(context_length, emb_dim)
      drop_emb = nn.Dropout(drop_rate)
      trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(n_layers)])
      final_norm = LayerNorm(emb_dim)
      out_head = nn.Linear(emb_dim, vocab_size, bias=False)

  `LlmScratch.DummyGPTModel` remains unchanged; this module uses
  `LlmScratch.TransformerBlock` for the actual attention and feed-forward stack.
  """

  alias LlmScratch.{
    DummyLayerNorm,
    EmbeddingNative,
    GPTConfig,
    MultiheadAttention,
    SelfAttentionV2,
    TransformerBlock
  }

  defstruct [:cfg, :tok_emb, :pos_emb, :drop_emb, :trf_blocks, :final_norm, :out_head]

  @type linear_no_bias :: %{kernel: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          cfg: GPTConfig.t(),
          tok_emb: EmbeddingNative.t(),
          pos_emb: EmbeddingNative.t(),
          drop_emb: float(),
          trf_blocks: [TransformerBlock.t()],
          final_norm: DummyLayerNorm.t(),
          out_head: linear_no_bias()
        }

  @spec new(GPTConfig.t(), keyword()) :: t()
  @doc """
  Creates a GPT model from a `%LlmScratch.GPTConfig{}`.

  ## Options

    * `:seed` - deterministic seed used for embeddings, transformer blocks,
      and the output projection.
    * `:norm_eps` - epsilon used by the final layer norm and transformer block
      layer norms.
  """
  def new(%GPTConfig{} = cfg, opts \\ []) when is_list(opts) do
    seed = normalize_seed(Keyword.get(opts, :seed))
    norm_eps = Keyword.get(opts, :norm_eps, 1.0e-5)

    %__MODULE__{
      cfg: cfg,
      tok_emb: EmbeddingNative.new(cfg.vocab_size, cfg.emb_dim, seed: seed),
      pos_emb: EmbeddingNative.new(cfg.context_length, cfg.emb_dim, seed: seed + 1),
      drop_emb: validate_dropout!(cfg.drop_rate),
      trf_blocks: transformer_blocks(cfg, seed + 2, norm_eps),
      final_norm: DummyLayerNorm.new(cfg.emb_dim, eps: norm_eps),
      out_head: init_out_head(cfg, seed + 2 + cfg.n_layers)
    }
  end

  @spec forward(t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  @doc """
  Runs a forward pass over token ids shaped `{batch_size, seq_len}`.

  The returned logits have shape `{batch_size, seq_len, vocab_size}`.

  Forward options:

    * `:mode` - `:inference` or `:train`. Defaults to `:inference`.
    * `:key` - optional `Nx.Random` key used by dropout in train mode.
  """
  def forward(%__MODULE__{} = model, %Nx.Tensor{} = in_idx, opts \\ []) do
    {_batch_size, seq_len} = validate_input_shape!(in_idx)
    validate_context_length!(seq_len, model.cfg.context_length)

    tok_embeds = EmbeddingNative.forward(model.tok_emb, in_idx)

    pos_embeds =
      model.pos_emb
      |> EmbeddingNative.forward(positional_indices(seq_len))

    x =
      tok_embeds
      |> Nx.add(pos_embeds)
      |> MultiheadAttention.maybe_dropout(
        %{dropout: model.drop_emb, seed: model.pos_emb.seed + 1},
        dropout_opts(opts)
      )

    x =
      Enum.reduce(model.trf_blocks, x, fn block, acc ->
        TransformerBlock.forward(block, acc, opts)
      end)

    x = DummyLayerNorm.forward(model.final_norm, x)

    linear(x, model.out_head)
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Alias for `forward/3` using default forward options.
  """
  def call(model, in_idx), do: forward(model, in_idx)

  defp transformer_blocks(cfg, _seed, _norm_eps) when cfg.n_layers == 0, do: []

  defp transformer_blocks(cfg, seed, norm_eps) do
    for layer_idx <- 0..(cfg.n_layers - 1) do
      TransformerBlock.new(cfg, seed: seed + layer_idx, norm_eps: norm_eps)
    end
  end

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

  defp linear(x, %{kernel: kernel}), do: Nx.dot(x, [-1], kernel, [0])

  defp positional_indices(seq_len), do: Nx.iota({seq_len}, type: {:s, 64})

  defp dropout_opts(opts) do
    opts
    |> Keyword.take([:mode, :key])
    |> Keyword.put(:mode, mode!(opts))
  end

  defp mode!(opts) do
    case Keyword.get(opts, :mode, :inference) do
      mode when mode in [:train, :inference] -> mode
      mode -> raise ArgumentError, "mode must be :train or :inference, got: #{inspect(mode)}"
    end
  end

  defp validate_input_shape!(in_idx) do
    case Nx.shape(in_idx) do
      {batch_size, seq_len} ->
        {batch_size, seq_len}

      shape ->
        raise ArgumentError, "expected input shape {batch_size, seq_len}, got: #{inspect(shape)}"
    end
  end

  defp validate_context_length!(seq_len, context_length) when seq_len <= context_length,
    do: :ok

  defp validate_context_length!(seq_len, context_length) do
    raise ArgumentError, "seq_len (#{seq_len}) exceeds context_length (#{context_length})"
  end

  defp validate_dropout!(value) when is_number(value) and value >= 0 and value < 1,
    do: value * 1.0

  defp validate_dropout!(value) do
    raise ArgumentError, "expected drop_rate to be a number in [0, 1), got: #{inspect(value)}"
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
  end
end
