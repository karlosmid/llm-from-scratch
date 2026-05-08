defmodule LlmScratch.TransformerBlock do
  @moduledoc """
  Real GPT-style transformer block implemented with Nx tensors.

  This mirrors the book's PyTorch `TransformerBlock`:

      shortcut = x
      x = norm1(x)
      x = attention(x)
      x = dropout(x)
      x = x + shortcut

      shortcut = x
      x = norm2(x)
      x = feed_forward(x)
      x = dropout(x)
      x = x + shortcut

  The block uses pre-normalization, causal multi-head self-attention, a
  position-wise feed-forward network, and residual shortcut connections.
  """

  alias LlmScratch.{
    DummyLayerNorm,
    FeedForward,
    GPTConfig,
    MultiheadAttention
  }

  defstruct [:att, :ff, :norm1, :norm2, :drop_shortcut, :cfg]

  @type t :: %__MODULE__{
          att: MultiheadAttention.t(),
          ff: FeedForward.t(),
          norm1: DummyLayerNorm.t(),
          norm2: DummyLayerNorm.t(),
          drop_shortcut: float(),
          cfg: GPTConfig.t() | map()
        }

  @spec new(GPTConfig.t() | map(), keyword()) :: t()
  @doc """
  Creates a GPT transformer block from a config.

  `cfg` can be a `%LlmScratch.GPTConfig{}` or a map with atom/string keys:
  `:emb_dim`, `:context_length`, `:n_heads`, `:drop_rate`, and `:qkv_bias`.
  `:attn_drop_rate` and `:shortcut_drop_rate` may be supplied to control those
  dropout sites independently; otherwise `:drop_rate` is used for compatibility.

  ## Options

    * `:seed` - deterministic seed used for attention and feed-forward
      initialization.
    * `:norm_eps` - epsilon for both layer-normalization modules.
  """
  def new(cfg, opts \\ []) when is_map(cfg) and is_list(opts) do
    params = params_from_cfg!(cfg)
    seed = normalize_seed(Keyword.get(opts, :seed))
    norm_eps = Keyword.get(opts, :norm_eps, 1.0e-5)

    %__MODULE__{
      cfg: cfg,
      att:
        MultiheadAttention.new(
          params.emb_dim,
          params.emb_dim,
          params.context_length,
          params.attn_drop_rate,
          params.n_heads,
          params.qkv_bias,
          seed: seed
        ),
      ff: FeedForward.new(cfg, seed: seed + 1),
      norm1: DummyLayerNorm.new(params.emb_dim, eps: norm_eps),
      norm2: DummyLayerNorm.new(params.emb_dim, eps: norm_eps),
      drop_shortcut: params.shortcut_drop_rate
    }
  end

  @spec forward(t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  @doc """
  Runs a forward pass over hidden states shaped `{batch_size, seq_len, emb_dim}`.

  Forward options:

    * `:mode` - `:train` or `:inference`. Dropout is skipped in inference mode.
    * `:key` - optional `Nx.Random` key used by dropout in train mode.
    * `:attention_key`, `:shortcut_key1`, `:shortcut_key2` - optional keys for
      controlling individual dropout calls.
  """
  def forward(%__MODULE__{} = block, %Nx.Tensor{} = x, opts \\ []) do
    shortcut = x

    x =
      block.norm1
      |> DummyLayerNorm.forward(x)
      |> then(&MultiheadAttention.forward(block.att, &1, attention_opts(opts)))
      |> MultiheadAttention.maybe_dropout(
        shortcut_dropout(block, block.att.seed + 1),
        shortcut_opts(opts, :shortcut_key1)
      )
      |> Nx.add(shortcut)

    shortcut = x

    block.norm2
    |> DummyLayerNorm.forward(x)
    |> then(&FeedForward.forward(block.ff, &1))
    |> MultiheadAttention.maybe_dropout(
      shortcut_dropout(block, block.att.seed + 2),
      shortcut_opts(opts, :shortcut_key2)
    )
    |> Nx.add(shortcut)
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Alias for `forward/3` using default forward options.

  This mirrors the callable-style API used by other modules in the project.
  Dropout is skipped because `forward/3` defaults to inference mode.
  """
  def call(block, x), do: forward(block, x)

  defp attention_opts(opts) do
    opts
    |> opts_with_default_mode()
    |> Keyword.put(:key, Keyword.get(opts, :attention_key, Keyword.get(opts, :key)))
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
  end

  defp shortcut_opts(opts, key_name) do
    opts
    |> opts_with_default_mode()
    |> Keyword.put(:key, Keyword.get(opts, key_name, Keyword.get(opts, :key)))
  end

  defp shortcut_dropout(block, seed), do: %{dropout: block.drop_shortcut, seed: seed}

  defp opts_with_default_mode(opts),
    do: Keyword.put(Keyword.take(opts, [:mode]), :mode, mode!(opts))

  defp mode!(opts) do
    case Keyword.get(opts, :mode, :inference) do
      mode when mode in [:train, :inference] -> mode
      mode -> raise ArgumentError, "mode must be :train or :inference, got: #{inspect(mode)}"
    end
  end

  defp params_from_cfg!(%GPTConfig{} = cfg) do
    %{
      emb_dim: validate_positive_integer!(:emb_dim, cfg.emb_dim),
      context_length: validate_positive_integer!(:context_length, cfg.context_length),
      n_heads: validate_positive_integer!(:n_heads, cfg.n_heads),
      attn_drop_rate: GPTConfig.attention_dropout(cfg),
      shortcut_drop_rate: GPTConfig.shortcut_dropout(cfg),
      qkv_bias: validate_boolean!(:qkv_bias, cfg.qkv_bias)
    }
  end

  defp params_from_cfg!(cfg) do
    %{
      emb_dim: cfg_value!(cfg, :emb_dim),
      context_length: cfg_value!(cfg, :context_length),
      n_heads: cfg_value!(cfg, :n_heads),
      drop_rate: cfg_value!(cfg, :drop_rate),
      attn_drop_rate: cfg_value(cfg, :attn_drop_rate),
      shortcut_drop_rate: cfg_value(cfg, :shortcut_drop_rate),
      qkv_bias: cfg_value!(cfg, :qkv_bias)
    }
    |> then(fn params ->
      drop_rate = validate_dropout!(:drop_rate, params.drop_rate)

      %{
        emb_dim: validate_positive_integer!(:emb_dim, params.emb_dim),
        context_length: validate_positive_integer!(:context_length, params.context_length),
        n_heads: validate_positive_integer!(:n_heads, params.n_heads),
        attn_drop_rate: dropout_or_default!(:attn_drop_rate, params.attn_drop_rate, drop_rate),
        shortcut_drop_rate:
          dropout_or_default!(:shortcut_drop_rate, params.shortcut_drop_rate, drop_rate),
        qkv_bias: validate_boolean!(:qkv_bias, params.qkv_bias)
      }
    end)
  end

  defp cfg_value(cfg, key) do
    Map.get(cfg, key, Map.get(cfg, Atom.to_string(key)))
  end

  defp cfg_value!(cfg, key) do
    cfg_value(cfg, key)
  end

  defp validate_positive_integer!(_field, value) when is_integer(value) and value > 0, do: value

  defp validate_positive_integer!(field, value) do
    raise ArgumentError, "expected #{field} to be a positive integer, got: #{inspect(value)}"
  end

  defp dropout_or_default!(_field, nil, default), do: default
  defp dropout_or_default!(field, value, _default), do: validate_dropout!(field, value)

  defp validate_dropout!(_field, value) when is_number(value) and value >= 0 and value < 1,
    do: value * 1.0

  defp validate_dropout!(field, value) do
    raise ArgumentError, "expected #{field} to be a number in [0, 1), got: #{inspect(value)}"
  end

  defp validate_boolean!(_field, value) when is_boolean(value), do: value

  defp validate_boolean!(field, value) do
    raise ArgumentError, "expected #{field} to be a boolean, got: #{inspect(value)}"
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
  end
end
