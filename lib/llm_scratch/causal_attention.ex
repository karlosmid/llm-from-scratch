defmodule LlmScratch.CausalAttention do
  @moduledoc """
  Causal self-attention over batched token sequences.

  Mirrors the PyTorch `CausalAttention` module:

    * query/key/value dense projections
    * causal upper-triangular mask stored in the module state
    * attention dropout
  """
  alias LlmScratch.SelfAttentionV2

  defstruct [
    :w_q,
    :w_k,
    :w_v,
    :mask,
    :d_in,
    :d_out,
    :context_length,
    :dropout,
    :qkv_bias,
    :seed
  ]

  @type dense_weights :: %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          w_q: dense_weights(),
          w_k: dense_weights(),
          w_v: dense_weights(),
          mask: Nx.Tensor.t(),
          d_in: pos_integer(),
          d_out: pos_integer(),
          context_length: pos_integer(),
          dropout: float(),
          qkv_bias: boolean(),
          seed: integer()
        }

  @spec new(pos_integer(), pos_integer(), pos_integer(), number()) :: t()
  @doc """
  Creates a causal attention module with default `qkv_bias: false`.

  Equivalent to `new(d_in, d_out, context_length, dropout, false, [])`.
  """
  def new(d_in, d_out, context_length, dropout),
    do: new(d_in, d_out, context_length, dropout, false, [])

  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), boolean()) :: t()
  @doc """
  Creates a causal attention module with explicit `qkv_bias`.

  Equivalent to `new(d_in, d_out, context_length, dropout, qkv_bias, [])`.
  """
  def new(d_in, d_out, context_length, dropout, qkv_bias),
    do: new(d_in, d_out, context_length, dropout, qkv_bias, [])

  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), boolean(), keyword()) :: t()
  @doc """
  Creates a causal attention module.

  ## Arguments

    * `d_in` - input feature size.
    * `d_out` - projection/output feature size.
    * `context_length` - maximum sequence length for the causal mask.
    * `dropout` - dropout rate in `[0, 1)`, applied to attention weights.
    * `qkv_bias` - whether query/key/value dense layers use bias.
    * `opts` - keyword options:
      `:seed` (optional, deterministic initialization).

  ## Notes

  `mask` in the struct is the Elixir/Nx equivalent of PyTorch `register_buffer`
  for the upper-triangular causal mask.
  """
  def new(d_in, d_out, context_length, dropout, qkv_bias, opts)
      when is_integer(d_in) and d_in > 0 and is_integer(d_out) and d_out > 0 and
             is_integer(context_length) and context_length > 0 do
    seed = normalize_seed(Keyword.get(opts, :seed))
    qkv_bias = SelfAttentionV2.normalize_qkv_bias(qkv_bias)
    dropout = normalize_dropout(dropout)

    w_q = SelfAttentionV2.init_dense_weights(d_in, d_out, seed, qkv_bias, "q_proj")
    w_k = SelfAttentionV2.init_dense_weights(d_in, d_out, seed, qkv_bias, "k_proj")
    w_v = SelfAttentionV2.init_dense_weights(d_in, d_out, seed, qkv_bias, "v_proj")

    mask =
      Nx.broadcast(1.0, {context_length, context_length})
      |> Nx.triu(k: 1)
      |> Nx.as_type({:f, 32})

    %__MODULE__{
      w_q: w_q,
      w_k: w_k,
      w_v: w_v,
      mask: mask,
      d_in: d_in,
      d_out: d_out,
      context_length: context_length,
      dropout: dropout,
      qkv_bias: qkv_bias,
      seed: seed
    }
  end

  @spec forward(t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  @doc """
  Computes causal self-attention context vectors for a batched input.

  ## Arguments

    * `ca` - `%LlmScratch.CausalAttention{}` module state.
    * `x` - input tensor of shape `{batch_size, num_tokens, d_in}`.
    * `opts` - keyword options:
      `:mode` (`:train` or `:inference`, default `:train`),
      `:key` (optional Nx random key for dropout when in train mode).

  ## Returns

    * context tensor of shape `{batch_size, num_tokens, d_out}`.
  """
  def forward(%__MODULE__{} = ca, %Nx.Tensor{} = x, opts \\ []) do
    {batch_size, num_tokens, _} = validate_input_shape!(x, ca.d_in)
    validate_context_length!(num_tokens, ca.context_length)

    keys = SelfAttentionV2.dense_project(x, ca.w_k)
    queries = SelfAttentionV2.dense_project(x, ca.w_q)
    values = SelfAttentionV2.dense_project(x, ca.w_v)

    attn_scores = Nx.dot(queries, [2], [0], keys, [2], [0])

    mask =
      ca.mask
      |> Nx.slice([0, 0], [num_tokens, num_tokens])
      |> Nx.greater(0.0)
      |> Nx.broadcast({batch_size, num_tokens, num_tokens})

    neg_inf = Nx.broadcast(:neg_infinity, Nx.shape(attn_scores))
    masked_scores = Nx.select(mask, neg_inf, attn_scores)

    attn_weights =
      masked_scores
      |> Nx.divide(Nx.sqrt(Nx.axis_size(keys, -1)))
      |> Axon.Activations.softmax(axis: -1)
      |> maybe_dropout(ca, opts)

    Nx.dot(attn_weights, [2], [0], values, [1], [0])
  end

  defp maybe_dropout(attn_weights, %{dropout: dropout}, _opts) when dropout <= 0.0,
    do: attn_weights

  defp maybe_dropout(attn_weights, %{dropout: dropout, seed: seed}, opts) do
    mode = Keyword.get(opts, :mode, :train)

    if mode == :train do
      key = Keyword.get(opts, :key) || Nx.Random.key(seed)

      %Axon.StatefulOutput{output: dropped, state: %{"key" => _new_key}} =
        Axon.Layers.dropout(attn_weights, key, rate: dropout, mode: :train)

      dropped
    else
      attn_weights
    end
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
  end

  defp normalize_dropout(dropout) when is_number(dropout) and dropout >= 0 and dropout < 1 do
    dropout * 1.0
  end

  defp normalize_dropout(dropout) do
    raise ArgumentError, "dropout must be a number in [0, 1), got: #{inspect(dropout)}"
  end

  defp validate_input_shape!(inputs, expected_d_in) do
    case Nx.shape(inputs) do
      {batch_size, num_tokens, ^expected_d_in} ->
        {batch_size, num_tokens, expected_d_in}

      shape ->
        raise ArgumentError,
              "expected inputs shape {batch_size, num_tokens, #{expected_d_in}}, got: #{inspect(shape)}"
    end
  end

  defp validate_context_length!(num_tokens, context_length) when num_tokens <= context_length,
    do: :ok

  defp validate_context_length!(num_tokens, context_length) do
    raise ArgumentError,
          "num_tokens (#{num_tokens}) exceeds context_length (#{context_length})"
  end
end
