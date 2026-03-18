defmodule LlmScratch.MultiheadAttention do
  @moduledoc """
  Multi-head causal self-attention with shared Q/K/V projections and an output
  projection.

  Mirrors the PyTorch `MultiHeadAttention` module from chapter 3:

    * one query/key/value projection each with total size `d_out`
    * head splitting into `num_heads` chunks of size `head_dim`
    * causal masking over token positions
    * attention dropout
    * output projection that mixes the concatenated head outputs

  This module is different from
  `LlmScratch.MultiheadAttentionWrapper`:

    * `MultiheadAttentionWrapper` builds `num_heads` separate
      `LlmScratch.CausalAttention` modules and concatenates their outputs
    * `MultiheadAttention` uses one set of Q/K/V projections of size `d_out`,
      splits those projections into heads, runs attention per head, then applies
      one final output projection

  The expected input shape is:

      {batch_size, num_tokens, d_in}

  and the returned output shape is:

      {batch_size, num_tokens, d_out}

  Internally, projected queries, keys, and values move through these shapes:

      {batch_size, num_tokens, d_out}
      -> {batch_size, num_tokens, num_heads, head_dim}
      -> {batch_size, num_heads, num_tokens, head_dim}

  where `head_dim = div(d_out, num_heads)`.
  """

  alias LlmScratch.SelfAttentionV2

  defstruct [
    :w_q,
    :w_k,
    :w_v,
    :out_proj,
    :mask,
    :d_in,
    :d_out,
    :context_length,
    :dropout,
    :num_heads,
    :head_dim,
    :qkv_bias,
    :seed
  ]

  @type dense_weights :: %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          w_q: dense_weights(),
          w_k: dense_weights(),
          w_v: dense_weights(),
          out_proj: dense_weights(),
          mask: Nx.Tensor.t(),
          d_in: pos_integer(),
          d_out: pos_integer(),
          context_length: pos_integer(),
          dropout: float(),
          num_heads: pos_integer(),
          head_dim: pos_integer(),
          qkv_bias: boolean(),
          seed: integer()
        }

  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), pos_integer()) :: t()
  def new(d_in, d_out, context_length, dropout, num_heads),
    do: new(d_in, d_out, context_length, dropout, num_heads, false, [])

  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), pos_integer(), boolean()) ::
          t()
  def new(d_in, d_out, context_length, dropout, num_heads, qkv_bias),
    do: new(d_in, d_out, context_length, dropout, num_heads, qkv_bias, [])

  @spec new(
          pos_integer(),
          pos_integer(),
          pos_integer(),
          number(),
          pos_integer(),
          boolean(),
          keyword()
        ) ::
          t()
  @doc """
  Creates a multi-head causal attention module.

  `d_out` must be divisible by `num_heads`, because each head receives an equal
  slice of the projected feature dimension.

  The Q/K/V projections are initialized with output size `d_out`, the
  per-head size is stored in `head_dim`, and `out_proj` maps the concatenated
  head outputs back into `d_out`.

  ## Arguments

    * `d_in` - input embedding dimension for each token
    * `d_out` - total projected output dimension across all heads
    * `context_length` - maximum sequence length supported by the causal mask
    * `dropout` - dropout rate applied to attention weights during training
    * `num_heads` - number of attention heads
    * `qkv_bias` - whether the query, key, and value projections use bias
    * `opts` - keyword options for initialization

  ## Options

    * `:seed` - deterministic initialization seed
  """
  def new(d_in, d_out, context_length, dropout, num_heads, qkv_bias, opts)
      when is_integer(d_in) and d_in > 0 and is_integer(d_out) and d_out > 0 and
             is_integer(context_length) and context_length > 0 and is_integer(num_heads) and
             num_heads > 0 do
    if rem(d_out, num_heads) != 0 do
      raise ArgumentError, "d_out must be divisible by num_heads"
    end

    seed = normalize_seed(Keyword.get(opts, :seed))
    qkv_bias = SelfAttentionV2.normalize_qkv_bias(qkv_bias)
    dropout = normalize_dropout(dropout)
    head_dim = div(d_out, num_heads)

    w_q = SelfAttentionV2.init_dense_weights(d_in, d_out, seed, qkv_bias, "q_proj")
    w_k = SelfAttentionV2.init_dense_weights(d_in, d_out, seed, qkv_bias, "k_proj")
    w_v = SelfAttentionV2.init_dense_weights(d_in, d_out, seed, qkv_bias, "v_proj")
    out_proj = SelfAttentionV2.init_dense_weights(d_out, d_out, seed, true, "out_proj")

    mask =
      Nx.broadcast(1.0, {context_length, context_length})
      |> Nx.triu(k: 1)
      |> Nx.as_type({:f, 32})

    %__MODULE__{
      w_q: w_q,
      w_k: w_k,
      w_v: w_v,
      out_proj: out_proj,
      mask: mask,
      d_in: d_in,
      d_out: d_out,
      context_length: context_length,
      dropout: dropout,
      num_heads: num_heads,
      head_dim: head_dim,
      qkv_bias: qkv_bias,
      seed: seed
    }
  end

  @spec forward(t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  @doc """
  Computes causal multi-head self-attention for batched token sequences.

  ## Arguments

    * `mha` - `%LlmScratch.MultiheadAttention{}`
    * `x` - input tensor of shape `{batch_size, num_tokens, d_in}`
    * `opts` - forward options

  ## Forward Options

    * `:mode` - `:train` or `:inference`; dropout is only applied in train mode
    * `:key` - optional `Nx.Random` key used by dropout in train mode

  ## Steps

    * project `x` into queries, keys, and values of shape
      `{batch_size, num_tokens, d_out}`
    * split the last dimension into `num_heads * head_dim`
    * compute attention scores independently for each head
    * apply the causal mask so tokens cannot attend to future tokens
    * softmax and optionally apply dropout to the attention weights
    * combine attended values across heads and reshape back to
      `{batch_size, num_tokens, d_out}`
    * apply `out_proj`

  ## Returns

    * tensor of shape `{batch_size, num_tokens, d_out}`
  """
  def forward(%__MODULE__{} = mha, %Nx.Tensor{} = x, opts \\ []) do
    {batch_size, num_tokens, _} = validate_input_shape!(x, mha.d_in)
    validate_context_length!(num_tokens, mha.context_length)

    keys =
      x
      |> SelfAttentionV2.dense_project(mha.w_k)
      |> split_heads(batch_size, num_tokens, mha.num_heads, mha.head_dim)

    queries =
      x
      |> SelfAttentionV2.dense_project(mha.w_q)
      |> split_heads(batch_size, num_tokens, mha.num_heads, mha.head_dim)

    values =
      x
      |> SelfAttentionV2.dense_project(mha.w_v)
      |> split_heads(batch_size, num_tokens, mha.num_heads, mha.head_dim)

    attn_scores = Nx.dot(queries, [3], [0, 1], keys, [3], [0, 1])

    mask =
      mha.mask
      |> Nx.slice([0, 0], [num_tokens, num_tokens])
      |> Nx.greater(0.0)
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch_size, mha.num_heads, num_tokens, num_tokens})

    neg_inf = Nx.broadcast(:neg_infinity, Nx.shape(attn_scores))
    masked_scores = Nx.select(mask, neg_inf, attn_scores)

    context =
      masked_scores
      |> Nx.divide(Nx.sqrt(mha.head_dim))
      |> Axon.Activations.softmax(axis: -1)
      |> maybe_dropout(mha, opts)
      |> Nx.dot([3], [0, 1], values, [2], [0, 1])
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch_size, num_tokens, mha.d_out})

    SelfAttentionV2.dense_project(context, mha.out_proj)
  end

  @doc false
  defp split_heads(tensor, batch_size, num_tokens, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch_size, num_tokens, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
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
