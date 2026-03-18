defmodule LlmScratch.MultiheadAttentionWrapper do
  @moduledoc """
  Thin wrapper that builds multiple independent causal-attention heads and
  concatenates their outputs on the last axis.

  Mirrors the PyTorch module:

      class MultiHeadAttentionWrapper(nn.Module):
          def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
              super().__init__()
              self.heads = nn.ModuleList(
                  [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                   for _ in range(num_heads)]
              )

          def forward(self, x):
              return torch.cat([head(x) for head in self.heads], dim=-1)
  """
  alias LlmScratch.CausalAttention

  defstruct [
    :heads,
    :d_in,
    :d_out,
    :context_length,
    :dropout,
    :num_heads,
    :qkv_bias,
    :seed
  ]

  @type t :: %__MODULE__{
          heads: [CausalAttention.t()],
          d_in: pos_integer(),
          d_out: pos_integer(),
          context_length: pos_integer(),
          dropout: float(),
          num_heads: pos_integer(),
          qkv_bias: boolean(),
          seed: integer()
        }

  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), pos_integer()) :: t()
  def new(d_in, d_out, context_length, dropout, num_heads),
    do: new(d_in, d_out, context_length, dropout, num_heads, false, [])

  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), pos_integer(), boolean()) :: t()
  def new(d_in, d_out, context_length, dropout, num_heads, qkv_bias),
    do: new(d_in, d_out, context_length, dropout, num_heads, qkv_bias, [])

  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), pos_integer(), boolean(), keyword()) ::
          t()
  @doc """
  Creates a wrapper containing `num_heads` independent causal-attention heads.

  ## Options

    * `:seed` - deterministic base seed. Each head uses `seed + head_index`.
  """
  def new(d_in, d_out, context_length, dropout, num_heads, qkv_bias, opts)
      when is_integer(d_in) and d_in > 0 and is_integer(d_out) and d_out > 0 and
             is_integer(context_length) and context_length > 0 and is_integer(num_heads) and
             num_heads > 0 do
    seed = normalize_seed(Keyword.get(opts, :seed))
    dropout = normalize_dropout(dropout)
    qkv_bias = normalize_qkv_bias(qkv_bias)

    heads =
      for index <- 0..(num_heads - 1) do
        CausalAttention.new(
          d_in,
          d_out,
          context_length,
          dropout,
          qkv_bias,
          seed: seed + index
        )
      end

    %__MODULE__{
      heads: heads,
      d_in: d_in,
      d_out: d_out,
      context_length: context_length,
      dropout: dropout,
      num_heads: num_heads,
      qkv_bias: qkv_bias,
      seed: seed
    }
  end

  @spec forward(t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  @doc """
  Runs all heads on the same input and concatenates their context vectors on the
  last axis.

  Accepts the same forward options as `LlmScratch.CausalAttention.forward/3`.
  """
  def forward(%__MODULE__{heads: heads}, %Nx.Tensor{} = x, opts \\ []) do
    heads
    |> Enum.map(&CausalAttention.forward(&1, x, opts))
    |> Nx.concatenate(axis: -1)
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

  defp normalize_qkv_bias(qkv_bias) when is_boolean(qkv_bias), do: qkv_bias

  defp normalize_qkv_bias(qkv_bias) do
    raise ArgumentError, "qkv_bias must be a boolean, got: #{inspect(qkv_bias)}"
  end
end
