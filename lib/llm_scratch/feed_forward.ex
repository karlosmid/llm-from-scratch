defmodule LlmScratch.FeedForward do
  @moduledoc """
  Position-wise feed-forward network used inside GPT transformer blocks.

  This mirrors the book's PyTorch module:

      nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
      )

  The first dense layer expands the embedding dimension by a factor of four,
  `LlmScratch.GELU` applies the nonlinearity element-wise, and the second dense
  layer projects the activations back to the original embedding dimension.

  The input can have any number of leading dimensions as long as its final axis
  is `emb_dim`. For GPT hidden states shaped `{batch_size, seq_len, emb_dim}`,
  the output has the same shape.
  """

  import Nx.Defn

  alias LlmScratch.{GELU, GPTConfig, SelfAttentionV2}

  defstruct [:emb_dim, :layers]

  @type dense_weights :: %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          emb_dim: pos_integer(),
          layers: %{
            first: dense_weights(),
            gelu: GELU.t(),
            second: dense_weights()
          }
        }

  @spec new(GPTConfig.t() | map(), keyword()) :: t()
  @doc """
  Creates a feed-forward network from a GPT config.

  `cfg` can be a `%LlmScratch.GPTConfig{}` or a map with either `:emb_dim` or
  `"emb_dim"`.

  ## Options

    * `:seed` - deterministic seed for dense layer initialization.

  ## Example

      iex> cfg = %{emb_dim: 4}
      iex> ff = LlmScratch.FeedForward.new(cfg, seed: 123)
      iex> Nx.shape(ff.layers.first.kernel)
      {4, 16}
      iex> Nx.shape(ff.layers.second.kernel)
      {16, 4}
  """
  def new(cfg, opts \\ []) when is_map(cfg) and is_list(opts) do
    emb_dim = emb_dim_from_cfg!(cfg)
    seed = normalize_seed(Keyword.get(opts, :seed))

    %__MODULE__{
      emb_dim: emb_dim,
      layers: %{
        first: SelfAttentionV2.init_dense_weights(emb_dim, 4 * emb_dim, seed, true, "ff_first"),
        gelu: GELU.new(),
        second:
          SelfAttentionV2.init_dense_weights(4 * emb_dim, emb_dim, seed + 1, true, "ff_second")
      }
    }
  end

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Runs the feed-forward network over `x`.

  GELU and both dense layers are applied over the last dimension. The output
  preserves all leading dimensions and restores the last dimension to `emb_dim`.

  Raises `ArgumentError` if the final axis of `x` is not the configured
  embedding dimension.

  ## Example

      iex> ff = LlmScratch.FeedForward.new(%{emb_dim: 4}, seed: 123)
      iex> x = Nx.broadcast(1.0, {2, 3, 4})
      iex> LlmScratch.FeedForward.forward(ff, x) |> Nx.shape()
      {2, 3, 4}
  """
  def forward(%__MODULE__{} = feed_forward, %Nx.Tensor{} = x) do
    validate_last_axis!(x, feed_forward.emb_dim)

    x
    |> linear_defn(feed_forward.layers.first)
    |> GELU.forward()
    |> linear_defn(feed_forward.layers.second)
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def call(feed_forward, x), do: forward(feed_forward, x)

  @doc """
  Applies a dense layer over the last axis of `x`.

  The dense weights are maps with `:kernel` shaped `{d_in, d_out}` and `:bias`
  shaped `{d_out}`. The function is defined with `Nx.Defn`, so it can be reused
  inside differentiable code such as `Nx.Defn.value_and_grad/2`.

  ## Example

      iex> layer = %{kernel: Nx.broadcast(1.0, {3, 2}), bias: Nx.tensor([0.1, -0.1])}
      iex> x = Nx.tensor([[1.0, 2.0, 3.0]])
      iex> LlmScratch.FeedForward.linear_defn(x, layer) |> Nx.to_flat_list()
      [6.1, 5.9]
  """
  defn linear_defn(x, %{kernel: kernel, bias: bias}) do
    Nx.add(Nx.dot(x, [-1], kernel, [0]), bias)
  end

  defp emb_dim_from_cfg!(%GPTConfig{emb_dim: emb_dim}), do: validate_emb_dim!(emb_dim)

  defp emb_dim_from_cfg!(cfg) do
    cfg
    |> Map.get(:emb_dim, Map.get(cfg, "emb_dim"))
    |> validate_emb_dim!()
  end

  defp validate_emb_dim!(emb_dim) when is_integer(emb_dim) and emb_dim > 0, do: emb_dim

  defp validate_emb_dim!(emb_dim) do
    raise ArgumentError,
          "expected cfg to contain a positive integer emb_dim, got: #{inspect(emb_dim)}"
  end

  defp validate_last_axis!(x, emb_dim) do
    last_axis =
      x
      |> Nx.shape()
      |> Tuple.to_list()
      |> List.last()

    unless last_axis == emb_dim do
      raise ArgumentError,
            "expected input last dimension to be #{emb_dim}, got #{inspect(last_axis)}"
    end
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
  end
end
