defmodule LlmScratch.DummyLayerNorm do
  @moduledoc """
  Layer normalization implemented with Nx tensors.

  Layer normalization normalizes each input vector over its last dimension. For
  GPT-style hidden states shaped `{batch_size, seq_len, emb_dim}`, this means
  every token embedding is normalized independently across its `emb_dim`
  features.

  This mirrors the book's PyTorch implementation:

      mean = x.mean(dim=-1, keepdim=True)
      var = x.var(dim=-1, keepdim=True, unbiased=False)
      norm_x = (x - mean) / torch.sqrt(var + eps)
      scale * norm_x + shift

  The Nx equivalents are:

      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)

  `Nx.variance/2` computes population variance, which matches
  `torch.var(..., unbiased=False)`. The `scale` and `shift` tensors correspond
  to PyTorch's trainable parameters initialized with `torch.ones(emb_dim)` and
  `torch.zeros(emb_dim)`.
  """

  defstruct [:emb_dim, :eps, :scale, :shift]

  @type t :: %__MODULE__{
          emb_dim: pos_integer(),
          eps: float(),
          scale: Nx.Tensor.t(),
          shift: Nx.Tensor.t()
        }

  @spec new(pos_integer(), keyword()) :: t()
  @doc """
  Creates a layer normalization module for vectors with `emb_dim` features.

  ## Options

    * `:eps` - small value added to the variance before the square root.
      Defaults to `1.0e-5`, matching the PyTorch example.

  ## Returns

  A `%LlmScratch.DummyLayerNorm{}` struct containing:

    * `:emb_dim` - expected size of the input tensor's last dimension.
    * `:eps` - numerical stability value.
    * `:scale` - tensor of ones shaped `{emb_dim}`.
    * `:shift` - tensor of zeros shaped `{emb_dim}`.

  ## Example

      iex> ln = LlmScratch.DummyLayerNorm.new(5)
      iex> Nx.shape(ln.scale)
      {5}
      iex> Nx.shape(ln.shift)
      {5}
  """
  def new(emb_dim, opts \\ []) when is_integer(emb_dim) and emb_dim > 0 do
    eps = Keyword.get(opts, :eps, 1.0e-5)

    %__MODULE__{
      emb_dim: emb_dim,
      eps: eps,
      scale: Nx.broadcast(1.0, {emb_dim}) |> Nx.as_type({:f, 32}),
      shift: Nx.broadcast(0.0, {emb_dim}) |> Nx.as_type({:f, 32})
    }
  end

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Applies layer normalization over the last axis of `x`.

  The input tensor's last dimension must equal the layer's `emb_dim`. Any number
  of leading dimensions is supported, for example `{batch_size, emb_dim}` or
  `{batch_size, seq_len, emb_dim}`.

  ## Example

      {batch_example, _key} =
        Nx.Random.normal(Nx.Random.key(123), 0.0, 1.0, shape: {2, 5})

      ln = LlmScratch.DummyLayerNorm.new(5)
      out_ln = LlmScratch.DummyLayerNorm.forward(ln, batch_example)

      Nx.mean(out_ln, axes: [-1], keep_axes: true)
      Nx.variance(out_ln, axes: [-1], keep_axes: true)

  The resulting mean is close to zero and the population variance is close to
  one for each row. Because `eps` is included in the denominator, variance can
  be slightly below `1.0` before rounding.

  Raises `ArgumentError` if `x` does not have `emb_dim` features on its last
  axis.
  """
  def forward(%__MODULE__{} = layer_norm, %Nx.Tensor{} = x) do
    validate_last_axis!(x, layer_norm.emb_dim)

    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.variance(x, axes: [-1], keep_axes: true)
    norm_x = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(var, layer_norm.eps)))

    Nx.add(Nx.multiply(layer_norm.scale, norm_x), layer_norm.shift)
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
end
