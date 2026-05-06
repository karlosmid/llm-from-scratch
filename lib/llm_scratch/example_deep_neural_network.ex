defmodule LlmScratch.ExampleDeepNeuralNetwork do
  @moduledoc """
  Small deep network used to demonstrate gradient flow with and without shortcut
  connections.

  This module mirrors the book's PyTorch example:

      nn.ModuleList([
        nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
        nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
        nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
        nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
        nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
      ])

  The network is intentionally small and fixed to five dense-plus-GELU layers
  because the surrounding chapter example focuses on comparing gradient flow,
  not on building a general model container.

  Unlike PyTorch, Nx tensors do not accumulate gradients in mutable `.grad`
  fields. `backward/3` returns `{loss, gradients}` directly, where `gradients`
  has the same nested structure as the model layers.

  Shortcut behavior follows the example's shape check. For `use_shortcut: true`,
  the first four layers in `layer_sizes = [3, 3, 3, 3, 3, 1]` add their output
  back to the input because their input and output shapes match. The final
  `3 -> 1` layer cannot use a shortcut and is applied normally.
  """

  import Nx.Defn

  alias LlmScratch.{FeedForward, GELU, SelfAttentionV2, SimpleGradient}

  defstruct [:layers, :use_shortcut]

  @doc """
  Builds a five-layer dense network.

  ## Parameters

    * `layer_sizes` - six positive integers describing the input and output
      width of each dense layer. For example, `[3, 3, 3, 3, 3, 1]` creates
      layers `3 -> 3`, `3 -> 3`, `3 -> 3`, `3 -> 3`, and `3 -> 1`.

    * `opts` - keyword options:
      `:use_shortcut` is required and controls residual connections;
      `:seed` is optional and defaults to `123`.

  Dense kernels and biases are initialized through Axon so their layout and
  initialization behavior match the rest of this project.

  ## Example

      iex> model = LlmScratch.ExampleDeepNeuralNetwork.new([3, 3, 3, 3, 3, 1],
      ...>   use_shortcut: false,
      ...>   seed: 123
      ...> )
      iex> tuple_size(model.layers)
      5
      iex> Nx.shape(elem(model.layers, 0).kernel)
      {3, 3}
      iex> Nx.shape(elem(model.layers, 4).kernel)
      {3, 1}
  """
  def new(layer_sizes, opts \\ []) do
    use_shortcut = Keyword.fetch!(opts, :use_shortcut)
    seed = Keyword.get(opts, :seed, 123)

    layers =
      layer_sizes
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.map(fn {[d_in, d_out], idx} ->
        SelfAttentionV2.init_dense_weights(
          d_in,
          d_out,
          seed + idx,
          true,
          "example_layer_#{idx}"
        )
      end)
      |> List.to_tuple()

    %__MODULE__{layers: layers, use_shortcut: use_shortcut}
  end

  @doc """
  Runs the network on `x`.

  `x` is expected to be a rank-2 tensor whose second axis matches the first
  layer input width. The returned tensor has the output width of the final
  layer.

  When `use_shortcut` is enabled, residual additions are applied for layers
  whose dense-plus-GELU output shape matches the incoming `x` shape.

  ## Example

      iex> model = LlmScratch.ExampleDeepNeuralNetwork.new([3, 3, 3, 3, 3, 1],
      ...>   use_shortcut: true,
      ...>   seed: 123
      ...> )
      iex> x = Nx.tensor([[1.0, 0.0, -1.0]], type: {:f, 32})
      iex> LlmScratch.ExampleDeepNeuralNetwork.forward(model, x) |> Nx.shape()
      {1, 1}
  """
  def forward(%__MODULE__{layers: layers, use_shortcut: false}, x),
    do: forward_without_shortcut(layers, x)

  def forward(%__MODULE__{layers: layers, use_shortcut: true}, x),
    do: forward_with_shortcut(layers, x)

  @doc """
  Computes mean-squared-error loss and gradients for the model layers.

  This is the Nx equivalent of a PyTorch forward pass followed by
  `loss.backward()`. Instead of mutating parameter gradients, it returns them:

      {loss, gradients} = backward(model, x, target)

  `loss` is a scalar tensor. `gradients` is a tuple with one map per layer. Each
  layer gradient map contains `:kernel` and `:bias` tensors matching the
  corresponding layer parameter shapes.
  """
  def backward(%__MODULE__{layers: layers, use_shortcut: false}, x, target),
    do: backward_without_shortcut(layers, x, target)

  def backward(%__MODULE__{layers: layers, use_shortcut: true}, x, target),
    do: backward_with_shortcut(layers, x, target)

  @doc """
  Returns the absolute mean of each layer's kernel gradient.

  This mirrors the Python example's:

      param.grad.abs().mean().item()

  Bias gradients are intentionally ignored because the chapter example prints
  only gradients for parameters whose names include `"weight"`.
  """
  def weight_gradient_means(gradients) do
    gradients
    |> Tuple.to_list()
    |> Enum.map(fn %{kernel: kernel} ->
      kernel
      |> Nx.abs()
      |> Nx.mean()
    end)
  end

  defn forward_without_shortcut({layer_0, layer_1, layer_2, layer_3, layer_4}, x) do
    x
    |> linear_gelu(layer_0)
    |> linear_gelu(layer_1)
    |> linear_gelu(layer_2)
    |> linear_gelu(layer_3)
    |> linear_gelu(layer_4)
  end

  defn forward_with_shortcut({layer_0, layer_1, layer_2, layer_3, layer_4}, x) do
    x
    |> residual_linear_gelu(layer_0)
    |> residual_linear_gelu(layer_1)
    |> residual_linear_gelu(layer_2)
    |> residual_linear_gelu(layer_3)
    |> linear_gelu(layer_4)
  end

  defn backward_without_shortcut(layers, x, target) do
    value_and_grad(layers, fn layers ->
      layers
      |> forward_without_shortcut(x)
      |> SimpleGradient.mse_loss(target)
    end)
  end

  defn backward_with_shortcut(layers, x, target) do
    value_and_grad(layers, fn layers ->
      layers
      |> forward_with_shortcut(x)
      |> SimpleGradient.mse_loss(target)
    end)
  end

  defnp residual_linear_gelu(x, layer) do
    x + linear_gelu(x, layer)
  end

  defnp linear_gelu(x, layer) do
    x
    |> FeedForward.linear_defn(layer)
    |> GELU.forward_defn()
  end
end
