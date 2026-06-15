defmodule LlmScratch.LinearWithLoRA do
  @moduledoc """
  Dense projection wrapped with a LoRA adapter.

  `LinearWithLoRA` is the bridge between an existing dense projection and the
  standalone `%LlmScratch.LoRALayer{}` adapter. It keeps the original dense
  layer available as `:linear`, creates a matching adapter as `:lora`, and
  computes:

      dense(x) + lora(x)

  This is the usual LoRA wiring: the base projection provides the original model
  behavior, and the low-rank adapter learns an additive update during
  fine-tuning.

  This mirrors the book's PyTorch wrapper:

      class LinearWithLoRA(torch.nn.Module):
          def __init__(self, linear, rank, alpha):
              super().__init__()
              self.linear = linear
              self.lora = LoRALayer(
                  linear.in_features, linear.out_features, rank, alpha
              )

          def forward(self, x):
              return self.linear(x) + self.lora(x)

  In this project, dense layers are represented as maps:

      %{kernel: kernel, bias: bias}

  where `kernel` has shape `{in_dim, out_dim}` and `bias` is optional with shape
  `{out_dim}`. Bias-free layers are represented as `%{kernel: kernel}`. This
  wrapper keeps that dense layer unchanged and adds a `%LlmScratch.LoRALayer{}`
  whose input and output dimensions are inferred from the dense kernel.

  Because `LoRALayer` initializes `B` to zeros, the wrapped layer initially
  produces the same output as the original dense layer.

  The input can have any number of leading dimensions as long as its final axis
  is `in_dim`. For example:

    * `{batch_size, in_dim}` becomes `{batch_size, out_dim}`
    * `{batch_size, seq_len, in_dim}` becomes `{batch_size, seq_len, out_dim}`

  The struct implements `Nx.Container`, so both the wrapped dense layer and the
  LoRA adapter can be traversed by the project's training and checkpoint code.
  If a caller wants to freeze the base projection and train only LoRA weights,
  that freezing should be handled by the optimizer/training setup that consumes
  this container.
  """

  import Nx.Defn

  alias LlmScratch.LoRALayer

  @enforce_keys [:linear, :lora]
  defstruct [:linear, :lora]

  @type dense_layer :: %{required(:kernel) => Nx.Tensor.t(), optional(:bias) => Nx.Tensor.t()}
  @type t :: %__MODULE__{
          linear: dense_layer(),
          lora: LoRALayer.t()
        }

  @doc """
  Wraps a dense layer with a LoRA adapter.

  The dense layer's kernel shape determines the adapter dimensions:

      {in_dim, out_dim} = Nx.shape(linear.kernel)

  The created adapter is equivalent to:

      LoRALayer.new(in_dim, out_dim, rank, alpha, opts)

  `rank` controls the size of the adapter bottleneck. `alpha` scales the adapter
  contribution. Additional options are forwarded to `LoRALayer.new/5`, for
  example `:seed`, `:a`, or `:b`.

  The dense layer is validated before the wrapper is returned:

    * `:kernel` must be a 2D tensor with shape `{in_dim, out_dim}`
    * `:bias`, when present, must have shape `{out_dim}`

  The dense weights are not copied or modified; they are stored as provided.
  """
  @spec new(dense_layer(), pos_integer(), number(), keyword()) :: t()
  def new(%{kernel: %Nx.Tensor{} = kernel} = linear, rank, alpha, opts \\ [])
      when is_integer(rank) and rank > 0 and is_number(alpha) and is_list(opts) do
    {in_dim, out_dim} = validate_linear!(linear, kernel)

    %__MODULE__{
      linear: linear,
      lora: LoRALayer.new(in_dim, out_dim, rank, alpha, opts)
    }
  end

  @doc """
  Applies the wrapped dense layer plus the LoRA adapter to `x`.

  The final axis of `x` must match the dense kernel input dimension. The output
  has the same leading shape with final axis equal to the dense output
  dimension.

  This function validates the input shape before calling `forward_defn/2`.
  The initial output equals the wrapped dense layer output when the adapter uses
  its default initialization, because `LoRALayer.B` starts as zeros.
  """
  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{} = layer, %Nx.Tensor{} = x) do
    validate_input!(layer, x)
    forward_defn(layer, x)
  end

  @doc """
  Defn-compatible forward pass.

  This skips runtime validation and is intended for composition inside larger
  `Nx.Defn` functions. It computes:

      linear_defn(linear, x) + LoRALayer.forward_defn(lora, x)

  using `Nx.dot/4` over the final axis.
  """
  defn forward_defn(layer, x) do
    Nx.add(linear_defn(layer.linear, x), LoRALayer.forward_defn(layer.lora, x))
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def call(layer, x), do: forward(layer, x)

  deftransformp linear_defn(%{kernel: kernel} = linear, x) do
    y = Nx.dot(x, [-1], kernel, [0])

    if Map.has_key?(linear, :bias) do
      Nx.add(y, linear.bias)
    else
      y
    end
  end

  defp validate_linear!(linear, kernel) do
    case Nx.shape(kernel) do
      {in_dim, out_dim} ->
        validate_bias!(linear, out_dim)
        {in_dim, out_dim}

      shape ->
        raise ArgumentError,
              "expected linear kernel shape {in_dim, out_dim}, got #{inspect(shape)}"
    end
  end

  defp validate_bias!(%{bias: %Nx.Tensor{} = bias}, out_dim) do
    if Nx.shape(bias) != {out_dim} do
      raise ArgumentError,
            "expected linear bias shape #{inspect({out_dim})}, got #{inspect(Nx.shape(bias))}"
    end
  end

  defp validate_bias!(_linear, _out_dim), do: :ok

  defp validate_input!(%__MODULE__{} = layer, x) do
    {in_dim, _out_dim} = Nx.shape(layer.linear.kernel)
    last_axis = tuple_size(Nx.shape(x)) - 1

    if Nx.axis_size(x, last_axis) != in_dim do
      raise ArgumentError,
            "expected input final axis to be #{in_dim}, got shape #{inspect(Nx.shape(x))}"
    end
  end
end

defimpl Nx.Container, for: LlmScratch.LinearWithLoRA do
  def traverse(layer, acc, fun) do
    {linear, acc} = fun.(layer.linear, acc)
    {lora, acc} = fun.(layer.lora, acc)
    {%{layer | linear: linear, lora: lora}, acc}
  end

  def reduce(layer, acc, fun) do
    acc
    |> then(&fun.(layer.linear, &1))
    |> then(&fun.(layer.lora, &1))
  end

  def serialize(layer) do
    {__MODULE__, [linear: layer.linear, lora: layer.lora], %{}}
  end

  def deserialize([linear: linear, lora: lora], %{}) do
    struct!(LlmScratch.LinearWithLoRA, %{linear: linear, lora: lora})
  end
end
