defmodule LlmScratch.LoRALayer do
  @moduledoc """
  Low-rank adaptation layer.

  LoRA adds a trainable low-rank update next to an existing dense projection.
  For a frozen projection `x @ W`, the adapter contribution is:

      alpha * (x @ A @ B)

  where `A` has shape `{in_dim, rank}` and `B` has shape `{rank, out_dim}`.
  In the usual LoRA setup, the base projection weights stay frozen while only
  `A` and `B` are trained. This module implements only the adapter branch; the
  caller is responsible for adding it to the base layer output when wiring LoRA
  into a model.

  This mirrors the book's PyTorch implementation:

      self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
      torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
      self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
      self.alpha = alpha

  The layer computes:

      alpha * (x @ A @ B)

  over the final axis of `x`. The input may have any number of leading axes.
  For example, an input with shape `{batch_size, seq_len, in_dim}` produces an
  output with shape `{batch_size, seq_len, out_dim}`.

  `B` is initialized to zeros so the adapter initially contributes exactly zero
  to the model output. This preserves the behavior of the base model at the
  moment LoRA is attached, while still allowing the adapter to learn during
  fine-tuning.

  The struct implements `Nx.Container`, so the `A` and `B` tensors can be found
  by the existing training and checkpoint traversal code.
  """

  import Nx.Defn

  alias LlmScratch.Random

  @enforce_keys [:in_dim, :out_dim, :rank, :alpha, :a, :b]
  defstruct [:in_dim, :out_dim, :rank, :alpha, :a, :b]

  @type t :: %__MODULE__{
          in_dim: pos_integer(),
          out_dim: pos_integer(),
          rank: pos_integer(),
          alpha: number(),
          a: Nx.Tensor.t(),
          b: Nx.Tensor.t()
        }

  @doc """
  Creates a LoRA layer.

  `in_dim` must match the final input dimension. `out_dim` is the adapter
  output dimension, usually the same output dimension as the dense layer being
  adapted. `rank` controls the size of the low-rank bottleneck and therefore
  the number of trainable adapter parameters:

      in_dim * rank + rank * out_dim

  `alpha` scales the adapter output. The book implementation uses `alpha`
  directly as:

      alpha * (x @ A @ B)

  This module follows that behavior exactly and does not divide by `rank`.

  `A` is initialized with the same bound produced by
  `torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))` for the `{in_dim, rank}`
  matrix. `B` is initialized to zeros.

  ## Options

    * `:seed` - deterministic seed for initializing `A`. Defaults to `123`.
    * `:a` - optional pre-initialized `{in_dim, rank}` tensor.
    * `:b` - optional pre-initialized `{rank, out_dim}` tensor.

  Passing `:a` or `:b` is useful for loading adapter weights or for tests that
  need deterministic, hand-written tensors. Their shapes are validated.
  """
  @spec new(pos_integer(), pos_integer(), pos_integer(), number(), keyword()) :: t()
  def new(in_dim, out_dim, rank, alpha, opts \\ [])
      when is_integer(in_dim) and in_dim > 0 and
             is_integer(out_dim) and out_dim > 0 and
             is_integer(rank) and rank > 0 and
             is_number(alpha) and is_list(opts) do
    %__MODULE__{
      in_dim: in_dim,
      out_dim: out_dim,
      rank: rank,
      alpha: alpha,
      a:
        Keyword.get_lazy(opts, :a, fn -> init_a(in_dim, rank, Keyword.get(opts, :seed, 123)) end),
      b: Keyword.get_lazy(opts, :b, fn -> init_b(rank, out_dim) end)
    }
    |> validate_parameters!()
  end

  @doc """
  Applies the LoRA layer to `x`.

  The final axis of `x` must be `in_dim`; the output has the same leading shape
  with final axis `out_dim`.

  This function performs shape validation before dispatching to
  `forward_defn/2`. Use `forward_defn/2` directly only from code already running
  inside an `Nx.Defn` context.
  """
  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{} = layer, %Nx.Tensor{} = x) do
    validate_input!(layer, x)
    forward_defn(layer, x)
  end

  @doc """
  Defn-compatible LoRA forward pass.

  This version skips runtime shape validation so it can be composed inside
  larger `defn` functions. It expects `layer` to contain `A` and `B` tensors
  with compatible shapes.
  """
  defn forward_defn(layer, x) do
    x
    |> Nx.dot([-1], layer.a, [0])
    |> Nx.dot([-1], layer.b, [0])
    |> Nx.multiply(layer.alpha)
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def call(layer, x), do: forward(layer, x)

  defp init_a(in_dim, rank, seed) do
    # PyTorch's kaiming_uniform_(tensor, a=sqrt(5)) gives bound = 1 / sqrt(fan_in).
    # For the requested A shape {in_dim, rank}, PyTorch treats fan_in as rank.
    bound = :math.sqrt(1.0 / rank)
    key = Random.manual_seed(seed)

    {a, _key} = Nx.Random.uniform(key, -bound, bound, shape: {in_dim, rank}, type: {:f, 32})
    Nx.as_type(a, {:f, 32})
  end

  defp init_b(rank, out_dim) do
    0.0
    |> Nx.broadcast({rank, out_dim})
    |> Nx.as_type({:f, 32})
  end

  defp validate_parameters!(%__MODULE__{} = layer) do
    assert_shape!(layer.a, {layer.in_dim, layer.rank}, "A")
    assert_shape!(layer.b, {layer.rank, layer.out_dim}, "B")
    layer
  end

  defp validate_input!(%__MODULE__{} = layer, x) do
    last_axis = tuple_size(Nx.shape(x)) - 1

    if Nx.axis_size(x, last_axis) != layer.in_dim do
      raise ArgumentError,
            "expected input final axis to be #{layer.in_dim}, got shape #{inspect(Nx.shape(x))}"
    end
  end

  defp assert_shape!(tensor, expected_shape, name) do
    if Nx.shape(tensor) != expected_shape do
      raise ArgumentError,
            "expected LoRA #{name} shape #{inspect(expected_shape)}, got #{inspect(Nx.shape(tensor))}"
    end
  end
end

defimpl Nx.Container, for: LlmScratch.LoRALayer do
  def traverse(layer, acc, fun) do
    {a, acc} = fun.(layer.a, acc)
    {b, acc} = fun.(layer.b, acc)
    {%{layer | a: a, b: b}, acc}
  end

  def reduce(layer, acc, fun) do
    acc
    |> then(&fun.(layer.a, &1))
    |> then(&fun.(layer.b, &1))
  end

  def serialize(layer) do
    metadata = Map.take(layer, [:in_dim, :out_dim, :rank, :alpha])
    {__MODULE__, [a: layer.a, b: layer.b], metadata}
  end

  def deserialize([a: a, b: b], metadata) do
    struct!(LlmScratch.LoRALayer, Map.merge(metadata, %{a: a, b: b}))
  end
end
