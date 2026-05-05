defmodule LlmScratch.GELU do
  @moduledoc """
  Gaussian Error Linear Unit activation implemented with Nx tensors.

  GELU is a smooth nonlinear activation commonly used in transformer blocks.
  Instead of zeroing all negative inputs like ReLU, it smoothly gates values
  based on their magnitude. GPT-style models typically use the tanh-based
  approximation rather than the exact error-function version.

  This module mirrors the PyTorch implementation:

      0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
      ))

  The equivalent Nx formula is:

      0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

  The activation is stateless, so the struct only exists to match the
  module-style shape used by other examples in this project.
  """

  defstruct []

  @type t :: %__MODULE__{}

  @sqrt_2_over_pi :math.sqrt(2.0 / :math.pi())

  @spec new() :: t()
  @doc """
  Creates a stateless GELU activation module.

  ## Example

      iex> gelu = LlmScratch.GELU.new()
      iex> LlmScratch.GELU.forward(gelu, Nx.tensor([0.0])) |> Nx.to_flat_list()
      [0.0]
  """
  def new, do: %__MODULE__{}

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Applies GELU to `x` using a `%LlmScratch.GELU{}` module struct.

  This form is useful when composing modules that all expose a `forward/2`
  function. The input can have any shape; GELU is applied element-wise and the
  output keeps the same shape.

  ## Example

      iex> gelu = LlmScratch.GELU.new()
      iex> x = Nx.tensor([-1.0, 0.0, 1.0])
      iex> LlmScratch.GELU.forward(gelu, x) |> Nx.round(6) |> Nx.to_flat_list()
      [-0.158808, 0.0, 0.841192]
  """
  def forward(%__MODULE__{}, %Nx.Tensor{} = x), do: forward(x)

  @spec forward(Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Applies GELU to `x` directly.

  Since GELU has no parameters, this convenience form produces the same result
  as `forward(LlmScratch.GELU.new(), x)`.

  ## Example

      iex> x = Nx.tensor([[-3.0, 0.0, 3.0]])
      iex> LlmScratch.GELU.forward(x) |> Nx.round(6) |> Nx.to_flat_list()
      [-0.003637, 0.0, 2.996363]
  """
  def forward(%Nx.Tensor{} = x) do
    x_cubed = Nx.pow(x, 3)
    inner = Nx.multiply(@sqrt_2_over_pi, Nx.add(x, Nx.multiply(0.044715, x_cubed)))

    x
    |> Nx.multiply(0.5)
    |> Nx.multiply(Nx.add(1.0, Nx.tanh(inner)))
  end
end
