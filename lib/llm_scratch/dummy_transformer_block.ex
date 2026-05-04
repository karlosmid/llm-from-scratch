defmodule LlmScratch.DummyTransformerBlock do
  @moduledoc """
  Placeholder transformer block.

  It mirrors the book's dummy PyTorch block and returns its input unchanged.
  """

  defstruct [:cfg]

  @type t :: %__MODULE__{cfg: map()}

  @spec new(map()) :: t()
  def new(cfg) when is_map(cfg), do: %__MODULE__{cfg: cfg}

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{}, %Nx.Tensor{} = x), do: x
end
