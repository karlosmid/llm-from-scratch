defmodule SelfAttentionV2 do
  @moduledoc """
  Compatibility wrapper around `LlmScratch.SelfAttentionV2`.
  """

  alias LlmScratch.SelfAttentionV2, as: Impl

  @spec new(pos_integer(), pos_integer()) :: Impl.t()
  defdelegate new(d_in, d_out), to: Impl

  @spec new(pos_integer(), pos_integer(), keyword()) :: Impl.t()
  defdelegate new(d_in, d_out, opts), to: Impl

  @spec forward(Impl.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defdelegate forward(self_attention, inputs), to: Impl
end
