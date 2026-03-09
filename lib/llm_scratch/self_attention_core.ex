defmodule LlmScratch.SelfAttentionCore do
  @moduledoc false

  @spec context_from_qkv(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), pos_integer()) ::
          Nx.Tensor.t()
  def context_from_qkv(q, k, v, d_out) do
    attn_scores = Nx.dot(q, [1], k, [1])
    d_k = Nx.tensor(d_out, type: {:f, 32})

    attn_weights =
      attn_scores
      |> Nx.divide(Nx.sqrt(d_k))
      |> Axon.Activations.softmax(axis: -1)

    Nx.dot(attn_weights, [1], v, [0])
  end
end
