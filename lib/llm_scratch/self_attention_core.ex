defmodule LlmScratch.SelfAttentionCore do
  @moduledoc """
  has function that is shared in LlmScratch.SelfAttentionV1 and LlmScratch.SelfAttentionV2

  """

  @spec context_from_qkv(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), pos_integer()) ::
          Nx.Tensor.t()
  @doc """
  Computes self-attention context vectors from projected query, key, and value
  tensors.

  ## Arguments

    * `q` - query tensor of shape `{num_tokens, d_out}`.
    * `k` - key tensor of shape `{num_tokens, d_out}`.
    * `v` - value tensor of shape `{num_tokens, d_out}`.
    * `d_out` - feature size used for score scaling by `sqrt(d_out)`.

  ## Returns

    * context tensor of shape `{num_tokens, d_out}`.
  """
  def context_from_qkv(q, k, v, d_out) do
    # {no_of_tokens, d_out} dot {no_of_tokens, d_out} = {no_of_tokens, no_of_tokens}
    # dot over second dimension d_out

    attn_scores = Nx.dot(q, [1], k, [1])
    d_k = Nx.tensor(d_out, type: {:f, 32})

    attn_weights =
      attn_scores
      |> Nx.divide(Nx.sqrt(d_k))
      |> Axon.Activations.softmax(axis: -1)

    # {no_of_tokens, no_of_tokens} dot {no_of_tokens, d_out} = {no_of_tokens, d_out}
    # because we are dotting row (over second dimension) with column (over first dimension)

    Nx.dot(attn_weights, [1], v, [0])
  end
end
