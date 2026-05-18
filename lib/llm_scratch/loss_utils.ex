defmodule LlmScratch.LossUtils do
  @moduledoc """
  Loss helpers for GPT-style logits and target token tensors.
  """

  @doc """
  Returns the predicted probabilities for the expected target token ids.

  `probas` should be shaped `{batch_size, seq_len, vocab_size}` and `targets`
  should be shaped `{batch_size, seq_len}`. The returned tensor is flattened to
  `{batch_size * seq_len}`.

  ## Examples

      iex> probas = Nx.tensor([[[0.5, 0.25, 0.25], [0.125, 0.375, 0.5]]])
      iex> targets = Nx.tensor([[0, 2]])
      iex> LlmScratch.LossUtils.target_token_probas(probas, targets) |> Nx.to_flat_list()
      [0.5, 0.5]
  """
  @spec target_token_probas(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def target_token_probas(%Nx.Tensor{} = probas, %Nx.Tensor{} = targets) do
    vocab_size = elem(Nx.shape(probas), 2)

    #{2, 3, 50257} => {6, 50257}
    probas
    |> Nx.reshape({:auto, vocab_size})
    |> Nx.take_along_axis(
      #{2, 3} => {6, 1}
      Nx.reshape(targets, {:auto, 1}), axis: 1
    )
    |> Nx.squeeze(axes: [1])
  end

  @doc """
  Calculates mean cross entropy loss from logits and target token ids.
  step1: logits
  step2: probabilities
  step3: target probabilities
  step4: logarithmic probabilities
  step5: average logarithmic probabilities
  step6: negative average log probabilities

  `logits` should be shaped `{batch_size, seq_len, vocab_size}` and `targets`
  should be shaped `{batch_size, seq_len}`.

  ## Examples

      iex> logits = Nx.log(Nx.tensor([[[0.5, 0.25, 0.25], [0.125, 0.375, 0.5]]]))
      iex> targets = Nx.tensor([[0, 2]])
      iex> loss = LlmScratch.LossUtils.cross_entropy_loss(logits, targets)
      iex> Float.round(Nx.to_number(loss), 6)
      0.693147
  """
  @spec cross_entropy_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def cross_entropy_loss(%Nx.Tensor{} = logits, %Nx.Tensor{} = targets) do
    logits #step1
    |> Axon.Activations.softmax(axis: -1) #step2
    |> target_token_probas(targets) #step3
    |> Nx.log() #step4
    |> Nx.negate() #step6
    |> Nx.mean() #step5
  end
end
