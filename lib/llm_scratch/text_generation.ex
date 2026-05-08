defmodule LlmScratch.TextGeneration do
  @moduledoc """
  Greedy token generation helpers for GPT-style models.

  `generate_text_simple/4` mirrors the Python loop from the book:

      idx_cond = idx[:, -context_size:]
      logits = model(idx_cond)
      logits = logits[:, -1, :]
      probas = torch.softmax(logits, dim=-1)
      idx_next = torch.argmax(probas, dim=-1, keepdim=True)
      idx = torch.cat((idx, idx_next), dim=1)

  The model is expected to be a struct whose module exports `forward/2`.
  Existing modules such as `LlmScratch.GPTModel` and
  `LlmScratch.DummyGPTModel` satisfy that contract.
  """

  @spec generate_text_simple(struct(), Nx.Tensor.t(), non_neg_integer(), pos_integer()) ::
          Nx.Tensor.t()
  @doc """
  Generates new token ids with greedy decoding.

  ## Arguments

    * `model` - a GPT-style model struct whose module exports `forward/2`.
      `forward/2` receives token ids shaped `{batch_size, seq_len}` and must
      return logits shaped `{batch_size, seq_len, vocab_size}`.

    * `idx` - token ids shaped `{batch_size, seq_len}`. The returned tensor
      keeps the same batch size and token id type.

    * `max_new_tokens` - number of new tokens to generate and append. Use `0`
      to return `idx` unchanged.

    * `context_size` - maximum number of latest tokens to pass to the model on
      each generation step.

  ## Returns

  A token id tensor shaped `{batch_size, seq_len + max_new_tokens}`.

  The next token is selected with `argmax` over the final-position softmax
  probabilities, then appended to `idx` along the sequence axis.

  ## Examples

      generated =
        LlmScratch.TextGeneration.generate_text_simple(
          model,
          Nx.tensor([[6109, 3626]], type: {:s, 64}),
          5,
          1024
        )

      Nx.shape(generated)
      #=> {1, 7}
  """
  def generate_text_simple(model, %Nx.Tensor{} = idx, max_new_tokens, context_size)
      when is_integer(max_new_tokens) and max_new_tokens >= 0 and is_integer(context_size) and
             context_size > 0 do
    validate_idx_shape!(idx)

    Enum.reduce(1..max_new_tokens//1, idx, fn _step, acc ->
      idx_cond = last_tokens(acc, context_size)
      logits = forward!(model, idx_cond)
      logits = last_position_logits(logits)
      probas = Axon.Activations.softmax(logits, axis: -1)

      idx_next =
        probas
        |> Nx.argmax(axis: -1, keep_axis: true)
        |> Nx.as_type(Nx.type(acc))

      Nx.concatenate([acc, idx_next], axis: 1)
    end)
  end

  defp last_tokens(idx, context_size) do
    {_batch_size, seq_len} = Nx.shape(idx)
    length = min(seq_len, context_size)
    start = seq_len - length

    Nx.slice_along_axis(idx, start, length, axis: 1)
  end

  defp last_position_logits(logits) do
    case Nx.shape(logits) do
      {_batch_size, seq_len, _vocab_size} ->
        logits
        |> Nx.slice_along_axis(seq_len - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])

      shape ->
        raise ArgumentError,
              "expected logits shape {batch_size, seq_len, vocab_size}, got: #{inspect(shape)}"
    end
  end

  defp forward!(%module{} = model, idx_cond) do
    if function_exported?(module, :forward, 2) do
      module.forward(model, idx_cond)
    else
      raise ArgumentError, "expected #{inspect(module)} to export forward/2"
    end
  end

  defp validate_idx_shape!(idx) do
    case Nx.shape(idx) do
      {_batch_size, _seq_len} ->
        :ok

      shape ->
        raise ArgumentError, "expected idx shape {batch_size, seq_len}, got: #{inspect(shape)}"
    end
  end
end
