defmodule LlmScratch.TextGeneration do
  @moduledoc """
  Token generation helpers for GPT-style models.

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

  @default_sampling_seed 123

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
   ## Algorithm:

      Start with prompt tokens.
      Keep only the last context_size tokens.
      Run the model.
      Take logits from the last position.
      Convert logits to probabilities.
      Pick the highest-probability token with argmax.
      Append it.
      Repeat max_new_tokens times.
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

  @spec generate(
          struct(),
          Nx.Tensor.t(),
          non_neg_integer(),
          pos_integer(),
          number(),
          pos_integer() | nil,
          integer() | nil,
          keyword()
        ) :: Nx.Tensor.t()
  @doc """
  Generates new token ids with greedy or temperature-scaled sampling.

  This mirrors the book's `generate` function:

      idx_cond = idx[:, -context_size:]
      logits = model(idx_cond)
      logits = logits[:, -1, :]

      if top_k is not None:
          top_logits, _ = torch.topk(logits, top_k)
          min_val = top_logits[:, -1]
          logits = torch.where(logits < min_val, -inf, logits)

      if temperature > 0.0:
          logits = logits / temperature
          probs = torch.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)
      else:
          idx_next = torch.argmax(logits, dim=-1, keepdim=True)

  ## Arguments

    * `model` - a GPT-style model struct whose module exports `forward/2`.

    * `idx` - token ids shaped `{batch_size, seq_len}`.

    * `max_new_tokens` - maximum number of new tokens to append.

    * `context_size` - maximum number of latest tokens to pass to the model.

    * `temperature` - `0.0` uses deterministic argmax decoding. Values greater
      than `0.0` divide logits by the temperature and sample from the softmax
      distribution.

    * `top_k` - when set, keeps only the `top_k` highest logits per batch row
      before decoding.

    * `eos_id` - when set, generation stops before appending an EOS token once
      all batch rows predict that id.

  ## Options

    * `:seed` - deterministic sampling seed used when `temperature > 0.0`.
      Defaults to `#{@default_sampling_seed}`, matching the chapter examples.

  ## Examples

      token_ids =
        LlmScratch.TextGeneration.generate(
          model,
          LlmScratch.TextUtils.text_to_token_ids("Every effort moves you", "code-davinci-002"),
          15,
          model.cfg.context_length,
          1.4,
          25
        )

  ## Algorithm

    Start with prompt token ids.
    Keep only the latest context_size tokens.
    Run the model.
    Take logits for the last position.
    Optionally keep only top-k logits.
    If temperature == 0, pick the highest logit.
    If temperature > 0, softmax and sample.
    Stop if EOS is predicted.
  """
  def generate(
        model,
        %Nx.Tensor{} = idx,
        max_new_tokens,
        context_size,
        temperature \\ 0.0,
        top_k \\ nil,
        eos_id \\ nil,
        opts \\ []
      )
      when is_integer(max_new_tokens) and max_new_tokens >= 0 and is_integer(context_size) and
             context_size > 0 and is_number(temperature) and temperature >= 0.0 and
             is_list(opts) do
    validate_idx_shape!(idx)
    validate_top_k!(top_k)

    seed = Keyword.get(opts, :seed, @default_sampling_seed)
    key = LlmScratch.Random.manual_seed(seed)

    {_key, generated_idx} =
      Enum.reduce_while(1..max_new_tokens//1, {key, idx}, fn _step, {key, acc} ->
        idx_cond = last_tokens(acc, context_size)
        logits = model |> forward!(idx_cond) |> last_position_logits() |> mask_top_k(top_k)

        {idx_next, key} =
          next_token(logits, temperature, key)

        idx_next = Nx.as_type(idx_next, Nx.type(acc))

        if eos_reached?(idx_next, eos_id) do
          {:halt, {key, acc}}
        else
          {:cont, {key, Nx.concatenate([acc, idx_next], axis: 1)}}
        end
      end)

    generated_idx
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

  defp mask_top_k(logits, nil), do: logits

  defp mask_top_k(logits, top_k) do
    {_batch_size, vocab_size} = Nx.shape(logits)

    if top_k > vocab_size do
      raise ArgumentError, "top_k must be <= vocab size #{vocab_size}, got: #{inspect(top_k)}"
    end

    {top_logits, _top_pos} = Nx.top_k(logits, k: top_k)

    min_top_logits =
      top_logits
      |> Nx.slice_along_axis(top_k - 1, 1, axis: -1)
      |> Nx.broadcast(Nx.shape(logits))

    Nx.select(
      Nx.less(logits, min_top_logits),
      Nx.broadcast(:neg_infinity, Nx.shape(logits)),
      logits
    )
  end

  defp next_token(logits, temperature, key) when temperature > 0.0 do
    probas =
      logits
      |> Nx.divide(temperature)
      |> Axon.Activations.softmax(axis: -1)

    sample_from_batch(probas, key)
  end

  defp next_token(logits, _temperature, key) do
    {Nx.argmax(logits, axis: -1, keep_axis: true), key}
  end

  defp sample_from_batch(probas, key) do
    probas = Nx.backend_transfer(probas, Nx.BinaryBackend)
    {batch_size, vocab_size} = Nx.shape(probas)

    {sampled_ids, key} =
      probas
      |> Nx.to_list()
      |> Enum.map_reduce(key, fn row_probas, key ->
        {sampled_token_ids, key} =
          Nx.Random.choice(
            key,
            Nx.iota({vocab_size}),
            Nx.tensor(row_probas),
            samples: 1
          )

        {[Nx.to_number(sampled_token_ids[0])], key}
      end)

    {Nx.tensor(sampled_ids, type: {:s, 64}) |> Nx.reshape({batch_size, 1}), key}
  end

  defp eos_reached?(_idx_next, nil), do: false

  defp eos_reached?(idx_next, eos_id) do
    idx_next
    |> Nx.equal(eos_id)
    |> Nx.all()
    |> Nx.to_number()
    |> Kernel.==(1)
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

  defp validate_top_k!(nil), do: :ok

  defp validate_top_k!(top_k) when is_integer(top_k) and top_k > 0, do: :ok

  defp validate_top_k!(top_k) do
    raise ArgumentError, "top_k must be a positive integer or nil, got: #{inspect(top_k)}"
  end
end
