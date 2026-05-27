defmodule LlmScratch.TemperatureScaling do
  @moduledoc """
  Sampling helpers used by temperature-scaled text generation examples.

  `multinomial/2` mirrors `torch.multinomial(probas, num_samples: n)` for a
  one-dimensional probability tensor. It samples token ids from `0` to
  `Nx.size(probas) - 1`.
  """

  @doc """
  Applies temperature scaling to `logits`, then returns softmax probabilities.

  This mirrors:

      scaled_logits = logits / temperature
      torch.softmax(scaled_logits, dim=0)

  A lower temperature sharpens the distribution; a higher temperature flattens
  it. `temperature` must be greater than zero.

  ## Example

      logits = Nx.tensor([1.0, 2.0, 3.0])
      LlmScratch.TemperatureScaling.softmax_with_temperature(logits, 1.0)

  """
  @spec softmax_with_temperature(Nx.Tensor.t(), number()) :: Nx.Tensor.t()
  def softmax_with_temperature(logits, temperature)
      when is_number(temperature) and temperature > 0 do
    logits
    |> Nx.divide(temperature)
    |> Axon.Activations.softmax()
  end

  @doc """
  Samples token ids from `probas`.

  Options:

    * `:samples` - number of token ids to sample.
    * `:seed` - deterministic seed for `Nx.Random.key/1`.

  ## Example

      probas = Nx.tensor([0.25, 0.75])
      LlmScratch.TemperatureScaling.multinomial(probas, samples: 3, seed: 123)

  """
  @spec multinomial(Nx.Tensor.t(), keyword()) :: [integer()]
  def multinomial(probas, opts) do
    samples = Keyword.fetch!(opts, :samples)
    seed = Keyword.fetch!(opts, :seed)

    {sampled_token_ids, _key} =
      Nx.Random.choice(Nx.Random.key(seed), Nx.iota({Nx.size(probas)}), probas, samples: samples)

    Nx.to_flat_list(sampled_token_ids)
  end

  @doc """
  Samples repeatedly from `probas` and returns `{frequency, token}` pairs.

  `inverse_vocab` maps token ids to token strings, for example
  `%{0 => "closer", 1 => "every"}`.
  """
  @spec sampled_token_frequencies(Nx.Tensor.t(), %{integer() => String.t()}, keyword()) :: [
          {non_neg_integer(), String.t()}
        ]
  def sampled_token_frequencies(probas, inverse_vocab, opts) when is_map(inverse_vocab) do
    sampled_ids = multinomial(probas, opts)
    frequencies = Enum.frequencies(sampled_ids)

    0..(map_size(inverse_vocab) - 1)
    |> Enum.map(fn token_id -> {Map.get(frequencies, token_id, 0), inverse_vocab[token_id]} end)
  end
end
