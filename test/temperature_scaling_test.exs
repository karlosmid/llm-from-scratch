defmodule LlmScratch.TemperatureScalingTest do
  use ExUnit.Case

  alias LlmScratch.TemperatureScaling

  test "softmax_with_temperature scales logits before softmax" do
    logits = Nx.tensor([1.0, 2.0, 3.0])

    assert TemperatureScaling.softmax_with_temperature(logits, 1.0)
           |> Nx.all_close(Axon.Activations.softmax(logits), atol: 1.0e-6)
           |> Nx.to_number() == 1

    assert TemperatureScaling.softmax_with_temperature(logits, 2.0)
           |> Nx.all_close(Axon.Activations.softmax(Nx.divide(logits, 2.0)), atol: 1.0e-6)
           |> Nx.to_number() == 1
  end

  test "multinomial samples token ids from probabilities with a seed" do
    probas =
      Nx.tensor([
        0.060907062,
        0.0016312535,
        0.0001001936,
        0.57212007,
        0.003419003,
        0.0001325691,
        0.0001012005,
        0.3575764,
        0.0040122364
      ])

    assert TemperatureScaling.multinomial(probas, samples: 1, seed: 123) == [7]
  end

  test "sampled_token_frequencies counts sampled ids by inverse vocabulary" do
    probas =
      Nx.tensor([
        0.060907062,
        0.0016312535,
        0.0001001936,
        0.57212007,
        0.003419003,
        0.0001325691,
        0.0001012005,
        0.3575764,
        0.0040122364
      ])

    inverse_vocab = %{
      0 => "closer",
      1 => "every",
      2 => "effort",
      3 => "forward",
      4 => "inches",
      5 => "moves",
      6 => "pizza",
      7 => "toward",
      8 => "you"
    }

    assert TemperatureScaling.sampled_token_frequencies(probas, inverse_vocab,
             seed: 123,
             samples: 1_000
           ) == [
             {64, "closer"},
             {2, "every"},
             {0, "effort"},
             {572, "forward"},
             {3, "inches"},
             {0, "moves"},
             {0, "pizza"},
             {357, "toward"},
             {2, "you"}
           ]
  end
end
