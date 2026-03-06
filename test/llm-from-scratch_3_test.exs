defmodule LlmFromScratch3Test do
  use ExUnit.Case

  test "simple self-attention mechanism" do
    inputs =
      Nx.tensor(
        [
          # Your (x^1)
          [0.43, 0.15, 0.89],
          # journey (x^2)
          [0.55, 0.87, 0.66],
          # starts (x^3)
          [0.57, 0.85, 0.64],
          # with (x^4)
          [0.22, 0.58, 0.33],
          # one (x^5)
          [0.77, 0.25, 0.10],
          # step (x^6)
          [0.05, 0.80, 0.55]
        ],
        type: {:f, 32}
      )

    assert Nx.shape(inputs) == {6, 3}

    query =
      inputs
      |> Nx.slice_along_axis(1, 1, axis: 0)
      |> Nx.squeeze(axes: [0])

    assert Nx.shape(query) == {3}

    attn_scores_2 = Nx.dot(inputs, [1], query, [0])

    assert Nx.shape(attn_scores_2) == {6}

    expected_attn_scores_2 =
      Nx.tensor(
        [
          0.9544000029563904,
          1.4950000047683716,
          1.4754000902175903,
          0.8434000015258789,
          0.7070000171661377,
          1.0865000486373901
        ],
        type: :f32
      )

    assert Nx.all_close(attn_scores_2, expected_attn_scores_2, atol: 1.0e-6) |> Nx.to_number() ==
             1,
           "Attention scores should match expected values exactly"

    attn_scores_2_normalized =
      Nx.divide(attn_scores_2, Nx.sum(attn_scores_2, axes: [0]))

    expected_attn_scores_2_normalized =
      Nx.tensor(
        [
          0.14545010030269623,
          0.22783729434013367,
          0.22485026717185974,
          0.1285337507724762,
          0.10774646699428558,
          0.1655820906162262
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(attn_scores_2_normalized, expected_attn_scores_2_normalized, atol: 1.0e-6)
           |> Nx.to_number() == 1,
           "Normalized attention scores should match expected values exactly"

    attn_scores_2_normalized_sum = Nx.sum(attn_scores_2_normalized, axes: [0])

    assert Nx.all_close(attn_scores_2_normalized_sum, Nx.tensor([1.0], type: {:f, 32}),
             atol: 1.0e-6
           )
           |> Nx.to_number() == 1,
           "Sum of normalized attention scores should be 1.0"

    attn_scores_2_softmax_naive = softmax_naive(attn_scores_2)

    expected_attn_scores_2_softmax =
      Nx.tensor(
        [
          0.13854756951332092,
          0.237891286611557,
          0.23327402770519257,
          0.12399158626794815,
          0.10818187147378922,
          0.15811361372470856
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(
             attn_scores_2_softmax_naive,
             expected_attn_scores_2_softmax,
             atol: 1.0e-6
           )
           |> Nx.to_number() == 1,
           "Softmax of attention scores should match expected values exactly"

    attn_scores_2_softmax_axon = Axon.Activations.softmax(attn_scores_2)

    expected_attn_scores_2_softmax_axon =
      Nx.tensor(
        [
          0.13854756951332092,
          0.237891286611557,
          0.23327402770519257,
          0.12399158626794815,
          0.10818187147378922,
          0.15811361372470856
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(attn_scores_2_softmax_axon, expected_attn_scores_2_softmax_axon,
             atol: 1.0e-6
           )
           |> Nx.to_number() == 1,
           "Axon Softmax of attention scores should match expected values exactly"

    context_vec_2 =
      attn_scores_2_softmax_axon
      |> Nx.new_axis(-1)
      |> Nx.multiply(inputs)
      |> Nx.sum(axes: [0])

    expected_context_vec_2 =
      Nx.tensor(
        [0.4418657422065735, 0.6514819860458374, 0.5683088898658752],
        type: {:f, 32}
      )

    assert Nx.all_close(context_vec_2, expected_context_vec_2, atol: 1.0e-6) |> Nx.to_number() ==
             1,
           "Context vector should match expected values exactly"
  end

  defp softmax_naive(%Nx.Tensor{} = x) do
    exp_x = Nx.exp(x)
    Nx.divide(exp_x, Nx.sum(exp_x, axes: [0]))
  end
end
