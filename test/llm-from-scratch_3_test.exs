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

  test "attention weights for all tokens" do
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

    # Scores for every query against every key: Q @ K^T
    attn_scores = Nx.dot(inputs, [1], inputs, [1])
    assert Nx.shape(attn_scores) == {6, 6}

    expected_attn_scores =
      Nx.tensor(
        [
          [
            0.9994999766349792,
            0.9544000029563904,
            0.9422000050544739,
            0.47530001401901245,
            0.4575999975204468,
            0.6309999823570251
          ],
          [
            0.9544000029563904,
            1.4950000047683716,
            1.4754000902175903,
            0.8434000015258789,
            0.7070000171661377,
            1.0865000486373901
          ],
          [
            0.9422000050544739,
            1.4754000902175903,
            1.4570000171661377,
            0.8295999765396118,
            0.715399980545044,
            1.0605000257492065
          ],
          [
            0.47530001401901245,
            0.8434000015258789,
            0.8295999765396118,
            0.4936999976634979,
            0.3473999798297882,
            0.656499981880188
          ],
          [
            0.4575999975204468,
            0.7070000171661377,
            0.715399980545044,
            0.3473999798297882,
            0.665399968624115,
            0.29350000619888306
          ],
          [
            0.6309999823570251,
            1.0865000486373901,
            1.0605000257492065,
            0.656499981880188,
            0.29350000619888306,
            0.9450000524520874
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(attn_scores, expected_attn_scores, atol: 1.0e-6) |> Nx.to_number() == 1,
           "Attention scores should match expected values exactly"

    # Row-wise softmax gives attention weights per query token
    attn_weights_softmax = Axon.Activations.softmax(attn_scores, axis: -1)

    expected_attn_weights_softmax =
      Nx.tensor(
        [
          [
            0.2098347693681717,
            0.20058146119117737,
            0.19814923405647278,
            0.12422822415828705,
            0.12204873561859131,
            0.14515765011310577
          ],
          [
            0.13854758441448212,
            0.237891286611557,
            0.23327404260635376,
            0.12399159371852875,
            0.10818187147378922,
            0.15811361372470856
          ],
          [
            0.1390075981616974,
            0.23692145943641663,
            0.23260195553302765,
            0.12420440465211868,
            0.11080020666122437,
            0.15646442770957947
          ],
          [
            0.14352688193321228,
            0.20739442110061646,
            0.20455202460289001,
            0.14619223773479462,
            0.12629525363445282,
            0.172039195895195
          ],
          [
            0.15261085331439972,
            0.19583867490291595,
            0.1974906474351883,
            0.13668666779994965,
            0.18785890936851501,
            0.12951429188251495
          ],
          [
            0.13847115635871887,
            0.2183637171983719,
            0.21275943517684937,
            0.14204756915569305,
            0.09880637377500534,
            0.18955175578594208
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(attn_weights_softmax, expected_attn_weights_softmax, atol: 1.0e-6)
           |> Nx.to_number() ==
             1

    assert Nx.shape(attn_weights_softmax) == {6, 6}

    row_sums = Nx.sum(attn_weights_softmax, axes: [1])
    ones = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {6})

    assert Nx.all_close(row_sums, ones, atol: 1.0e-6) |> Nx.to_number() == 1

    context_vecs = Nx.dot(attn_weights_softmax, [1], inputs, [0])

    expected_context_vecs =
      Nx.tensor(
        [
          [0.4420594274997711, 0.5930986404418945, 0.5789890885353088],
          [0.4418657422065735, 0.6514819860458374, 0.5683088898658752],
          [0.4431275427341461, 0.6495946049690247, 0.5670731067657471],
          [0.43038973212242126, 0.6298280954360962, 0.5510270595550537],
          [0.4671017527580261, 0.5909927487373352, 0.5265965461730957],
          [0.41772449016571045, 0.650323212146759, 0.5645352005958557]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(context_vecs, expected_context_vecs, atol: 1.0e-6) |> Nx.to_number() ==
             1,
           "Context vectors should match expected values exactly"

    assert Nx.shape(context_vecs) == {6, 3}
  end

  test "self-attention mechanism with trainable weights" do
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

    x_2 = Nx.slice_along_axis(inputs, 1, 1, axis: 0) |> Nx.squeeze(axes: [0])
    d_in = Nx.shape(x_2) |> elem(0)
    d_out = 2
    key = LlmScratch.Random.manual_seed(123)

    query_weights = Axon.Initializers.uniform(scale: 1.0).({d_in, d_out}, {:f, 32}, key)
    key_weights = Axon.Initializers.uniform(scale: 1.0).({d_in, d_out}, {:f, 32}, key)
    value_weights = Axon.Initializers.uniform(scale: 1.0).({d_in, d_out}, {:f, 32}, key)

    # 1x3 dot 3x2 = 1x2
    query_2 = Nx.dot(x_2, query_weights)
    assert Nx.shape(query_2) == {d_out}
    expected_query_2 = Nx.tensor([-0.20726783573627472, -0.3094936013221741], type: {:f, 32})
    expected_query_2_book = Nx.tensor([0.4306, 1.4551], type: {:f, 32})

    assert Nx.all_close(query_2, expected_query_2, atol: 1.0e-6) |> Nx.to_number() == 1,
           "query_2 should match expected values exactly"

    refute Nx.all_close(query_2, expected_query_2_book, atol: 1.0e-6) |> Nx.to_number() == 1,
           "query_2 should not match expected values exactly due to different random number generators in PyTorch and Nx"

    key_2 = Nx.dot(x_2, key_weights)
    assert Nx.shape(key_2) == {d_out}

    value_2 = Nx.dot(x_2, value_weights)
    assert Nx.shape(value_2) == {d_out}

    # 6x3 dot 3x2 = 6x2
    keys = Nx.dot(inputs, key_weights)
    assert Nx.shape(keys) == {6, d_out}

    # 6x3 dot 3x2 = 6x2
    values = Nx.dot(inputs, value_weights)
    assert Nx.shape(values) == {6, d_out}

    keys_2 = Nx.slice_along_axis(keys, 1, 1, axis: 0) |> Nx.squeeze(axes: [0])
    # 1x2 dot 2x1 = 1x1
    attn_scores_22 = Nx.dot(query_2, keys_2)
    assert Nx.shape(attn_scores_22) == {}
    expected_attn_scores_22 = Nx.tensor([0.1387462466955185], type: {:f, 32})

    assert Nx.all_close(attn_scores_22, expected_attn_scores_22, atol: 1.0e-6) |> Nx.to_number() ==
             1,
           "attn_scores_22 should match expected values exactly"

    expected_attn_scores_22_book = Nx.tensor([1.8524], type: {:f, 32})

    refute Nx.all_close(attn_scores_22, expected_attn_scores_22_book, atol: 1.0e-6)
           |> Nx.to_number() == 1,
           "attn_scores_22 should not match expected values exactly due to different random number generators in PyTorch and Nx"

    # {2} dot {6,2} over feature dim -> {6}
    attn_scores_2 = Nx.dot(query_2, [0], keys, [1])
    assert Nx.shape(attn_scores_2) == {6}

    expected_attn_scores_2 =
      Nx.tensor(
        [
          -0.17058327794075012,
          0.1387462466955185,
          0.14079777896404266,
          0.10855000466108322,
          0.13786746561527252,
          0.09178745746612549
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(attn_scores_2, expected_attn_scores_2, atol: 1.0e-6) |> Nx.to_number() ==
             1,
           "attn_scores_2 should match expected values"

    d_k = Nx.axis_size(keys, -1) |> Nx.tensor(type: {:f, 32})

    attn_weights_2 =
      attn_scores_2
      |> Nx.divide(Nx.sqrt(d_k))
      |> Axon.Activations.softmax(axis: -1)

    assert Nx.shape(attn_weights_2) == {6}

    expected_attn_weights_2 =
      Nx.tensor(
        [
          0.1397317796945572,
          0.17389537394046783,
          0.1741478145122528,
          0.1702217161655426,
          0.17378734052181244,
          0.16821600496768951
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(attn_weights_2, expected_attn_weights_2, atol: 1.0e-6) |> Nx.to_number() ==
             1,
           "attn_weights_2 should match expected values"

    context_vec_2 = Nx.dot(attn_weights_2, [0], values, [0])
    assert Nx.shape(context_vec_2) == {d_out}

    expected_context_vec_2 =
      Nx.tensor(
        [-0.11537063866853714, -0.18990936875343323],
        type: {:f, 32}
      )

    assert Nx.all_close(context_vec_2, expected_context_vec_2, atol: 1.0e-6) |> Nx.to_number() ==
             1,
           "context_vec_2 should match expected values"
  end

  test "Implementing a compact self-attention module" do
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

    sa = LlmScratch.SelfAttentionV1.new(3, 2, seed: 123)

    context_vecs = LlmScratch.SelfAttentionV1.forward(sa, inputs)

    assert Nx.shape(context_vecs) == {6, 2}

    expected_context_vecs =
      Nx.tensor(
        [
          [-0.07548463344573975, -0.15017275512218475],
          [-0.11537063866853714, -0.18990936875343323],
          [-0.11561498790979385, -0.19015151262283325],
          [-0.11222726106643677, -0.1867683380842209],
          [-0.11577533185482025, -0.1902845799922943],
          [-0.11009891331195831, -0.18466056883335114]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(context_vecs, expected_context_vecs, atol: 1.0e-6) |> Nx.to_number() == 1,
           "context_vecs should match expected values exactly"
  end

  test "compact self-attention module v2 uses Axon dense initialization" do
    inputs =
      Nx.tensor(
        [
          [0.43, 0.15, 0.89],
          [0.55, 0.87, 0.66],
          [0.57, 0.85, 0.64],
          [0.22, 0.58, 0.33],
          [0.77, 0.25, 0.10],
          [0.05, 0.80, 0.55]
        ],
        type: {:f, 32}
      )

    sa = LlmScratch.SelfAttentionV2.new(3, 2, seed: 789)

    context_vecs = LlmScratch.SelfAttentionV2.forward(sa, inputs)

    assert Nx.shape(context_vecs) == {6, 2}

    expected_context_vecs =
      Nx.tensor(
        [
          [0.20869487524032593, -0.11512904614210129],
          [0.1995905637741089, -0.10041604191064835],
          [0.197800412774086, -0.09748103469610214],
          [0.20753224194049835, -0.11311019212007523],
          [0.16690319776535034, -0.04650232568383217],
          [0.22278699278831482, -0.1379932463169098]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(context_vecs, expected_context_vecs, atol: 1.0e-6) |> Nx.to_number() == 1,
           "context_vecs should match expected values exactly"
  end

  test "exercise 3.1 - transfer v2 weights into v1 and match outputs" do
    inputs =
      Nx.tensor(
        [
          [0.43, 0.15, 0.89],
          [0.55, 0.87, 0.66],
          [0.57, 0.85, 0.64],
          [0.22, 0.58, 0.33],
          [0.77, 0.25, 0.10],
          [0.05, 0.80, 0.55]
        ],
        type: {:f, 32}
      )

    sa_v2 = LlmScratch.SelfAttentionV2.new(3, 2, seed: 123)

    # Axon dense kernels in this project are already shaped {d_in, d_out},
    # matching V1's expected projection weight layout.
    sa_v1 =
      LlmScratch.SelfAttentionV1.new(3, 2,
        w_q: sa_v2.w_q.kernel,
        w_k: sa_v2.w_k.kernel,
        w_v: sa_v2.w_v.kernel
      )

    context_v2 = LlmScratch.SelfAttentionV2.forward(sa_v2, inputs)
    context_v1 = LlmScratch.SelfAttentionV1.forward(sa_v1, inputs)

    assert Nx.shape(context_v1) == {6, 2}
    assert Nx.shape(context_v2) == {6, 2}

    assert Nx.all_close(context_v1, context_v2, atol: 1.0e-6) |> Nx.to_number() == 1,
           "after copying weights, v1 and v2 should produce the same outputs"
  end
end
