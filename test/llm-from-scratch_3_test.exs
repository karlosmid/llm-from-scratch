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

  test "casual attention mask" do
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

    queries = LlmScratch.SelfAttentionV2.dense_project(inputs, sa_v2.w_q)
    keys = LlmScratch.SelfAttentionV2.dense_project(inputs, sa_v2.w_k)
    attn_scores = Nx.dot(queries, [1], keys, [1])

    attn_weights =
      Nx.divide(attn_scores, Nx.sqrt(Nx.axis_size(keys, -1)))
      |> Axon.Activations.softmax(axis: -1)

    expected_attn_weights =
      Nx.tensor(
        [
          [
            0.1531660407781601,
            0.1543799340724945,
            0.1536969095468521,
            0.1883421689271927,
            0.1566835194826126,
            0.1937314122915268
          ],
          [
            0.14633320271968842,
            0.14933642745018005,
            0.14835353195667267,
            0.19792792201042175,
            0.1513950079679489,
            0.2066539078950882
          ],
          [
            0.14623598754405975,
            0.14969584345817566,
            0.1487070620059967,
            0.19760626554489136,
            0.15132319927215576,
            0.20643165707588196
          ],
          [
            0.15648190677165985,
            0.15691247582435608,
            0.15638375282287598,
            0.1835598349571228,
            0.15919820964336395,
            0.18746380507946014
          ],
          [
            0.15005655586719513,
            0.1612723171710968,
            0.1604372262954712,
            0.18309712409973145,
            0.15431715548038483,
            0.1908196359872818
          ],
          [
            0.15678176283836365,
            0.1527770310640335,
            0.15225917100906372,
            0.1877351701259613,
            0.15941700339317322,
            0.1910298764705658
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(attn_weights, expected_attn_weights, atol: 1.0e-6) |> Nx.to_number() == 1,
           "attn_weights should match expected values"

    assert Nx.shape(attn_weights) == {6, 6}

    context_length = Nx.axis_size(attn_scores, 0)

    mask_simple =
      Nx.broadcast(1.0, {context_length, context_length})
      |> Nx.tril()

    mask_simple_expected =
      Nx.tensor(
        [
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(mask_simple, mask_simple_expected, atol: 1.0e-6) |> Nx.to_number() == 1,
           "mask_simple should match expected values"

    masked_attn_weights = Nx.multiply(attn_weights, mask_simple)

    assert Nx.shape(masked_attn_weights) == {6, 6}

    masked_attn_weights_expected =
      Nx.tensor(
        [
          [
            [0.1531660407781601, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.14633320271968842, 0.14933642745018005, 0.0, 0.0, 0.0, 0.0],
            [0.14623598754405975, 0.14969584345817566, 0.1487070620059967, 0.0, 0.0, 0.0],
            [
              0.15648190677165985,
              0.15691247582435608,
              0.15638375282287598,
              0.1835598349571228,
              0.0,
              0.0
            ],
            [
              0.15005655586719513,
              0.1612723171710968,
              0.1604372262954712,
              0.18309712409973145,
              0.15431715548038483,
              0.0
            ],
            [
              0.15678176283836365,
              0.1527770310640335,
              0.15225917100906372,
              0.1877351701259613,
              0.15941700339317322,
              0.1910298764705658
            ]
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(masked_attn_weights, masked_attn_weights_expected, atol: 1.0e-6)
           |> Nx.to_number() == 1,
           "masked_attn_weights should match expected values"

    row_sums = Nx.sum(masked_attn_weights, axes: [-1], keep_axes: true)

    masked_attn_weights_norm =
      Nx.divide(masked_attn_weights, row_sums)

    expected_masked_attn_weights_norm =
      Nx.tensor(
        [
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.49492135643959045, 0.5050787329673767, 0.0, 0.0, 0.0, 0.0],
          [0.32888707518577576, 0.33666834235191345, 0.3344445526599884, 0.0, 0.0, 0.0],
          [
            0.2395114302635193,
            0.2401704639196396,
            0.23936119675636292,
            0.2809569537639618,
            0.0,
            0.0
          ],
          [
            0.18544265627861023,
            0.1993032991886139,
            0.1982712745666504,
            0.2262747883796692,
            0.1907079815864563,
            0.0
          ],
          [
            0.15678176283836365,
            0.1527770310640335,
            0.15225917100906372,
            0.1877351701259613,
            0.15941700339317322,
            0.1910298764705658
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(masked_attn_weights_norm, expected_masked_attn_weights_norm, atol: 1.0e-6)
           |> Nx.to_number() == 1,
           "masked_attn_weights_norm should match expected values"

    mask =
      Nx.broadcast(1.0, {context_length, context_length})
      # upper triangle above diagonal
      |> Nx.triu(k: 1)

    mask_bool = Nx.greater(mask, 0.0)
    neg_inf = Nx.broadcast(:neg_infinity, Nx.shape(attn_scores))
    masked = Nx.select(mask_bool, neg_inf, attn_scores)

    expected_masked =
      Nx.tensor(
        [
          [
            -0.5037099719047546,
            :neg_infinity,
            :neg_infinity,
            :neg_infinity,
            :neg_infinity,
            :neg_infinity
          ],
          [
            -0.7201937437057495,
            -0.69146329164505,
            :neg_infinity,
            :neg_infinity,
            :neg_infinity,
            :neg_infinity
          ],
          [
            -0.7123136520385742,
            -0.6792438626289368,
            -0.6886162161827087,
            :neg_infinity,
            :neg_infinity,
            :neg_infinity
          ],
          [
            -0.3948274254798889,
            -0.3909415006637573,
            -0.39571473002433777,
            -0.16911853849887848,
            :neg_infinity,
            :neg_infinity
          ],
          [
            -0.3698960840702057,
            -0.2679566442966461,
            -0.2752986252307892,
            -0.08846122026443481,
            -0.3303014636039734,
            :neg_infinity
          ],
          [
            -0.4973980784416199,
            -0.533991277217865,
            -0.53879314661026,
            -0.24258869886398315,
            -0.473825067281723,
            -0.21798484027385712
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all(Nx.equal(masked, expected_masked)) |> Nx.to_number() == 1,
           "masked should match expected values"

    masked_attn_weights_causal =
      Nx.divide(masked, Nx.sqrt(Nx.axis_size(keys, -1)))
      |> Axon.Activations.softmax(axis: -1)

    expected_masked_attn_weights_causal =
      Nx.tensor(
        [
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.4949212968349457, 0.5050786733627319, 0.0, 0.0, 0.0, 0.0],
          [0.32888710498809814, 0.33666837215423584, 0.3344445526599884, 0.0, 0.0, 0.0],
          [
            0.23951144516468048,
            0.2401704639196396,
            0.2393612116575241,
            0.2809569537639618,
            0.0,
            0.0
          ],
          [
            0.18544265627861023,
            0.1993032991886139,
            0.1982712745666504,
            0.22627480328083038,
            0.1907079666852951,
            0.0
          ],
          [
            0.15678176283836365,
            0.1527770310640335,
            0.15225917100906372,
            0.1877351701259613,
            0.15941700339317322,
            0.1910298764705658
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(masked_attn_weights_causal, expected_masked_attn_weights_causal,
             atol: 1.0e-6
           )
           |> Nx.to_number() == 1,
           "masked_attn_weights_causal should match expected values"
  end

  test "dropout" do
    key = Nx.Random.key(123)
    example = Nx.broadcast(1.0, {6, 6})
    %Axon.StatefulOutput{output: dropped, state: %{"key" => _new_key}} =
      Axon.Layers.dropout(example, key, rate: 0.5, mode: :train)

    expected_dropped =
      Nx.tensor(
        [
          [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
          [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
          [2.0, 2.0, 0.0, 0.0, 0.0, 2.0],
          [2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
          [0.0, 0.0, 2.0, 0.0, 2.0, 2.0]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(dropped, expected_dropped, atol: 1.0e-6) |> Nx.to_number() == 1,
           "dropped should match expected values"


    masked_attn_weights_causal =
      Nx.tensor(
        [
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.4949212968349457, 0.5050786733627319, 0.0, 0.0, 0.0, 0.0],
          [0.32888710498809814, 0.33666837215423584, 0.3344445526599884, 0.0, 0.0, 0.0],
          [
            0.23951144516468048,
            0.2401704639196396,
            0.2393612116575241,
            0.2809569537639618,
            0.0,
            0.0
          ],
          [
            0.18544265627861023,
            0.1993032991886139,
            0.1982712745666504,
            0.22627480328083038,
            0.1907079666852951,
            0.0
          ],
          [
            0.15678176283836365,
            0.1527770310640335,
            0.15225917100906372,
            0.1877351701259613,
            0.15941700339317322,
            0.1910298764705658
          ]
        ],
        type: {:f, 32}
      )

    %Axon.StatefulOutput{output: masked_attn_weights_causal_dropped, state: %{"key" => _new_key}} =
      Axon.Layers.dropout(masked_attn_weights_causal, key, rate: 0.5, mode: :train)

    expected_masked_attn_weights_causal_dropped =
      Nx.tensor(
        [
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.47902289032936096, 0.4803409278392792, 0.0, 0.0, 0.0, 0.0],
          [0.37088531255722046, 0.3986065983772278, 0.0, 0.0, 0.3814159333705902, 0.0],
          [0.0, 0.0, 0.30451834201812744, 0.0, 0.31883400678634644, 0.3820597529411316]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(
             masked_attn_weights_causal_dropped,
             expected_masked_attn_weights_causal_dropped,
             atol: 1.0e-6
           )
           |> Nx.to_number() == 1,
           "masked_attn_weights_causal_dropped should match expected values"

  end

  test "causal attention matches stacked batch example" do
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

    d_in = 3
    d_out = 2
    batch = Nx.stack([inputs, inputs], axis: 0)
    context_length = elem(Nx.shape(batch), 1)
    ca = LlmScratch.CausalAttention.new(d_in, d_out, context_length, 0.0, false, seed: 123)

    context_vecs = LlmScratch.CausalAttention.forward(ca, batch, mode: :inference)

    assert Nx.shape(context_vecs) == {2, 6, 2}

    expected_context_vecs =
      Nx.tensor(
        [
          [
            [-0.49523380398750305, -0.17632800340652466],
            [-0.07537277787923813, -0.13790269196033478],
            [0.06633053719997406, -0.12039512395858765],
            [0.11786159127950668, -0.10831516981124878],
            [0.1877504140138626, -0.04864511638879776],
            [0.1768769919872284, -0.08047633618116379]
          ],
          [
            [-0.49523380398750305, -0.17632800340652466],
            [-0.07537277787923813, -0.13790269196033478],
            [0.06633053719997406, -0.12039512395858765],
            [0.11786159127950668, -0.10831516981124878],
            [0.1877504140138626, -0.04864511638879776],
            [0.1768769919872284, -0.08047633618116379]
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(context_vecs, expected_context_vecs, atol: 1.0e-6) |> Nx.to_number() == 1
  end
end
