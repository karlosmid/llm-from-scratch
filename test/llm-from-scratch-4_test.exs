defmodule LlmFromScratch4Test do
  use ExUnit.Case

  alias LlmScratch.{
    DummyGPTModel,
    DummyLayerNorm,
    ExampleDeepNeuralNetwork,
    FeedForward,
    GELU,
    GPTConfig,
    SimpleGradient,
    TransformerBlock
  }

  test "dummy GPT model returns logits for GPT-2 tokenized batch" do
    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 1024,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.1,
      qkv_bias: false
    }

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    {:ok, tokens1} = Tiktoken.encode("code-davinci-002", txt1)
    {:ok, tokens2} = Tiktoken.encode("code-davinci-002", txt2)

    batch =
      [tokens1, tokens2]
      |> Enum.map(&Nx.tensor(&1, type: {:s, 64}))
      |> Nx.stack()

    expected_batch =
      Nx.tensor(
        [
          [6109, 3626, 6100, 345],
          [6109, 1110, 6622, 257]
        ],
        type: {:s, 64}
      )

    assert Nx.equal(batch, expected_batch) |> Nx.all() |> Nx.to_number() == 1

    model = DummyGPTModel.new(gpt_config_124m, seed: 123)
    logits = DummyGPTModel.forward(model, batch)

    assert Nx.shape(logits) == {2, 4, 50_257}

    # we only check first 5 dimensions of logits, as their full dimensions of 50_257 is too large
    assert_close(logits[[0, 0, 0..4]], [
      0.09030798077583313,
      -0.020954661071300507,
      0.14234550297260284,
      -0.003979288041591644,
      -0.07776273787021637
    ])

    assert_close(logits[[0, 1, 0..4]], [
      0.03524987772107124,
      -0.2750282883644104,
      0.20687848329544067,
      0.03522858768701553,
      0.037489306181669235
    ])

    assert_close(logits[[1, 3, 0..4]], [
      -0.23411811888217926,
      0.2425902932882309,
      0.18158411979675293,
      0.2798754572868347,
      -0.35763686895370483
    ])
  end

  defp assert_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-6)
    expected = Nx.tensor(expected, type: {:f, 32})

    assert Nx.all_close(actual, expected, atol: atol) |> Nx.to_number() == 1
  end

  test "simple gradient example matches book data" do
    y = Nx.tensor([1.0])
    x1 = Nx.tensor([1.1])
    w1 = Nx.tensor([2.2])
    b = Nx.tensor([0.0])

    grad_l_w1 = SimpleGradient.grad_w1(w1, b, x1, y)
    grad_l_b = SimpleGradient.grad_b(w1, b, x1, y)

    {loss, {w1_grad, b_grad}} = SimpleGradient.backward(w1, b, x1, y)

    assert_close(loss, Nx.tensor(0.08518788))
    assert_close(w1_grad, [-0.0898], atol: 1.0e-4)
    assert_close(b_grad, [-0.0817], atol: 1.0e-4)
    assert_close(grad_l_w1, [-0.0898], atol: 1.0e-4)
    assert_close(grad_l_b, [-0.0817], atol: 1.0e-4)
  end

  test "layer norm normalizes over the last dimension" do
    {batch_example, _key} = Nx.Random.normal(Nx.Random.key(123), 0.0, 1.0, shape: {2, 5})
    layer_norm = DummyLayerNorm.new(5)
    out_ln = DummyLayerNorm.forward(layer_norm, batch_example)
    mean = Nx.mean(out_ln, axes: [-1], keep_axes: true)
    var = Nx.variance(out_ln, axes: [-1], keep_axes: true)

    assert Nx.shape(layer_norm.scale) == {5}
    assert Nx.shape(layer_norm.shift) == {5}
    assert_close(layer_norm.scale, [1.0, 1.0, 1.0, 1.0, 1.0])
    assert_close(layer_norm.shift, [0.0, 0.0, 0.0, 0.0, 0.0])

    assert_close(mean, [[0.0], [0.0]], atol: 1.0e-6)
    assert_close(var, [[1.0], [1.0]], atol: 1.0e-4)
  end

  test "gelu applies the GPT approximate activation" do
    gelu = GELU.new()
    input = Nx.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]], type: {:f, 32})

    output = GELU.forward(gelu, input)

    assert_close(output, [[-0.00363743, -0.158808, 0.0, 0.841192, 2.9963627]], atol: 1.0e-6)
    assert_close(GELU.forward(input), output, atol: 1.0e-6)
  end

  test "feed forward expands through gelu and projects back to embedding dimension" do
    cfg = %GPTConfig{
      vocab_size: 50_257,
      context_length: 1024,
      emb_dim: 4,
      n_heads: 2,
      n_layers: 1,
      drop_rate: 0.1,
      qkv_bias: false
    }

    feed_forward = FeedForward.new(cfg, seed: 123)
    x = Nx.tensor([[[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, 1.3]]], type: {:f, 32})

    y = FeedForward.forward(feed_forward, x)

    assert Nx.shape(feed_forward.layers.first.kernel) == {4, 16}
    assert Nx.shape(feed_forward.layers.first.bias) == {16}
    assert Nx.shape(feed_forward.layers.second.kernel) == {16, 4}
    assert Nx.shape(feed_forward.layers.second.bias) == {4}
    assert Nx.shape(y) == {1, 2, 4}

    assert_close(FeedForward.call(feed_forward, x), y)
  end

  test "feed forward preserves GPT-124M hidden state shape with EXLA" do
    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 1024,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.1,
      qkv_bias: false
    }

    {x, _key} = Nx.Random.uniform(Nx.Random.key(123), 0.0, 1.0, shape: {2, 3, 768})
    feed_forward = FeedForward.new(gpt_config_124m, seed: 123)

    out = FeedForward.forward(feed_forward, x)

    assert Nx.shape(out) == {2, 3, 768}
  end

  test "deep neural network asserts layer weight gradient means without shortcuts" do
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = Nx.tensor([[1.0, 0.0, -1.0]], type: {:f, 32})
    target = Nx.tensor([[0.0]], type: {:f, 32})

    model = ExampleDeepNeuralNetwork.new(layer_sizes, use_shortcut: false, seed: 123)

    {loss, gradients} = ExampleDeepNeuralNetwork.backward(model, sample_input, target)
    gradient_means = ExampleDeepNeuralNetwork.weight_gradient_means(gradients)

    assert Nx.shape(ExampleDeepNeuralNetwork.forward(model, sample_input)) == {1, 1}
    assert_close(loss, Nx.tensor(2.9010625e-6, type: {:f, 32}), atol: 1.0e-12)

    assert_close(Enum.at(gradient_means, 0), Nx.tensor(1.9688005e-5), atol: 1.0e-10)
    assert_close(Enum.at(gradient_means, 1), Nx.tensor(8.4962267e-6), atol: 1.0e-10)
    assert_close(Enum.at(gradient_means, 2), Nx.tensor(1.0740861e-5), atol: 1.0e-10)
    assert_close(Enum.at(gradient_means, 3), Nx.tensor(1.0382744e-5), atol: 1.0e-10)
    assert_close(Enum.at(gradient_means, 4), Nx.tensor(5.419739e-6), atol: 1.0e-10)
  end

  test "deep neural network asserts larger gradient flow with shortcuts" do
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = Nx.tensor([[1.0, 0.0, -1.0]], type: {:f, 32})
    target = Nx.tensor([[0.0]], type: {:f, 32})

    model = ExampleDeepNeuralNetwork.new(layer_sizes, use_shortcut: true, seed: 123)

    {loss, gradients} = ExampleDeepNeuralNetwork.backward(model, sample_input, target)
    gradient_means = ExampleDeepNeuralNetwork.weight_gradient_means(gradients)

    assert Nx.shape(ExampleDeepNeuralNetwork.forward(model, sample_input)) == {1, 1}
    assert_close(loss, Nx.tensor(0.02248338, type: {:f, 32}), atol: 1.0e-8)

    assert_close(Enum.at(gradient_means, 0), Nx.tensor(0.0034418134), atol: 1.0e-8)
    assert_close(Enum.at(gradient_means, 1), Nx.tensor(0.008453862), atol: 1.0e-8)
    assert_close(Enum.at(gradient_means, 2), Nx.tensor(0.0049184095), atol: 1.0e-8)
    assert_close(Enum.at(gradient_means, 3), Nx.tensor(0.003547175), atol: 1.0e-8)
    assert_close(Enum.at(gradient_means, 4), Nx.tensor(0.00962014), atol: 1.0e-8)

    without_shortcut =
      ExampleDeepNeuralNetwork.new(layer_sizes, use_shortcut: false, seed: 123)

    {_loss, without_shortcut_gradients} =
      ExampleDeepNeuralNetwork.backward(without_shortcut, sample_input, target)

    without_shortcut_gradient_means =
      ExampleDeepNeuralNetwork.weight_gradient_means(without_shortcut_gradients)

    Enum.zip(gradient_means, without_shortcut_gradient_means)
    |> Enum.each(fn {with_shortcut_mean, without_shortcut_mean} ->
      assert Nx.greater(with_shortcut_mean, without_shortcut_mean) |> Nx.to_number() == 1
    end)
  end

  test "transformer block preserves GPT-124M hidden state shape" do
    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 1024,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.1,
      qkv_bias: false
    }

    {x, _key} = Nx.Random.uniform(Nx.Random.key(123), 0.0, 1.0, shape: {2, 4, 768})
    block = TransformerBlock.new(gpt_config_124m, seed: 123)
    output = TransformerBlock.forward(block, x)

    assert Nx.shape(x) == {2, 4, 768}
    assert Nx.shape(output) == {2, 4, 768}
  end
end
