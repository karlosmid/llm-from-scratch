defmodule LlmFromScratch4Test do
  use ExUnit.Case

  alias LlmScratch.{
    DummyGPTModel,
    DummyLayerNorm,
    ExampleDeepNeuralNetwork,
    FeedForward,
    GELU,
    GPTModel,
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

  test "GPT-124M model returns logits for input batch" do
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

    batch =
      Nx.tensor(
        [
          [6109, 3626, 6100, 345],
          [6109, 1110, 6622, 257]
        ],
        type: {:s, 64}
      )

    model = GPTModel.new(gpt_config_124m, seed: 123)
    total_params = total_parameters(model)
    total_size_mb = total_params * 4 / 1024 / 1024
    total_params_with_weight_tying = total_params - count_tensor(model.out_head.kernel)
    out = GPTModel.forward(model, batch)

    assert Nx.equal(
             batch,
             Nx.tensor(
               [
                 [6109, 3626, 6100, 345],
                 [6109, 1110, 6622, 257]
               ],
               type: {:s, 64}
             )
           )
           |> Nx.all()
           |> Nx.to_number() == 1

    assert total_params == 163_009_536
    assert total_size_mb == 621.83203125
    assert Nx.shape(out) == {2, 4, 50_257}
    assert %Nx.Tensor{} = out
    assert total_params_with_weight_tying == 124_412_160
  end

  test "exercise 4.1 compares feed forward and multi-head attention parameter counts" do
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

    block = TransformerBlock.new(gpt_config_124m, seed: 123)

    feed_forward_params = count_feed_forward(block.ff)
    attention_params = count_attention(block.att)

    assert feed_forward_params == 4_722_432
    assert attention_params == 2_360_064
    assert feed_forward_params == 2 * attention_params + 2_304
    assert feed_forward_params > attention_params
  end

  test "exercise 4.2 initializes GPT-2 medium config and counts parameters" do
    gpt2_medium_config = %GPTConfig{
      vocab_size: 50_257,
      context_length: 1024,
      emb_dim: 1024,
      n_heads: 16,
      n_layers: 24,
      drop_rate: 0.1,
      qkv_bias: false
    }

    assert gpt2_medium_config.emb_dim == 1024
    assert gpt2_medium_config.n_layers == 24
    assert gpt2_medium_config.n_heads == 16
    assert rem(gpt2_medium_config.emb_dim, gpt2_medium_config.n_heads) == 0
    assert gpt_parameter_count(gpt2_medium_config) == 406_212_608
  end

  test "exercise 4.2 initializes GPT-2 large config and counts parameters" do
    gpt2_large_config = %GPTConfig{
      vocab_size: 50_257,
      context_length: 1024,
      emb_dim: 1280,
      n_heads: 20,
      n_layers: 36,
      drop_rate: 0.1,
      qkv_bias: false
    }

    assert gpt2_large_config.emb_dim == 1280
    assert gpt2_large_config.n_layers == 36
    assert gpt2_large_config.n_heads == 20
    assert rem(gpt2_large_config.emb_dim, gpt2_large_config.n_heads) == 0
    assert gpt_parameter_count(gpt2_large_config) == 838_220_800
  end

  test "exercise 4.2 initializes GPT-2 XL config and counts parameters" do
    gpt2_xl_config = %GPTConfig{
      vocab_size: 50_257,
      context_length: 1024,
      emb_dim: 1600,
      n_heads: 25,
      n_layers: 48,
      drop_rate: 0.1,
      qkv_bias: false
    }

    assert gpt2_xl_config.emb_dim == 1600
    assert gpt2_xl_config.n_layers == 48
    assert gpt2_xl_config.n_heads == 25
    assert rem(gpt2_xl_config.emb_dim, gpt2_xl_config.n_heads) == 0
    assert gpt_parameter_count(gpt2_xl_config) == 1_637_792_000
  end

  defp total_parameters(%GPTModel{} = model) do
    model.tok_emb.weight
    |> count_tensor()
    |> Kernel.+(count_tensor(model.pos_emb.weight))
    |> Kernel.+(Enum.reduce(model.trf_blocks, 0, &(&2 + count_transformer_block(&1))))
    |> Kernel.+(count_layer_norm(model.final_norm))
    |> Kernel.+(count_tensor(model.out_head.kernel))
  end

  defp count_transformer_block(block) do
    count_attention(block.att) +
      count_feed_forward(block.ff) +
      count_layer_norm(block.norm1) +
      count_layer_norm(block.norm2)
  end

  defp count_attention(attention) do
    attention.w_q
    |> count_dense(attention.qkv_bias)
    |> Kernel.+(count_dense(attention.w_k, attention.qkv_bias))
    |> Kernel.+(count_dense(attention.w_v, attention.qkv_bias))
    |> Kernel.+(count_dense(attention.out_proj, true))
  end

  defp count_feed_forward(feed_forward) do
    count_dense(feed_forward.layers.first, true) +
      count_dense(feed_forward.layers.second, true)
  end

  defp count_layer_norm(layer_norm) do
    count_tensor(layer_norm.scale) + count_tensor(layer_norm.shift)
  end

  defp count_dense(%{kernel: kernel, bias: bias}, true),
    do: count_tensor(kernel) + count_tensor(bias)

  defp count_dense(%{kernel: kernel}, false), do: count_tensor(kernel)

  defp count_tensor(tensor), do: Nx.size(tensor)

  defp gpt_parameter_count(%GPTConfig{} = cfg) do
    embedding_params = 2 * cfg.vocab_size * cfg.emb_dim + cfg.context_length * cfg.emb_dim
    block_params = cfg.n_layers * (12 * cfg.emb_dim * cfg.emb_dim + 10 * cfg.emb_dim)
    final_norm_params = 2 * cfg.emb_dim

    embedding_params + block_params + final_norm_params
  end
end
