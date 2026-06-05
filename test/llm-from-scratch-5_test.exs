defmodule LlmFromScratch5Test do
  use ExUnit.Case

  import LlmScratch.TestHelpers

  alias LlmScratch.{
    GPTConfig,
    GPT2OpenAI,
    GPTModel,
    GptDatasetV1,
    LossUtils,
    ModelCheckpoint,
    TemperatureScaling,
    TextGeneration,
    TextUtils,
    Training
  }

  test "5.1.1 generate_text_simple generates text from a GPT-124M start context" do
    # set EXLA for faster computing

    use_accelerated_backend()

    # we set context_length to 256 so we can train this model in this century on m3 chip

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 256,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.0,
      qkv_bias: false
    }

    start_context = "Every effort moves you"
    encoded_tensor = TextUtils.text_to_token_ids(start_context, "code-davinci-002")
    model = GPTModel.new(gpt_config_124m, seed: 123)

    out =
      TextGeneration.generate_text_simple(
        model,
        encoded_tensor,
        6,
        gpt_config_124m.context_length
      )

    output_tokens = Nx.to_flat_list(out)
    decoded_text = TextUtils.token_ids_to_text(out, "code-davinci-002")

    assert Nx.to_flat_list(encoded_tensor) == [6109, 3626, 6100, 345]
    assert Nx.shape(encoded_tensor) == {1, 4}

    assert output_tokens == [
             6109,
             3626,
             6100,
             345,
             49_770,
             30_538,
             17_021,
             41_246,
             47_931,
             14_609
           ]

    assert Nx.shape(out) == {1, 10}
    assert length(output_tokens) == 10
    assert decoded_text == "Every effort moves youTea labelled proves insensitive Niño Zombie"
  end

  test "5.1.2 calculating the text generation loss" do
    use_accelerated_backend()

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 256,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.0,
      qkv_bias: false
    }

    input_texts = ["every effort moves", "I really like"]
    target_texts = [" effort moves you", " really like chocolate"]

    inputs = TextUtils.texts_to_token_ids(input_texts, "code-davinci-002")
    targets = TextUtils.texts_to_token_ids(target_texts, "code-davinci-002")

    # initialize our gpt model
    model = GPTModel.new(gpt_config_124m, seed: 123)
    # predict next input token
    logits = GPTModel.forward(model, inputs)
    # using softmax get probability values (this is redundant step)
    probas = Axon.Activations.softmax(logits, axis: -1)
    # get token ids that have maximal logit values
    token_ids = Nx.argmax(probas, axis: -1, keep_axis: true)

    assert Nx.to_list(inputs) == [
             [16_833, 3626, 6100],
             [40, 1107, 588]
           ]

    assert Nx.to_list(targets) == [
             [3626, 6100, 345],
             [1107, 588, 11_311]
           ]

    assert Nx.shape(logits) == {2, 3, 50_257}
    assert Nx.shape(targets) == {2, 3}

    assert Nx.to_list(token_ids) == [
             [[19_107], [28_919], [47_017]],
             [[24_695], [16_007], [16_023]]
           ]

    # targets what model should predict
    assert TextUtils.token_ids_to_text(targets[0] |> Nx.new_axis(0), "code-davinci-002") ==
             " effort moves you"

    # what model actually predicted
    assert TextUtils.token_ids_to_text(
             token_ids[0] |> Nx.flatten() |> Nx.new_axis(0),
             "code-davinci-002"
           ) ==
             "identifiedreements Buffer"

    # probabilities that model generated for targets
    target_probas = LossUtils.target_token_probas(probas, targets)

    assert_close(
      target_probas,
      Nx.tensor([
        5.6883127e-5,
        3.5589568e-5,
        2.4848670e-5,
        4.5891375e-5,
        2.5597232e-5,
        1.9225208e-5
      ]),
      atol: 1.0e-10
    )

    # we calculate loss as difference what model actually predicted for target tokens
    # this loss should be minimized
    # we have big loss result, and that is expected as we did not train the model
    # goal is to have loss close to zero
    loss = LossUtils.cross_entropy_loss(logits, targets)
    assert_close(loss, Nx.tensor(10.340371), atol: 1.0e-6)
  end

  test "5.1.3 Calculating the training and validation set losses" do
    device = use_accelerated_backend()

    file_content = File.read!("the-verdict.txt")
    {:ok, encoded_tokens} = Tiktoken.encode("code-davinci-002", file_content)

    assert String.length(file_content) == 20_479
    assert length(encoded_tokens) == 5_145

    train_ratio = 0.90
    split_idx = trunc(train_ratio * String.length(file_content))
    train_data = String.slice(file_content, 0, split_idx)
    val_data = String.slice(file_content, split_idx, String.length(file_content) - split_idx)

    {:ok, train_tokens} = Tiktoken.encode("code-davinci-002", train_data, ["<|endoftext|>"])
    {:ok, val_tokens} = Tiktoken.encode("code-davinci-002", val_data, ["<|endoftext|>"])

    assert String.length(train_data) == 18_431
    assert String.length(val_data) == 2_048
    assert length(train_tokens) == 4_612
    assert length(val_tokens) == 534

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 256,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.0,
      qkv_bias: false
    }

    train_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: train_data,
        batch_size: 2,
        max_length: gpt_config_124m.context_length,
        stride: gpt_config_124m.context_length,
        drop_last: true,
        shuffle: true,
        num_workers: 0
      )

    val_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: val_data,
        batch_size: 2,
        max_length: gpt_config_124m.context_length,
        stride: gpt_config_124m.context_length,
        drop_last: false,
        shuffle: false,
        num_workers: 0
      )

    assert %{batch_size: 2, drop_last: true, num_workers: 0, length: 9} = train_loader
    assert %{batch_size: 2, drop_last: false, num_workers: 0, length: 1} = val_loader

    train_batches = Enum.take(train_loader.stream, train_loader.length)
    val_batches = Enum.take(val_loader.stream, val_loader.length)

    assert length(train_batches) == 9
    assert length(val_batches) == 1

    Enum.each(train_batches, fn batch ->
      {inputs, targets} = Enum.unzip(batch)

      assert Nx.shape(Nx.stack(inputs)) == {2, 256}
      assert Nx.shape(Nx.stack(targets)) == {2, 256}
    end)

    Enum.each(val_batches, fn batch ->
      {inputs, targets} = Enum.unzip(batch)

      assert Nx.shape(Nx.stack(inputs)) == {2, 256}
      assert Nx.shape(Nx.stack(targets)) == {2, 256}
    end)

    model = GPTModel.new(gpt_config_124m, seed: 123)

    train_loss = LossUtils.calc_loss_loader(train_loader, model, device)
    val_loss = LossUtils.calc_loss_loader(val_loader, model, device)

    assert_in_delta train_loss, 11.028817, 1.0e-5
    assert_in_delta val_loss, 10.995487, 1.0e-5
  end

  @tag :emlx
  @tag :train
  @tag timeout: 900_000
  test "5.2 train an llm" do
    device = use_accelerated_backend()

    file_content = File.read!("the-verdict.txt")
    train_ratio = 0.90
    split_idx = trunc(train_ratio * String.length(file_content))
    train_data = String.slice(file_content, 0, split_idx)
    val_data = String.slice(file_content, split_idx, String.length(file_content) - split_idx)

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 256,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.0,
      qkv_bias: false
    }

    train_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: train_data,
        batch_size: 2,
        max_length: gpt_config_124m.context_length,
        stride: gpt_config_124m.context_length,
        drop_last: true,
        shuffle: true,
        num_workers: 0
      )

    val_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: val_data,
        batch_size: 2,
        max_length: gpt_config_124m.context_length,
        stride: gpt_config_124m.context_length,
        drop_last: false,
        shuffle: false,
        num_workers: 0
      )

    model = GPTModel.new(gpt_config_124m, seed: 123)

    optimizer =
      Training.adamw(
        0.0004,
        weight_decay: 0.1
      )

    {trained_model, train_losses, val_losses, tokens_seen} =
      Training.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        10,
        5,
        5,
        "Every effort moves you",
        "code-davinci-002"
      )

    assert length(train_losses) == 18
    assert length(val_losses) == 18

    assert tokens_seen == [
             512,
             3072,
             5632,
             8192,
             10_752,
             13_312,
             15_872,
             18_432,
             20_992,
             23_552,
             26_112,
             28_672,
             31_232,
             33_792,
             36_352,
             38_912,
             41_472,
             44_032
           ]

    assert Enum.all?(train_losses, &is_float/1)
    assert Enum.all?(val_losses, &is_float/1)
    assert Enum.all?(train_losses ++ val_losses, &(&1 > 0.0))
    assert %GPTModel{} = trained_model

    checkpoint_path = "ch5_2_gpt_124m.nx"
    ModelCheckpoint.save!(trained_model, checkpoint_path)

    assert File.exists?(checkpoint_path)
    assert File.stat!(checkpoint_path).size > 0
  end

  test "5.3 load trained model checkpoint and generate text" do
    use_accelerated_backend()

    checkpoint_path = "ch5_2_gpt_124m.nx"
    assert File.exists?(checkpoint_path)

    trained_model = ModelCheckpoint.load!(checkpoint_path)
    tokenizer = "code-davinci-002"

    token_ids =
      TextGeneration.generate_text_simple(
        trained_model,
        TextUtils.text_to_token_ids("Every effort moves you", tokenizer),
        25,
        trained_model.cfg.context_length
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    repeated_token_ids =
      TextGeneration.generate_text_simple(
        trained_model,
        TextUtils.text_to_token_ids("Every effort moves you", tokenizer),
        25,
        trained_model.cfg.context_length
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    decoded_text = TextUtils.token_ids_to_text(token_ids, tokenizer)
    repeated_decoded_text = TextUtils.token_ids_to_text(repeated_token_ids, tokenizer)

    assert Nx.shape(token_ids) == {1, 29}
    assert String.starts_with?(decoded_text, "Every effort moves you")
    assert decoded_text == repeated_decoded_text
  end

  test "5.3 samples next token from vocabulary probabilities" do
    vocab = %{
      "closer" => 0,
      "every" => 1,
      "effort" => 2,
      "forward" => 3,
      "inches" => 4,
      "moves" => 5,
      "pizza" => 6,
      "toward" => 7,
      "you" => 8
    }

    inverse_vocab = Map.new(vocab, fn {token, token_id} -> {token_id, token} end)

    # Fictional logits for input text "every effort moves you"
    next_token_logits =
      Nx.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    # logits to probability values
    probas = Axon.Activations.softmax(next_token_logits)

    # greedy algorithm
    next_token_id =
      probas
      |> Nx.argmax()
      |> Nx.to_number()

    assert inverse_vocab[next_token_id] == "forward"

    # this is equivalent to torch.multinomial
    sampled_token_id = TemperatureScaling.multinomial(probas, samples: 1, seed: 123) |> hd()

    assert inverse_vocab[sampled_token_id] == "toward"

    # lets repeat sampling more times to get token distribution
    # Our GPTModel becomes more creative, or better to say none deterministic,
    # because it will generate according to this distribution:
    # Every effort moves you forward
    # Every effort moves you toward
    # Every effort move you closer
    # But it will never generate Every effort moves you pizza.
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

  test "exercise 5.1 counts pizza sampling frequency across temperatures" do
    vocab = %{
      "closer" => 0,
      "every" => 1,
      "effort" => 2,
      "forward" => 3,
      "inches" => 4,
      "moves" => 5,
      "pizza" => 6,
      "toward" => 7,
      "you" => 8
    }

    inverse_vocab = Map.new(vocab, fn {token, token_id} -> {token_id, token} end)
    next_token_logits = Nx.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    pizza_sample_counts =
      [1, 0.1, 5]
      |> Enum.map(fn temperature ->
        probas = TemperatureScaling.softmax_with_temperature(next_token_logits, temperature)

        {pizza_count, "pizza"} =
          probas
          |> TemperatureScaling.sampled_token_frequencies(inverse_vocab,
            seed: 123,
            samples: 1_000
          )
          |> Enum.find(fn {_count, token} -> token == "pizza" end)

        {temperature, pizza_count}
      end)

    assert pizza_sample_counts == [{1, 0}, {0.1, 0}, {5, 39}]

    # faster and acurate count for pizza
    pizza_expected_counts =
      [1, 0.1, 5]
      |> Enum.map(fn temperature ->
        probas = TemperatureScaling.softmax_with_temperature(next_token_logits, temperature)
        # multiply probas value for pizza with sampling size
        {temperature, Nx.to_number(probas[vocab["pizza"]]) * 1_000}
      end)

    assert [{1, temperature_1_count}, {0.1, nearly_zero}, {5, temperature_5_count}] =
             pizza_expected_counts

    assert_in_delta temperature_1_count, 0.10120050865225494, 1.0e-10
    assert nearly_zero < 1.0e-30
    assert_in_delta temperature_5_count, 42.99795627593994, 1.0e-10
  end

  test "5.3.2 top-k sampling masks logits outside top k" do
    next_token_logits =
      Nx.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    top_k = 3
    {top_logits, top_pos} = Nx.top_k(next_token_logits, k: top_k)

    assert Nx.all_close(top_logits, Nx.tensor([6.75, 6.28, 4.51]))
    assert Nx.to_flat_list(top_pos) == [3, 7, 0]

    min_top_logit = top_logits[top_k - 1]

    new_logits =
      Nx.select(
        Nx.less(next_token_logits, min_top_logit),
        Nx.broadcast(:neg_infinity, Nx.shape(next_token_logits)),
        next_token_logits
      )

    assert Nx.all_close(
             new_logits,
             Nx.tensor([
               4.51,
               :neg_infinity,
               :neg_infinity,
               6.75,
               :neg_infinity,
               :neg_infinity,
               :neg_infinity,
               6.28,
               :neg_infinity
             ])
           )

    topk_probas = Axon.Activations.softmax(new_logits)

    assert Nx.all_close(
             topk_probas,
             Nx.tensor([
               0.0615,
               0.0000,
               0.0000,
               0.5775,
               0.0000,
               0.0000,
               0.0000,
               0.3610,
               0.0000
             ]),
             atol: 1.0e-4
           )
  end

  test "5.3.3 generate uses temperature scaling and top-k sampling" do
    use_accelerated_backend()

    checkpoint_path = "ch5_2_gpt_124m.nx"
    assert File.exists?(checkpoint_path)

    model = ModelCheckpoint.load!(checkpoint_path)
    tokenizer = "code-davinci-002"

    token_ids =
      TextGeneration.generate(
        model,
        TextUtils.text_to_token_ids("Every effort moves you", tokenizer),
        15,
        model.cfg.context_length,
        1.4,
        25
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    decoded_text = TextUtils.token_ids_to_text(token_ids, tokenizer)

    assert Nx.shape(token_ids) == {1, 19}
    assert Enum.take(Nx.to_flat_list(token_ids), 4) == [6109, 3626, 6100, 345]
    assert String.starts_with?(decoded_text, "Every effort moves you")
  end

  @tag :train
  @tag timeout: 180_000
  test "5.4 load model and optimizer checkpoint and continue pretraining" do
    device = use_accelerated_backend()

    checkpoint_path = Path.join(System.tmp_dir!(), "ch5_4_model_and_optimizer.nx")
    on_exit(fn -> File.rm(checkpoint_path) end)

    tokenizer = "code-davinci-002"
    raw_text = String.duplicate("Every effort moves you forward. ", 120)

    gpt_config = %GPTConfig{
      vocab_size: 50_257,
      context_length: 16,
      emb_dim: 32,
      n_heads: 4,
      n_layers: 1,
      drop_rate: 0.0,
      qkv_bias: false
    }

    train_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: raw_text,
        batch_size: 2,
        max_length: gpt_config.context_length,
        stride: gpt_config.context_length,
        drop_last: true,
        shuffle: false,
        num_workers: 0
      )

    val_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: raw_text,
        batch_size: 2,
        max_length: gpt_config.context_length,
        stride: gpt_config.context_length,
        drop_last: false,
        shuffle: false,
        num_workers: 0
      )

    model = GPTModel.new(gpt_config, seed: 123)
    optimizer = Training.adamw(0.0004, weight_decay: 0.1)

    {trained_model, trained_optimizer, train_losses, val_losses, tokens_seen} =
      Training.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        1,
        2,
        1,
        "Every effort moves you",
        tokenizer,
        return_optimizer: true
      )

    assert %GPTModel{} = trained_model
    assert %Training.AdamW{m: m, v: v} = trained_optimizer
    assert trained_optimizer.step == train_loader.length
    assert length(m) > 0
    assert length(v) == length(m)
    assert Enum.all?(train_losses ++ val_losses, &is_float/1)
    assert Enum.all?(tokens_seen, &is_integer/1)

    ModelCheckpoint.save_training_state!(trained_model, trained_optimizer, checkpoint_path)

    %{
      model_state_dict: loaded_model,
      optimizer_state_dict: loaded_optimizer
    } = ModelCheckpoint.load_training_state!(checkpoint_path)

    assert %GPTModel{} = loaded_model
    assert %Training.AdamW{} = loaded_optimizer
    assert loaded_optimizer.step == trained_optimizer.step

    {_continued_model, continued_optimizer, continued_train_losses, continued_val_losses,
     continued_tokens_seen} =
      Training.train_model_simple(
        loaded_model,
        train_loader,
        val_loader,
        loaded_optimizer,
        device,
        1,
        2,
        1,
        "Every effort moves you",
        tokenizer,
        return_optimizer: true
      )

    assert continued_optimizer.step == trained_optimizer.step + train_loader.length
    assert Enum.all?(continued_train_losses ++ continued_val_losses, &is_float/1)
    assert Enum.all?(continued_tokens_seen, &is_integer/1)
  end

  @tag :download
  @tag timeout: 900_000
  test "5.5 downloads and loads public OpenAI GPT-2 124M settings and params" do
    use_accelerated_backend()

    {settings, params} =
      GPT2OpenAI.download_and_load_gpt2(
        "124M",
        models_dir: "gpt2"
      )

    assert settings == %{
             "n_vocab" => 50_257,
             "n_ctx" => 1024,
             "n_embd" => 768,
             "n_head" => 12,
             "n_layer" => 12
           }

    assert Map.keys(params) |> Enum.sort() == ["b", "blocks", "g", "wpe", "wte"]

    assert Nx.shape(params["wte"]) == {50257, 768}

    model =
      settings
      |> GPT2OpenAI.config_from_settings()
      |> GPTModel.new(seed: 123, norm_eps: 1.0e-5)
      |> GPT2OpenAI.load_weights_into_gpt(params)

    tokenizer = "code-davinci-002"

    token_ids =
      TextGeneration.generate(
        model,
        TextUtils.text_to_token_ids("Every effort moves you", tokenizer),
        25,
        model.cfg.context_length,
        1.5,
        50
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    decoded_text = TextUtils.token_ids_to_text(token_ids, tokenizer)

    assert decoded_text ==
             "Every effort moves you closer to victory!\" he told the story of his friend. But if all else failed, he was determined to defeat, his"
  end

  @tag :download
  @tag timeout: 900_000
  test "5.5 exercise calculates The Verdict losses with OpenAI GPT-2 124M weights" do
    device = use_accelerated_backend()

    model = GPT2OpenAI.load_model("124M", models_dir: "gpt2")

    file_content = File.read!("the-verdict.txt")
    train_ratio = 0.90
    split_idx = trunc(train_ratio * String.length(file_content))
    train_data = String.slice(file_content, 0, split_idx)
    val_data = String.slice(file_content, split_idx, String.length(file_content) - split_idx)

    train_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: train_data,
        batch_size: 2,
        max_length: 256,
        stride: 256,
        drop_last: true,
        shuffle: false,
        num_workers: 0
      )

    val_loader =
      GptDatasetV1.create_dataloader_v1(
        raw_text: val_data,
        batch_size: 2,
        max_length: 256,
        stride: 256,
        drop_last: false,
        shuffle: false,
        num_workers: 0
      )

    assert train_loader.length == 9
    assert val_loader.length == 1

    train_loss = LossUtils.calc_loss_loader(train_loader, model, device)
    val_loss = LossUtils.calc_loss_loader(val_loader, model, device)

    assert_in_delta train_loss, 3.7547634177737765, 1.0e-5
    assert_in_delta val_loss, 3.5596354007720947, 1.0e-5
  end

  @tag :download
  @tag timeout: 3_600_000
  test "5.6 exercise compares generated text from GPT-2 124M and 1558M" do
    use_accelerated_backend()

    prompt = "Every effort moves you"
    tokenizer = "code-davinci-002"

    model = GPT2OpenAI.load_model("1558M", models_dir: "gpt2")

    text =
      model
      |> TextGeneration.generate(
        TextUtils.text_to_token_ids(prompt, tokenizer),
        25,
        model.cfg.context_length,
        1.5,
        50
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> TextUtils.token_ids_to_text(tokenizer)

    assert text ==
             "Every effort moves you farther and farther closer to it. I don't need to try all their tricks.\n\n\n(END OF TRANSCRIPT"
  end
end
