defmodule LlmFromScratch7Test do
  use ExUnit.Case

  import LlmScratch.TestHelpers

  alias LlmScratch.{
    DataLoader,
    DummyGPTModel,
    FineTuneDataLoader,
    GPTConfig,
    GPT2OpenAI,
    GPTModel,
    InstructionDataset,
    InstructionsEvaluation,
    LoRAUtils,
    LossUtils,
    ModelCheckpoint,
    OllamaUtils,
    TextGeneration,
    TextUtils,
    Training
  }

  @instruction_data_url "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

  @tag :download
  test "7.1 downloads and loads instruction data" do
    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    assert length(data) == 1_100

    assert Enum.at(data, 50) == %{
             "instruction" => "Identify the correct spelling of the following word.",
             "input" => "Ocassion",
             "output" => "The correct spelling is 'Occasion.'"
           }

    assert Enum.at(data, 999) == %{
             "instruction" => "What is an antonym of 'complicated'?",
             "input" => "",
             "output" => "An antonym of 'complicated' is 'simple'."
           }

    entry_50 = Enum.at(data, 50)
    model_input_50 = FineTuneDataLoader.format_input(entry_50)
    desired_response_50 = "\n\n### Response:\n#{entry_50["output"]}"

    assert model_input_50 <> desired_response_50 ==
             "Below is an instruction that describes a task. " <>
               "Write a response that appropriately completes the request." <>
               "\n\n### Instruction:\nIdentify the correct spelling of the following word." <>
               "\n\n### Input:\nOcassion" <>
               "\n\n### Response:\nThe correct spelling is 'Occasion.'"

    entry_999 = Enum.at(data, 999)
    model_input_999 = FineTuneDataLoader.format_input(entry_999)
    desired_response_999 = "\n\n### Response:\n#{entry_999["output"]}"

    assert model_input_999 <> desired_response_999 ==
             "Below is an instruction that describes a task. " <>
               "Write a response that appropriately completes the request." <>
               "\n\n### Instruction:\nWhat is an antonym of 'complicated'?" <>
               "\n\n### Response:\nAn antonym of 'complicated' is 'simple'."

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    val_portion = length(data) - train_portion - test_portion

    train_data = Enum.slice(data, 0, train_portion)
    test_data = Enum.slice(data, train_portion, test_portion)
    val_data = Enum.slice(data, train_portion + test_portion, val_portion)

    assert length(train_data) == 935
    assert length(val_data) == 55
    assert length(test_data) == 110
  end

  test "7.2 creates pre-tokenized instruction dataset" do
    entry = %{
      "instruction" => "Classify the sentiment of the text.",
      "input" => "I loved it.",
      "output" => "Positive."
    }

    tokenizer = "code-davinci-002"
    dataset = InstructionDataset.new([entry], tokenizer)

    expected_text =
      "Below is an instruction that describes a task. " <>
        "Write a response that appropriately completes the request." <>
        "\n\n### Instruction:\nClassify the sentiment of the text." <>
        "\n\n### Input:\nI loved it." <>
        "\n\n### Response:\nPositive."

    {:ok, expected_tokens} = Tiktoken.encode(tokenizer, expected_text, ["<|endoftext|>"])

    assert dataset.data == [entry]
    assert dataset.encoded_texts == [expected_tokens]
    assert dataset.prompt_style == :alpaca
    assert dataset.mask_out_instructions_in_target == false
    assert InstructionDataset.get(dataset, 0) == expected_tokens
    assert InstructionDataset.length(dataset) == 1

    phi3_dataset = InstructionDataset.new([entry], tokenizer, prompt_style: :phi3)

    expected_phi3_text =
      "<|user|>\n" <>
        "Classify the sentiment of the text.\n" <>
        "I loved it.\n" <>
        "<|end|>\n" <>
        "<|assistant|>\n" <>
        "Positive.\n" <>
        "<|end|>"

    {:ok, expected_phi3_tokens} =
      Tiktoken.encode(tokenizer, expected_phi3_text, ["<|endoftext|>"])

    assert phi3_dataset.data == [entry]
    assert phi3_dataset.encoded_texts == [expected_phi3_tokens]
    assert phi3_dataset.prompt_style == :phi3
    assert phi3_dataset.mask_out_instructions_in_target == false
    assert InstructionDataset.get(phi3_dataset, 0) == expected_phi3_tokens
    assert InstructionDataset.length(phi3_dataset) == 1

    masked_dataset =
      InstructionDataset.new([entry], tokenizer, mask_out_instructions_in_target: true)

    assert [%{token_ids: ^expected_tokens, prompt_length: prompt_length}] =
             masked_dataset.encoded_texts

    assert masked_dataset.prompt_style == :alpaca
    assert masked_dataset.mask_out_instructions_in_target == true

    {:ok, expected_prompt_tokens} =
      Tiktoken.encode(tokenizer, FineTuneDataLoader.format_input(entry), ["<|endoftext|>"])

    assert prompt_length == length(expected_prompt_tokens)
  end

  test "7.2 custom collate pads batch inputs" do
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = {inputs_1, inputs_2, inputs_3}

    {inputs_tensor, _targets_tensor} = InstructionDataset.custom_collate(batch)

    assert inputs_tensor ==
             Nx.tensor(
               [
                 [0, 1, 2, 3, 4],
                 [5, 6, 50_256, 50_256, 50_256],
                 [7, 8, 9, 50_256, 50_256]
               ],
               type: {:s, 64}
             )
  end

  test "7.3 custom collate creates shifted targets with ignored padding" do
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = {inputs_1, inputs_2, inputs_3}

    {inputs, targets} = InstructionDataset.custom_collate(batch)

    assert inputs ==
             Nx.tensor(
               [
                 [0, 1, 2, 3, 4],
                 [5, 6, 50_256, 50_256, 50_256],
                 [7, 8, 9, 50_256, 50_256]
               ],
               type: {:s, 64}
             )

    assert targets ==
             Nx.tensor(
               [
                 [1, 2, 3, 4, 50_256],
                 [6, 50_256, -100, -100, -100],
                 [8, 9, 50_256, -100, -100]
               ],
               type: {:s, 64}
             )
  end

  test "7.3 cross entropy ignores masked instruction targets" do
    logits_1 =
      Nx.tensor([
        [-1.0, 1.0],
        [-0.5, 1.5]
      ])

    targets_1 = Nx.tensor([0, 1], type: {:s, 64})
    loss_1 = LossUtils.cross_entropy_loss(logits_1, targets_1)

    logits_2 =
      Nx.tensor([
        [-1.0, 1.0],
        [-0.5, 1.5],
        [-0.5, 1.5]
      ])

    targets_2 = Nx.tensor([0, 1, 1], type: {:s, 64})
    loss_2 = LossUtils.cross_entropy_loss(logits_2, targets_2)

    targets_3 = Nx.tensor([0, 1, -100], type: {:s, 64})
    loss_3 = LossUtils.cross_entropy_loss(logits_2, targets_3)

    assert_in_delta Nx.to_number(loss_1), 1.1269, 1.0e-4
    assert_in_delta Nx.to_number(loss_2), 0.7936, 1.0e-4
    assert Nx.equal(loss_1, loss_3) |> Nx.to_number() == 1
  end

  test "7.3 loss loader skips batches with only ignored instruction targets" do
    model =
      %GPTConfig{
        vocab_size: 16,
        context_length: 4,
        emb_dim: 8,
        n_heads: 1,
        n_layers: 1,
        drop_rate: 0.0,
        qkv_bias: false
      }
      |> DummyGPTModel.new(seed: 123)

    ignored_batch = {
      Nx.tensor([[1, 2, 3, 4]], type: {:s, 64}),
      Nx.tensor([[-100, -100, -100, -100]], type: {:s, 64})
    }

    valid_batch = {
      Nx.tensor([[1, 2, 3, 4]], type: {:s, 64}),
      Nx.tensor([[-100, -100, 5, 6]], type: {:s, 64})
    }

    data_loader = %{
      stream: [ignored_batch, valid_batch],
      length: 2
    }

    expected_loss =
      valid_batch
      |> then(fn {inputs, targets} -> LossUtils.calc_loss_batch(inputs, targets, model) end)
      |> Nx.to_number()

    assert LossUtils.calc_loss_loader(data_loader, model) == expected_loss
  end

  test "exercise 7.2 custom collate masks instruction and input targets" do
    batch = [
      %{token_ids: [10, 11, 12, 13], prompt_length: 3},
      %{token_ids: [20, 21], prompt_length: 1}
    ]

    {inputs, targets} = InstructionDataset.custom_collate(batch)

    assert inputs ==
             Nx.tensor(
               [
                 [10, 11, 12, 13],
                 [20, 21, 50_256, 50_256]
               ],
               type: {:s, 64}
             )

    assert targets ==
             Nx.tensor(
               [
                 [-100, -100, 13, 50_256],
                 [21, 50_256, -100, -100]
               ],
               type: {:s, 64}
             )
  end

  test "instruction collate can cap batches to model context length" do
    batch = [
      %{token_ids: Enum.to_list(0..9), prompt_length: 0},
      %{token_ids: Enum.to_list(20..24), prompt_length: 0}
    ]

    {inputs, targets} = binary_instruction_collate(batch, 6)

    assert Nx.shape(inputs) == {2, 6}
    assert Nx.shape(targets) == {2, 6}

    assert inputs ==
             Nx.tensor(
               [
                 [0, 1, 2, 3, 4, 5],
                 [20, 21, 22, 23, 24, 50_256]
               ],
               type: {:s, 64}
             )

    assert targets ==
             Nx.tensor(
               [
                 [1, 2, 3, 4, 5, 6],
                 [21, 22, 23, 24, 50_256, -100]
               ],
               type: {:s, 64}
             )
  end

  @tag :download
  test "7.4 creates instruction data loaders with custom collate function" do
    tokenizer = "code-davinci-002"
    batch_size = 8
    num_workers = 3
    :rand.seed(:exsss, {123, 123, 123})

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    val_portion = length(data) - train_portion - test_portion

    train_data = Enum.slice(data, 0, train_portion)
    test_data = Enum.slice(data, train_portion, test_portion)
    val_data = Enum.slice(data, train_portion + test_portion, val_portion)

    train_dataset = InstructionDataset.new(train_data, tokenizer)
    val_dataset = InstructionDataset.new(val_data, tokenizer)
    test_dataset = InstructionDataset.new(test_data, tokenizer)

    train_loader =
      DataLoader.new(train_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &InstructionDataset.custom_collate/1,
        shuffle: true,
        drop_last: true,
        num_workers: num_workers
      )

    val_loader =
      DataLoader.new(val_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &InstructionDataset.custom_collate/1,
        shuffle: false,
        drop_last: false,
        num_workers: num_workers
      )

    test_loader =
      DataLoader.new(test_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &InstructionDataset.custom_collate/1,
        shuffle: false,
        drop_last: false,
        num_workers: num_workers
      )

    assert InstructionDataset.length(train_dataset) == 935
    assert InstructionDataset.length(val_dataset) == 55
    assert InstructionDataset.length(test_dataset) == 110

    assert train_loader.length == 116
    assert val_loader.length == 7
    assert test_loader.length == 14
    assert train_loader.num_workers == 3
    assert val_loader.num_workers == 3
    assert test_loader.num_workers == 3

    assert [{train_inputs, train_targets} | _] = train_loader.batches
    assert [{val_inputs, val_targets} | _] = val_loader.batches
    assert [{test_inputs, test_targets} | _] = test_loader.batches

    # As we are randomly shuffle data before creating batches, we got different result than in the book
    assert Nx.shape(train_inputs) == {8, 91}
    assert Nx.shape(train_targets) == {8, 91}
    assert Nx.shape(val_inputs) == {8, 74}
    assert Nx.shape(val_targets) == {8, 74}
    assert Nx.shape(test_inputs) == {8, 64}
    assert Nx.shape(test_targets) == {8, 64}
  end

  @tag :download
  @tag :train_long
  @tag timeout: 3_600_000
  test "7.5 and 7.6 evaluates and trains OpenAI GPT-2 355M on instruction data" do
    device = use_accelerated_backend()

    tokenizer = "code-davinci-002"
    batch_size = 8
    num_workers = 3

    model =
      "355M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> Nx.backend_transfer(device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    val_portion = length(data) - train_portion - test_portion
    train_data = Enum.slice(data, 0, train_portion)
    val_data = Enum.slice(data, train_portion + test_portion, val_portion)

    train_dataset = InstructionDataset.new(train_data, tokenizer)
    val_dataset = InstructionDataset.new(val_data, tokenizer)

    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      DataLoader.new(train_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &binary_instruction_collate/1,
        shuffle: true,
        drop_last: true,
        num_workers: num_workers
      )

    val_loader =
      DataLoader.new(val_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &binary_instruction_collate/1,
        shuffle: false,
        drop_last: false,
        num_workers: num_workers
      )

    train_loss = LossUtils.calc_loss_loader(train_loader, model, device, 5)
    val_loss = LossUtils.calc_loss_loader(val_loader, model, device, 5)

    input_text =
      val_data
      |> List.first()
      |> FineTuneDataLoader.format_input()

    token_ids =
      TextGeneration.generate(
        model,
        input_text
        |> TextUtils.text_to_token_ids(tokenizer)
        |> Nx.backend_transfer(device),
        35,
        model.cfg.context_length,
        0.0,
        nil,
        50_256
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    generated_text = TextUtils.token_ids_to_text(token_ids, tokenizer)

    response_text =
      generated_text |> String.slice(String.length(input_text)..-1//1) |> String.trim()

    assert model.cfg.context_length == 1024
    assert model.cfg.emb_dim == 1024
    assert model.cfg.n_layers == 24
    assert model.cfg.n_heads == 16
    assert_in_delta train_loss, 3.7422078609466554, 1.0e-5
    assert_in_delta val_loss, 3.7619348049163817, 1.0e-5
    assert input_text == FineTuneDataLoader.format_input(List.first(val_data))
    assert String.starts_with?(generated_text, input_text)

    assert response_text ==
             "### Response:\n\nThe chef cooks the meal every day.\n\n### Instruction:\n\nConvert the active sentence to passive: 'The chef cooks the"

    num_epochs = 2
    optimizer = Training.adamw(0.00005, weight_decay: 0.1)

    {trained_model, trained_optimizer, train_losses, val_losses, tokens_seen} =
      Training.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        5,
        5,
        input_text,
        tokenizer,
        generate_samples: true,
        return_optimizer: true
      )

    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_model_and_optimizer.nx"
    ModelCheckpoint.save_training_state!(trained_model, trained_optimizer, checkpoint_path)

    metrics_path = "ch7_instruction_finetuning_metrics.json"

    write_instruction_training_metrics!(
      metrics_path,
      num_epochs,
      train_losses,
      val_losses,
      tokens_seen
    )

    assert trained_model.__struct__ == model.__struct__
    assert %Training.AdamW{} = trained_optimizer
    assert File.exists?(checkpoint_path)
    assert File.stat!(checkpoint_path).size > 0
    assert File.exists?(metrics_path)
    assert File.stat!(metrics_path).size > 0
    assert length(train_losses) == 47
    assert length(val_losses) == 47
    assert length(tokens_seen) == 47
    assert Enum.all?(train_losses ++ val_losses, &is_float/1)
    assert Enum.all?(tokens_seen, &is_integer/1)
    assert tokens_seen == Enum.sort(tokens_seen)
    assert List.last(tokens_seen) > List.first(tokens_seen)
  end

  @tag :train
  @tag timeout: 900_000
  test "7.7 evaluates fine-tuned instruction model on test samples" do
    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_model_and_optimizer.nx"

    assert File.exists?(checkpoint_path)

    device = use_accelerated_backend()
    tokenizer = "code-davinci-002"
    :rand.seed(:exsss, {123, 123, 123})

    %{model_state_dict: model} = ModelCheckpoint.load_training_state!(checkpoint_path)
    model = Nx.backend_transfer(model, device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    test_data = Enum.slice(data, train_portion, test_portion)

    expected_samples = [
      %{
        input_text:
          "Below is an instruction that describes a task. " <>
            "Write a response that appropriately completes the request." <>
            "\n\n### Instruction:\nRewrite the sentence using a simile." <>
            "\n\n### Input:\nThe car is very fast.",
        correct_response: "The car is as fast as lightning.",
        model_response: "The car is as fast as a cheetah."
      },
      %{
        input_text:
          "Below is an instruction that describes a task. " <>
            "Write a response that appropriately completes the request." <>
            "\n\n### Instruction:\nWhat type of cloud is typically associated with thunderstorms?",
        correct_response:
          "The type of cloud typically associated with thunderstorms is cumulonimbus.",
        model_response:
          "A thunderstorm is a type of cloud that typically forms when thunderstorms produce a dense, convective layer of air that is at least 10 miles thick."
      },
      %{
        input_text:
          "Below is an instruction that describes a task. " <>
            "Write a response that appropriately completes the request." <>
            "\n\n### Instruction:\nName the author of 'Pride and Prejudice'.",
        correct_response: "Jane Austen.",
        model_response: "The author of 'Pride and Prejudice' is Jane Austen."
      }
    ]

    output_path = "instruction-data-with-response.json"

    enriched_data =
      test_data
      |> InstructionsEvaluation.write_responses!(model, tokenizer, device,
        output_path: output_path
      )

    assert File.exists?(output_path)

    enriched_data
    |> Enum.take(3)
    |> Enum.zip(expected_samples)
    |> Enum.each(fn {entry, expected} ->
      input_text = FineTuneDataLoader.format_input(entry)

      assert input_text == expected.input_text
      assert entry["output"] == expected.correct_response
      assert entry["model_response"] == expected.model_response
    end)
  end

  @tag :ollama
  test "7.8 queries local Ollama llama3 model for response scores" do
    assert OllamaUtils.ollama_running?()

    result = OllamaUtils.query_model("What do Llamas eat?", "llama3")

    assert String.contains?(result, "Llamas are herbivores")
    assert String.contains?(result, "Minerals")
    assert String.contains?(result, "Grasses")

    response_data_path = "instruction-data-with-response.json"

    assert File.exists?(response_data_path)

    test_data =
      response_data_path
      |> File.read!()
      |> Jason.decode!()
      |> Enum.take(3)

    expected_samples = [
      %{
        output: "The car is as fast as lightning.",
        model_response: "The car is as fast as a cheetah.",
        score: 85
      },
      %{
        output: "The type of cloud typically associated with thunderstorms is cumulonimbus.",
        model_response:
          "A thunderstorm is a type of cloud that typically forms when thunderstorms produce a dense, convective layer of air that is at least 10 miles thick.",
        score: 20
      },
      %{
        output: "Jane Austen.",
        model_response: "The author of 'Pride and Prejudice' is Jane Austen.",
        score: 95
      }
    ]

    test_data
    |> Enum.take(3)
    |> Enum.zip(expected_samples)
    |> Enum.each(fn {entry, expected} ->
      prompt =
        "Given the input `#{FineTuneDataLoader.format_input(entry)}` " <>
          "and correct output `#{entry["output"]}`, " <>
          "score the model response `#{entry["model_response"]}`" <>
          " on a scale from 0 to 100, where 100 is the best score. "

      score_response = OllamaUtils.query_model(prompt, "llama3")

      report =
        "\nDataset response:\n" <>
          ">> #{entry["output"]}\n" <>
          "\nModel response:\n" <>
          ">> #{entry["model_response"]}\n" <>
          "\nScore:\n" <>
          ">> #{score_response}\n" <>
          "\n-------------------------\n"

      assert entry["output"] == expected.output
      assert entry["model_response"] == expected.model_response
      assert response_score(score_response) == expected.score

      assert report =~ "\nDataset response:\n>> #{expected.output}\n"
      assert report =~ "\nModel response:\n>> #{expected.model_response}\n"
      assert report =~ "\nScore:\n>> #{score_response}\n"
      assert String.ends_with?(report, "\n-------------------------\n")
    end)
  end

  @tag :ollama
  @tag timeout: 900_000
  test "7.8 scores all saved instruction responses with Ollama" do
    assert OllamaUtils.ollama_running?()

    response_data_path = "instruction-data-with-response.json"

    assert File.exists?(response_data_path)

    test_data =
      response_data_path
      |> File.read!()
      |> Jason.decode!()

    scores = InstructionsEvaluation.generate_model_scores(test_data, "model_response")
    average_score = Enum.sum(scores) / length(scores)

    assert length(scores) == 110
    assert length(scores) == length(test_data)
    assert_in_delta average_score, 50.32, 0.4
  end

  @tag :download
  @tag :train_long
  @tag timeout: 3_600_000
  test "exercise 7.1 trains instruction model with Phi-3 prompt style" do
    device = use_accelerated_backend()

    tokenizer = "code-davinci-002"
    batch_size = 8
    num_workers = 3

    model =
      "355M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> Nx.backend_transfer(device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    val_portion = length(data) - train_portion - test_portion
    train_data = Enum.slice(data, 0, train_portion)
    val_data = Enum.slice(data, train_portion + test_portion, val_portion)

    train_dataset = InstructionDataset.new(train_data, tokenizer, prompt_style: :phi3)
    val_dataset = InstructionDataset.new(val_data, tokenizer, prompt_style: :phi3)

    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      DataLoader.new(train_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &binary_instruction_collate/1,
        shuffle: true,
        drop_last: true,
        num_workers: num_workers
      )

    val_loader =
      DataLoader.new(val_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &binary_instruction_collate/1,
        shuffle: false,
        drop_last: false,
        num_workers: num_workers
      )

    train_loss = LossUtils.calc_loss_loader(train_loader, model, device, 5)
    val_loss = LossUtils.calc_loss_loader(val_loader, model, device, 5)

    input_text =
      val_data
      |> List.first()
      |> FineTuneDataLoader.format_text(:phi3)

    token_ids =
      TextGeneration.generate(
        model,
        input_text
        |> TextUtils.text_to_token_ids(tokenizer)
        |> Nx.backend_transfer(device),
        35,
        model.cfg.context_length,
        0.0,
        nil,
        50_256
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    generated_text = TextUtils.token_ids_to_text(token_ids, tokenizer)

    num_epochs = 2
    optimizer = Training.adamw(0.00005, weight_decay: 0.1)

    {trained_model, trained_optimizer, train_losses, val_losses, tokens_seen} =
      Training.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        5,
        5,
        input_text,
        tokenizer,
        generate_samples: true,
        return_optimizer: true
      )

    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_phi3_model_and_optimizer.nx"
    ModelCheckpoint.save_training_state!(trained_model, trained_optimizer, checkpoint_path)

    metrics_path = "ch7_instruction_finetuning_phi3_metrics.json"

    write_instruction_training_metrics!(
      metrics_path,
      num_epochs,
      train_losses,
      val_losses,
      tokens_seen
    )

    assert train_dataset.prompt_style == :phi3
    assert val_dataset.prompt_style == :phi3
    assert String.starts_with?(input_text, "<|user|>\n")
    assert String.ends_with?(input_text, "\n<|end|>\n<|assistant|>\n")
    assert String.starts_with?(generated_text, input_text)
    assert is_float(train_loss)
    assert is_float(val_loss)
    assert trained_model.__struct__ == model.__struct__
    assert %Training.AdamW{} = trained_optimizer
    assert File.exists?(checkpoint_path)
    assert File.stat!(checkpoint_path).size > 0
    assert File.exists?(metrics_path)
    assert File.stat!(metrics_path).size > 0
    assert length(train_losses) == 47
    assert length(val_losses) == 47
    assert length(tokens_seen) == 47
    assert Enum.all?(train_losses ++ val_losses, &is_float/1)
    assert Enum.all?(tokens_seen, &is_integer/1)
    assert tokens_seen == Enum.sort(tokens_seen)
    assert List.last(tokens_seen) > List.first(tokens_seen)
  end

  @tag :download
  @tag :train
  @tag :ollama
  @tag timeout: 1_800_000
  test "exercise 7.1 scores Phi-3 prompt style instruction model with Ollama" do
    assert OllamaUtils.ollama_running?()

    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_phi3_model_and_optimizer.nx"

    assert File.exists?(checkpoint_path)

    device = use_accelerated_backend()
    tokenizer = "code-davinci-002"

    %{model_state_dict: model} = ModelCheckpoint.load_training_state!(checkpoint_path)
    model = Nx.backend_transfer(model, device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    test_data = Enum.slice(data, train_portion, test_portion)

    output_path = "instruction-data-with-response-phi3.json"

    enriched_data =
      test_data
      |> InstructionsEvaluation.write_responses!(model, tokenizer, device,
        output_path: output_path,
        prompt_style: :phi3
      )

    scores = InstructionsEvaluation.generate_model_scores(enriched_data, "model_response")
    average_score = Enum.sum(scores) / length(scores)
    metrics_path = "ch7_instruction_finetuning_phi3_ollama_scores.json"

    write_ollama_score_metrics!(metrics_path, scores, average_score)

    assert File.exists?(output_path)
    assert File.exists?(metrics_path)
    assert length(enriched_data) == 110
    assert length(scores) == 110
    assert length(scores) == length(enriched_data)
    assert_in_delta average_score, 46.67, 0.4
  end

  @tag :download
  @tag :train_long
  @tag timeout: 3_600_000
  test "exercise 7.2 trains instruction model with instruction and input masking" do
    device = use_accelerated_backend()

    tokenizer = "code-davinci-002"

    model =
      "355M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> Nx.backend_transfer(device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    val_portion = length(data) - train_portion - test_portion
    train_data = Enum.slice(data, 0, train_portion)
    val_data = Enum.slice(data, train_portion + test_portion, val_portion)

    train_dataset =
      InstructionDataset.new(train_data, tokenizer, mask_out_instructions_in_target: true)

    val_dataset =
      InstructionDataset.new(val_data, tokenizer, mask_out_instructions_in_target: true)

    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      instruction_data_loader(train_dataset, shuffle: true, drop_last: true)

    val_loader =
      instruction_data_loader(val_dataset, shuffle: false, drop_last: false)

    input_text =
      val_data
      |> List.first()
      |> FineTuneDataLoader.format_input()

    num_epochs = 2
    optimizer = Training.adamw(0.00005, weight_decay: 0.1)

    {trained_model, trained_optimizer, train_losses, val_losses, tokens_seen} =
      Training.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        5,
        5,
        input_text,
        tokenizer,
        generate_samples: true,
        return_optimizer: true
      )

    checkpoint_path =
      "ch7_instruction_finetuned_gpt2_355m_instruction_masked_model_and_optimizer.nx"

    ModelCheckpoint.save_training_state!(trained_model, trained_optimizer, checkpoint_path)

    metrics_path = "ch7_instruction_finetuning_instruction_masked_metrics.json"

    write_instruction_training_metrics!(
      metrics_path,
      num_epochs,
      train_losses,
      val_losses,
      tokens_seen
    )

    assert train_dataset.prompt_style == :alpaca
    assert train_dataset.mask_out_instructions_in_target == true
    assert val_dataset.prompt_style == :alpaca
    assert val_dataset.mask_out_instructions_in_target == true
    assert match?([%{token_ids: _, prompt_length: _} | _], train_dataset.encoded_texts)
    assert input_text == FineTuneDataLoader.format_input(List.first(val_data))
    assert trained_model.__struct__ == model.__struct__
    assert %Training.AdamW{} = trained_optimizer
    assert File.exists?(checkpoint_path)
    assert File.stat!(checkpoint_path).size > 0
    assert File.exists?(metrics_path)
    assert File.stat!(metrics_path).size > 0
    assert length(train_losses) == 47
    assert length(val_losses) == 47
    assert length(tokens_seen) == 47
    assert Enum.all?(train_losses ++ val_losses, &is_float/1)
    assert Enum.all?(tokens_seen, &is_integer/1)
    assert tokens_seen == Enum.sort(tokens_seen)
    assert List.last(tokens_seen) > List.first(tokens_seen)
  end

  @tag :download
  @tag :train
  @tag :ollama
  @tag timeout: 1_800_000
  test "exercise 7.2 scores instruction-masked instruction model with Ollama" do
    assert OllamaUtils.ollama_running?()

    checkpoint_path =
      "ch7_instruction_finetuned_gpt2_355m_instruction_masked_model_and_optimizer.nx"

    assert File.exists?(checkpoint_path)

    device = use_accelerated_backend()
    tokenizer = "code-davinci-002"

    %{model_state_dict: model} = ModelCheckpoint.load_training_state!(checkpoint_path)
    model = Nx.backend_transfer(model, device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    test_data = Enum.slice(data, train_portion, test_portion)

    output_path = "instruction-data-with-response-instruction-masked.json"

    enriched_data =
      test_data
      |> InstructionsEvaluation.write_responses!(model, tokenizer, device,
        output_path: output_path
      )

    scores = InstructionsEvaluation.generate_model_scores(enriched_data, "model_response")
    average_score = Enum.sum(scores) / length(scores)
    baseline_average_score = 50.32
    metrics_path = "ch7_instruction_finetuning_instruction_masked_ollama_scores.json"

    write_ollama_score_metrics!(metrics_path, scores, average_score, baseline_average_score)

    assert File.exists?(output_path)
    assert File.exists?(metrics_path)
    assert length(enriched_data) == 110
    assert length(scores) == 110
    assert length(scores) == length(enriched_data)
    assert_in_delta average_score, 49.78, 0.4
  end

  @tag :train_long
  @tag timeout: 72_000_000
  test "exercise 7.3 fine-tunes instruction model on Alpaca data" do
    device = use_accelerated_backend()
    tokenizer = "code-davinci-002"

    model =
      "355M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> Nx.backend_transfer(device)

    alpaca_data = load_instruction_json!("alpaca_data.json")
    %{train: train_data, val: val_data} = instruction_data_partitions(alpaca_data)

    train_dataset = InstructionDataset.new(train_data, tokenizer)
    val_dataset = InstructionDataset.new(val_data, tokenizer)

    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      instruction_data_loader(train_dataset,
        shuffle: true,
        drop_last: true,
        allowed_max_length: model.cfg.context_length
      )

    val_loader =
      instruction_data_loader(val_dataset,
        shuffle: false,
        drop_last: false,
        allowed_max_length: model.cfg.context_length
      )

    input_text = val_data |> List.first() |> FineTuneDataLoader.format_input()
    optimizer = Training.adamw(0.00005, weight_decay: 0.1)
    num_epochs = 2

    {trained_model, trained_optimizer, train_losses, val_losses, tokens_seen} =
      Training.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        5,
        5,
        input_text,
        tokenizer,
        generate_samples: false,
        return_optimizer: true
      )

    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_alpaca_model_and_optimizer.nx"
    ModelCheckpoint.save_training_state!(trained_model, trained_optimizer, checkpoint_path)

    metrics_path = "ch7_instruction_finetuning_alpaca_metrics.json"

    write_instruction_training_metrics!(
      metrics_path,
      num_epochs,
      train_losses,
      val_losses,
      tokens_seen
    )

    assert length(alpaca_data) > 50_000
    assert trained_model.__struct__ == model.__struct__
    assert %Training.AdamW{} = trained_optimizer
    assert File.exists?(checkpoint_path)
    assert File.stat!(checkpoint_path).size > 0
    assert File.exists?(metrics_path)
    assert File.stat!(metrics_path).size > 0
    assert Enum.all?(train_losses ++ val_losses, &is_float/1)
    assert Enum.all?(tokens_seen, &is_integer/1)
  end

  @tag :train
  @tag :ollama
  @tag timeout: 1_800_000
  test "exercise 7.3 scores Alpaca instruction model with Ollama" do
    assert OllamaUtils.ollama_running?()

    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_alpaca_model_and_optimizer.nx"

    assert File.exists?(checkpoint_path)

    device = use_accelerated_backend()
    tokenizer = "code-davinci-002"

    %{model_state_dict: model} = ModelCheckpoint.load_training_state!(checkpoint_path)
    model = Nx.backend_transfer(model, device)

    alpaca_data = load_instruction_json!("alpaca_data.json")

    test_data =
      alpaca_data |> instruction_data_partitions() |> Map.fetch!(:test) |> Enum.take(110)

    output_path = "alpaca-data-with-response.json"

    enriched_data =
      test_data
      |> InstructionsEvaluation.write_responses!(model, tokenizer, device,
        output_path: output_path
      )

    scores = InstructionsEvaluation.generate_model_scores(enriched_data, "model_response")
    average_score = Enum.sum(scores) / length(scores)
    metrics_path = "ch7_instruction_finetuning_alpaca_ollama_scores.json"

    write_ollama_score_metrics!(metrics_path, scores, average_score)

    assert File.exists?(output_path)
    assert File.exists?(metrics_path)
    assert length(enriched_data) == 110
    assert length(scores) == 110
    assert length(scores) == length(enriched_data)
    assert_in_delta average_score, 50.54, 0.4
  end

  @tag :download
  @tag timeout: 1_800_000
  test "exercise 7.4 freezes GPT-2 124M trainable parameters and adds LoRA adapters" do
    device = use_accelerated_backend()

    model =
      "124M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> GPTModel.replace_out_head(2, seed: 123, bias: true)
      |> Nx.backend_transfer(device)

    total_params_before = GPTModel.trainable_parameters(model)

    frozen_model = GPTModel.freeze(model)
    total_params_after = GPTModel.trainable_parameters(frozen_model)

    lora_model =
      LoRAUtils.replace_linear_with_lora(frozen_model, 16, 16, include_output_head: true)

    total_lora_params = GPTModel.trainable_parameters(lora_model)
    inspected = inspect(lora_model)

    assert total_params_before == GPTModel.total_parameters(model)
    assert total_params_before == 124_441_346
    assert GPTModel.frozen?(frozen_model)
    assert total_params_after == 0
    assert lora_model.trainable == [:lora]
    assert total_lora_params == 2_666_528
    assert inspected =~ "GPTModel("
    assert inspected =~ "(W_query): LinearWithLoRA("

    assert inspected =~
             "linear=Linear(in_features=768, out_features=768, bias=true)"

    assert inspected =~ "lora=LoRALayer(in_features=768, out_features=768, rank=16, alpha=16)"
    assert inspected =~ "(0): LinearWithLoRA("
    assert inspected =~ "lora=LoRALayer(in_features=768, out_features=3072, rank=16, alpha=16)"
    assert inspected =~ "(out_head): LinearWithLoRA("
    assert inspected =~ "lora=LoRALayer(in_features=768, out_features=2, rank=16, alpha=16)"
  end

  @tag :download
  @tag timeout: 3_600_000
  test "exercise 7.4 fine-tunes instruction model with LoRA" do
    device = use_accelerated_backend()
    tokenizer = "code-davinci-002"
    batch_size = 8
    num_workers = 3

    model =
      "355M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> GPTModel.freeze()
      |> LoRAUtils.replace_linear_with_lora(16, 16, include_output_head: true)
      |> Nx.backend_transfer(device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    val_portion = length(data) - train_portion - test_portion
    train_data = Enum.slice(data, 0, train_portion)
    val_data = Enum.slice(data, train_portion + test_portion, val_portion)

    train_dataset = InstructionDataset.new(train_data, tokenizer)
    val_dataset = InstructionDataset.new(val_data, tokenizer)

    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      DataLoader.new(train_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &binary_instruction_collate/1,
        shuffle: true,
        drop_last: true,
        num_workers: num_workers
      )

    val_loader =
      DataLoader.new(val_dataset.encoded_texts,
        batch_size: batch_size,
        collate_fn: &binary_instruction_collate/1,
        shuffle: false,
        drop_last: false,
        num_workers: num_workers
      )

    input_text = val_data |> List.first() |> FineTuneDataLoader.format_input()
    train_loss = LossUtils.calc_loss_loader(train_loader, model, device, 5)
    val_loss = LossUtils.calc_loss_loader(val_loader, model, device, 5)
    optimizer = Training.adamw(0.00005, weight_decay: 0.1)
    num_epochs = 2

    {trained_model, trained_optimizer, train_losses, val_losses, tokens_seen} =
      Training.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        5,
        5,
        input_text,
        tokenizer,
        generate_samples: false,
        return_optimizer: true
      )

    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_lora_model_and_optimizer.nx"
    ModelCheckpoint.save_training_state!(trained_model, trained_optimizer, checkpoint_path)

    metrics_path = "ch7_instruction_finetuning_lora_metrics.json"

    write_instruction_training_metrics!(
      metrics_path,
      num_epochs,
      train_losses,
      val_losses,
      tokens_seen
    )

    assert length(data) == 1_100
    assert length(train_data) == 935
    assert length(val_data) == 55
    assert train_loader.length == 116
    assert val_loader.length == 7
    assert model.trainable == [:lora]
    assert GPTModel.trainable_parameters(model) == 7_898_384
    assert_in_delta train_loss, 3.7422078609466554, 1.0e-5
    assert_in_delta val_loss, 3.7619348049163817, 1.0e-5
    assert input_text == FineTuneDataLoader.format_input(List.first(val_data))
    assert trained_model.__struct__ == model.__struct__
    assert trained_model.trainable == [:lora]
    assert %Training.AdamW{} = trained_optimizer
    assert File.exists?(checkpoint_path)
    assert File.stat!(checkpoint_path).size > 0
    assert File.exists?(metrics_path)
    assert File.stat!(metrics_path).size > 0
    assert length(train_losses) == 47
    assert length(val_losses) == 47
    assert length(tokens_seen) == 47
    assert Enum.all?(train_losses ++ val_losses, &is_float/1)
    assert Enum.all?(tokens_seen, &is_integer/1)
    assert tokens_seen == Enum.sort(tokens_seen)
    assert List.last(tokens_seen) > List.first(tokens_seen)
  end

  @tag :download
  @tag :train
  @tag :ollama
  @tag timeout: 1_800_000
  test "exercise 7.4 scores LoRA instruction model with Ollama" do
    assert OllamaUtils.ollama_running?()

    checkpoint_path = "ch7_instruction_finetuned_gpt2_355m_lora_model_and_optimizer.nx"

    assert File.exists?(checkpoint_path)

    device = use_accelerated_backend()
    tokenizer = "code-davinci-002"

    %{model_state_dict: model} = ModelCheckpoint.load_training_state!(checkpoint_path)
    model = Nx.backend_transfer(model, device)

    file_path =
      System.tmp_dir!()
      |> Path.join("llm_scratch_instruction_data")
      |> Path.join("instruction-data.json")

    data =
      FineTuneDataLoader.download_and_load_instructions_file(
        file_path,
        @instruction_data_url
      )

    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    test_data = Enum.slice(data, train_portion, test_portion)

    output_path = "instruction-data-with-response-lora.json"

    enriched_data =
      test_data
      |> InstructionsEvaluation.write_responses!(model, tokenizer, device,
        output_path: output_path
      )

    scores = InstructionsEvaluation.generate_model_scores(enriched_data, "model_response")
    average_score = Enum.sum(scores) / length(scores)
    baseline_average_score = 50.32
    metrics_path = "ch7_instruction_finetuning_lora_ollama_scores.json"

    write_ollama_score_metrics!(metrics_path, scores, average_score, baseline_average_score)

    assert_in_delta average_score, 49.9, 0.4
  end

  defp binary_instruction_collate(batch) do
    binary_instruction_collate(batch, nil)
  end

  defp binary_instruction_collate(batch, allowed_max_length) do
    previous_backend = Nx.default_backend()

    try do
      Nx.default_backend(Nx.BinaryBackend)
      InstructionDataset.custom_collate(batch, 50256, -100, allowed_max_length)
    after
      Nx.default_backend(previous_backend)
    end
  end

  defp instruction_data_loader(dataset, opts) do
    allowed_max_length = Keyword.get(opts, :allowed_max_length)

    DataLoader.new(dataset.encoded_texts,
      batch_size: 8,
      collate_fn: &binary_instruction_collate(&1, allowed_max_length),
      shuffle: Keyword.fetch!(opts, :shuffle),
      drop_last: Keyword.fetch!(opts, :drop_last),
      num_workers: 3
    )
  end

  defp load_instruction_json!(path) do
    path
    |> File.read!()
    |> Jason.decode!()
  end

  defp instruction_data_partitions(data) do
    train_portion = trunc(length(data) * 0.85)
    test_portion = trunc(length(data) * 0.1)
    val_portion = length(data) - train_portion - test_portion

    %{
      train: Enum.slice(data, 0, train_portion),
      test: Enum.slice(data, train_portion, test_portion),
      val: Enum.slice(data, train_portion + test_portion, val_portion)
    }
  end

  defp write_instruction_training_metrics!(
         path,
         num_epochs,
         train_losses,
         val_losses,
         tokens_seen
       ) do
    metrics = %{
      num_epochs: num_epochs,
      losses: %{
        tokens_seen: tokens_seen,
        train_values: train_losses,
        val_values: val_losses
      }
    }

    {:ok, encoded_metrics} = Jason.encode(metrics, pretty: true)
    File.write!(path, encoded_metrics)
  end

  defp write_ollama_score_metrics!(path, scores, average_score, baseline_average_score \\ nil) do
    metrics =
      %{
        score_count: length(scores),
        average_score: average_score,
        scores: scores
      }
      |> maybe_put_baseline_score(baseline_average_score)

    {:ok, encoded_metrics} = Jason.encode(metrics, pretty: true)
    File.write!(path, encoded_metrics)
  end

  defp maybe_put_baseline_score(metrics, nil), do: metrics

  defp maybe_put_baseline_score(metrics, baseline_average_score) do
    metrics
    |> Map.put(:baseline_average_score, baseline_average_score)
    |> Map.put(:score_delta, metrics.average_score - baseline_average_score)
  end

  defp response_score(response) do
    response
    |> then(&Regex.run(~r/\b(100|[1-9]?\d)\s*(?:out of|\/)\s*100\b/, &1, capture: :all_but_first))
    |> case do
      [score] -> String.to_integer(score)
      nil -> flunk("Expected Ollama response to include a score out of 100, got: #{response}")
    end
  end
end
