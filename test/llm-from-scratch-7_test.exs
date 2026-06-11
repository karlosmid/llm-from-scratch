defmodule LlmFromScratch7Test do
  use ExUnit.Case

  import LlmScratch.TestHelpers

  alias LlmScratch.{
    DataLoader,
    FineTuneDataLoader,
    GPT2OpenAI,
    InstructionDataset,
    LossUtils,
    TextGeneration,
    TextUtils
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
    assert InstructionDataset.get(dataset, 0) == expected_tokens
    assert InstructionDataset.length(dataset) == 1
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
  @tag timeout: 3_600_000
  test "7.5 generates response for validation instruction with OpenAI GPT-2 355M" do
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
  end
end
