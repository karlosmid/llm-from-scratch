defmodule LlmFromScratch7Test do
  use ExUnit.Case

  alias LlmScratch.{FineTuneDataLoader, InstructionDataset}

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
end
