defmodule LlmFromScratch6Test do
  use ExUnit.Case

  alias LlmScratch.{
    DataLoader,
    FineTuneDataLoader,
    GPT2OpenAI,
    GPTConfig,
    SpamDataset,
    TextGeneration,
    TextUtils
  }

  @tag :download
  test "6.1 loads SMS spam TSV rows and counts labels" do
    tmp_dir = Path.join(System.tmp_dir!(), "llm_scratch_sms_spam_collection")
    data_file_path = Path.expand("SMSSpamCollection.tsv")

    data_file_path =
      FineTuneDataLoader.download_and_unzip_spam_data(
        zip_path: Path.join(tmp_dir, "sms_spam_collection.zip"),
        extracted_path: Path.join(tmp_dir, "sms_spam_collection"),
        data_file_path: data_file_path
      )

    records = FineTuneDataLoader.load_spam_data(data_file_path)
    label_counts = FineTuneDataLoader.label_counts(records)

    assert length(records) == 5_574
    assert label_counts["ham"] == 4_827
    assert label_counts["spam"] == 747

    balanced_records = FineTuneDataLoader.create_balanced_dataset(records)
    balanced_label_counts = FineTuneDataLoader.label_counts(balanced_records)

    assert length(balanced_records) == 1_494
    assert balanced_label_counts["ham"] == 747
    assert balanced_label_counts["spam"] == 747

    mapped_records = FineTuneDataLoader.map_labels_to_ids(balanced_records)
    mapped_label_counts = FineTuneDataLoader.label_counts(mapped_records)

    assert mapped_label_counts[0] == 747
    assert mapped_label_counts[1] == 747

    {train_records, validation_records, test_records} =
      FineTuneDataLoader.random_split(mapped_records, 0.7, 0.1)

    write_csv!("train.csv", train_records)
    write_csv!("validation.csv", validation_records)
    write_csv!("test.csv", test_records)

    assert length(train_records) == 1_045
    assert length(validation_records) == 149
    assert length(test_records) == 300
    assert File.exists?("train.csv")
    assert File.exists?("validation.csv")
    assert File.exists?("test.csv")
  end

  defp write_csv!(path, records) do
    rows =
      records
      |> Enum.map(fn record ->
        [csv_field(record.label), ",", csv_field(record.text), "\n"]
      end)

    File.write!(path, ["Label,Text\n", rows])
  end

  defp csv_field(value) do
    value
    |> to_string()
    |> String.replace("\"", "\"\"")
    |> then(&"\"#{&1}\"")
  end

  test "6.3 creating data loaders" do
    tokenizer = "code-davinci-002"
    {:ok, encoded_tokens} = Tiktoken.encode(tokenizer, "<|endoftext|>", ["<|endoftext|>"])

    assert encoded_tokens == [50_256]

    train_dataset = SpamDataset.new("train.csv", tokenizer, max_length: nil)

    assert train_dataset.max_length == 120

    val_dataset =
      SpamDataset.new("validation.csv", tokenizer, max_length: train_dataset.max_length)

    test_dataset =
      SpamDataset.new("test.csv", tokenizer, max_length: train_dataset.max_length)

    num_workers = 0
    batch_size = 8
    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      train_dataset
      |> dataset_samples()
      |> DataLoader.new(
        batch_size: batch_size,
        shuffle: true,
        num_workers: num_workers,
        drop_last: true
      )

    val_loader =
      val_dataset
      |> dataset_samples()
      |> DataLoader.new(
        batch_size: batch_size,
        num_workers: num_workers,
        drop_last: false
      )

    test_loader =
      test_dataset
      |> dataset_samples()
      |> DataLoader.new(
        batch_size: batch_size,
        num_workers: num_workers,
        drop_last: false
      )

    {input_batch, target_batch} =
      train_loader.batches
      |> List.first()
      |> collate_batch()

    assert Nx.shape(input_batch) == {8, 120}
    assert Nx.shape(target_batch) == {8}

    assert train_loader.length == 130
    assert val_loader.length == 19
    assert test_loader.length == 38
  end

  defp dataset_samples(dataset) do
    0..(SpamDataset.length(dataset) - 1)
    |> Enum.map(&SpamDataset.get(dataset, &1))
  end

  defp collate_batch(batch) do
    {inputs, labels} = Enum.unzip(batch)

    {
      Nx.stack(inputs),
      Nx.stack(labels)
    }
  end

  @tag :download
  @tag timeout: 900_000
  test "6.4 loads OpenAI GPT-2 and generates classification prompts" do
    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    choose_model = "gpt2-small (124M)"
    tokenizer = "code-davinci-002"
    config = GPTConfig.openai_gpt2(choose_model)

    model_size =
      choose_model
      |> String.split(" ")
      |> List.last()
      |> String.trim_leading("(")
      |> String.trim_trailing(")")

    model = GPT2OpenAI.load_model(model_size, models_dir: "gpt2")

    assert config.vocab_size == 50_257
    assert config.context_length == 1024
    assert config.drop_rate == 0.0
    assert config.qkv_bias == true
    assert config.emb_dim == 768
    assert config.n_layers == 12
    assert config.n_heads == 12

    decoded_text_1 =
      model
      |> TextGeneration.generate_text_simple(
        TextUtils.text_to_token_ids("Every effort moves you", tokenizer),
        15,
        config.context_length
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> TextUtils.token_ids_to_text(tokenizer)

    assert decoded_text_1 ==
             "Every effort moves you forward.\n\nThe first step is to understand the importance of your work"

    text_2 =
      "Is the following text 'spam'? Answer with 'yes' or 'no':" <>
        " 'You are a winner you have been specially" <>
        " selected to receive $1000 cash or a $2000 award.'"

    decoded_text_2 =
      model
      |> TextGeneration.generate_text_simple(
        TextUtils.text_to_token_ids(text_2, tokenizer),
        23,
        config.context_length
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> TextUtils.token_ids_to_text(tokenizer)

    assert decoded_text_2 ==
             "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'\n\nThe following text 'spam'? Answer with 'yes' or 'no': 'You are a winner"
  end


end
