defmodule LlmFromScratch6Test do
  use ExUnit.Case

  alias LlmScratch.{
    DataLoader,
    FineTuneDataLoader,
    GPT2OpenAI,
    GPTConfig,
    GPTModel,
    LossClassificationUtils,
    LossUtils,
    ModelCheckpoint,
    SpamDataset,
    TextGeneration,
    TextUtils,
    Training
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

  defp gpt2_compatible_token_ids(text) do
    {:ok, token_ids} = Tiktoken.encode("code-davinci-002", text, ["<|endoftext|>"])

    Enum.map(token_ids, &min(&1, 50_256))
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

  @tag :download
  @tag timeout: 900_000
  test "6.5 inspects saved GPT-2 small like a PyTorch module tree" do
    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    model = GPT2OpenAI.load_model("124M", models_dir: "gpt2")
    inspected = inspect(model)

    assert inspected =~ "GPTModel("
    assert inspected =~ "(tok_emb): Embedding(50257, 768)"
    assert inspected =~ "(pos_emb): Embedding(1024, 768)"
    assert inspected =~ "(drop_emb): Dropout(p=0.0, inplace=false)"
    assert inspected =~ "(trf_blocks): Sequential("
    assert inspected =~ "(0): TransformerBlock("
    assert inspected =~ "(11): TransformerBlock("
    assert inspected =~ "(W_query): Linear(in_features=768, out_features=768, bias=true)"
    assert inspected =~ "(W_key): Linear(in_features=768, out_features=768, bias=true)"
    assert inspected =~ "(W_value): Linear(in_features=768, out_features=768, bias=true)"
    assert inspected =~ "(out_proj): Linear(in_features=768, out_features=768, bias=true)"
    assert inspected =~ "(0): Linear(in_features=768, out_features=3072, bias=true)"
    assert inspected =~ "(1): GELU()"
    assert inspected =~ "(2): Linear(in_features=3072, out_features=768, bias=true)"
    assert inspected =~ "(norm1): LayerNorm()"
    assert inspected =~ "(norm2): LayerNorm()"
    assert inspected =~ "(drop_resid): Dropout(p=0.0, inplace=false)"
    assert inspected =~ "(final_norm): LayerNorm()"
    assert inspected =~ "(out_head): Linear(in_features=768, out_features=50257, bias=false)"

    assert Regex.scan(~r/\(\d+\): TransformerBlock\(/, inspected) |> length() == 12

    # by frozing model blocks, we skip their training
    frozen_model = GPTModel.freeze(model)
    assert GPTModel.frozen?(frozen_model)
    assert frozen_model.trainable == []

    # we change output block second dimension to two, as this we have two classes, ham/spam
    # we want to train final_norm block and last transformer block
    # Sebastian states that based on his experiments, we will get better results
    num_classes = 2

    classification_model =
      frozen_model
      |> GPTModel.replace_out_head(num_classes, seed: 123, bias: true)
      |> GPTModel.set_trainable([:out_head, :final_norm, {:trf_block, 11}])

    assert classification_model.trainable == [:out_head, :final_norm, {:trf_block, 11}]

    assert inspect(classification_model) =~
             "(out_head): Linear(in_features=768, out_features=2, bias=true)"

    # input message that we want to classify
    inputs =
      "Do you have time"
      |> TextUtils.text_to_token_ids("code-davinci-002")
      |> Nx.backend_transfer(Nx.BinaryBackend)

    assert Nx.to_list(inputs) == [[5211, 345, 423, 640]]
    assert Nx.shape(inputs) == {1, 4}

    outputs =
      classification_model
      |> GPTModel.forward(Nx.backend_transfer(inputs, EXLA.Backend))
      |> Nx.backend_transfer(Nx.BinaryBackend)

    assert Nx.shape(outputs) == {1, 4, 2}

    expected_outputs =
      Nx.tensor([
        [
          [0.41287035, 1.2672385],
          [-3.6814282, 4.7981105],
          [-3.9752169, 4.006133],
          [-1.0933434, 3.5170393]
        ]
      ])

    assert Nx.all_close(outputs, expected_outputs, atol: 1.0e-5) |> Nx.to_number() == 1

    # it is enough to do fine tune training only for last input token!
    # Why? Attention mechanism is using casual attention mask, where we hide from current token its next token
    # Because of that, only the last message token has attention weights of all previous token.
    last_output_token = outputs[[.., -1, ..]]
    assert Nx.shape(last_output_token) == {1, 2}

    assert Nx.all_close(last_output_token, Nx.tensor([[-1.0933434, 3.5170393]]), atol: 1.0e-5)
           |> Nx.to_number() == 1
  end

  @tag :download
  @tag timeout: 900_000
  test "6.6 calculates classification accuracy before fine-tuning" do
    previous_backend = Nx.default_backend()
    device = Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    tokenizer = &gpt2_compatible_token_ids/1

    train_dataset = SpamDataset.new("train.csv", tokenizer, max_length: nil)

    val_dataset =
      SpamDataset.new("validation.csv", tokenizer, max_length: train_dataset.max_length)

    test_dataset =
      SpamDataset.new("test.csv", tokenizer, max_length: train_dataset.max_length)

    batch_size = 8
    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      train_dataset
      |> dataset_samples()
      |> DataLoader.new(batch_size: batch_size, shuffle: true, num_workers: 0, drop_last: true)

    val_loader =
      val_dataset
      |> dataset_samples()
      |> DataLoader.new(batch_size: batch_size, num_workers: 0, drop_last: false)

    test_loader =
      test_dataset
      |> dataset_samples()
      |> DataLoader.new(batch_size: batch_size, num_workers: 0, drop_last: false)

    model =
      "124M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> GPTModel.freeze()
      |> GPTModel.replace_out_head(2, seed: 123, bias: true)
      |> GPTModel.set_trainable([:out_head, :final_norm, {:trf_block, 11}])

    # returns the number of correctly predicted classification divided with number of messages in a batch
    train_accuracy = LossClassificationUtils.calc_accuracy_loader(train_loader, model, device, 10)
    val_accuracy = LossClassificationUtils.calc_accuracy_loader(val_loader, model, device, 10)
    test_accuracy = LossClassificationUtils.calc_accuracy_loader(test_loader, model, device, 10)

    train_loss = LossUtils.calc_loss_loader(train_loader, model, device, 10, target: :last_token)
    val_loss = LossUtils.calc_loss_loader(val_loader, model, device, 10, target: :last_token)
    test_loss = LossUtils.calc_loss_loader(test_loader, model, device, 10, target: :last_token)

    # we have almost 50:50 chance for ham/spam, as we still have not trained our model for classification
    assert train_accuracy == 0.5125
    assert val_accuracy == 0.525
    assert test_accuracy == 0.5625

    assert_in_delta train_loss, 1.5194151997566223, 1.0e-6
    assert_in_delta val_loss, 1.4858015120029449, 1.0e-6
    assert_in_delta test_loss, 1.3634233981370927, 1.0e-6
  end

  @tag :download
  @tag :train
  @tag timeout: 3_600_000
  test "6.7 fine-tunes classifier on spam dataset" do
    previous_backend = Nx.default_backend()
    device = Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    tokenizer = &gpt2_compatible_token_ids/1

    train_dataset = SpamDataset.new("train.csv", tokenizer, max_length: nil)

    val_dataset =
      SpamDataset.new("validation.csv", tokenizer, max_length: train_dataset.max_length)

    batch_size = 8
    :rand.seed(:exsss, {123, 123, 123})

    train_loader =
      train_dataset
      |> dataset_samples()
      |> DataLoader.new(batch_size: batch_size, shuffle: true, num_workers: 0, drop_last: true)

    val_loader =
      val_dataset
      |> dataset_samples()
      |> DataLoader.new(batch_size: batch_size, num_workers: 0, drop_last: false)

    model =
      "124M"
      |> GPT2OpenAI.load_model(models_dir: "gpt2")
      |> GPTModel.freeze()
      |> GPTModel.replace_out_head(2, seed: 123, bias: true)
      |> GPTModel.set_trainable([:out_head, :final_norm, {:trf_block, 11}])

    optimizer = Training.adamw(5.0e-5, weight_decay: 0.1)
    num_epochs = 5

    {trained_model, trained_optimizer, train_losses, val_losses, train_accs, val_accs,
     examples_seen} =
      Training.train_classifier_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        50,
        5,
        return_optimizer: true
      )

    checkpoint_path = "ch6_spam_classifier_model_and_optimizer.nx"
    ModelCheckpoint.save_training_state!(trained_model, trained_optimizer, checkpoint_path)

    metrics_path = "ch6_spam_classifier_training_metrics.json"

    write_training_metrics!(
      metrics_path,
      num_epochs,
      examples_seen,
      train_losses,
      val_losses,
      train_accs,
      val_accs
    )

    assert %GPTModel{} = trained_model
    assert %Training.AdamW{} = trained_optimizer
    assert File.exists?(checkpoint_path)
    assert File.stat!(checkpoint_path).size > 0
    assert File.exists?(metrics_path)
    assert File.stat!(metrics_path).size > 0
    assert length(train_losses) == 13
    assert length(val_losses) == 13
    assert length(train_accs) == num_epochs
    assert length(val_accs) == num_epochs
    assert examples_seen == train_loader.length * batch_size * num_epochs
    assert Enum.all?(train_losses ++ val_losses, &is_float/1)
    assert Enum.all?(train_accs ++ val_accs, &(&1 >= 0.0 and &1 <= 1.0))
  end

  defp write_training_metrics!(
         path,
         num_epochs,
         examples_seen,
         train_losses,
         val_losses,
         train_accs,
         val_accs
       ) do
    metrics = %{
      num_epochs: num_epochs,
      examples_seen: examples_seen,
      losses: %{
        epochs_seen: linspace(0.0, num_epochs * 1.0, length(train_losses)),
        examples_seen: linspace(0.0, examples_seen * 1.0, length(train_losses)),
        train_values: train_losses,
        val_values: val_losses
      },
      accuracies: %{
        epochs_seen: linspace(1.0, num_epochs * 1.0, length(train_accs)),
        examples_seen:
          linspace(examples_seen / num_epochs, examples_seen * 1.0, length(train_accs)),
        train_values: train_accs,
        val_values: val_accs
      }
    }

    {:ok, encoded_metrics} = Jason.encode(metrics, pretty: true)
    File.write!(path, encoded_metrics)
  end

  defp linspace(_start, stop, 1), do: [stop * 1.0]

  defp linspace(start, stop, count) do
    step = (stop - start) / (count - 1)

    Enum.map(0..(count - 1), fn index ->
      start + index * step
    end)
  end
end
