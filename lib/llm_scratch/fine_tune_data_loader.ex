defmodule LlmScratch.FineTuneDataLoader do
  @moduledoc """
  Helpers for loading small supervised fine-tuning datasets.

  The SMS spam dataset is a two-column TSV file with labels (`ham` or `spam`)
  and text. Records can be loaded as maps or converted to Nx tensor pairs for
  use with `LlmScratch.DataLoader`.
  """

  @spam_url "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
  @zip_path "sms_spam_collection.zip"
  @extracted_path "sms_spam_collection"
  @data_filename "SMSSpamCollection.tsv"
  @raw_filename "SMSSpamCollection"
  @label_ids %{"ham" => 0, "spam" => 1}
  @pad_token_id 50_256

  @type spam_record :: %{
          label: String.t(),
          label_id: 0 | 1,
          text: String.t()
        }

  @doc """
  Downloads and extracts the UCI SMS spam dataset if the TSV is not present.

  Accepts a keyword list of path and URL options.

  ## Input Parameters

    * `opts` - optional keyword list controlling where the dataset is downloaded,
      extracted, and saved.

  ## Options

    * `:url` - dataset zip URL. Defaults to the UCI SMS spam collection URL.
    * `:zip_path` - local zip cache path. Defaults to
      `"sms_spam_collection.zip"`.
    * `:extracted_path` - directory where the zip is extracted. Defaults to
      `"sms_spam_collection"`.
    * `:data_file_path` - final TSV path. Defaults to
      `"sms_spam_collection/SMSSpamCollection.tsv"`.

  ## Output

  Returns the final TSV path as a string. If `:data_file_path` already exists,
  no download or extraction is performed. If the zip exists but the TSV does
  not, the existing zip is reused.
  """
  @spec download_and_unzip_spam_data(keyword()) :: Path.t()
  def download_and_unzip_spam_data(opts \\ []) do
    url = Keyword.get(opts, :url, @spam_url)
    zip_path = Keyword.get(opts, :zip_path, @zip_path)
    extracted_path = Keyword.get(opts, :extracted_path, @extracted_path)
    data_file_path = Keyword.get(opts, :data_file_path, Path.join(extracted_path, @data_filename))

    if File.exists?(data_file_path) do
      data_file_path
    else
      File.mkdir_p!(Path.dirname(zip_path))
      File.mkdir_p!(extracted_path)

      unless File.exists?(zip_path) do
        download_file!(url, zip_path)
      end

      unzip_file!(zip_path, extracted_path)
      rename_raw_dataset!(extracted_path, data_file_path)

      data_file_path
    end
  end

  @doc """
  Loads the SMS spam TSV into a list of maps.

  ## Input Parameters

    * `data_file_path` - path to a two-column TSV file. Each row must have a
      label (`ham` or `spam`) in the first column and the message text in the
      second column.

  ## Output

  Returns a list of `%{label: label, label_id: label_id, text: text}` maps.
  `label_id` is `0` for `ham` and `1` for `spam`.
  """
  @spec load_spam_data(Path.t()) :: [spam_record()]
  def load_spam_data(data_file_path) do
    data_file_path
    |> File.stream!([], :line)
    |> Stream.map(&String.trim_trailing(&1, "\n"))
    |> Stream.map(&String.trim_trailing(&1, "\r"))
    |> Stream.reject(&(&1 == ""))
    |> Enum.map(&parse_spam_row!/1)
  end

  @doc """
  Counts records by label.

  ## Input Parameters

    * `records` - list of records returned by `load_spam_data/1`, or any list
      of maps containing a `:label` key.

  ## Output

  Returns a map where each key is a label value and each value is the number
  of records with that label.
  """
  @spec label_counts([map()]) :: %{optional(term()) => non_neg_integer()}
  def label_counts(records) do
    Enum.frequencies_by(records, & &1.label)
  end

  @doc """
  Creates a balanced SMS spam dataset by undersampling ham records.

  This mirrors the pandas example:

      num_spam = df[df["Label"] == "spam"].shape[0]
      ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
      balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

  ## Input Parameters

    * `records` - list of records returned by `load_spam_data/1`.
    * `opts` - optional keyword list controlling the deterministic sample.

  ## Options

    * `:random_state` - integer seed used for sampling ham records. Defaults
      to `123`.

  ## Output

  Returns a list containing a deterministic sample of ham records followed by
  all spam records. The returned dataset has the same number of ham and spam
  records.
  """
  @spec create_balanced_dataset([spam_record()], keyword()) :: [spam_record()]
  def create_balanced_dataset(records, opts \\ []) do
    random_state = Keyword.get(opts, :random_state, 123)

    spam_records = Enum.filter(records, &(&1.label == "spam"))
    ham_records = Enum.filter(records, &(&1.label == "ham"))
    num_spam = length(spam_records)

    if length(ham_records) < num_spam do
      raise ArgumentError,
            "expected at least #{num_spam} ham records, got #{length(ham_records)}"
    end

    ham_subset =
      ham_records
      |> deterministic_sample(num_spam, random_state)

    ham_subset ++ spam_records
  end

  @doc """
  Converts string class labels to integer class labels.

  This mirrors the pandas example:

      balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

  ## Input Parameters

    * `records` - list of maps with `:label` set to `"ham"` or `"spam"`.

  ## Output

  Returns a list of records with `:label` replaced by `0` for ham and `1` for
  spam. Other fields, including `:text` and `:label_id`, are preserved.
  """
  @spec map_labels_to_ids([spam_record()]) :: [map()]
  def map_labels_to_ids(records) do
    Enum.map(records, fn %{label: label} = record ->
      %{record | label: Map.fetch!(@label_ids, label)}
    end)
  end

  @doc """
  Randomly shuffles records and splits them into train, validation, and test sets.

  This mirrors the pandas example:

      df = df.sample(frac=1, random_state=123).reset_index(drop=True)
      train_end = int(len(df) * train_frac)
      validation_end = train_end + int(len(df) * validation_frac)
      train_df = df[:train_end]
      validation_df = df[train_end:validation_end]
      test_df = df[validation_end:]

  The test split is implied as the remaining records after the train and
  validation splits.

  ## Input Parameters

    * `records` - list of records to shuffle and split.
    * `train_frac` - fraction of records assigned to the train split.
    * `validation_frac` - fraction of records assigned to the validation split.
    * `opts` - optional keyword list controlling deterministic shuffling.

  ## Options

    * `:random_state` - integer seed used for shuffling. Defaults to `123`.

  ## Output

  Returns `{train_records, validation_records, test_records}`. The train and
  validation split sizes are truncated with `trunc/1`, matching Python's
  `int(...)`; the test split receives the remainder.
  """
  @spec random_split([map()], number(), number(), keyword()) :: {[map()], [map()], [map()]}
  def random_split(records, train_frac, validation_frac, opts \\ []) do
    validate_split_fractions!(train_frac, validation_frac)

    random_state = Keyword.get(opts, :random_state, 123)
    shuffled_records = deterministic_sample(records, length(records), random_state)
    train_end = trunc(length(shuffled_records) * train_frac)
    validation_end = train_end + trunc(length(shuffled_records) * validation_frac)

    {
      Enum.slice(shuffled_records, 0, train_end),
      Enum.slice(shuffled_records, train_end, validation_end - train_end),
      Enum.slice(shuffled_records, validation_end, length(shuffled_records) - validation_end)
    }
  end

  @doc """
  Builds a reusable dataloader for the SMS spam dataset.

  By default, records are tokenized into `{input_ids, label}` Nx tensor pairs.
  Pass `tokenize: false` to batch raw record maps.

  ## Input Parameters

    * `opts` - optional keyword list controlling dataset location,
      tokenization, and batching.

  ## Options

    * `:data_file_path` - existing TSV path to load. When omitted,
      `download_and_unzip_spam_data/1` is called with the same options.
    * `:url` - dataset zip URL used only when `:data_file_path` is omitted.
    * `:zip_path` - local zip cache path used only when downloading or
      extracting.
    * `:extracted_path` - extraction directory used only when downloading or
      extracting.
    * `:tokenize` - when `true`, returns batches of `{input_ids, label}` Nx
      tensor pairs. When `false`, batches raw record maps. Defaults to `true`.
    * `:tokenizer_model` - Tiktoken model name used when tokenizing. Defaults
      to `"code-davinci-002"`.
    * `:max_length` - fixed token length for each text tensor. Longer messages
      are truncated and shorter messages are padded. Defaults to `120`.
    * `:pad_token_id` - token id used for padding. Defaults to GPT-2's
      end-of-text token id, `50256`.
    * `:batch_size` - number of examples per batch. Defaults to `32`.
    * `:shuffle` - whether to shuffle records before batching. Defaults to
      `true`.
    * `:drop_last` - whether to drop an incomplete final batch. Defaults to
      `false`.
    * `:num_workers` - worker count passed through to `LlmScratch.DataLoader`.
      Defaults to `0`.

  ## Output

  Returns the map produced by `LlmScratch.DataLoader.new/2`, including
  `:stream`, `:batches`, `:length`, `:batch_size`, `:shuffle`, `:drop_last`,
  and `:num_workers`.
  """
  @spec create_spam_dataloader(keyword()) :: map()
  def create_spam_dataloader(opts \\ []) do
    data_file_path =
      Keyword.get_lazy(opts, :data_file_path, fn ->
        download_and_unzip_spam_data(opts)
      end)

    records = load_spam_data(data_file_path)

    dataset =
      if Keyword.get(opts, :tokenize, true) do
        to_tensor_dataset(records, opts)
      else
        records
      end

    LlmScratch.DataLoader.new(dataset,
      batch_size: Keyword.get(opts, :batch_size, 32),
      shuffle: Keyword.get(opts, :shuffle, true),
      drop_last: Keyword.get(opts, :drop_last, false),
      num_workers: Keyword.get(opts, :num_workers, 0)
    )
  end

  @doc """
  Converts records to `{input_ids, label}` tensor pairs.

  Texts are encoded with Tiktoken, truncated to `:max_length`, and padded with
  GPT-2's end-of-text token id.

  ## Input Parameters

    * `records` - list of `%{label_id: label_id, text: text}` maps.
    * `opts` - optional keyword list controlling tokenization.

  ## Options

    * `:tokenizer_model` - Tiktoken model name used for encoding text. Defaults
      to `"code-davinci-002"`.
    * `:max_length` - fixed number of token ids in each input tensor. Defaults
      to `120`.
    * `:pad_token_id` - token id appended until each input reaches
      `:max_length`. Defaults to `50256`.

  ## Output

  Returns a list of `{input_ids, label}` tuples. `input_ids` is an Nx tensor of
  shape `{max_length}` and signed 64-bit integer type. `label` is a scalar Nx
  tensor containing `0` for `ham` or `1` for `spam`.
  """
  @spec to_tensor_dataset([spam_record()], keyword()) :: [{Nx.Tensor.t(), Nx.Tensor.t()}]
  def to_tensor_dataset(records, opts \\ []) do
    model = Keyword.get(opts, :tokenizer_model, "code-davinci-002")
    max_length = Keyword.get(opts, :max_length, 120)
    pad_token_id = Keyword.get(opts, :pad_token_id, @pad_token_id)

    Enum.map(records, fn %{label_id: label_id, text: text} ->
      input_ids =
        text
        |> encode_text!(model)
        |> pad_or_truncate(max_length, pad_token_id)
        |> Nx.tensor(type: {:s, 64})

      {input_ids, Nx.tensor(label_id, type: {:s, 64})}
    end)
  end

  defp parse_spam_row!(row) do
    case String.split(row, "\t", parts: 2) do
      [label, text] when is_map_key(@label_ids, label) ->
        %{label: label, label_id: Map.fetch!(@label_ids, label), text: text}

      [label, _text] ->
        raise ArgumentError, "unknown SMS spam label #{inspect(label)}"

      _ ->
        raise ArgumentError, "expected a two-column TSV row, got #{inspect(row)}"
    end
  end

  defp encode_text!(text, model) do
    {:ok, token_ids} = Tiktoken.encode(model, text, ["<|endoftext|>"])
    token_ids
  end

  defp pad_or_truncate(token_ids, max_length, pad_token_id) do
    token_ids
    |> Enum.take(max_length)
    |> then(fn ids -> ids ++ List.duplicate(pad_token_id, max_length - length(ids)) end)
  end

  defp deterministic_sample(records, count, random_state) do
    seed = {random_state, random_state, random_state}
    initial_state = :rand.seed_s(:exsss, seed)

    records
    |> Enum.map_reduce(initial_state, fn record, state ->
      {sort_key, next_state} = :rand.uniform_s(state)
      {{sort_key, record}, next_state}
    end)
    |> elem(0)
    |> Enum.sort_by(fn {sort_key, _record} -> sort_key end)
    |> Enum.take(count)
    |> Enum.map(fn {_sort_key, record} -> record end)
  end

  defp validate_split_fractions!(train_frac, validation_frac) do
    cond do
      train_frac < 0 or validation_frac < 0 ->
        raise ArgumentError, "split fractions must be greater than or equal to 0"

      train_frac + validation_frac > 1 ->
        raise ArgumentError, "train and validation fractions must sum to 1 or less"

      true ->
        :ok
    end
  end

  defp download_file!(url, destination) do
    tmp_destination = destination <> ".download"
    File.rm(tmp_destination)

    response =
      Req.get!(
        url,
        into: File.stream!(tmp_destination),
        receive_timeout: 120_000,
        retry: false
      )

    if response.status in 200..299 do
      File.rename!(tmp_destination, destination)
    else
      File.rm(tmp_destination)
      raise "failed to download #{url}: HTTP #{response.status}"
    end
  rescue
    error ->
      File.rm(destination <> ".download")
      reraise error, __STACKTRACE__
  end

  defp unzip_file!(zip_path, extracted_path) do
    zip_path
    |> String.to_charlist()
    |> :zip.unzip([{:cwd, String.to_charlist(extracted_path)}])
    |> case do
      {:ok, _files} -> :ok
      {:error, reason} -> raise "failed to unzip #{zip_path}: #{inspect(reason)}"
    end
  end

  defp rename_raw_dataset!(extracted_path, data_file_path) do
    raw_file_path = Path.join(extracted_path, @raw_filename)

    if File.exists?(raw_file_path) do
      File.rename!(raw_file_path, data_file_path)
    else
      raise "expected extracted dataset file at #{raw_file_path}"
    end
  end
end
