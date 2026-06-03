defmodule LlmScratch.SpamDataset do
  @moduledoc """
  CSV-backed spam classification dataset.

  This mirrors the PyTorch `SpamDataset` example from chapter 6. A dataset is
  built from a CSV file containing `Label` and `Text` columns, pre-tokenizes all
  texts during construction, truncates sequences that exceed the configured
  maximum length, and pads shorter sequences with a fixed token id.
  """

  @enforce_keys [:data, :encoded_texts, :max_length, :pad_token_id]
  defstruct [:data, :encoded_texts, :max_length, :pad_token_id]

  @end_of_text "<|endoftext|>"
  @pad_token_id 50_256

  @type row :: %{label: integer(), text: String.t()}
  @type tokenizer :: String.t() | (String.t() -> [integer()])
  @type t :: %__MODULE__{
          data: [row()],
          encoded_texts: [[integer()]],
          max_length: non_neg_integer(),
          pad_token_id: integer()
        }

  @doc """
  Creates a spam dataset from a CSV file.

  ## Input Parameters

    * `csv_file` - path to a CSV file with a header row containing `Label` and
      `Text` columns. Labels may be integer strings (`"0"`, `"1"`) or the SMS
      spam labels (`"ham"`, `"spam"`).
    * `tokenizer` - either a Tiktoken model name accepted by `Tiktoken.encode/3`
      or a one-argument function that receives text and returns a list of token
      ids.
    * `opts` - optional keyword list controlling sequence length and padding.

  ## Options

    * `:max_length` - fixed sequence length. When omitted, the longest
      pre-tokenized sequence length in the CSV is used.
    * `:pad_token_id` - token id appended to shorter sequences. Defaults to
      GPT-2's end-of-text token id, `50256`.
    * `:allowed_special` - special tokens allowed when `tokenizer` is a
      Tiktoken model name. Defaults to `["<|endoftext|>"]`.

  ## Output

  Returns a `%LlmScratch.SpamDataset{}`. The `:data` field contains parsed
  `%{label: integer, text: text}` rows, and `:encoded_texts` contains one
  padded/truncated token-id list per row.
  """
  @spec new(Path.t(), tokenizer(), keyword()) :: t()
  def new(csv_file, tokenizer, opts \\ []) when is_binary(csv_file) do
    data = read_csv!(csv_file)
    encoded_texts = Enum.map(data, &encode_text!(&1.text, tokenizer, opts))
    max_length = Keyword.get(opts, :max_length) || longest_length(encoded_texts)
    pad_token_id = Keyword.get(opts, :pad_token_id, @pad_token_id)

    encoded_texts =
      Enum.map(encoded_texts, fn encoded_text ->
        encoded_text
        |> Enum.take(max_length)
        |> pad_to_length(max_length, pad_token_id)
      end)

    %__MODULE__{
      data: data,
      encoded_texts: encoded_texts,
      max_length: max_length,
      pad_token_id: pad_token_id
    }
  end

  @doc """
  Returns one encoded text and label pair by index.

  ## Input Parameters

    * `dataset` - `%LlmScratch.SpamDataset{}` returned by `new/3`.
    * `index` - zero-based row index.

  ## Output

  Returns `{encoded, label}` where `encoded` is an Nx tensor of shape
  `{dataset.max_length}` and signed 64-bit integer type, and `label` is a scalar
  signed 64-bit integer Nx tensor.
  """
  @spec get(t(), non_neg_integer()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def get(%__MODULE__{} = dataset, index) when is_integer(index) and index >= 0 do
    encoded = Enum.fetch!(dataset.encoded_texts, index)
    label = dataset.data |> Enum.fetch!(index) |> Map.fetch!(:label)

    {
      Nx.tensor(encoded, type: {:s, 64}),
      Nx.tensor(label, type: {:s, 64})
    }
  end

  @doc """
  Returns the number of rows in the dataset.

  ## Input Parameters

    * `dataset` - `%LlmScratch.SpamDataset{}` returned by `new/3`.

  ## Output

  Returns the number of CSV data rows as a non-negative integer.
  """
  @spec length(t()) :: non_neg_integer()
  def length(%__MODULE__{} = dataset) do
    Kernel.length(dataset.data)
  end

  @doc """
  Returns the longest stored encoded sequence length.

  ## Input Parameters

    * `dataset` - `%LlmScratch.SpamDataset{}` returned by `new/3`.

  ## Output

  Returns the maximum token count among `dataset.encoded_texts`. Because
  `new/3` stores padded/truncated sequences, this matches `dataset.max_length`
  for datasets created by this module.
  """
  @spec longest_encoded_length(t()) :: non_neg_integer()
  def longest_encoded_length(%__MODULE__{} = dataset) do
    longest_length(dataset.encoded_texts)
  end

  defp encode_text!(text, tokenizer, _opts) when is_function(tokenizer, 1) do
    tokenizer.(text)
  end

  defp encode_text!(text, tokenizer, opts) when is_binary(tokenizer) do
    allowed_special = Keyword.get(opts, :allowed_special, [@end_of_text])
    {:ok, token_ids} = Tiktoken.encode(tokenizer, text, allowed_special)
    token_ids
  end

  defp longest_length(encoded_texts) do
    encoded_texts
    |> Enum.map(&Kernel.length/1)
    |> Enum.max(fn -> 0 end)
  end

  defp pad_to_length(token_ids, max_length, pad_token_id) do
    token_ids ++ List.duplicate(pad_token_id, max_length - Kernel.length(token_ids))
  end

  defp read_csv!(csv_file) do
    csv_file
    |> File.stream!([], :line)
    |> Stream.map(&String.trim_trailing(&1, "\n"))
    |> Stream.map(&String.trim_trailing(&1, "\r"))
    |> Enum.to_list()
    |> parse_csv_rows!()
  end

  defp parse_csv_rows!([]), do: []

  defp parse_csv_rows!([header | rows]) do
    header_indexes =
      header
      |> parse_csv_line!()
      |> Enum.with_index()
      |> Map.new()

    label_index = Map.fetch!(header_indexes, "Label")
    text_index = Map.fetch!(header_indexes, "Text")

    rows
    |> Enum.reject(&(&1 == ""))
    |> Enum.map(fn row ->
      columns = parse_csv_line!(row)

      %{
        label: columns |> Enum.fetch!(label_index) |> parse_label!(),
        text: Enum.fetch!(columns, text_index)
      }
    end)
  end

  defp parse_label!("ham"), do: 0
  defp parse_label!("spam"), do: 1

  defp parse_label!(label) do
    case Integer.parse(label) do
      {label_id, ""} ->
        label_id

      _ ->
        raise ArgumentError, "expected integer, ham, or spam label, got: #{inspect(label)}"
    end
  end

  defp parse_csv_line!(line) do
    line
    |> String.to_charlist()
    |> parse_csv_chars([], [], false)
    |> Enum.map(&(Enum.reverse(&1) |> List.to_string()))
    |> Enum.reverse()
  end

  defp parse_csv_chars([], field, fields, false), do: [field | fields]

  defp parse_csv_chars([], _field, _fields, true),
    do: raise(ArgumentError, "unterminated CSV quote")

  defp parse_csv_chars([?", ?" | rest], field, fields, true) do
    parse_csv_chars(rest, [?" | field], fields, true)
  end

  defp parse_csv_chars([?" | rest], field, fields, in_quotes) do
    parse_csv_chars(rest, field, fields, not in_quotes)
  end

  defp parse_csv_chars([?, | rest], field, fields, false) do
    parse_csv_chars(rest, [], [field | fields], false)
  end

  defp parse_csv_chars([char | rest], field, fields, in_quotes) do
    parse_csv_chars(rest, [char | field], fields, in_quotes)
  end
end
