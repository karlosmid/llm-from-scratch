defmodule LlmScratch.InstructionDataset do
  @moduledoc """
  Pre-tokenized instruction fine-tuning dataset.

  This mirrors the PyTorch `InstructionDataset` example from chapter 7. A
  dataset is built from decoded instruction maps, formats each map as an
  Alpaca-style prompt plus response, and pre-tokenizes the full text during
  construction.

  ## Examples

      iex> data = [
      ...>   %{
      ...>     "instruction" => "Name the capital of France.",
      ...>     "input" => "",
      ...>     "output" => "Paris."
      ...>   }
      ...> ]
      iex> dataset = LlmScratch.InstructionDataset.new(data, "code-davinci-002")
      iex> LlmScratch.InstructionDataset.length(dataset)
      1
      iex> LlmScratch.InstructionDataset.get(dataset, 0)
      [21106, 318, 281, 12064, 326, 8477, 257, 4876, 13, 19430, 257,
       2882, 326, 20431, 32543, 262, 2581, 13, 198, 198, 21017, 46486,
       25, 198, 5376, 262, 3139, 286, 4881, 13, 198, 198, 21017, 18261,
       25, 198, 40313, 13]
  """

  @enforce_keys [:data, :encoded_texts]
  defstruct [:data, :encoded_texts]

  @end_of_text "<|endoftext|>"

  @type instruction_record :: LlmScratch.FineTuneDataLoader.instruction_record()
  @type tokenizer :: String.t()
  @type t :: %__MODULE__{
          data: [instruction_record()],
          encoded_texts: [[integer()]]
        }

  @doc """
  Creates an instruction dataset from decoded instruction records.

  Each record is formatted with `LlmScratch.FineTuneDataLoader.format_input/1`,
  then the target response is appended as:

      \\n\\n### Response:\\n...

  The resulting full text is encoded immediately and stored in
  `dataset.encoded_texts`.

  ## Input Parameters

    * `data` - list of decoded instruction maps. Each map must contain string
      keys `"instruction"`, `"input"`, and `"output"`.
    * `tokenizer` - Tiktoken model name accepted by `Tiktoken.encode/3`.
    * `opts` - optional keyword list controlling tokenization.

  ## Options

    * `:allowed_special` - special tokens allowed when `tokenizer` is a
      Tiktoken model name. Defaults to `["<|endoftext|>"]`.

  ## Output

  Returns a `%LlmScratch.InstructionDataset{}`. The `:data` field contains the
  original records, and `:encoded_texts` contains one pre-tokenized token-id
  list per record.

  ## Examples

      iex> data = [
      ...>   %{
      ...>     "instruction" => "Classify the sentiment.",
      ...>     "input" => "I loved it.",
      ...>     "output" => "Positive."
      ...>   }
      ...> ]
      iex> dataset = LlmScratch.InstructionDataset.new(data, "code-davinci-002")
      iex> dataset.data
      [
        %{
          "input" => "I loved it.",
          "instruction" => "Classify the sentiment.",
          "output" => "Positive."
        }
      ]
      iex> dataset.encoded_texts
      [
        [21106, 318, 281, 12064, 326, 8477, 257, 4876, 13, 19430, 257,
         2882, 326, 20431, 32543, 262, 2581, 13, 198, 198, 21017, 46486,
         25, 198, 9487, 1958, 262, 15598, 13, 198, 198, 21017, 23412, 25,
         198, 40, 6151, 340, 13, 198, 198, 21017, 18261, 25, 198, 21604,
         1800, 13]
      ]
  """
  @spec new([instruction_record()], tokenizer(), keyword()) :: t()
  def new(data, tokenizer, opts \\ []) when is_list(data) do
    # Pre-tokenize each complete training example once so repeated dataset
    # access does not rebuild the prompt or call the tokenizer again.
    encoded_texts =
      Enum.map(data, fn entry ->
        entry
        # The model trains on the instruction/input prompt followed by the
        # expected response target, matching the chapter 7 PyTorch dataset.
        |> full_text()
        |> encode_text!(tokenizer, opts)
      end)

    # Keep the decoded records for inspection while serving encoded examples
    # from `encoded_texts`, which mirrors the PyTorch dataset fields.
    %__MODULE__{
      data: data,
      encoded_texts: encoded_texts
    }
  end

  @doc """
  Returns one encoded instruction example by index.

  ## Input Parameters

    * `dataset` - `%LlmScratch.InstructionDataset{}` returned by `new/3`.
    * `index` - zero-based row index.

  ## Output

  Returns the pre-tokenized token-id list for the requested record.
  """
  @spec get(t(), non_neg_integer()) :: [integer()]
  def get(%__MODULE__{} = dataset, index) when is_integer(index) and index >= 0 do
    Enum.fetch!(dataset.encoded_texts, index)
  end

  @doc """
  Returns the number of records in the dataset.

  ## Input Parameters

    * `dataset` - `%LlmScratch.InstructionDataset{}` returned by `new/3`.

  ## Output

  Returns the number of instruction records as a non-negative integer.
  """
  @spec length(t()) :: non_neg_integer()
  def length(%__MODULE__{} = dataset) do
    Kernel.length(dataset.data)
  end

  defp full_text(%{"output" => output} = entry) when is_binary(output) do
    instruction_plus_input = LlmScratch.FineTuneDataLoader.format_input(entry)
    response_text = "\n\n### Response:\n#{output}"

    instruction_plus_input <> response_text
  end

  defp encode_text!(text, tokenizer, opts) when is_binary(tokenizer) do
    allowed_special = Keyword.get(opts, :allowed_special, [@end_of_text])
    {:ok, token_ids} = Tiktoken.encode(tokenizer, text, allowed_special)
    token_ids
  end
end
