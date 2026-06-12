defmodule LlmScratch.InstructionDataset do
  @moduledoc """
  Pre-tokenized instruction fine-tuning dataset.

  This mirrors the PyTorch `InstructionDataset` example from chapter 7. A
  dataset is built from decoded instruction maps, formats each map as a prompt
  plus response, and pre-tokenizes the full text during construction. Alpaca
  prompt formatting is used by default; Phi-3-style chat formatting can be
  selected with `prompt_style: :phi3`.

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

  @enforce_keys [:data, :encoded_texts, :prompt_style, :mask_out_instructions_in_target]
  defstruct [:data, :encoded_texts, :prompt_style, :mask_out_instructions_in_target]

  @end_of_text "<|endoftext|>"
  @pad_token_id 50_256

  @type instruction_record :: LlmScratch.FineTuneDataLoader.instruction_record()
  @type encoded_example ::
          [integer()] | %{token_ids: [integer()], prompt_length: non_neg_integer()}
  @type tokenizer :: String.t()
  @type prompt_style :: :alpaca | :phi3
  @type t :: %__MODULE__{
          data: [instruction_record()],
          encoded_texts: [encoded_example()],
          prompt_style: prompt_style(),
          mask_out_instructions_in_target: boolean()
        }

  @doc """
  Creates an instruction dataset from decoded instruction records.

  By default, each record is formatted with
  `LlmScratch.FineTuneDataLoader.format_input/1`, then the target response is
  appended as:

      \\n\\n### Response:\\n...

  With `prompt_style: :phi3`, each record is formatted as:

      <|user|>
      ...
      <|end|>
      <|assistant|>
      ...
      <|end|>

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
    * `:prompt_style` - prompt format for full training examples. Supported
      values are `:alpaca` and `:phi3`. Defaults to `:alpaca`.
    * `:mask_out_instructions_in_target` - when `true`, instruction/input
      prompt targets are masked with `-100` so the loss is computed only after
      the prompt. Padding targets are still masked independently by
      `custom_collate/4`. Defaults to `false`.

  ## Output

  Returns a `%LlmScratch.InstructionDataset{}`. The `:data` field contains the
  original records, and `:encoded_texts` contains one pre-tokenized example per
  record. With `mask_out_instructions_in_target: false`, each example is a
  token-id list. With `mask_out_instructions_in_target: true`, each example
  also carries prompt-length metadata used by `custom_collate/4`.

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
    prompt_style = Keyword.get(opts, :prompt_style, :alpaca)
    mask_out_instructions_in_target = Keyword.get(opts, :mask_out_instructions_in_target, false)

    # Pre-tokenize each complete training example once so repeated dataset
    # access does not rebuild the prompt or call the tokenizer again.
    encoded_texts =
      Enum.map(data, fn entry ->
        encode_example(entry, tokenizer, opts, prompt_style, mask_out_instructions_in_target)
      end)

    # Keep the decoded records for inspection while serving encoded examples
    # from `encoded_texts`, which mirrors the PyTorch dataset fields.
    %__MODULE__{
      data: data,
      encoded_texts: encoded_texts,
      prompt_style: prompt_style,
      mask_out_instructions_in_target: mask_out_instructions_in_target
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

  @doc """
  Pads a batch of token-id sequences and creates input/target tensors.

  This mirrors the chapter 7 Python collate function:

      inputs_1 = [0, 1, 2, 3, 4]
      inputs_2 = [5, 6]
      inputs_3 = [7, 8, 9]
      batch = (inputs_1, inputs_2, inputs_3)
      inputs, targets = custom_collate_fn(batch)
      print(inputs)
      print(targets)

  The function finds the longest sequence length in the batch after adding one
  extra pad token, pads every item to that length, then creates next-token
  prediction pairs. Inputs drop the final token from each padded row; targets
  drop the first token. In target rows, all padding tokens after the first one
  are replaced with `ignore_index` so they do not contribute to the loss.

  ## Input Parameters

    * `batch` - tuple or list of token-id lists.
    * `pad_token_id` - token id used for padding. Defaults to GPT-2's
      end-of-text token id, `50256`.
    * `ignore_index` - label value used for ignored target positions. Defaults
      to `-100`.
    * `allowed_max_length` - optional maximum row length after inputs and
      targets are created. Defaults to `nil`, which keeps the full batch
      length.

  ## Output

  Returns `{inputs, targets}` where both values are signed 64-bit Nx tensors
  with shape `{batch_size, max_sequence_length}`.

  ## Examples

      iex> batch = {[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]}
      iex> {inputs, targets} = LlmScratch.InstructionDataset.custom_collate(batch)
      iex> inputs
      #Nx.Tensor<
        s64[3][5]
        [
          [0, 1, 2, 3, 4],
          [5, 6, 50256, 50256, 50256],
          [7, 8, 9, 50256, 50256]
        ]
      >
      iex> targets
      #Nx.Tensor<
        s64[3][5]
        [
          [1, 2, 3, 4, 50256],
          [6, 50256, -100, -100, -100],
          [8, 9, 50256, -100, -100]
        ]
      >
  """
  @spec custom_collate(
          tuple() | [encoded_example()],
          integer(),
          integer(),
          nil | pos_integer()
        ) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def custom_collate(
        batch,
        pad_token_id \\ @pad_token_id,
        ignore_index \\ -100,
        allowed_max_length \\ nil
      )

  def custom_collate(batch, pad_token_id, ignore_index, allowed_max_length)
      when is_tuple(batch) do
    batch
    |> Tuple.to_list()
    |> custom_collate(pad_token_id, ignore_index, allowed_max_length)
  end

  def custom_collate(batch, pad_token_id, ignore_index, allowed_max_length)
      when is_list(batch) do
    examples = Enum.map(batch, &normalize_example/1)

    # Find the longest sequence length after the one extra pad token that the
    # Python collate function appends to every item.
    batch_max_length =
      examples
      |> Enum.map(&(Kernel.length(&1.token_ids) + 1))
      |> Enum.max(fn -> 0 end)

    {inputs, targets} =
      Enum.map(examples, fn %{token_ids: item, prompt_length: prompt_length} ->
        # Elixir data is immutable, so this builds the equivalent of
        # `new_item = item.copy(); new_item += [pad_token_id]`.
        new_item = item ++ [pad_token_id]

        # Pad the copied item to the longest sequence length in the batch.
        padded = pad_to_length(new_item, batch_max_length, pad_token_id)

        # Match `padded[:-1]` for inputs and `padded[1:]` for next-token
        # targets.
        inputs = Enum.slice(padded, 0, batch_max_length - 1)

        targets =
          padded
          |> Enum.slice(1, batch_max_length - 1)
          |> mask_extra_padding_targets(pad_token_id, ignore_index)
          |> mask_prompt_targets(prompt_length, ignore_index)

        # Optionally cap both rows to the model context length.
        {
          maybe_truncate(inputs, allowed_max_length),
          maybe_truncate(targets, allowed_max_length)
        }
      end)
      |> Enum.unzip()

    {
      Nx.tensor(inputs, type: {:s, 64}),
      Nx.tensor(targets, type: {:s, 64})
    }
  end

  defp encode_example(entry, tokenizer, opts, prompt_style, false) do
    entry
    # The model trains on the instruction/input prompt followed by the
    # expected response target, matching the chapter 7 PyTorch dataset.
    |> full_text(prompt_style)
    |> encode_text!(tokenizer, opts)
  end

  defp encode_example(entry, tokenizer, opts, prompt_style, true) do
    token_ids =
      entry
      |> full_text(prompt_style)
      |> encode_text!(tokenizer, opts)

    prompt_length =
      entry
      |> prompt_text(prompt_style)
      |> encode_text!(tokenizer, opts)
      |> Kernel.length()

    %{token_ids: token_ids, prompt_length: prompt_length}
  end

  defp encode_example(
         _entry,
         _tokenizer,
         _opts,
         _prompt_style,
         mask_out_instructions_in_target
       ) do
    raise ArgumentError,
          "expected :mask_out_instructions_in_target to be a boolean, got: #{inspect(mask_out_instructions_in_target)}"
  end

  defp normalize_example(token_ids) when is_list(token_ids) do
    %{token_ids: token_ids, prompt_length: nil}
  end

  defp normalize_example(%{token_ids: token_ids, prompt_length: prompt_length})
       when is_list(token_ids) and is_integer(prompt_length) and prompt_length >= 0 do
    %{token_ids: token_ids, prompt_length: prompt_length}
  end

  defp full_text(%{"output" => output} = entry, :alpaca) when is_binary(output) do
    instruction_plus_input = prompt_text(entry, :alpaca)
    response_text = "\n\n### Response:\n#{output}"

    instruction_plus_input <> response_text
  end

  defp full_text(%{"output" => output} = entry, :phi3) when is_binary(output) do
    entry
    |> LlmScratch.FineTuneDataLoader.format_text(:phi3)
    |> Kernel.<>(output)
    |> Kernel.<>("\n<|end|>")
  end

  defp full_text(_entry, prompt_style) do
    raise ArgumentError, "unsupported prompt style: #{inspect(prompt_style)}"
  end

  defp prompt_text(entry, prompt_style) do
    LlmScratch.FineTuneDataLoader.format_text(entry, prompt_style)
  end

  defp encode_text!(text, tokenizer, opts) when is_binary(tokenizer) do
    allowed_special = Keyword.get(opts, :allowed_special, [@end_of_text])
    {:ok, token_ids} = Tiktoken.encode(tokenizer, text, allowed_special)
    token_ids
  end

  defp pad_to_length(token_ids, max_length, pad_token_id) do
    token_ids ++ List.duplicate(pad_token_id, max_length - Kernel.length(token_ids))
  end

  defp mask_extra_padding_targets(targets, pad_token_id, ignore_index) do
    {masked_targets, _seen_first_pad?} =
      Enum.map_reduce(targets, false, fn
        ^pad_token_id, false ->
          {pad_token_id, true}

        ^pad_token_id, true ->
          {ignore_index, true}

        token_id, seen_first_pad? ->
          {token_id, seen_first_pad?}
      end)

    masked_targets
  end

  defp mask_prompt_targets(targets, nil, _ignore_index), do: targets

  defp mask_prompt_targets(targets, prompt_length, ignore_index) do
    mask_count = max(prompt_length - 1, 0)

    targets
    |> Enum.with_index()
    |> Enum.map(fn
      {_target, index} when index < mask_count -> ignore_index
      {target, _index} -> target
    end)
  end

  defp maybe_truncate(token_ids, nil), do: token_ids
  defp maybe_truncate(token_ids, max_length), do: Enum.take(token_ids, max_length)
end
