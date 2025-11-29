defmodule LlmScratch.GptDatasetV1 do
  def chunk_dataset(txt, model, max_length, stride) do
    {:ok, token_ids} = Tiktoken.encode(model, txt, ["<|endoftext|>"])

    if length(token_ids) < max_length + 1,
      do: raise("Number of tokenized inputs must be at least max_length + 1")

    num_chunks = length(token_ids) - max_length

    chunks =
      0..num_chunks//stride
      |> Enum.reduce([input_chunks: [], target_chunks: []], fn i, acc ->
        input_chunk = Enum.slice(token_ids, i..(i + max_length - 1))
        target_chunk = Enum.slice(token_ids, (i + 1)..(i + max_length))

        [
          input_chunks: [Nx.tensor(input_chunk) | acc[:input_chunks]],
          target_chunks: [Nx.tensor(target_chunk) | acc[:target_chunks]]
        ]
      end)

    [
      input_chunks: Enum.reverse(chunks[:input_chunks]),
      target_chunks: Enum.reverse(chunks[:target_chunks])
    ]
  end

  def create_dataloader_v1(opts) do
    raw_text = Keyword.fetch!(opts, :raw_text)
    batch_size = Keyword.get(opts, :batch_size, 4)
    max_length = Keyword.get(opts, :max_length, 256)
    stride = Keyword.get(opts, :stride, 128)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, true)
    num_workers = Keyword.get(opts, :num_workers, 0)

    # Create dataset
    [input_chunks: input_chunks, target_chunks: target_chunks] =
      LlmScratch.GptDatasetV1.chunk_dataset(raw_text, "code-davinci-002", max_length, stride)

    # Zip input and target chunks together so each item is {input, target}
    dataset = Enum.zip(input_chunks, target_chunks)

    LlmScratch.DataLoader.new(dataset,
      batch_size: batch_size,
      shuffle: shuffle,
      drop_last: drop_last,
      num_workers: num_workers
    )
  end
end
