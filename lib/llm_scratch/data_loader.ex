defmodule LlmScratch.DataLoader do
  @doc """
  Builds a data loader map from an in-memory dataset.

  Returns a map with:
  - `:stream` - an infinite stream of batches
  - `:batches` - the finite batch list used for one epoch
  - `:length` - number of batches in one pass over the dataset
  - `:batch_size` - configured batch size (default `32`)
  - `:shuffle` - whether batches are shuffled
  - `:drop_last` - whether incomplete batches are dropped
  - `:num_workers` - concurrency used by `iterate/2`
  - `:collate_fn` - function used to transform each list of examples into a
    batch. Defaults to keeping the list unchanged.

  ## Options
  - `:batch_size` - number of samples per batch (default: `32`)
  - `:shuffle` - shuffles dataset once before cycling (default: `true`)
  - `:drop_last` - drops batches smaller than `:batch_size` (default: `false`)
  - `:num_workers` - parallel workers for iteration (default: `0`)
  - `:collate_fn` - one-argument function called for each batch after
    chunking and `:drop_last` filtering. Defaults to identity.
  """
  def new(dataset, opts \\ []) when is_list(dataset) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, false)
    num_workers = Keyword.get(opts, :num_workers, 0)
    collate_fn = Keyword.get(opts, :collate_fn, &identity/1)

    batches =
      dataset
      |> prepare_dataset(shuffle)
      |> Stream.chunk_every(batch_size)
      |> filter_incomplete_batches(drop_last, batch_size)
      |> Stream.map(collate_fn)
      |> Enum.to_list()

    stream = Stream.cycle(batches)

    %{
      stream: stream,
      batches: batches,
      length: length(batches),
      batch_size: batch_size,
      shuffle: shuffle,
      drop_last: drop_last,
      num_workers: num_workers,
      collate_fn: collate_fn
    }
  end

  @doc """
  Iterates over loader batches and applies `fun` to each batch.

  When `num_workers` is `0`, batches are processed sequentially.
  When `num_workers` is greater than `0`, batches are processed concurrently
  using `Task.async_stream/3`.
  """
  def iterate(%{stream: stream, num_workers: 0}, fun) when is_function(fun, 1) do
    Enum.each(stream, fun)
  end

  def iterate(%{stream: stream, num_workers: num_workers}, fun)
      when is_function(fun, 1) and num_workers > 0 do
    stream
    |> Task.async_stream(fun, max_concurrency: num_workers, ordered: false)
    |> Stream.run()
  end

  defp prepare_dataset(dataset, true) do
    Enum.shuffle(dataset)
  end

  defp prepare_dataset(dataset, false) do
    dataset
  end

  defp filter_incomplete_batches(stream, true, batch_size) do
    Stream.filter(stream, &(length(&1) == batch_size))
  end

  defp filter_incomplete_batches(stream, false, _batch_size) do
    stream
  end

  defp identity(batch), do: batch
end
