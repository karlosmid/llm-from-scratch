defmodule LlmScratch.DataLoader do
  def new(dataset, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, false)
    num_workers = Keyword.get(opts, :num_workers, 0)

    stream =
      dataset
      |> prepare_dataset(shuffle)
      |> Stream.chunk_every(batch_size)
      |> filter_incomplete_batches(drop_last, batch_size)

    %{
      stream: stream,
      batch_size: batch_size,
      drop_last: drop_last,
      num_workers: num_workers
    }
  end

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
    dataset
    |> Enum.to_list()
    |> Enum.shuffle()
    |> Stream.cycle()
  end

  defp prepare_dataset(dataset, false) do
    Stream.cycle(dataset)
  end

  defp filter_incomplete_batches(stream, true, batch_size) do
    Stream.filter(stream, &(length(&1) == batch_size))
  end

  defp filter_incomplete_batches(stream, false, _batch_size) do
    stream
  end
end
