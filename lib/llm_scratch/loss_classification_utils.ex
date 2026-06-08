defmodule LlmScratch.LossClassificationUtils do
  @moduledoc """
  Classification evaluation helpers.

  These utilities are used after replacing a GPT language-model output head with
  a classification head. For sequence classification, only the logits from the
  final input token are used because that token can attend to all previous
  tokens through causal self-attention.
  """

  @type device :: nil | :default | atom() | tuple()

  @doc """
  Calculates classification accuracy over batches from a data loader.

  This is the Nx/Elixir counterpart of the PyTorch loop:

      model.eval()
      logits = model(input_batch)[:, -1, :]
      predicted_labels = torch.argmax(logits, dim=-1)
      accuracy = correct_predictions / num_examples

  `data_loader` should be a map with `:stream` and `:length`, such as the value
  returned by `LlmScratch.DataLoader.new/2`. Each batch may be either:

    * `{input_batch, target_batch}` with already-stacked tensors
    * a list of `{input_tensor, target_tensor}` examples

  `model` must be a struct whose module exports `forward/2`. The model's
  ordinary forward path is used, which is the project's evaluation path for
  GPT-style models.

  `device` accepts an Nx backend, such as `EXLA.Backend` or
  `{EXLA.Backend, client: :cuda}`. Use `nil` or `:default` to keep tensors on
  their current backend.

  Pass `nil` for `num_batches` to evaluate one full pass through the loader.
  When an empty loader, or `num_batches: 0`, produces no examples, returns
  `:nan`.
  """
  @spec calc_accuracy_loader(map(), struct(), device(), nil | non_neg_integer(), keyword()) ::
          float() | :nan
  def calc_accuracy_loader(data_loader, model, device \\ :default, num_batches \\ nil, opts \\ []) do
    loader_length = Map.get(data_loader, :length, 0)
    num_batches = normalize_num_batches(num_batches, loader_length)

    # Match the Python implementation's "evaluate at most num_batches" behavior
    # by taking only that many batches from the loader's epoch stream.
    {correct_predictions, num_examples} =
      data_loader.stream
      |> Stream.take(num_batches)
      |> Enum.reduce({0, 0}, fn batch, {correct_predictions, num_examples} ->
        {input_batch, target_batch} = stack_batch(batch)

        # Move both tensors together so model computation and label comparison
        # happen on the requested backend, just like `tensor.to(device)`.
        input_batch = maybe_transfer(input_batch, device)
        target_batch = maybe_transfer(target_batch, device)

        # A classification head emits logits for every sequence position.
        logits =
          model
          |> forward_model(input_batch)
          |> select_logits(opts)

        # `argmax(axis: -1)` turns the selected-token class scores into integer
        # labels. Comparing against the target tensor yields one boolean per row.
        predicted_labels = Nx.argmax(logits, axis: -1)
        # Nx.tensor([0, 1, 1, 0]) and
        # Nx.tensor([0, 0, 1, 0]) =
        # Nx.tensor([1, 0, 1, 1], type: {:u, 8})
        batch_correct = predicted_labels |> Nx.equal(target_batch) |> Nx.sum() |> Nx.to_number()
        batch_examples = Nx.axis_size(predicted_labels, 0)

        {correct_predictions + batch_correct, num_examples + batch_examples}
      end)

    if num_examples == 0 do
      :nan
    else
      correct_predictions / num_examples
    end
  end

  defp maybe_transfer(tensor, nil), do: tensor
  defp maybe_transfer(tensor, :default), do: tensor
  defp maybe_transfer(tensor, device), do: Nx.backend_transfer(tensor, device)

  defp normalize_num_batches(nil, loader_length), do: loader_length
  defp normalize_num_batches(num_batches, loader_length), do: min(num_batches, loader_length)

  defp select_logits(logits, opts) do
    case Keyword.get(opts, :target, :last_token) do
      :first_token -> logits[[.., 0, ..]]
      :last_token -> logits[[.., -1, ..]]
    end
  end

  defp stack_batch({input_batch, target_batch}) do
    {input_batch, target_batch}
  end

  defp stack_batch(batch) do
    {inputs, targets} = Enum.unzip(batch)

    {Nx.stack(inputs), Nx.stack(targets)}
  end

  defp forward_model(model, input_batch) do
    module = model.__struct__
    apply(module, :forward, [model, input_batch])
  end
end
