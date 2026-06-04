defmodule LlmScratch.LossUtils do
  @moduledoc """
  Loss helpers for GPT-style logits and target token tensors.
  """

  import Nx.Defn

  @doc """
  Returns the predicted probabilities for the expected target token ids.

  `probas` should be shaped `{batch_size, seq_len, vocab_size}` and `targets`
  should be shaped `{batch_size, seq_len}`. The returned tensor is flattened to
  `{batch_size * seq_len}`.

  ## Examples

      iex> probas = Nx.tensor([[[0.5, 0.25, 0.25], [0.125, 0.375, 0.5]]])
      iex> targets = Nx.tensor([[0, 2]])
      iex> LlmScratch.LossUtils.target_token_probas(probas, targets) |> Nx.to_flat_list()
      [0.5, 0.5]
  """
  @spec target_token_probas(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def target_token_probas(%Nx.Tensor{} = probas, %Nx.Tensor{} = targets) do
    vocab_size = elem(Nx.shape(probas), 2)

    # {2, 3, 50257} => {6, 50257}
    probas
    |> Nx.reshape({:auto, vocab_size})
    |> Nx.take_along_axis(
      # {2, 3} => {6, 1}
      Nx.reshape(targets, {:auto, 1}),
      axis: 1
    )
    |> Nx.squeeze(axes: [1])
  end

  @doc """
  Calculates mean cross entropy loss from logits and target token ids.
  step1: logits
  step2: probabilities
  step3: target probabilities
  step4: logarithmic probabilities
  step5: average logarithmic probabilities
  step6: negative average log probabilities

  `logits` should be shaped `{batch_size, seq_len, vocab_size}` and `targets`
  should be shaped `{batch_size, seq_len}`.

  ## Examples

      iex> logits = Nx.log(Nx.tensor([[[0.5, 0.25, 0.25], [0.125, 0.375, 0.5]]]))
      iex> targets = Nx.tensor([[0, 2]])
      iex> loss = LlmScratch.LossUtils.cross_entropy_loss(logits, targets)
      iex> Float.round(Nx.to_number(loss), 6)
      0.693147
  """
  @spec cross_entropy_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def cross_entropy_loss(%Nx.Tensor{} = logits, %Nx.Tensor{} = targets) do
    cross_entropy_loss_defn(logits, targets)
  end

  @doc """
  Defn-compatible mean cross entropy loss from logits and target token ids.
  """
  defn cross_entropy_loss_defn(logits, targets) do
    if Nx.rank(logits) == 3 do
      # Language-model flow:
      # logits shape is {batch_size, seq_len, vocab_size}
      # targets shape is {batch_size, seq_len}
      vocab_size = Nx.axis_size(logits, 2)

      # step1: logits
      logits
      # step2: probabilities
      |> Axon.Activations.softmax(axis: -1)
      # step3: target probabilities
      |> Nx.reshape({:auto, vocab_size})
      |> Nx.take_along_axis(Nx.reshape(targets, {:auto, 1}), axis: 1)
      |> Nx.squeeze(axes: [1])
      # step4: logarithmic probabilities
      |> Nx.log()
      # step6: negative average log probabilities
      |> Nx.negate()
      # step5: average logarithmic probabilities
      |> Nx.mean()
    else
      # Classification flow:
      # logits shape is {batch_size, num_classes}
      # targets shape is {batch_size}
      # step1: logits
      logits
      # step2: probabilities
      |> Axon.Activations.softmax(axis: -1)
      # step3: target probabilities
      |> Nx.take_along_axis(Nx.reshape(targets, {:auto, 1}), axis: 1)
      |> Nx.squeeze(axes: [1])
      # step4: logarithmic probabilities
      |> Nx.log()
      # step6: negative average log probabilities
      |> Nx.negate()
      # step5: average logarithmic probabilities
      |> Nx.mean()
    end
  end

  @doc """
  Calculates cross entropy loss for an input/target batch and model.

  This is the Nx counterpart to:

      logits = model(input_batch)
      loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
      )

  `input_batch` and `target_batch` should be shaped `{batch_size, seq_len}`.
  The model must be a struct whose module exports `forward/2` or `forward/3`.

  The optional `device` argument accepts an Nx backend, such as `EXLA.Backend`
  or `{EXLA.Backend, client: :cuda}`. Use `nil` or `:default` to leave tensors
  on their current backend.

  Options:

    * `:target` - use `:all_tokens` or omit it for language-model loss over all
      token positions. Use `:last_token` for sequence classification, where the
      model output is sliced to `model(input_batch)[:, -1, :]` before computing
      cross entropy.
  """
  @spec calc_loss_batch(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          struct(),
          nil | :default | atom() | tuple(),
          keyword()
        ) ::
          Nx.Tensor.t()
  def calc_loss_batch(input_batch, target_batch, model, device \\ :default, opts \\ []) do
    input_batch = maybe_transfer(input_batch, device)
    target_batch = maybe_transfer(target_batch, device)

    model
    |> forward_model(input_batch)
    |> select_loss_logits(opts)
    |> cross_entropy_loss(target_batch)
  end

  @doc """
  Calculates average loss over batches from a data loader.

  `data_loader` should be a `%{stream: stream, length: length}` map, such as the
  value returned by `LlmScratch.DataLoader.new/2`. Each batch may be either:

    * `{input_batch, target_batch}` with already-stacked tensors
    * a list of `{input_tensor, target_tensor}` examples

  Pass `nil` for `num_batches` to evaluate one full pass through the loader.
  Returns `:nan` when the loader has no batches.

  Accepts the same options as `calc_loss_batch/5`; use `target: :last_token` for
  classification loss.
  """
  @spec calc_loss_loader(
          map(),
          struct(),
          nil | :default | atom() | tuple(),
          nil | non_neg_integer(),
          keyword()
        ) ::
          float() | :nan
  def calc_loss_loader(data_loader, model, device \\ :default, num_batches \\ nil, opts \\ []) do
    loader_length = Map.get(data_loader, :length, 0)

    cond do
      loader_length == 0 ->
        :nan

      true ->
        num_batches = normalize_num_batches(num_batches, loader_length)

        data_loader.stream
        |> Stream.take(num_batches)
        |> Enum.reduce(0.0, fn batch, total_loss ->
          {input_batch, target_batch} = stack_batch(batch)
          loss = calc_loss_batch(input_batch, target_batch, model, device, opts)

          total_loss + Nx.to_number(loss)
        end)
        |> Kernel./(num_batches)
    end
  end

  defp maybe_transfer(tensor, nil), do: tensor
  defp maybe_transfer(tensor, :default), do: tensor
  defp maybe_transfer(tensor, device), do: Nx.backend_transfer(tensor, device)

  defp normalize_num_batches(nil, loader_length), do: loader_length
  defp normalize_num_batches(num_batches, loader_length), do: min(num_batches, loader_length)

  defp select_loss_logits(logits, opts) do
    if Keyword.get(opts, :target) == :last_token do
      # Classification uses only the final token because it has causal attention
      # over the full input message.
      logits[[.., -1, ..]]
    else
      logits
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
