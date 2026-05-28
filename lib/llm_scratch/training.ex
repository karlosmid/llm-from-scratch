defimpl Nx.Container, for: LlmScratch.EmbeddingNative do
  def traverse(embedding, acc, fun) do
    {weight, acc} = fun.(embedding.weight, acc)
    {%{embedding | weight: weight}, acc}
  end

  def reduce(embedding, acc, fun), do: fun.(embedding.weight, acc)

  def serialize(embedding) do
    metadata = Map.take(embedding, [:vocab_size, :embedding_dim, :seed])
    {__MODULE__, [weight: embedding.weight], metadata}
  end

  def deserialize([weight: weight], metadata) do
    struct!(LlmScratch.EmbeddingNative, Map.put(metadata, :weight, weight))
  end
end

defimpl Nx.Container, for: LlmScratch.DummyLayerNorm do
  def traverse(layer_norm, acc, fun) do
    {scale, acc} = fun.(layer_norm.scale, acc)
    {shift, acc} = fun.(layer_norm.shift, acc)
    {%{layer_norm | scale: scale, shift: shift}, acc}
  end

  def reduce(layer_norm, acc, fun) do
    acc
    |> then(&fun.(layer_norm.scale, &1))
    |> then(&fun.(layer_norm.shift, &1))
  end

  def serialize(layer_norm) do
    metadata = Map.take(layer_norm, [:emb_dim, :eps])
    {__MODULE__, [scale: layer_norm.scale, shift: layer_norm.shift], metadata}
  end

  def deserialize([scale: scale, shift: shift], metadata) do
    struct!(LlmScratch.DummyLayerNorm, Map.merge(metadata, %{scale: scale, shift: shift}))
  end
end

defimpl Nx.Container, for: LlmScratch.GELU do
  def traverse(gelu, acc, _fun), do: {gelu, acc}
  def reduce(_gelu, acc, _fun), do: acc
  def serialize(_gelu), do: {__MODULE__, [], :ok}
  def deserialize([], :ok), do: LlmScratch.GELU.new()
end

defimpl Nx.Container, for: LlmScratch.FeedForward do
  def traverse(feed_forward, acc, fun) do
    {layers, acc} = fun.(feed_forward.layers, acc)
    {%{feed_forward | layers: layers}, acc}
  end

  def reduce(feed_forward, acc, fun), do: fun.(feed_forward.layers, acc)

  def serialize(feed_forward) do
    {__MODULE__, [layers: feed_forward.layers], %{emb_dim: feed_forward.emb_dim}}
  end

  def deserialize([layers: layers], metadata) do
    struct!(LlmScratch.FeedForward, Map.put(metadata, :layers, layers))
  end
end

defimpl Nx.Container, for: LlmScratch.MultiheadAttention do
  @container_fields [:w_q, :w_k, :w_v, :out_proj, :mask]
  @metadata_fields [
    :d_in,
    :d_out,
    :context_length,
    :dropout,
    :num_heads,
    :head_dim,
    :qkv_bias,
    :seed
  ]

  def traverse(attention, acc, fun) do
    Enum.reduce(@container_fields, {attention, acc}, fn field, {attention, acc} ->
      {value, acc} = fun.(Map.fetch!(attention, field), acc)
      {Map.put(attention, field, value), acc}
    end)
  end

  def reduce(attention, acc, fun) do
    Enum.reduce(@container_fields, acc, fn field, acc ->
      fun.(Map.fetch!(attention, field), acc)
    end)
  end

  def serialize(attention) do
    pairs = Enum.map(@container_fields, &{&1, Map.fetch!(attention, &1)})
    {__MODULE__, pairs, Map.take(attention, @metadata_fields)}
  end

  def deserialize(pairs, metadata) do
    struct!(LlmScratch.MultiheadAttention, Map.merge(metadata, Map.new(pairs)))
  end
end

defimpl Nx.Container, for: LlmScratch.TransformerBlock do
  @container_fields [:att, :ff, :norm1, :norm2]
  @metadata_fields [:drop_shortcut, :cfg]

  def traverse(block, acc, fun) do
    Enum.reduce(@container_fields, {block, acc}, fn field, {block, acc} ->
      {value, acc} = fun.(Map.fetch!(block, field), acc)
      {Map.put(block, field, value), acc}
    end)
  end

  def reduce(block, acc, fun) do
    Enum.reduce(@container_fields, acc, fn field, acc ->
      fun.(Map.fetch!(block, field), acc)
    end)
  end

  def serialize(block) do
    pairs = Enum.map(@container_fields, &{&1, Map.fetch!(block, &1)})
    {__MODULE__, pairs, Map.take(block, @metadata_fields)}
  end

  def deserialize(pairs, metadata) do
    struct!(LlmScratch.TransformerBlock, Map.merge(metadata, Map.new(pairs)))
  end
end

defimpl Nx.Container, for: LlmScratch.GPTModel do
  @container_fields [:tok_emb, :pos_emb, :trf_blocks, :final_norm, :out_head]
  @metadata_fields [:cfg, :drop_emb]

  def traverse(model, acc, fun) do
    {tok_emb, acc} = fun.(model.tok_emb, acc)
    {pos_emb, acc} = fun.(model.pos_emb, acc)
    {trf_blocks, acc} = Enum.map_reduce(model.trf_blocks, acc, fun)
    {final_norm, acc} = fun.(model.final_norm, acc)
    {out_head, acc} = fun.(model.out_head, acc)

    {%{
       model
       | tok_emb: tok_emb,
         pos_emb: pos_emb,
         trf_blocks: trf_blocks,
         final_norm: final_norm,
         out_head: out_head
     }, acc}
  end

  def reduce(model, acc, fun) do
    acc
    |> then(&fun.(model.tok_emb, &1))
    |> then(&fun.(model.pos_emb, &1))
    |> then(fn acc ->
      Enum.reduce(model.trf_blocks, acc, fn block, acc -> fun.(block, acc) end)
    end)
    |> then(&fun.(model.final_norm, &1))
    |> then(&fun.(model.out_head, &1))
  end

  def serialize(model) do
    pairs =
      @container_fields
      |> Enum.map(fn field -> {field, Map.fetch!(model, field)} end)
      |> Keyword.update!(:trf_blocks, &List.to_tuple/1)

    {__MODULE__, pairs, Map.take(model, @metadata_fields)}
  end

  def deserialize(pairs, metadata) do
    pairs =
      pairs
      |> Keyword.update!(:trf_blocks, &Tuple.to_list/1)
      |> Map.new()

    struct!(LlmScratch.GPTModel, Map.merge(metadata, pairs))
  end
end

defmodule LlmScratch.Training do
  @moduledoc """
  Simple GPT training loop helpers mirroring the book's `train_model_simple`.
  """

  import Nx.Defn

  alias LlmScratch.{GPTModel, LossUtils, TextGeneration, TextUtils}

  defmodule AdamW do
    @moduledoc """
    Minimal AdamW optimizer state.

    In the training loop, AdamW is responsible for turning the gradients from
    `loss_and_grad/4` into a new model. Each batch computes how the loss changes
    with respect to the model's trainable tensors; AdamW decides how large each
    parameter update should be, applies regularization through weight decay, and
    carries optimizer state forward to the next batch.

    AdamW keeps two moving averages for each trainable tensor:

      * `m` tracks the exponentially decayed average gradient.
      * `v` tracks the exponentially decayed average squared gradient.

    During each optimization step, the averages are bias-corrected with the
    current `step`, normalized with `eps`, combined with decoupled weight
    decay, and subtracted from the parameter using `learning_rate`.

    ## Fields

      * `:learning_rate` - positive scalar that controls the size of each
        parameter update.
      * `:weight_decay` - decoupled decay factor applied directly to
        parameters, separate from the gradient moments.
      * `:beta1` - exponential decay rate for `m`, the first-moment gradient
        estimate.
      * `:beta2` - exponential decay rate for `v`, the second-moment squared
        gradient estimate.
      * `:eps` - small constant added to the denominator for numerical
        stability.
      * `:step` - number of optimizer updates already applied; used for bias
        correction.
      * `:m` - first-moment tensors, lazily initialized to zeros with the same
        shapes as the trainable parameters.
      * `:v` - second-moment tensors, lazily initialized to zeros with the same
        shapes as the trainable parameters.
    """

    defstruct learning_rate: nil,
              weight_decay: 0.0,
              beta1: 0.9,
              beta2: 0.999,
              eps: 1.0e-8,
              step: 0,
              m: nil,
              v: nil
  end

  @type optimizer :: %AdamW{} | (struct(), struct() -> struct())

  @spec adamw(number(), keyword()) :: AdamW.t()
  @doc """
  Creates a minimal AdamW optimizer.

  This mirrors the optimizer used by the Python chapter example. Moment state is
  initialized lazily on the first optimization step because it must match the
  model's trainable parameter structure.

  AdamW adapts each parameter update using the history of recent gradients:

      m = beta1 * m + (1 - beta1) * gradient
      v = beta2 * v + (1 - beta2) * gradient ** 2
      m_hat = m / (1 - beta1 ** step)
      v_hat = v / (1 - beta2 ** step)
      update = m_hat / (sqrt(v_hat) + eps) + weight_decay * parameter
      parameter = parameter - learning_rate * update

  The `m` and `v` tensors start as zeros with the same shapes as the model's
  trainable tensors. `weight_decay` is decoupled from the gradient moments,
  which is the distinguishing behavior of AdamW compared to classic Adam with
  L2 regularization folded into the gradient.

  ## Options

    * `:weight_decay` - decoupled weight decay. Defaults to `0.0`.
    * `:beta1` - first-moment decay. Defaults to `0.9`.
    * `:beta2` - second-moment decay. Defaults to `0.999`.
    * `:eps` - numerical stability constant. Defaults to `1.0e-8`.
  """
  def adamw(learning_rate, opts \\ []) when is_number(learning_rate) and learning_rate > 0 do
    %AdamW{
      learning_rate: learning_rate * 1.0,
      weight_decay: Keyword.get(opts, :weight_decay, 0.0) * 1.0,
      beta1: Keyword.get(opts, :beta1, 0.9) * 1.0,
      beta2: Keyword.get(opts, :beta2, 0.999) * 1.0,
      eps: Keyword.get(opts, :eps, 1.0e-8) * 1.0
    }
  end

  @spec train_model_simple(
          struct(),
          map(),
          map(),
          optimizer(),
          nil | :default | atom() | tuple(),
          non_neg_integer(),
          pos_integer(),
          pos_integer(),
          String.t(),
          String.t(),
          keyword()
        ) ::
          {struct(), [float()], [float()], [non_neg_integer()]}
          | {struct(), optimizer(), [float()], [float()], [non_neg_integer()]}
  @doc """
  Trains a GPT-style model with a simple batch loop.

  This mirrors the book's Python `train_model_simple` example. Since Elixir
  data is immutable, the updated model is returned instead of being mutated in
  place. The training path calls `GPTModel.train/3`, which enables dropout and
  threads an explicit `Nx.Random` key through the model.

  ## Parameters

    * `model` - GPT-style model struct whose trainable tensors can be traversed
      by `Nx.Container`.
    * `train_loader` - data loader map with `:stream` and `:length`.
    * `val_loader` - validation data loader map with `:stream` and `:length`.
    * `optimizer` - either `Training.adamw/2` output or a two-argument function
      `(model, gradients -> updated_model)`. The built-in AdamW optimizer
      returns updated optimizer state after each batch, so its moment estimates
      are carried through the full training loop.
    * `device` - Nx backend target. Use `:default` or `nil` to keep tensors on
      their current backend.
    * `num_epochs` - number of full passes over `train_loader`.
    * `eval_freq` - evaluate every `eval_freq` global training steps.
    * `eval_iter` - maximum number of batches from each loader during
      evaluation.
    * `start_context` - prompt used by `generate_and_print_sample/4`.
    * `tokenizer` - tokenizer name passed to `Tiktoken`.
    * `opts` - optional settings:
      * `:return_optimizer` - when `true`, include the final optimizer state in
        the return tuple for training checkpoints.
      * `:generate_samples` - when `false`, skip text sample generation during
        evaluation. Defaults to `true`.

  ## Returns

    * `{model, train_losses, val_losses, track_tokens_seen}` by default.
    * `{model, optimizer, train_losses, val_losses, track_tokens_seen}` when
      `return_optimizer: true`.
  """
  def train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
        opts \\ []
      )
      when is_struct(model) and is_map(train_loader) and is_map(val_loader) and
             is_integer(num_epochs) and num_epochs >= 0 and is_integer(eval_freq) and
             eval_freq > 0 and is_integer(eval_iter) and eval_iter > 0 and is_list(opts) do
    # Start dropout RNG from a fresh positive runtime integer, then move the
    # key to the same backend requested for the model and batches.
    key =
      System.unique_integer([:positive])
      |> Nx.Random.key()
      |> maybe_transfer_tensor(device)

    # Training state is carried through the epoch/batch reducers because Elixir
    # data is immutable. It keeps the current model parameters, AdamW optimizer
    # memory, dropout RNG key, progress counters, and evaluation metrics that
    # are accumulated while learning.
    state = %{
      # Current model parameters; replaced after each optimizer step.
      model: maybe_transfer_model(model, device),
      # Periodic evaluation losses, stored newest-first during training.
      train_losses: [],
      val_losses: [],
      # Token counts captured at the same evaluation points as the losses.
      track_tokens_seen: [],
      # Running count of all tokens consumed by training batches.
      tokens_seen: 0,
      # Batch counter; starts at -1 so the first batch becomes step 0.
      global_step: -1,
      # Dropout RNG key; GPTModel.train/3 returns the next key for the next batch.
      key: key,
      # Optimizer state; AdamW carries step count and moment tensors across batches.
      optimizer: optimizer
    }

    # Main training loop by number of epocs
    state =
      Enum.reduce(1..num_epochs//1, state, fn epoch, state ->
        train_epoch(
          state,
          train_loader,
          val_loader,
          device,
          epoch,
          eval_freq,
          eval_iter,
          start_context,
          tokenizer,
          opts
        )
      end)

    train_losses = Enum.reverse(state.train_losses)
    val_losses = Enum.reverse(state.val_losses)
    tokens_seen = Enum.reverse(state.track_tokens_seen)

    if Keyword.get(opts, :return_optimizer, false) do
      {state.model, state.optimizer, train_losses, val_losses, tokens_seen}
    else
      {state.model, train_losses, val_losses, tokens_seen}
    end
  end

  @doc """
  Calculates training and validation losses for a fixed number of batches.

  Evaluation uses the model's dropout-free forward path, matching PyTorch
  `model.eval()`.

  ## Parameters

    * `model` - GPT-style model used for inference.
    * `train_loader` - training data loader map with `:stream` and `:length`.
    * `val_loader` - validation data loader map with `:stream` and `:length`.
    * `device` - Nx backend target passed through to
      `LossUtils.calc_loss_loader/4`.
    * `eval_iter` - maximum number of batches evaluated from each loader.

  ## Returns

    * `{train_loss, val_loss}` as floats.
  """
  def evaluate_model(model, train_loader, val_loader, device, eval_iter)
      when is_struct(model) and is_integer(eval_iter) and eval_iter > 0 do
    {
      LossUtils.calc_loss_loader(train_loader, model, device, eval_iter),
      LossUtils.calc_loss_loader(val_loader, model, device, eval_iter)
    }
  end

  @doc """
  Generates and prints a text sample from `start_context`.

  The prompt is encoded with `tokenizer`, moved to `device` when requested, and
  decoded back to text after greedy generation. The generated text is printed
  and also returned.

  ## Parameters

    * `model` - GPT-style model used for greedy text generation.
    * `tokenizer` - tokenizer name passed to `Tiktoken`.
    * `device` - Nx backend target for the encoded prompt.
    * `start_context` - prompt text to continue.

  ## Returns

    * generated text as a string.
  """
  def generate_and_print_sample(model, tokenizer, device, start_context)
      when is_struct(model) and is_binary(tokenizer) and is_binary(start_context) do
    idx =
      start_context
      |> TextUtils.text_to_token_ids(tokenizer)
      |> maybe_transfer_tensor(device)

    context_size = model.cfg.context_length

    generated =
      model
      |> TextGeneration.generate_text_simple(idx, 50, context_size)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    text = TextUtils.token_ids_to_text(generated, tokenizer)
    IO.puts(text)
    text
  end

  @doc """
  Computes training loss and gradients for one input/target batch.

  This is the `loss.backward()` counterpart from the Python example. It calls
  `GPTModel.train/3`, so dropout is enabled, and returns the next RNG key for
  the following batch.

  ## Parameters

    * `model` - GPT model whose trainable tensors implement `Nx.Container`.
    * `input_batch` - token id tensor shaped `{batch_size, seq_len}`.
    * `target_batch` - target token id tensor shaped `{batch_size, seq_len}`.
    * `key` - `Nx.Random` key used for dropout.

  ## Returns

    * `{loss, gradients, key}` where `loss` is a scalar tensor, `gradients`
      matches the model structure, and `key` is the advanced RNG key.
  """
  defn loss_and_grad(model, input_batch, target_batch, key) do
    {{loss, key}, {gradients, _input_gradients, _target_gradients, _key_gradients}} =
      value_and_grad(
        {model, input_batch, target_batch, key},
        fn {model, input_batch, target_batch, key} ->
          {logits, key} = GPTModel.train(model, input_batch, key)
          {LossUtils.cross_entropy_loss_defn(logits, target_batch), key}
        end,
        &elem(&1, 0)
      )

    {loss, gradients, key}
  end

  defp train_epoch(
         state,
         train_loader,
         val_loader,
         device,
         epoch,
         eval_freq,
         eval_iter,
         start_context,
         tokenizer,
         opts
       ) do
    train_loader
    # shufle batches
    |> epoch_batches()
    |> Enum.reduce(state, fn batch, state ->
      # separate batch inputs and targets
      {input_batch, target_batch} = stack_batch(batch, device)
      # calculate loss and gradinets
      {_loss, gradients, key} = loss_and_grad(state.model, input_batch, target_batch, state.key)
      # avoid overfitting and penalized larger weights
      {model, optimizer} = optimizer_step(state.optimizer, state.model, gradients)
      # tokens that we have processed so far
      tokens_seen = state.tokens_seen + Nx.size(input_batch)
      global_step = state.global_step + 1

      state = %{
        state
        | model: model,
          tokens_seen: tokens_seen,
          global_step: global_step,
          key: key,
          optimizer: optimizer
      }

      # evaluate model on evaluation frequency
      if rem(global_step, eval_freq) == 0 do
        {train_loss, val_loss} =
          evaluate_model(model, train_loader, val_loader, device, eval_iter)

        IO.puts(
          "Ep #{epoch} (Step #{pad_step(global_step)}): " <>
            "Train loss #{format_loss(train_loss)}, Val loss #{format_loss(val_loss)}"
        )

        if Keyword.get(opts, :generate_samples, true) do
          # generate tokens based on current model to see what model actually predicts
          # we do not want gibberish text!
          generate_and_print_sample(model, tokenizer, device, start_context)
        end

        %{
          state
          | train_losses: [train_loss | state.train_losses],
            val_losses: [val_loss | state.val_losses],
            track_tokens_seen: [tokens_seen | state.track_tokens_seen]
        }
      else
        state
      end
    end)
  end

  defp epoch_batches(%{batches: batches, shuffle: true}) do
    Enum.shuffle(batches)
  end

  defp epoch_batches(%{batches: batches}) do
    batches
  end

  defp epoch_batches(%{stream: stream, length: length}) do
    Enum.take(stream, length)
  end

  defp stack_batch({%Nx.Tensor{} = input_batch, %Nx.Tensor{} = target_batch}, device) do
    {maybe_transfer_tensor(input_batch, device), maybe_transfer_tensor(target_batch, device)}
  end

  defp stack_batch(batch, device) when is_list(batch) do
    {inputs, targets} = Enum.unzip(batch)

    {
      inputs |> Nx.stack() |> maybe_transfer_tensor(device),
      targets |> Nx.stack() |> maybe_transfer_tensor(device)
    }
  end

  # here we update model weights, central part of training algorithm
  # we have one gradient for each model parameter (weight)x
  defp optimizer_step(%AdamW{} = optimizer, model, gradients) do
    # extract model weights
    parameter_tensors = trainable_tensors(model)
    # extract gradient weights
    gradient_tensors = trainable_tensors(gradients)
    # adamw m and v tensors
    {m_tensors, v_tensors} = adamw_moments(optimizer, gradient_tensors)
    step = optimizer.step + 1

    # update parameters using AdamW optimization
    {updated_parameters, {updated_m, updated_v}} =
      Enum.zip([parameter_tensors, gradient_tensors, m_tensors, v_tensors])
      |> Enum.map_reduce({[], []}, fn {parameter, gradient, m, v}, {updated_m, updated_v} ->
        m = Nx.add(Nx.multiply(optimizer.beta1, m), Nx.multiply(1.0 - optimizer.beta1, gradient))

        v =
          Nx.add(
            Nx.multiply(optimizer.beta2, v),
            Nx.multiply(1.0 - optimizer.beta2, Nx.pow(gradient, 2))
          )

        m_hat = Nx.divide(m, 1.0 - :math.pow(optimizer.beta1, step))
        v_hat = Nx.divide(v, 1.0 - :math.pow(optimizer.beta2, step))

        update =
          m_hat
          |> Nx.divide(Nx.add(Nx.sqrt(v_hat), optimizer.eps))
          |> Nx.add(Nx.multiply(optimizer.weight_decay, parameter))

        parameter = Nx.subtract(parameter, Nx.multiply(optimizer.learning_rate, update))

        {parameter, {[m | updated_m], [v | updated_v]}}
      end)

    # update model with new weight values
    {model, []} = put_trainable_tensors(model, updated_parameters)

    # the next batch can continue from the accumulated AdamW moment history.
    optimizer = %{
      optimizer
      | step: step,
        m: Enum.reverse(updated_m),
        v: Enum.reverse(updated_v)
    }

    {model, optimizer}
  end

  defp optimizer_step(optimizer, model, gradients) when is_function(optimizer, 2) do
    {optimizer.(model, gradients), optimizer}
  end

  defp adamw_moments(%AdamW{m: nil, v: nil}, gradient_tensors) do
    moments = Enum.map(gradient_tensors, &Nx.broadcast(0.0, Nx.shape(&1)))
    {moments, moments}
  end

  defp adamw_moments(%AdamW{m: m, v: v}, _gradient_tensors), do: {m, v}

  defp trainable_tensors(%LlmScratch.EmbeddingNative{weight: weight}), do: [weight]

  defp trainable_tensors(%LlmScratch.DummyLayerNorm{scale: scale, shift: shift}) do
    [scale, shift]
  end

  defp trainable_tensors(%LlmScratch.GELU{}), do: []

  defp trainable_tensors(%LlmScratch.FeedForward{layers: layers}) do
    dense_tensors(layers.first, true) ++ dense_tensors(layers.second, true)
  end

  defp trainable_tensors(%LlmScratch.MultiheadAttention{} = attention) do
    dense_tensors(attention.w_q, attention.qkv_bias) ++
      dense_tensors(attention.w_k, attention.qkv_bias) ++
      dense_tensors(attention.w_v, attention.qkv_bias) ++
      dense_tensors(attention.out_proj, true)
  end

  defp trainable_tensors(%LlmScratch.TransformerBlock{} = block) do
    trainable_tensors(block.att) ++
      trainable_tensors(block.ff) ++
      trainable_tensors(block.norm1) ++
      trainable_tensors(block.norm2)
  end

  defp trainable_tensors(%GPTModel{} = model) do
    trainable_tensors(model.tok_emb) ++
      trainable_tensors(model.pos_emb) ++
      Enum.flat_map(model.trf_blocks, &trainable_tensors/1) ++
      trainable_tensors(model.final_norm) ++
      dense_tensors(model.out_head, false)
  end

  defp dense_tensors(%{kernel: kernel, bias: bias}, true), do: [kernel, bias]
  defp dense_tensors(%{kernel: kernel}, false), do: [kernel]

  defp put_trainable_tensors(%LlmScratch.EmbeddingNative{} = embedding, [weight | rest]) do
    {%{embedding | weight: weight}, rest}
  end

  defp put_trainable_tensors(%LlmScratch.DummyLayerNorm{} = layer_norm, [
         scale,
         shift | rest
       ]) do
    {%{layer_norm | scale: scale, shift: shift}, rest}
  end

  defp put_trainable_tensors(%LlmScratch.GELU{} = gelu, rest), do: {gelu, rest}

  defp put_trainable_tensors(%LlmScratch.FeedForward{} = feed_forward, tensors) do
    {first, tensors} = put_dense_tensors(feed_forward.layers.first, true, tensors)
    {second, tensors} = put_dense_tensors(feed_forward.layers.second, true, tensors)

    {
      %{feed_forward | layers: %{feed_forward.layers | first: first, second: second}},
      tensors
    }
  end

  defp put_trainable_tensors(%LlmScratch.MultiheadAttention{} = attention, tensors) do
    {w_q, tensors} = put_dense_tensors(attention.w_q, attention.qkv_bias, tensors)
    {w_k, tensors} = put_dense_tensors(attention.w_k, attention.qkv_bias, tensors)
    {w_v, tensors} = put_dense_tensors(attention.w_v, attention.qkv_bias, tensors)
    {out_proj, tensors} = put_dense_tensors(attention.out_proj, true, tensors)

    {
      %{attention | w_q: w_q, w_k: w_k, w_v: w_v, out_proj: out_proj},
      tensors
    }
  end

  defp put_trainable_tensors(%LlmScratch.TransformerBlock{} = block, tensors) do
    {att, tensors} = put_trainable_tensors(block.att, tensors)
    {ff, tensors} = put_trainable_tensors(block.ff, tensors)
    {norm1, tensors} = put_trainable_tensors(block.norm1, tensors)
    {norm2, tensors} = put_trainable_tensors(block.norm2, tensors)

    {%{block | att: att, ff: ff, norm1: norm1, norm2: norm2}, tensors}
  end

  defp put_trainable_tensors(%GPTModel{} = model, tensors) do
    {tok_emb, tensors} = put_trainable_tensors(model.tok_emb, tensors)
    {pos_emb, tensors} = put_trainable_tensors(model.pos_emb, tensors)
    {trf_blocks, tensors} = Enum.map_reduce(model.trf_blocks, tensors, &put_trainable_tensors/2)
    {final_norm, tensors} = put_trainable_tensors(model.final_norm, tensors)
    {out_head, tensors} = put_dense_tensors(model.out_head, false, tensors)

    {
      %{
        model
        | tok_emb: tok_emb,
          pos_emb: pos_emb,
          trf_blocks: trf_blocks,
          final_norm: final_norm,
          out_head: out_head
      },
      tensors
    }
  end

  defp put_dense_tensors(%{kernel: _kernel, bias: _bias} = dense, true, [kernel, bias | rest]) do
    {%{dense | kernel: kernel, bias: bias}, rest}
  end

  defp put_dense_tensors(%{kernel: _kernel} = dense, false, [kernel | rest]) do
    {%{dense | kernel: kernel}, rest}
  end

  defp maybe_transfer_model(model, device) when device in [nil, :default], do: model
  defp maybe_transfer_model(model, device), do: Nx.backend_transfer(model, device)

  defp maybe_transfer_tensor(tensor, device) when device in [nil, :default], do: tensor
  defp maybe_transfer_tensor(tensor, device), do: Nx.backend_transfer(tensor, device)

  defp format_loss(:nan), do: "nan"
  defp format_loss(loss), do: :erlang.float_to_binary(loss * 1.0, decimals: 3)

  defp pad_step(step) do
    step
    |> Integer.to_string()
    |> String.pad_leading(6, "0")
  end
end

defimpl Nx.Container, for: LlmScratch.Training.AdamW do
  @metadata_fields [:learning_rate, :weight_decay, :beta1, :beta2, :eps, :step]

  def traverse(%{m: nil, v: nil} = optimizer, acc, _fun), do: {optimizer, acc}

  def traverse(optimizer, acc, fun) do
    {m, acc} = Enum.map_reduce(optimizer.m || [], acc, fun)
    {v, acc} = Enum.map_reduce(optimizer.v || [], acc, fun)

    {%{optimizer | m: empty_to_nil(m), v: empty_to_nil(v)}, acc}
  end

  def reduce(%{m: nil, v: nil}, acc, _fun), do: acc

  def reduce(optimizer, acc, fun) do
    acc
    |> reduce_tensors(optimizer.m, fun)
    |> reduce_tensors(optimizer.v, fun)
  end

  def serialize(%{m: nil, v: nil} = optimizer) do
    {__MODULE__, [], Map.take(optimizer, @metadata_fields)}
  end

  def serialize(optimizer) do
    pairs =
      []
      |> maybe_put_tuple(:m, optimizer.m)
      |> maybe_put_tuple(:v, optimizer.v)

    {__MODULE__, pairs, Map.take(optimizer, @metadata_fields)}
  end

  def deserialize(pairs, metadata) do
    optimizer_state =
      pairs
      |> Map.new()
      |> Map.update(:m, nil, &Tuple.to_list/1)
      |> Map.update(:v, nil, &Tuple.to_list/1)

    struct!(LlmScratch.Training.AdamW, Map.merge(metadata, optimizer_state))
  end

  defp reduce_tensors(acc, nil, _fun), do: acc

  defp reduce_tensors(acc, tensors, fun),
    do: Enum.reduce(tensors, acc, fn tensor, acc -> fun.(tensor, acc) end)

  defp maybe_put_tuple(pairs, _field, nil), do: pairs
  defp maybe_put_tuple(pairs, field, tensors), do: [{field, List.to_tuple(tensors)} | pairs]

  defp empty_to_nil([]), do: nil
  defp empty_to_nil(tensors), do: tensors
end
