defmodule LlmScratch.GPTModel do
  @moduledoc """
  GPT model implemented with Nx tensors and the real transformer block.

  This mirrors the book's PyTorch `GPTModel`:

      tok_emb = nn.Embedding(vocab_size, emb_dim)
      pos_emb = nn.Embedding(context_length, emb_dim)
      drop_emb = nn.Dropout(emb_drop_rate)
      trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(n_layers)])
      final_norm = LayerNorm(emb_dim)
      out_head = nn.Linear(emb_dim, vocab_size, bias=False)

  `LlmScratch.DummyGPTModel` remains unchanged; this module uses
  `LlmScratch.TransformerBlock` for the actual attention and feed-forward stack.
  """

  import Nx.Defn

  alias LlmScratch.{
    DummyLayerNorm,
    EmbeddingNative,
    GPTConfig,
    LinearWithLoRA,
    MultiheadAttention,
    SelfAttentionV2,
    TransformerBlock
  }

  @trainable_fields [:tok_emb, :pos_emb, :trf_blocks, :final_norm, :out_head, :lora]

  defstruct [
    :cfg,
    :tok_emb,
    :pos_emb,
    :drop_emb,
    :trf_blocks,
    :final_norm,
    :out_head,
    trainable: :all
  ]

  @type linear_no_bias :: %{kernel: Nx.Tensor.t()}
  @type linear :: linear_no_bias() | %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          cfg: GPTConfig.t(),
          tok_emb: EmbeddingNative.t(),
          pos_emb: EmbeddingNative.t(),
          drop_emb: float(),
          trf_blocks: [TransformerBlock.t()],
          final_norm: DummyLayerNorm.t(),
          out_head: linear(),
          trainable: :all | [atom() | {:trf_block, non_neg_integer()}]
        }

  @spec new(GPTConfig.t(), keyword()) :: t()
  @doc """
  Creates a GPT model from a `%LlmScratch.GPTConfig{}`.

  ## Options

    * `:seed` - deterministic seed used for embeddings, transformer blocks,
      and the output projection.
    * `:norm_eps` - epsilon used by the final layer norm and transformer block
      layer norms.
  """
  def new(%GPTConfig{} = cfg, opts \\ []) when is_list(opts) do
    seed = normalize_seed(Keyword.get(opts, :seed))
    norm_eps = Keyword.get(opts, :norm_eps, 1.0e-5)

    %__MODULE__{
      cfg: cfg,
      tok_emb: EmbeddingNative.new(cfg.vocab_size, cfg.emb_dim, seed: seed),
      pos_emb: EmbeddingNative.new(cfg.context_length, cfg.emb_dim, seed: seed + 1),
      drop_emb: GPTConfig.embedding_dropout(cfg),
      trf_blocks: transformer_blocks(cfg, seed + 2, norm_eps),
      final_norm: DummyLayerNorm.new(cfg.emb_dim, eps: norm_eps),
      out_head: init_out_head(cfg, seed + 2 + cfg.n_layers * 8),
      trainable: :all
    }
  end

  @doc """
  Freezes all model parameters for fine-tuning workflows.

  Nx tensors do not have a mutable `requires_grad` flag like PyTorch tensors.
  Instead, this marks the model with an empty trainable layer list. The training
  optimizer respects that metadata and leaves all existing parameters
  unchanged.

  Returns an updated `%LlmScratch.GPTModel{}`.
  """
  @spec freeze(t()) :: t()
  def freeze(%__MODULE__{} = model), do: %{model | trainable: []}

  @doc """
  Restores all model parameters to trainable.

  Returns an updated `%LlmScratch.GPTModel{}` with the default `:all`
  trainable setting.
  """
  @spec unfreeze(t()) :: t()
  def unfreeze(%__MODULE__{} = model), do: %{model | trainable: :all}

  @doc """
  Marks selected top-level model layers as trainable.

  ## Parameters

    * `model` - `%LlmScratch.GPTModel{}` to update.
    * `fields` - list of trainable top-level layer names. Accepted fields are
      `:tok_emb`, `:pos_emb`, `:trf_blocks`, `:final_norm`, `:out_head`, and
      `:lora`.

  This is useful after calling `freeze/1`, for example to train only a newly
  added classification head.

  Returns an updated `%LlmScratch.GPTModel{}`.
  """
  @spec set_trainable(t(), [atom()]) :: t()
  def set_trainable(%__MODULE__{} = model, fields) when is_list(fields) do
    unknown_fields =
      Enum.reject(fields, fn
        field when field in @trainable_fields ->
          true

        {:trf_block, index} when is_integer(index) and index >= 0 ->
          index < length(model.trf_blocks)

        _field ->
          false
      end)

    if unknown_fields != [] do
      raise ArgumentError,
            "unknown trainable GPTModel fields #{inspect(unknown_fields)}; expected one or more of #{inspect(@trainable_fields)} or {:trf_block, index}"
    end

    %{model | trainable: Enum.uniq(fields)}
  end

  @doc """
  Returns `true` when no model parameters are trainable.
  """
  @spec frozen?(t()) :: boolean()
  def frozen?(%__MODULE__{trainable: []}), do: true
  def frozen?(%__MODULE__{}), do: false

  @doc """
  Replaces the output head with a dense classification or projection head.

  ## Parameters

    * `model` - `%LlmScratch.GPTModel{}` to update.
    * `out_features` - number of output logits produced for each token.
    * `opts` - optional keyword list.

  ## Options

    * `:seed` - deterministic initialization seed. Defaults to `123`.
    * `:bias` - whether to include a bias vector. Defaults to `true`, matching
      `torch.nn.Linear/2`.

  ## Output

  Returns an updated model whose `:out_head` has kernel shape
  `{model.cfg.emb_dim, out_features}` and, when enabled, bias shape
  `{out_features}`.
  """
  @spec replace_out_head(t(), pos_integer(), keyword()) :: t()
  def replace_out_head(%__MODULE__{} = model, out_features, opts \\ [])
      when is_integer(out_features) and out_features > 0 and is_list(opts) do
    seed = Keyword.get(opts, :seed, 123)
    bias = Keyword.get(opts, :bias, true)

    %{
      model
      | out_head:
          SelfAttentionV2.init_dense_weights(
            model.cfg.emb_dim,
            out_features,
            seed,
            bias,
            "out_head"
          )
    }
  end

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Runs an evaluation forward pass over token ids shaped `{batch_size, seq_len}`.

  The returned logits have shape `{batch_size, seq_len, vocab_size}`.
  Dropout is disabled, matching PyTorch `model.eval()`.
  """
  def forward(%__MODULE__{} = model, %Nx.Tensor{} = in_idx) do
    {_batch_size, seq_len} = validate_input_shape!(in_idx)
    validate_context_length!(seq_len, model.cfg.context_length)

    tok_embeds = EmbeddingNative.forward(model.tok_emb, in_idx)

    pos_embeds =
      model.pos_emb
      |> EmbeddingNative.forward(positional_indices(seq_len))

    x =
      tok_embeds
      |> Nx.add(pos_embeds)

    x =
      Enum.reduce(model.trf_blocks, x, fn block, acc ->
        TransformerBlock.forward(block, acc)
      end)

    x = DummyLayerNorm.forward(model.final_norm, x)

    linear(x, model.out_head)
  end

  @spec eval(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Alias for `forward/2`, named after PyTorch evaluation mode.
  """
  def eval(model, in_idx), do: forward(model, in_idx)

  @doc """
  Defn-compatible training forward pass.

  Dropout is enabled, matching PyTorch `model.train()`. The caller must pass an
  explicit `Nx.Random` key; the updated key is returned with the logits.
  """
  defn train(model, in_idx, key) do
    seq_len = Nx.axis_size(in_idx, 1)

    tok_embeds = EmbeddingNative.forward_defn(model.tok_emb, in_idx)

    pos_embeds =
      model.pos_emb
      |> EmbeddingNative.forward_defn(Nx.iota({seq_len}, type: {:s, 64}))

    x = Nx.add(tok_embeds, pos_embeds)
    {x, key} = MultiheadAttention.dropout_defn(x, model.drop_emb, key)
    {x, key} = transformer_blocks_train(model.trf_blocks, x, key)
    x = DummyLayerNorm.forward_defn(model.final_norm, x)

    {linear_defn(x, model.out_head), key}
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Alias for `forward/2`.
  """
  def call(model, in_idx), do: forward(model, in_idx)

  @spec total_parameters(t()) :: non_neg_integer()
  @doc """
  Counts all parameters in the GPT model.

  The count includes token embeddings, positional embeddings, every
  transformer block, the final layer norm, and the output projection.
  """
  def total_parameters(%__MODULE__{} = model) do
    model.tok_emb.weight
    |> tensor_parameters()
    |> Kernel.+(tensor_parameters(model.pos_emb.weight))
    |> Kernel.+(Enum.reduce(model.trf_blocks, 0, &(&2 + transformer_block_parameters(&1))))
    |> Kernel.+(layer_norm_parameters(model.final_norm))
    |> Kernel.+(dense_parameters(model.out_head, Map.has_key?(model.out_head, :bias)))
  end

  @spec trainable_parameters(t()) :: non_neg_integer()
  @doc """
  Counts parameters marked as trainable by the model metadata.

  Nx tensors do not expose PyTorch's mutable `requires_grad` flag, so
  `freeze/1`, `unfreeze/1`, and `set_trainable/2` store that information in
  `model.trainable`. This function mirrors the book's
  `sum(p.numel() for p in model.parameters() if p.requires_grad)` by counting
  only the fields currently selected for optimization.
  """
  def trainable_parameters(%__MODULE__{trainable: :all} = model), do: total_parameters(model)

  def trainable_parameters(%__MODULE__{trainable: []}), do: 0

  def trainable_parameters(%__MODULE__{trainable: fields} = model) when is_list(fields) do
    fields
    |> Enum.uniq()
    |> Enum.reduce(0, fn field, total -> total + trainable_field_parameters(model, field) end)
  end

  @spec transformer_block_parameters(map()) :: non_neg_integer()
  @doc """
  Counts parameters in one transformer block.
  """
  def transformer_block_parameters(block) do
    attention_parameters(block.att) +
      feed_forward_parameters(block.ff) +
      layer_norm_parameters(block.norm1) +
      layer_norm_parameters(block.norm2)
  end

  @spec attention_parameters(map()) :: non_neg_integer()
  @doc """
  Counts parameters in a multi-head attention module.
  """
  def attention_parameters(attention) do
    attention.w_q
    |> dense_parameters(attention.qkv_bias)
    |> Kernel.+(dense_parameters(attention.w_k, attention.qkv_bias))
    |> Kernel.+(dense_parameters(attention.w_v, attention.qkv_bias))
    |> Kernel.+(dense_parameters(attention.out_proj, true))
  end

  @spec feed_forward_parameters(map()) :: non_neg_integer()
  @doc """
  Counts parameters in a GPT feed-forward module.
  """
  def feed_forward_parameters(feed_forward) do
    dense_parameters(feed_forward.layers.first, true) +
      dense_parameters(feed_forward.layers.second, true)
  end

  @spec layer_norm_parameters(map()) :: non_neg_integer()
  @doc """
  Counts parameters in a layer norm module.
  """
  def layer_norm_parameters(layer_norm) do
    tensor_parameters(layer_norm.scale) + tensor_parameters(layer_norm.shift)
  end

  @spec tensor_parameters(Nx.Tensor.t()) :: non_neg_integer()
  @doc """
  Counts scalar values in a tensor.
  """
  def tensor_parameters(tensor), do: Nx.size(tensor)

  defp trainable_field_parameters(model, :tok_emb), do: tensor_parameters(model.tok_emb.weight)

  defp trainable_field_parameters(model, :pos_emb), do: tensor_parameters(model.pos_emb.weight)

  defp trainable_field_parameters(model, :trf_blocks) do
    Enum.reduce(model.trf_blocks, 0, &(&2 + transformer_block_parameters(&1)))
  end

  defp trainable_field_parameters(model, :final_norm), do: layer_norm_parameters(model.final_norm)

  defp trainable_field_parameters(model, :out_head) do
    dense_parameters(model.out_head, Map.has_key?(model.out_head, :bias))
  end

  defp trainable_field_parameters(model, :lora), do: lora_parameters(model)

  defp trainable_field_parameters(model, {:trf_block, index}) do
    model.trf_blocks
    |> Enum.at(index)
    |> transformer_block_parameters()
  end

  defp lora_parameters(%__MODULE__{} = model) do
    Enum.reduce(model.trf_blocks, 0, &(&2 + lora_parameters(&1))) +
      lora_parameters(model.out_head)
  end

  defp lora_parameters(%TransformerBlock{} = block) do
    lora_parameters(block.att) + lora_parameters(block.ff)
  end

  defp lora_parameters(%MultiheadAttention{} = attention) do
    lora_parameters(attention.w_q) +
      lora_parameters(attention.w_k) +
      lora_parameters(attention.w_v) +
      lora_parameters(attention.out_proj)
  end

  defp lora_parameters(%LlmScratch.FeedForward{} = feed_forward) do
    lora_parameters(feed_forward.layers.first) + lora_parameters(feed_forward.layers.second)
  end

  defp lora_parameters(%LinearWithLoRA{} = layer) do
    tensor_parameters(layer.lora.a) + tensor_parameters(layer.lora.b)
  end

  defp lora_parameters(_other), do: 0

  defp transformer_blocks(cfg, _seed, _norm_eps) when cfg.n_layers == 0, do: []

  defp transformer_blocks(cfg, seed, norm_eps) do
    for layer_idx <- 0..(cfg.n_layers - 1) do
      TransformerBlock.new(cfg, seed: seed + layer_idx * 8, norm_eps: norm_eps)
    end
  end

  defp init_out_head(cfg, seed) do
    %{kernel: kernel} =
      SelfAttentionV2.init_dense_weights(
        cfg.emb_dim,
        cfg.vocab_size,
        seed,
        false,
        "out_head"
      )

    %{kernel: kernel}
  end

  defp linear(x, %{kernel: kernel, bias: bias}), do: Nx.dot(x, [-1], kernel, [0]) |> Nx.add(bias)
  defp linear(x, %{kernel: kernel}), do: Nx.dot(x, [-1], kernel, [0])
  defp linear(x, %LinearWithLoRA{} = layer), do: LinearWithLoRA.forward(layer, x)

  deftransformp linear_defn(x, layer) do
    if match?(%LinearWithLoRA{}, layer) do
      LinearWithLoRA.forward_defn(layer, x)
    else
      y = Nx.dot(x, [-1], layer.kernel, [0])

      if Map.has_key?(layer, :bias) do
        Nx.add(y, layer.bias)
      else
        y
      end
    end
  end

  deftransformp transformer_blocks_train(blocks, x, key) do
    Enum.reduce(blocks, {x, key}, fn block, {x, key} ->
      TransformerBlock.train(block, x, key)
    end)
  end

  defp dense_parameters(%{kernel: kernel, bias: bias}, true),
    do: tensor_parameters(kernel) + tensor_parameters(bias)

  defp dense_parameters(%{kernel: kernel}, false), do: tensor_parameters(kernel)

  defp dense_parameters(%LinearWithLoRA{} = layer, _bias) do
    dense_parameters(layer.linear, Map.has_key?(layer.linear, :bias)) +
      tensor_parameters(layer.lora.a) +
      tensor_parameters(layer.lora.b)
  end

  defp positional_indices(seq_len), do: Nx.iota({seq_len}, type: {:s, 64})

  defp validate_input_shape!(in_idx) do
    case Nx.shape(in_idx) do
      {batch_size, seq_len} ->
        {batch_size, seq_len}

      shape ->
        raise ArgumentError, "expected input shape {batch_size, seq_len}, got: #{inspect(shape)}"
    end
  end

  defp validate_context_length!(seq_len, context_length) when seq_len <= context_length,
    do: :ok

  defp validate_context_length!(seq_len, context_length) do
    raise ArgumentError, "seq_len (#{seq_len}) exceeds context_length (#{context_length})"
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
  end
end
