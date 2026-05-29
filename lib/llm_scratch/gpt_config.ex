defmodule LlmScratch.GPTConfig do
  @moduledoc """
  GPT model configuration.

  This struct mirrors the config keys used by the Python examples.

  ## Fields

    * `:vocab_size` - number of tokens in the tokenizer vocabulary. This
      determines the number of rows in the token embedding table and the final
      logits dimension produced by the output head. For GPT-2 this is commonly
      `50_257`.

    * `:context_length` - maximum number of token positions the model can
      process in one forward pass. This determines the size of the positional
      embedding table. Inputs longer than this cannot be represented by the
      configured positional embeddings.

    * `:emb_dim` - width of each token and positional embedding vector. This is
      also the model's hidden size: tensors flowing through transformer blocks
      have shape `{batch_size, seq_len, emb_dim}`.

    * `:n_heads` - number of attention heads used by real transformer blocks.
      The dummy GPT model stores this value for config compatibility, but its
      placeholder transformer blocks do not use it yet.

    * `:n_layers` - number of transformer blocks in the model. In
      `LlmScratch.DummyGPTModel`, this controls how many dummy identity blocks
      are created.

    * `:drop_rate` - backward-compatible default dropout probability. It is
      used when a more specific dropout field is not set.

    * `:emb_drop_rate` - dropout probability used after adding token and
      positional embeddings.

    * `:shortcut_drop_rate` - dropout probability used on transformer-block
      shortcut branches after attention and feed-forward outputs.

    * `:attn_drop_rate` - dropout probability used on attention weights inside
      multi-head attention.

    * `:qkv_bias` - whether query, key, and value projections include bias
      terms in real attention layers. The dummy GPT model keeps this field for
      compatibility with the full GPT config shape, but does not use it because
      its transformer blocks are placeholders.
  """

  @enforce_keys [
    :vocab_size,
    :context_length,
    :emb_dim,
    :n_heads,
    :n_layers,
    :drop_rate,
    :qkv_bias
  ]

  defstruct [
    :vocab_size,
    :context_length,
    :emb_dim,
    :n_heads,
    :n_layers,
    :drop_rate,
    :emb_drop_rate,
    :shortcut_drop_rate,
    :attn_drop_rate,
    :qkv_bias
  ]

  @type t :: %__MODULE__{
          vocab_size: pos_integer(),
          context_length: pos_integer(),
          emb_dim: pos_integer(),
          n_heads: pos_integer(),
          n_layers: non_neg_integer(),
          drop_rate: float(),
          emb_drop_rate: float() | nil,
          shortcut_drop_rate: float() | nil,
          attn_drop_rate: float() | nil,
          qkv_bias: boolean()
        }

  @openai_gpt2_model_configs %{
    "gpt2-small (124M)" => %{emb_dim: 768, n_layers: 12, n_heads: 12},
    "gpt2-medium (355M)" => %{emb_dim: 1024, n_layers: 24, n_heads: 16},
    "gpt2-large (774M)" => %{emb_dim: 1280, n_layers: 36, n_heads: 20},
    "gpt2-xl (1558M)" => %{emb_dim: 1600, n_layers: 48, n_heads: 25},
    "124M" => %{emb_dim: 768, n_layers: 12, n_heads: 12},
    "355M" => %{emb_dim: 1024, n_layers: 24, n_heads: 16},
    "774M" => %{emb_dim: 1280, n_layers: 36, n_heads: 20},
    "1558M" => %{emb_dim: 1600, n_layers: 48, n_heads: 25}
  }

  @doc """
  Returns a GPT-2 config matching OpenAI's released checkpoint sizes.

  This mirrors the book's update pattern:

      NEW_CONFIG = GPT_CONFIG_124M.copy()
      NEW_CONFIG.update(model_configs[model_name])
      NEW_CONFIG.update({"context_length": 1024})
      NEW_CONFIG.update({"qkv_bias": True})

  Accepted names are the descriptive labels from the book and the short model
  sizes used by the downloader:

    * `"gpt2-small (124M)"` or `"124M"`
    * `"gpt2-medium (355M)"` or `"355M"`
    * `"gpt2-large (774M)"` or `"774M"`
    * `"gpt2-xl (1558M)"` or `"1558M"`
  """
  @spec openai_gpt2(String.t()) :: t()
  def openai_gpt2(model_name \\ "gpt2-small (124M)") when is_binary(model_name) do
    case Map.fetch(@openai_gpt2_model_configs, model_name) do
      {:ok, model_config} ->
        %__MODULE__{
          vocab_size: 50_257,
          context_length: 1024,
          emb_dim: model_config.emb_dim,
          n_heads: model_config.n_heads,
          n_layers: model_config.n_layers,
          drop_rate: 0.0,
          qkv_bias: true
        }

      :error ->
        raise ArgumentError,
              "unknown OpenAI GPT-2 model #{inspect(model_name)}; expected one of #{inspect(Map.keys(@openai_gpt2_model_configs))}"
    end
  end

  @spec embedding_dropout(t()) :: float()
  @doc """
  Returns the embedding dropout rate, falling back to `drop_rate` when unset.
  """
  def embedding_dropout(%__MODULE__{} = cfg), do: dropout_value!(cfg.emb_drop_rate, cfg.drop_rate)

  @spec shortcut_dropout(t()) :: float()
  @doc """
  Returns the shortcut dropout rate, falling back to `drop_rate` when unset.
  """
  def shortcut_dropout(%__MODULE__{} = cfg),
    do: dropout_value!(cfg.shortcut_drop_rate, cfg.drop_rate)

  @spec attention_dropout(t()) :: float()
  @doc """
  Returns the attention dropout rate, falling back to `drop_rate` when unset.
  """
  def attention_dropout(%__MODULE__{} = cfg),
    do: dropout_value!(cfg.attn_drop_rate, cfg.drop_rate)

  @spec estimated_parameter_count(t()) :: non_neg_integer()
  @doc """
  Estimates GPT model parameters directly from the configuration.

  This mirrors the architecture used by `LlmScratch.GPTModel` without
  allocating tensors:

    * token embeddings: `vocab_size * emb_dim`
    * positional embeddings: `context_length * emb_dim`
    * untied output projection: `emb_dim * vocab_size`
    * each transformer block: attention projections, feed-forward projections,
      and two layer norms
    * final layer norm: scale and shift, each shaped `{emb_dim}`

  Use `LlmScratch.GPTModel.total_parameters/1` when you already have a
  concrete model and want to count its actual tensors.
  """
  def estimated_parameter_count(%__MODULE__{} = cfg) do
    token_embedding_params = cfg.vocab_size * cfg.emb_dim
    positional_embedding_params = cfg.context_length * cfg.emb_dim
    output_projection_params = cfg.emb_dim * cfg.vocab_size
    block_params = cfg.n_layers * estimated_transformer_block_parameter_count(cfg)
    final_norm_params = estimated_layer_norm_parameter_count(cfg.emb_dim)

    token_embedding_params + positional_embedding_params + output_projection_params + block_params +
      final_norm_params
  end

  defp estimated_transformer_block_parameter_count(cfg) do
    estimated_attention_parameter_count(cfg) +
      estimated_feed_forward_parameter_count(cfg.emb_dim) +
      2 * estimated_layer_norm_parameter_count(cfg.emb_dim)
  end

  defp estimated_attention_parameter_count(cfg) do
    qkv_kernel_params = 3 * cfg.emb_dim * cfg.emb_dim
    qkv_bias_params = if cfg.qkv_bias, do: 3 * cfg.emb_dim, else: 0
    output_projection_params = cfg.emb_dim * cfg.emb_dim + cfg.emb_dim

    qkv_kernel_params + qkv_bias_params + output_projection_params
  end

  defp estimated_feed_forward_parameter_count(emb_dim) do
    expanded_dim = 4 * emb_dim
    first_projection_params = emb_dim * expanded_dim + expanded_dim
    second_projection_params = expanded_dim * emb_dim + emb_dim

    first_projection_params + second_projection_params
  end

  defp estimated_layer_norm_parameter_count(emb_dim), do: 2 * emb_dim

  defp dropout_value!(nil, fallback), do: validate_dropout!(fallback)
  defp dropout_value!(value, _fallback), do: validate_dropout!(value)

  defp validate_dropout!(value) when is_number(value) and value >= 0 and value < 1,
    do: value * 1.0

  defp validate_dropout!(value) do
    raise ArgumentError, "expected dropout value to be a number in [0, 1), got: #{inspect(value)}"
  end
end
