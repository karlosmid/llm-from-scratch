defimpl Inspect, for: LlmScratch.GPTModel do
  import Inspect.Algebra

  def inspect(model, _opts) do
    model
    |> inspect_model()
    |> string()
  end

  defp inspect_model(model) do
    lines = [
      "GPTModel(",
      "  (tok_emb): #{embedding(model.tok_emb)}",
      "  (pos_emb): #{embedding(model.pos_emb)}",
      "  (drop_emb): #{dropout(model.drop_emb)}",
      "  (trf_blocks): Sequential("
    ]

    block_lines =
      model.trf_blocks
      |> Enum.with_index()
      |> Enum.flat_map(fn {block, index} -> transformer_block(block, index) end)

    Enum.join(
      lines ++
        block_lines ++
        ["  )", "  (final_norm): LayerNorm()", "  (out_head): #{out_head(model.out_head)}", ")"],
      "\n"
    )
  end

  defp transformer_block(block, index) do
    [
      "    (#{index}): TransformerBlock(",
      "      (att): MultiHeadAttention(",
      "        (W_query): #{linear(block.att.w_q)}",
      "        (W_key): #{linear(block.att.w_k)}",
      "        (W_value): #{linear(block.att.w_v)}",
      "        (out_proj): #{linear(block.att.out_proj)}",
      "        (dropout): #{dropout(block.att.dropout)}",
      "      )",
      "      (ff): FeedForward(",
      "        (layers): Sequential(",
      "          (0): #{linear(block.ff.layers.first)}",
      "          (1): GELU()",
      "          (2): #{linear(block.ff.layers.second)}",
      "        )",
      "      )",
      "      (norm1): LayerNorm()",
      "      (norm2): LayerNorm()",
      "      (drop_resid): #{dropout(block.drop_shortcut)}",
      "    )"
    ]
  end

  defp embedding(embedding) do
    {vocab_size, embedding_dim} = Nx.shape(embedding.weight)
    "Embedding(#{vocab_size}, #{embedding_dim})"
  end

  defp linear(%LlmScratch.LinearWithLoRA{} = layer) do
    "LinearWithLoRA(linear=#{linear(layer.linear)}, lora=#{lora(layer.lora)})"
  end

  defp linear(%{kernel: kernel} = layer) do
    {in_features, out_features} = Nx.shape(kernel)

    "Linear(in_features=#{in_features}, out_features=#{out_features}, bias=#{Map.has_key?(layer, :bias)})"
  end

  defp lora(%LlmScratch.LoRALayer{} = layer) do
    "LoRALayer(in_features=#{layer.in_dim}, out_features=#{layer.out_dim}, rank=#{layer.rank}, alpha=#{layer.alpha})"
  end

  defp out_head(%LlmScratch.LinearWithLoRA{} = layer), do: linear(layer)

  defp out_head(%{kernel: kernel} = layer) do
    {in_features, out_features} = Nx.shape(kernel)

    "Linear(in_features=#{in_features}, out_features=#{out_features}, bias=#{Map.has_key?(layer, :bias)})"
  end

  defp dropout(dropout) do
    "Dropout(p=#{dropout * 1.0}, inplace=false)"
  end
end
