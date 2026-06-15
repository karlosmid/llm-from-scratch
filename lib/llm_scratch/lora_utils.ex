defmodule LlmScratch.LoRAUtils do
  @moduledoc """
  Utilities for attaching LoRA adapters to existing models.

  The main helper, `replace_linear_with_lora/4`, mirrors the recursive PyTorch
  pattern:

      for name, module in model.named_children():
          if isinstance(module, torch.nn.Linear):
              setattr(model, name, LinearWithLoRA(module, rank, alpha))
          else:
              replace_linear_with_lora(module, rank, alpha)

  The Elixir model is built from structs and maps rather than `nn.Module`
  objects, so this module performs the equivalent replacement for known dense
  projection locations in the GPT model:

    * attention projections: `w_q`, `w_k`, `w_v`, `out_proj`
    * feed-forward projections: `layers.first`, `layers.second`
    * output head only when `include_output_head: true`

  Embeddings, layer norms, dropout configuration, and masks are left unchanged.
  When a frozen `%LlmScratch.GPTModel{}` is adapted, its trainable metadata is
  updated to `[:lora]`, matching PyTorch's behavior where newly created LoRA
  parameters default to `requires_grad = True`.
  """

  alias LlmScratch.{
    FeedForward,
    GPTModel,
    LinearWithLoRA,
    MultiheadAttention,
    TransformerBlock
  }

  @doc """
  Recursively replaces dense projections with `%LinearWithLoRA{}` wrappers.

  `rank` and `alpha` are forwarded to `LinearWithLoRA.new/4`.

  ## Options

    * `:seed` - initial seed for LoRA `A` initialization. Defaults to `123`.
      Each wrapped layer receives the next integer seed.
    * `:include_output_head` - whether to wrap `GPTModel.out_head`. Defaults to
      `false`, so only transformer-block attention and feed-forward projections
      are adapted.

  Passing an already wrapped `%LinearWithLoRA{}` returns it unchanged.
  Passing a dense map directly wraps that single dense layer.
  """
  @spec replace_linear_with_lora(term(), pos_integer(), number(), keyword()) :: term()
  def replace_linear_with_lora(model, rank, alpha, opts \\ [])
      when is_integer(rank) and rank > 0 and is_number(alpha) and is_list(opts) do
    seed = Keyword.get(opts, :seed, 123)
    include_output_head = Keyword.get(opts, :include_output_head, false)

    {model, _next_seed} =
      replace(model, rank, alpha, %{
        seed: seed,
        include_output_head: include_output_head
      })

    model
  end

  defp replace(%GPTModel{} = model, rank, alpha, state) do
    {trf_blocks, state} = replace_list(model.trf_blocks, rank, alpha, state)

    {out_head, state} =
      if state.include_output_head do
        replace(model.out_head, rank, alpha, state)
      else
        {model.out_head, state}
      end

    {%{
       model
       | trf_blocks: trf_blocks,
         out_head: out_head,
         trainable: trainable_after_lora(model)
     }, state}
  end

  defp replace(%TransformerBlock{} = block, rank, alpha, state) do
    {att, state} = replace(block.att, rank, alpha, state)
    {ff, state} = replace(block.ff, rank, alpha, state)

    {%{block | att: att, ff: ff}, state}
  end

  defp replace(%MultiheadAttention{} = attention, rank, alpha, state) do
    {w_q, state} = wrap_dense(attention.w_q, rank, alpha, state)
    {w_k, state} = wrap_dense(attention.w_k, rank, alpha, state)
    {w_v, state} = wrap_dense(attention.w_v, rank, alpha, state)
    {out_proj, state} = wrap_dense(attention.out_proj, rank, alpha, state)

    {%{attention | w_q: w_q, w_k: w_k, w_v: w_v, out_proj: out_proj}, state}
  end

  defp replace(%FeedForward{} = feed_forward, rank, alpha, state) do
    {first, state} = wrap_dense(feed_forward.layers.first, rank, alpha, state)
    {second, state} = wrap_dense(feed_forward.layers.second, rank, alpha, state)

    {%{feed_forward | layers: %{feed_forward.layers | first: first, second: second}}, state}
  end

  defp replace(%LinearWithLoRA{} = layer, _rank, _alpha, state), do: {layer, state}

  defp replace(%{kernel: %Nx.Tensor{}} = dense, rank, alpha, state) do
    wrap_dense(dense, rank, alpha, state)
  end

  defp replace(other, _rank, _alpha, state), do: {other, state}

  defp replace_list(list, rank, alpha, state) do
    Enum.map_reduce(list, state, fn item, state ->
      replace(item, rank, alpha, state)
    end)
  end

  defp wrap_dense(%LinearWithLoRA{} = layer, _rank, _alpha, state), do: {layer, state}

  defp wrap_dense(%{kernel: %Nx.Tensor{}} = dense, rank, alpha, state) do
    layer = LinearWithLoRA.new(dense, rank, alpha, seed: state.seed)
    {layer, %{state | seed: state.seed + 1}}
  end

  defp trainable_after_lora(%GPTModel{trainable: :all}), do: :all
  defp trainable_after_lora(%GPTModel{trainable: []}), do: [:lora]

  defp trainable_after_lora(%GPTModel{trainable: fields}) when is_list(fields) do
    Enum.uniq(fields ++ [:lora])
  end
end
