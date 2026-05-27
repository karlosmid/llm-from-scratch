defmodule LlmScratch.ModelCheckpointTest do
  use ExUnit.Case

  alias LlmScratch.{GPTConfig, GPTModel, ModelCheckpoint}

  test "saves and loads a GPT model" do
    path = Path.join(System.tmp_dir!(), "llm_scratch_model_checkpoint_test.nx")
    on_exit(fn -> File.rm(path) end)

    cfg = %GPTConfig{
      vocab_size: 16,
      context_length: 4,
      emb_dim: 8,
      n_heads: 2,
      n_layers: 1,
      drop_rate: 0.0,
      qkv_bias: false
    }

    model = GPTModel.new(cfg, seed: 123)
    input = Nx.tensor([[1, 2, 3]], type: {:s, 64})

    ModelCheckpoint.save!(model, path)
    loaded_model = ModelCheckpoint.load!(path)

    assert %GPTModel{} = loaded_model
    assert loaded_model.cfg == model.cfg
    assert length(loaded_model.trf_blocks) == 1

    assert GPTModel.forward(loaded_model, input)
           |> Nx.all_close(GPTModel.forward(model, input), atol: 0.0)
           |> Nx.to_number() == 1
  end
end
