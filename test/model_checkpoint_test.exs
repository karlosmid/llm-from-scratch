defmodule LlmScratch.ModelCheckpointTest do
  use ExUnit.Case

  alias LlmScratch.{GPTConfig, GPTModel, ModelCheckpoint, Training}

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

  test "saves and loads model and optimizer training state" do
    path = Path.join(System.tmp_dir!(), "llm_scratch_training_checkpoint_test.nx")
    on_exit(fn -> File.rm(path) end)

    model = GPTModel.new(small_gpt_config(), seed: 123)

    optimizer = %{
      Training.adamw(0.001, weight_decay: 0.1)
      | step: 2,
        m: [Nx.tensor([0.1, 0.2]), Nx.tensor([[0.3]])],
        v: [Nx.tensor([0.01, 0.02]), Nx.tensor([[0.03]])]
    }

    ModelCheckpoint.save_training_state!(model, optimizer, path)
    loaded_state = ModelCheckpoint.load_training_state!(path)

    assert %{model_state_dict: %GPTModel{}, optimizer_state_dict: %Training.AdamW{}} =
             loaded_state

    assert loaded_state.model_state_dict.cfg == model.cfg
    assert loaded_state.optimizer_state_dict.learning_rate == optimizer.learning_rate
    assert loaded_state.optimizer_state_dict.weight_decay == optimizer.weight_decay
    assert loaded_state.optimizer_state_dict.step == optimizer.step

    Enum.zip(loaded_state.optimizer_state_dict.m, optimizer.m)
    |> Enum.each(fn {actual, expected} ->
      assert Nx.all_close(actual, expected, atol: 0.0) |> Nx.to_number() == 1
    end)

    Enum.zip(loaded_state.optimizer_state_dict.v, optimizer.v)
    |> Enum.each(fn {actual, expected} ->
      assert Nx.all_close(actual, expected, atol: 0.0) |> Nx.to_number() == 1
    end)
  end

  test "saves and loads model and fresh optimizer training state" do
    path = Path.join(System.tmp_dir!(), "llm_scratch_fresh_training_checkpoint_test.nx")
    on_exit(fn -> File.rm(path) end)

    model = GPTModel.new(small_gpt_config(), seed: 123)
    optimizer = Training.adamw(0.001, weight_decay: 0.1)

    ModelCheckpoint.save_training_state!(model, optimizer, path)
    loaded_state = ModelCheckpoint.load_training_state!(path)

    assert %Training.AdamW{m: nil, v: nil, step: 0} = loaded_state.optimizer_state_dict
  end

  defp small_gpt_config do
    %GPTConfig{
      vocab_size: 16,
      context_length: 4,
      emb_dim: 8,
      n_heads: 2,
      n_layers: 1,
      drop_rate: 0.0,
      qkv_bias: false
    }
  end
end
