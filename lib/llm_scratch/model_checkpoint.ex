defmodule LlmScratch.ModelCheckpoint do
  @moduledoc """
  Saves and loads Nx model containers.

  This is intended for trained models such as `LlmScratch.GPTModel`, whose
  tensors are serializable through `Nx.Container`.
  """

  @type path :: Path.t()

  @doc """
  Serializes `model` and writes it to `path`.

  Parent directories are created automatically. The model must implement
  `Nx.Container`; `LlmScratch.GPTModel` does through its container
  implementation in `LlmScratch.Training`.

  ## Example

      checkpoint_path = "ch5_2_gpt_124m.nx"
      LlmScratch.ModelCheckpoint.save!(trained_model, checkpoint_path)

  """
  @spec save!(struct(), path()) :: :ok
  def save!(model, path) when is_struct(model) and is_binary(path) do
    model
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> write_serialized!(path)
  end

  @doc """
  Reads a checkpoint from `path` and deserializes it back into a model struct.

  The checkpoint must have been written with `save!/2` or another compatible
  `Nx.serialize/1` call.

  ## Example

      trained_model = LlmScratch.ModelCheckpoint.load!("ch5_2_gpt_124m.nx")

  """
  @spec load!(path()) :: struct()
  def load!(path) when is_binary(path) do
    path
    |> File.read!()
    |> Nx.deserialize()
  end

  @doc """
  Serializes a training checkpoint containing both model and optimizer state.

  This mirrors the common PyTorch checkpoint shape:

      torch.save({
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
      }, "model_and_optimizer.pth")

  The returned Elixir checkpoint uses atom keys:

      %{
        model_state_dict: model,
        optimizer_state_dict: optimizer
      }

  Both values must implement `Nx.Container`. `LlmScratch.GPTModel` and
  `LlmScratch.Training.AdamW` do.

  ## Example

      LlmScratch.ModelCheckpoint.save_training_state!(
        trained_model,
        optimizer,
        "model_and_optimizer.nx"
      )

  """
  @spec save_training_state!(struct(), struct(), path()) :: :ok
  def save_training_state!(model, optimizer, path)
      when is_struct(model) and is_struct(optimizer) and is_binary(path) do
    write_serialized!(
      %{
        model_state_dict: Nx.backend_transfer(model, Nx.BinaryBackend),
        optimizer_state_dict: Nx.backend_transfer(optimizer, Nx.BinaryBackend)
      },
      path
    )
  end

  @doc """
  Reads a training checkpoint written by `save_training_state!/3`.

  Returns:

      %{
        model_state_dict: model,
        optimizer_state_dict: optimizer
      }

  """
  @spec load_training_state!(path()) :: %{
          model_state_dict: struct(),
          optimizer_state_dict: struct()
        }
  def load_training_state!(path) when is_binary(path) do
    path
    |> File.read!()
    |> Nx.deserialize()
  end

  defp write_serialized!(term, path) do
    path
    |> Path.dirname()
    |> File.mkdir_p!()

    File.write!(path, Nx.serialize(term))
  end
end
