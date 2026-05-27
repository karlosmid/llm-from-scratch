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
    path
    |> Path.dirname()
    |> File.mkdir_p!()

    File.write!(path, Nx.serialize(model))
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
end
