defmodule LlmScratch.TestHelpers do
  @moduledoc false

  import ExUnit.Assertions

  alias LlmScratch.{EMLXBackend, GPTConfig, SpamDataset}

  def use_accelerated_backend do
    context = EMLXBackend.apple_gpu_or_exla!()
    ExUnit.Callbacks.on_exit(fn -> EMLXBackend.restore!(context) end)
    context.backend
  end

  def decode_token_pieces(model, tokens) do
    Enum.map(tokens, fn token ->
      {:ok, text_piece} = Tiktoken.decode(model, [token])
      text_piece
    end)
  end

  def softmax_naive(%Nx.Tensor{} = x) do
    exp_x = Nx.exp(x)
    Nx.divide(exp_x, Nx.sum(exp_x, axes: [0]))
  end

  def assert_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-6)

    expected =
      case expected do
        %Nx.Tensor{} -> expected
        _ -> Nx.tensor(expected, type: {:f, 32})
      end

    assert Nx.all_close(actual, expected, atol: atol) |> Nx.to_number() == 1
  end

  def write_csv!(path, records) do
    rows =
      records
      |> Enum.map(fn record ->
        [csv_field(record.label), ",", csv_field(record.text), "\n"]
      end)

    File.write!(path, ["Label,Text\n", rows])
  end

  def csv_field(value) do
    value
    |> to_string()
    |> String.replace("\"", "\"\"")
    |> then(&"\"#{&1}\"")
  end

  def dataset_samples(dataset) do
    0..(SpamDataset.length(dataset) - 1)
    |> Enum.map(&SpamDataset.get(dataset, &1))
  end

  def collate_batch(batch) do
    {inputs, labels} = Enum.unzip(batch)

    {
      Nx.stack(inputs),
      Nx.stack(labels)
    }
  end

  def gpt2_compatible_token_ids(text) do
    {:ok, token_ids} = Tiktoken.encode("code-davinci-002", text, ["<|endoftext|>"])

    Enum.map(token_ids, &min(&1, 50_256))
  end

  def write_training_metrics!(
        path,
        num_epochs,
        examples_seen,
        train_losses,
        val_losses,
        train_accs,
        val_accs
      ) do
    metrics = %{
      num_epochs: num_epochs,
      examples_seen: examples_seen,
      losses: %{
        epochs_seen: linspace(0.0, num_epochs * 1.0, length(train_losses)),
        examples_seen: linspace(0.0, examples_seen * 1.0, length(train_losses)),
        train_values: train_losses,
        val_values: val_losses
      },
      accuracies: %{
        epochs_seen: linspace(1.0, num_epochs * 1.0, length(train_accs)),
        examples_seen:
          linspace(examples_seen / num_epochs, examples_seen * 1.0, length(train_accs)),
        train_values: train_accs,
        val_values: val_accs
      }
    }

    {:ok, encoded_metrics} = Jason.encode(metrics, pretty: true)
    File.write!(path, encoded_metrics)
  end

  def linspace(_start, stop, 1), do: [stop * 1.0]

  def linspace(start, stop, count) do
    step = (stop - start) / (count - 1)

    Enum.map(0..(count - 1), fn index ->
      start + index * step
    end)
  end

  def small_gpt_config do
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

  def tiny_gpt_config do
    %GPTConfig{
      vocab_size: 32,
      context_length: 8,
      emb_dim: 8,
      n_heads: 1,
      n_layers: 0,
      drop_rate: 0.0,
      qkv_bias: false
    }
  end

  def tiny_input_batch do
    [[1, 2, 3, 4], [4, 3, 2, 1]]
    |> Nx.tensor(type: {:s, 64})
    |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  def tiny_target_batch do
    [[2, 3, 4, 5], [3, 2, 1, 0]]
    |> Nx.tensor(type: {:s, 64})
    |> Nx.backend_transfer(Nx.BinaryBackend)
  end
end

ExUnit.start()
