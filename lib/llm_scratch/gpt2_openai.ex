defmodule LlmScratch.GPT2OpenAI do
  @moduledoc """
  Downloads and loads the public OpenAI GPT-2 checkpoints.

  The module is the bridge between OpenAI's original TensorFlow checkpoint
  layout and this repo's `%LlmScratch.GPTModel{}` structs. It handles three
  separate steps:

    * downloading the original checkpoint files
    * converting TensorFlow checkpoint tensors into `Nx.Tensor`s
    * assigning those tensors into the local GPT model implementation

  This mirrors the book's Python helper:

      settings, params = download_and_load_gpt2(
          model_size="124M",
          models_dir="gpt2"
      )

  The checkpoint files are downloaded from OpenAI's public GPT-2 storage and
  cached under `models_dir/model_size`. TensorFlow is used through `Pythonx` to
  read the original TensorFlow checkpoint and export its variables to a cached
  `model_params.npz` file in the same model directory.

  Supported checkpoint sizes are:

    * `"124M"`
    * `"355M"`
    * `"774M"`
    * `"1558M"`

  Python packages required by the checkpoint reader:

      pip install "tensorflow>=2.15.0" "tqdm>=4.66"

  At runtime, this Elixir module needs network access for the initial download
  and a Python environment with TensorFlow and NumPy for the checkpoint export.
  Subsequent loads reuse the downloaded files and cached `.npz` export when
  present.
  """

  alias LlmScratch.{DummyLayerNorm, EmbeddingNative, GPTConfig, GPTModel, ModelCheckpoint}

  @allowed_sizes ["124M", "355M", "774M", "1558M"]
  @filenames [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe"
  ]
  @base_url "https://openaipublic.blob.core.windows.net/gpt-2/models"
  @backup_base_url "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"

  @type settings :: map()
  @type params :: map()

  @doc """
  Downloads GPT-2 files and loads TensorFlow checkpoint params.

  `model_size` must be one of `"124M"`, `"355M"`, `"774M"`, or `"1558M"`.

  Options:

    * `:models_dir` - directory used to cache downloaded checkpoints. Defaults
      to `"gpt2"`.

  Returns `{settings, params}`. `settings` is decoded from `hparams.json`.
  `params` follows the book's shape:

      %{
        "wte" => token_embedding,
        "wpe" => position_embedding,
        "g" => final_norm_scale,
        "b" => final_norm_shift,
        "blocks" => [
          %{
            "attn" => %{"c_attn" => %{"w" => ..., "b" => ...}, ...},
            "mlp" => ...,
            "ln_1" => ...,
            "ln_2" => ...
          }
        ]
      }

  The tensor values are `Nx.Tensor`s.

  Raises if the model size is unsupported, a checkpoint file cannot be
  downloaded, TensorFlow cannot export the checkpoint, or a required tensor is
  missing from the exported `.npz` file.
  """
  @spec download_and_load_gpt2(String.t(), keyword()) :: {settings(), params()}
  def download_and_load_gpt2(model_size \\ "124M", opts \\ []) when is_list(opts) do
    models_dir = Keyword.get(opts, :models_dir, "gpt2")
    model_dir = download_gpt2_files!(model_size, models_dir)
    settings = load_settings!(model_dir)
    params = load_params_from_tf_checkpoint!(model_dir, settings)

    {settings, params}
  end

  @doc """
  Downloads GPT-2 files and returns the local model directory.

  If all expected files already exist and are non-empty, no download is
  attempted. Otherwise each checkpoint file is downloaded from OpenAI's public
  GPT-2 storage, with the book's Backblaze mirror used as a fallback.

  The returned directory is `Path.join(models_dir, model_size)`.
  """
  @spec download_gpt2_files!(String.t(), Path.t()) :: Path.t()
  def download_gpt2_files!(model_size, models_dir \\ "gpt2") when is_binary(models_dir) do
    validate_model_size!(model_size)

    model_dir = Path.join(models_dir, model_size)
    File.mkdir_p!(model_dir)

    unless all_checkpoint_files_present?(model_dir) do
      Enum.each(@filenames, fn filename ->
        destination = Path.join(model_dir, filename)
        primary_url = Enum.join([@base_url, model_size, filename], "/")
        backup_url = Enum.join([@backup_base_url, model_size, filename], "/")

        download_file!(primary_url, destination, backup_url)
      end)
    end

    model_dir
  end

  @doc """
  Builds a `%LlmScratch.GPTModel{}` with the public GPT-2 weights assigned.

  This is the highest-level loading helper when you want a ready-to-run model:
  it downloads missing checkpoint files, exports the TensorFlow checkpoint,
  creates the matching `%LlmScratch.GPTConfig{}`, initializes a local model, and
  assigns the downloaded weights.

  Options:

    * `:models_dir` - checkpoint cache directory. Defaults to `"gpt2"`.
    * `:seed` - seed used for the temporary model initialization before
      checkpoint weights are assigned. Defaults to `123`.

  The returned model uses `norm_eps: 1.0e-5`, matching GPT-2's layer norm
  epsilon.
  """
  @spec load_model(String.t(), keyword()) :: GPTModel.t()
  def load_model(model_size \\ "124M", opts \\ []) when is_list(opts) do
    {settings, params} = download_and_load_gpt2(model_size, opts)

    settings
    |> config_from_settings()
    |> GPTModel.new(seed: Keyword.get(opts, :seed, 123), norm_eps: 1.0e-5)
    |> load_weights_into_gpt(params)
  end

  @doc """
  Downloads, loads, assigns, and saves public GPT-2 weights as an Nx checkpoint.

  Returns the loaded `%LlmScratch.GPTModel{}`.

  This is useful when you want to pay the TensorFlow checkpoint conversion cost
  once and later load the model through `LlmScratch.ModelCheckpoint.load!/1`.
  Parent directories for `checkpoint_path` are created by
  `LlmScratch.ModelCheckpoint.save!/2`.
  """
  @spec save_model!(String.t(), Path.t(), keyword()) :: GPTModel.t()
  def save_model!(model_size \\ "124M", checkpoint_path, opts \\ [])
      when is_binary(checkpoint_path) and is_list(opts) do
    model = load_model(model_size, opts)
    ModelCheckpoint.save!(model, checkpoint_path)
    model
  end

  @doc """
  Validates a GPT-2 source tensor against a target tensor and returns the source.

  `left` is the tensor already present in the local model. `right` is the
  tensor loaded from the OpenAI checkpoint. The function raises when their
  shapes differ, then casts `right` to `left`'s type before returning it.

  This is the Nx equivalent of the book's PyTorch helper:

      def assign(left, right):
          if left.shape != right.shape:
              raise ValueError(...)
          return torch.nn.Parameter(torch.tensor(right))

  Elixir data is immutable and this repo stores parameters directly as
  `Nx.Tensor`s, so the function returns `right` converted to the target tensor's
  type after validating that the shapes match.
  """
  @spec assign(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def assign(%Nx.Tensor{} = left, %Nx.Tensor{} = right) do
    left_shape = Nx.shape(left)
    right_shape = Nx.shape(right)

    unless left_shape == right_shape do
      raise ArgumentError,
            "shape mismatch. Left: #{inspect(left_shape)}, Right: #{inspect(right_shape)}"
    end

    Nx.as_type(right, Nx.type(left))
  end

  @doc """
  Loads decoded OpenAI GPT-2 params into this repo's `%LlmScratch.GPTModel{}`.

  This mirrors the book's `load_weights_into_gpt(gpt, params)` function:

    * token and positional embeddings are copied from `"wte"` and `"wpe"`
    * Q/K/V weights and biases are split from `"attn"["c_attn"]`
    * attention output, feed-forward, and layer norm parameters are assigned
    * the output head is tied to token embeddings

  One layout difference from the PyTorch code is intentional: this repo's dense
  kernels are stored as `{in_dim, out_dim}`, matching the OpenAI TensorFlow
  checkpoint matrices. PyTorch `nn.Linear.weight` is `{out_dim, in_dim}`, so the
  Python book code uses `.T` where this Nx implementation does not.
  """
  @spec load_weights_into_gpt(GPTModel.t(), params()) :: GPTModel.t()
  def load_weights_into_gpt(%GPTModel{} = model, params) when is_map(params) do
    do_assign_params!(model, params)
  end

  @doc """
  Backward-compatible alias for `load_weights_into_gpt/2`.

  Prefer `load_weights_into_gpt/2` in new code.
  """
  @spec assign_params!(GPTModel.t(), params()) :: GPTModel.t()
  def assign_params!(%GPTModel{} = model, params) when is_map(params) do
    load_weights_into_gpt(model, params)
  end

  defp do_assign_params!(%GPTModel{} = model, params) when is_map(params) do
    blocks = Map.fetch!(params, "blocks")

    unless length(blocks) == length(model.trf_blocks) do
      raise ArgumentError,
            "expected #{length(model.trf_blocks)} GPT-2 blocks, got #{length(blocks)}"
    end

    token_embedding = Map.fetch!(params, "wte")

    %{
      model
      | tok_emb: put_embedding_weight!(model.tok_emb, token_embedding),
        pos_emb: put_embedding_weight!(model.pos_emb, Map.fetch!(params, "wpe")),
        final_norm:
          put_layer_norm!(
            model.final_norm,
            Map.fetch!(params, "g"),
            Map.fetch!(params, "b")
          ),
        out_head: put_output_head!(model.out_head, token_embedding),
        trf_blocks:
          Enum.zip(model.trf_blocks, blocks)
          |> Enum.map(fn {block, block_params} -> put_block!(block, block_params) end)
    }
  end

  @doc """
  Converts GPT-2 `hparams.json` settings into this repo's `%GPTConfig{}`.

  The OpenAI settings determine the model size from the triple
  `{n_embd, n_layer, n_head}`. The returned config keeps the checkpoint's
  vocabulary size and context length, and uses GPT-2's `qkv_bias: true`
  setting.
  """
  @spec config_from_settings(settings()) :: GPTConfig.t()
  def config_from_settings(settings) when is_map(settings) do
    %{
      GPTConfig.openai_gpt2(model_size_from_settings!(settings))
      | vocab_size: fetch_setting!(settings, "n_vocab"),
        context_length: fetch_setting!(settings, "n_ctx")
    }
  end

  @doc """
  Loads `hparams.json` from a downloaded GPT-2 model directory.

  `model_dir` should be the directory returned by `download_gpt2_files!/2`.
  """
  @spec load_settings!(Path.t()) :: settings()
  def load_settings!(model_dir) when is_binary(model_dir) do
    model_dir
    |> Path.join("hparams.json")
    |> File.read!()
    |> Jason.decode!()
  end

  @doc """
  Loads GPT-2 params from a downloaded TensorFlow checkpoint directory.

  This exports the TensorFlow checkpoint to `model_params.npz` if that cache
  file is missing, then reads the expected GPT-2 variables from the `.npz` file
  into the nested params map consumed by `load_weights_into_gpt/2`.
  """
  @spec load_params_from_tf_checkpoint!(Path.t(), settings()) :: params()
  def load_params_from_tf_checkpoint!(model_dir, settings)
      when is_binary(model_dir) and is_map(settings) do
    npz_path = export_tf_checkpoint_to_npz!(model_dir)

    %{
      "blocks" =>
        0..(fetch_setting!(settings, "n_layer") - 1)
        |> Enum.map(&load_block_params!(npz_path, &1)),
      "wte" => load_tensor!(npz_path, "model/wte"),
      "wpe" => load_tensor!(npz_path, "model/wpe"),
      "g" => load_tensor!(npz_path, "model/ln_f/g"),
      "b" => load_tensor!(npz_path, "model/ln_f/b")
    }
  end

  defp load_block_params!(npz_path, layer_idx) do
    prefix = "model/h#{layer_idx}"

    %{
      "attn" => %{
        "c_attn" => %{
          "w" => load_tensor!(npz_path, "#{prefix}/attn/c_attn/w"),
          "b" => load_tensor!(npz_path, "#{prefix}/attn/c_attn/b")
        },
        "c_proj" => %{
          "w" => load_tensor!(npz_path, "#{prefix}/attn/c_proj/w"),
          "b" => load_tensor!(npz_path, "#{prefix}/attn/c_proj/b")
        }
      },
      "mlp" => %{
        "c_fc" => %{
          "w" => load_tensor!(npz_path, "#{prefix}/mlp/c_fc/w"),
          "b" => load_tensor!(npz_path, "#{prefix}/mlp/c_fc/b")
        },
        "c_proj" => %{
          "w" => load_tensor!(npz_path, "#{prefix}/mlp/c_proj/w"),
          "b" => load_tensor!(npz_path, "#{prefix}/mlp/c_proj/b")
        }
      },
      "ln_1" => %{
        "g" => load_tensor!(npz_path, "#{prefix}/ln_1/g"),
        "b" => load_tensor!(npz_path, "#{prefix}/ln_1/b")
      },
      "ln_2" => %{
        "g" => load_tensor!(npz_path, "#{prefix}/ln_2/g"),
        "b" => load_tensor!(npz_path, "#{prefix}/ln_2/b")
      }
    }
  end

  defp load_tensor!(npz_path, name) do
    python_code = """
    import numpy as np

    if isinstance(npz_path, bytes):
        npz_path = npz_path.decode()

    if isinstance(name, bytes):
        name = name.decode()

    with np.load(npz_path) as data:
        array = data[name].astype(np.float32)

    (list(array.shape), array.tobytes(order="C"))
    """

    {result, _globals} =
      Pythonx.eval(python_code, %{
        "npz_path" => npz_path,
        "name" => name
      })

    {shape, bytes} = Pythonx.decode(result)

    bytes
    |> Nx.from_binary({:f, 32})
    |> Nx.reshape(List.to_tuple(shape))
  rescue
    error in [Pythonx.Error] ->
      raise """
      Failed to load GPT-2 variable #{inspect(name)} from #{inspect(npz_path)}.

      Make sure NumPy is available to Pythonx.

      Original error:

      #{Exception.message(error)}
      """
  end

  defp export_tf_checkpoint_to_npz!(model_dir) do
    npz_path = Path.join(model_dir, "model_params.npz")

    if File.exists?(npz_path) and File.stat!(npz_path).size > 0 do
      npz_path
    else
      ckpt_path = latest_checkpoint!(model_dir)
      python = python_executable!()

      script = """
      import sys
      import numpy as np
      import tensorflow as tf

      ckpt_path = sys.argv[1]
      npz_path = sys.argv[2]

      params = {}
      for name, _shape in tf.train.list_variables(ckpt_path):
          if name.startswith("model/"):
              params[name] = np.squeeze(tf.train.load_variable(ckpt_path, name)).astype(np.float32)

      np.savez_compressed(npz_path, **params)
      """

      case System.cmd(python, ["-c", script, ckpt_path, npz_path],
             stderr_to_stdout: true,
             env: [{"TF_CPP_MIN_LOG_LEVEL", "2"}]
           ) do
        {_output, 0} ->
          npz_path

        {output, status} ->
          File.rm(npz_path)

          raise """
          Failed to export TensorFlow checkpoint #{inspect(ckpt_path)} to #{inspect(npz_path)}.

          Make sure TensorFlow is available in the Python environment:

              pip install "tensorflow>=2.15.0" "tqdm>=4.66"

          Python exited with status #{status}.

          #{output}
          """
      end
    end
  end

  defp latest_checkpoint!(model_dir) do
    checkpoint_path = Path.join(model_dir, "checkpoint")

    checkpoint_file =
      checkpoint_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.find(&String.starts_with?(&1, "model_checkpoint_path:"))

    case checkpoint_file do
      nil ->
        raise ArgumentError, "no TensorFlow checkpoint found in #{inspect(model_dir)}"

      line ->
        [_key, quoted_path] = String.split(line, ":", parts: 2)

        ckpt_path =
          quoted_path
          |> String.trim()
          |> String.trim("\"")

        if Path.type(ckpt_path) == :absolute do
          ckpt_path
        else
          Path.join(model_dir, ckpt_path)
        end
    end
  end

  defp python_executable! do
    pythonx_python =
      :pythonx
      |> Application.app_dir("priv/uv/project/.venv/bin/python")
      |> to_string()

    cond do
      File.exists?(pythonx_python) ->
        pythonx_python

      python = System.find_executable("python3") ->
        python

      python = System.find_executable("python") ->
        python

      true ->
        raise "could not find a Python executable for TensorFlow checkpoint export"
    end
  end

  defp put_block!(block, block_params) do
    {q_w, k_w, v_w} = split_qkv(Map.fetch!(block_params["attn"]["c_attn"], "w"))
    {q_b, k_b, v_b} = split_qkv(Map.fetch!(block_params["attn"]["c_attn"], "b"))

    %{
      block
      | norm1:
          put_layer_norm!(
            block.norm1,
            block_params["ln_1"]["g"],
            block_params["ln_1"]["b"]
          ),
        norm2:
          put_layer_norm!(
            block.norm2,
            block_params["ln_2"]["g"],
            block_params["ln_2"]["b"]
          ),
        att: %{
          block.att
          | w_q: put_dense!(block.att.w_q, q_w, q_b),
            w_k: put_dense!(block.att.w_k, k_w, k_b),
            w_v: put_dense!(block.att.w_v, v_w, v_b),
            out_proj:
              put_dense!(
                block.att.out_proj,
                block_params["attn"]["c_proj"]["w"],
                block_params["attn"]["c_proj"]["b"]
              )
        },
        ff: %{
          block.ff
          | layers: %{
              block.ff.layers
              | first:
                  put_dense!(
                    block.ff.layers.first,
                    block_params["mlp"]["c_fc"]["w"],
                    block_params["mlp"]["c_fc"]["b"]
                  ),
                second:
                  put_dense!(
                    block.ff.layers.second,
                    block_params["mlp"]["c_proj"]["w"],
                    block_params["mlp"]["c_proj"]["b"]
                  )
            }
        }
    }
  end

  defp put_embedding_weight!(%EmbeddingNative{} = embedding, weight) do
    %{embedding | weight: assign(embedding.weight, weight)}
  end

  defp put_layer_norm!(%DummyLayerNorm{} = layer_norm, scale, shift) do
    %{layer_norm | scale: assign(layer_norm.scale, scale), shift: assign(layer_norm.shift, shift)}
  end

  defp put_dense!(dense, kernel, bias) do
    %{dense | kernel: assign(dense.kernel, kernel), bias: assign(dense.bias, bias)}
  end

  defp put_output_head!(out_head, token_embedding) do
    kernel = Nx.transpose(token_embedding)
    %{out_head | kernel: assign(out_head.kernel, kernel)}
  end

  defp split_qkv(tensor) do
    shape = Nx.shape(tensor)
    last_dim = shape |> Tuple.to_list() |> List.last()

    unless rem(last_dim, 3) == 0 do
      raise ArgumentError,
            "expected QKV tensor last dim to be divisible by 3, got #{inspect(shape)}"
    end

    width = div(last_dim, 3)

    {
      Nx.slice_along_axis(tensor, 0, width, axis: -1),
      Nx.slice_along_axis(tensor, width, width, axis: -1),
      Nx.slice_along_axis(tensor, 2 * width, width, axis: -1)
    }
  end

  defp download_file!(url, destination, backup_url) do
    if file_current?(url, destination) do
      :ok
    else
      url
      |> do_download_file(destination)
      |> case do
        :ok -> :ok
        {:error, primary_error} -> download_backup_file!(backup_url, destination, primary_error)
      end
    end
  end

  defp download_backup_file!(backup_url, destination, primary_error) do
    case do_download_file(backup_url, destination) do
      :ok ->
        :ok

      {:error, backup_error} ->
        raise """
        Failed to download GPT-2 checkpoint file #{Path.basename(destination)}.

        Primary error: #{inspect(primary_error)}
        Backup error: #{inspect(backup_error)}
        """
    end
  end

  defp do_download_file(url, destination) do
    tmp_destination = destination <> ".download"
    File.rm(tmp_destination)

    response =
      Req.get!(
        url,
        into: File.stream!(tmp_destination),
        receive_timeout: 120_000,
        retry: false
      )

    if response.status in 200..299 do
      File.rename!(tmp_destination, destination)
      :ok
    else
      File.rm(tmp_destination)
      {:error, {:http_status, response.status, url}}
    end
  rescue
    error ->
      File.rm(destination <> ".download")
      {:error, error}
  end

  defp file_current?(url, destination) do
    File.exists?(destination) and remote_content_length(url) == File.stat!(destination).size
  end

  defp all_checkpoint_files_present?(model_dir) do
    Enum.all?(@filenames, fn filename ->
      path = Path.join(model_dir, filename)
      File.exists?(path) and File.stat!(path).size > 0
    end)
  end

  defp remote_content_length(url) do
    response = Req.head!(url, receive_timeout: 30_000, retry: false)

    response
    |> Req.Response.get_header("content-length")
    |> List.first()
    |> case do
      nil -> :unknown
      size -> String.to_integer(size)
    end
  rescue
    _error -> :unknown
  end

  defp validate_model_size!(model_size) do
    unless model_size in @allowed_sizes do
      raise ArgumentError,
            "model_size must be one of #{inspect(@allowed_sizes)}, got: #{inspect(model_size)}"
    end
  end

  defp fetch_setting!(settings, key) do
    Map.fetch!(settings, key)
  end

  defp model_size_from_settings!(settings) do
    case {fetch_setting!(settings, "n_embd"), fetch_setting!(settings, "n_layer"),
          fetch_setting!(settings, "n_head")} do
      {768, 12, 12} -> "124M"
      {1024, 24, 16} -> "355M"
      {1280, 36, 20} -> "774M"
      {1600, 48, 25} -> "1558M"
      other -> raise ArgumentError, "unsupported GPT-2 settings dimensions: #{inspect(other)}"
    end
  end
end
