defmodule LlmScratch.SelfAttentionV2 do
  @moduledoc """
  Self-attention module initialized from Axon dense layers.

  API:

    * `new/2` and `new/3` - initialize `w_q`, `w_k`, and `w_v`
      using `Axon.input/2 |> Axon.dense/3`
    * `forward/2` - compute full context vectors for all tokens
  """

  defstruct [:w_q, :w_k, :w_v, :d_in, :d_out, :seed, :qkv_bias]

  @type dense_weights :: %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          w_q: dense_weights(),
          w_k: dense_weights(),
          w_v: dense_weights(),
          d_in: pos_integer(),
          d_out: pos_integer(),
          seed: integer(),
          qkv_bias: boolean()
        }

  @spec new(pos_integer(), pos_integer()) :: t()
  @doc """
  Creates a self-attention module whose projection weights are initialized via
  Axon dense layers.

  ## Arguments

    * `d_in` - input feature size.
    * `d_out` - projection/output feature size.

  ## Returns

    * `%LlmScratch.SelfAttentionV2{}` with dense-initialized `w_q`, `w_k`, and `w_v`.
  """
  def new(d_in, d_out), do: new(d_in, d_out, [])

  @spec new(pos_integer(), pos_integer(), keyword()) :: t()
  @doc """
  Creates a self-attention module with Axon dense-initialized projections.

  ## Arguments

    * `d_in` - input feature size.
    * `d_out` - projection/output feature size.
    * `opts` - keyword options:
      `:seed` (optional, deterministic initialization),
      `:qkv_bias` (optional, defaults to `true`).

  ## Returns

    * `%LlmScratch.SelfAttentionV2{}` with dense-initialized projections.
  """
  def new(d_in, d_out, opts)
      when is_integer(d_in) and d_in > 0 and is_integer(d_out) and d_out > 0 do
    seed = normalize_seed(Keyword.get(opts, :seed))
    qkv_bias = normalize_qkv_bias(Keyword.get(opts, :qkv_bias, true))

    w_q = init_dense_weights(d_in, d_out, seed, qkv_bias, "q_proj")
    w_k = init_dense_weights(d_in, d_out, seed, qkv_bias, "k_proj")
    w_v = init_dense_weights(d_in, d_out, seed, qkv_bias, "v_proj")

    %__MODULE__{
      w_q: w_q,
      w_k: w_k,
      w_v: w_v,
      d_in: d_in,
      d_out: d_out,
      seed: seed,
      qkv_bias: qkv_bias
    }
  end

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Computes context vectors for all tokens in `inputs`.

  ## Arguments

    * `sa` - `%LlmScratch.SelfAttentionV2{}` module state.
    * `inputs` - tensor of shape `{num_tokens, d_in}`.

  ## Returns

    * context tensor of shape `{num_tokens, d_out}`.
  """
  def forward(%__MODULE__{} = sa, %Nx.Tensor{} = inputs) do
    validate_input_shape!(inputs, sa.d_in)

    q = dense_project(inputs, sa.w_q)
    k = dense_project(inputs, sa.w_k)
    v = dense_project(inputs, sa.w_v)
    LlmScratch.SelfAttentionCore.context_from_qkv(q, k, v, sa.d_out)
  end

  @doc """
  Initializes Axon dense layer weights and returns `%{kernel, bias}` tensors.

  This helper is shared by attention modules that want dense initialization
  behavior consistent with `SelfAttentionV2`.

  ## Arguments

    * `d_in` - input feature size.
    * `d_out` - projection/output feature size.
    * `seed` - deterministic initialization seed.
    * `qkv_bias` - whether to include dense bias.
    * `layer_name` - Axon layer name used for parameter extraction.
  """
  def init_dense_weights(d_in, d_out, seed, qkv_bias, layer_name) do
    model =
      Axon.input("input", shape: {nil, d_in})
      |> Axon.dense(d_out, use_bias: qkv_bias, name: layer_name)

    {init_fn, _predict_fn} = Axon.build(model, seed: seed)
    params = init_fn.(Nx.template({1, d_in}, {:f, 32}), Axon.ModelState.empty())
    extract_dense_weights!(params, layer_name)
  end

  defp extract_dense_weights!(%Axon.ModelState{} = params, layer_name) do
    params
    |> Axon.ModelState.trainable_parameters()
    |> extract_dense_weights!(layer_name)
  end

  defp extract_dense_weights!(params, layer_name) when is_map(params) do
    layer_params =
      Map.get(params, layer_name) ||
        Enum.find_value(Map.values(params), fn
          layer when is_map(layer) ->
            kernel = Map.get(layer, "kernel") || Map.get(layer, :kernel)
            bias = Map.get(layer, "bias") || Map.get(layer, :bias)

            if match?(%Nx.Tensor{}, kernel) and (is_nil(bias) or match?(%Nx.Tensor{}, bias)) do
              layer
            else
              nil
            end

          _ ->
            nil
        end)

    kernel = layer_params && (Map.get(layer_params, "kernel") || Map.get(layer_params, :kernel))
    bias = layer_params && (Map.get(layer_params, "bias") || Map.get(layer_params, :bias))

    if match?(%Nx.Tensor{}, kernel) do
      kernel = Nx.as_type(kernel, {:f, 32})

      bias =
        case bias do
          %Nx.Tensor{} = bias -> Nx.as_type(bias, {:f, 32})
          nil -> zero_bias_from_kernel(kernel)
        end

      %{kernel: kernel, bias: bias}
    else
      raise ArgumentError,
            "could not extract dense kernel/bias params for layer #{inspect(layer_name)}"
    end
  end

  @doc """
  Projects the input tensor using the provided dense layer weights.

  ## Parameters

    - `inputs`: The input tensor of shape `{num_tokens, d_in}`.
    - `%{kernel: kernel, bias: bias}`: A map containing:
        - `kernel`: The projection weights of shape `{d_in, d_out}`.
        - `bias`: The bias vector of shape `{d_out}`.

  ## Returns

    - The projected tensor of shape `{num_tokens, d_out}`, computed as `Nx.dot(inputs, kernel) + bias`.

  """
  @spec dense_project(Nx.Tensor.t(), %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}) ::
          Nx.Tensor.t()

  def dense_project(inputs, %{kernel: kernel, bias: bias}) do
    Nx.add(Nx.dot(inputs, kernel), bias)
  end

  defp zero_bias_from_kernel(kernel) do
    {_, d_out} = Nx.shape(kernel)
    Nx.broadcast(0.0, {d_out}) |> Nx.as_type({:f, 32})
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
  end

  @doc """
  Validates and normalizes the `qkv_bias` option.

  ## Arguments

    * `qkv_bias` - boolean indicating whether Q/K/V projections use bias.

  ## Returns

    * the same boolean value when valid.

  ## Raises

    * `ArgumentError` if `qkv_bias` is not a boolean.
  """
  def normalize_qkv_bias(qkv_bias) when is_boolean(qkv_bias), do: qkv_bias

  def normalize_qkv_bias(qkv_bias) do
    raise ArgumentError, "qkv_bias must be a boolean, got: #{inspect(qkv_bias)}"
  end

  defp validate_input_shape!(inputs, expected_d_in) do
    case Nx.shape(inputs) do
      {_, ^expected_d_in} ->
        :ok

      shape ->
        raise ArgumentError,
              "expected inputs shape {num_tokens, #{expected_d_in}}, got: #{inspect(shape)}"
    end
  end
end
