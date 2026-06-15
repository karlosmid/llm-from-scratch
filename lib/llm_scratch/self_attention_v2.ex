defmodule LlmScratch.SelfAttentionV2 do
  @moduledoc """
  Self-attention module initialized from Axon dense layers.

  API:

    * `new/2` and `new/3` - initialize `w_q`, `w_k`, and `w_v`
      using `Axon.input/2 |> Axon.dense/3`
    * `forward/2` - compute full context vectors for all tokens
  """

  import Nx.Defn

  alias LlmScratch.LinearWithLoRA

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
    w_k = init_dense_weights(d_in, d_out, seed + 1, qkv_bias, "k_proj")
    w_v = init_dense_weights(d_in, d_out, seed + 2, qkv_bias, "v_proj")

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
    _layer_name = layer_name
    bound = :math.sqrt(1.0 / d_in)
    key = LlmScratch.Random.manual_seed(seed)

    {kernel, key} = Nx.Random.uniform(key, -bound, bound, shape: {d_in, d_out}, type: {:f, 32})

    bias =
      if qkv_bias do
        {bias, _key} = Nx.Random.uniform(key, -bound, bound, shape: {d_out}, type: {:f, 32})

        bias
      else
        Nx.broadcast(0.0, {d_out}) |> Nx.as_type({:f, 32})
      end

    %{kernel: Nx.as_type(kernel, {:f, 32}), bias: Nx.as_type(bias, {:f, 32})}
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
    dense_project_defn(inputs, %{kernel: kernel, bias: bias})
  end

  def dense_project(inputs, %LinearWithLoRA{} = layer) do
    LinearWithLoRA.forward(layer, inputs)
  end

  @doc """
  Defn-compatible dense projection over the last axis.
  """
  deftransform dense_project_defn(inputs, layer) do
    if match?(%LinearWithLoRA{}, layer) do
      LinearWithLoRA.forward_defn(layer, inputs)
    else
      Nx.add(Nx.dot(inputs, [-1], layer.kernel, [0]), layer.bias)
    end
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
