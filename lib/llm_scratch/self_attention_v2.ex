defmodule LlmScratch.SelfAttentionV2 do
  @moduledoc """
  Self-attention module initialized from Axon dense layers.

  API:

    * `new/2` and `new/3` - initialize `w_q`, `w_k`, and `w_v`
      using `Axon.input/2 |> Axon.dense/3`
    * `forward/2` - compute full context vectors for all tokens
  """

  defstruct [:w_q, :w_k, :w_v, :d_in, :d_out, :seed]

  @type dense_weights :: %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}

  @type t :: %__MODULE__{
          w_q: dense_weights(),
          w_k: dense_weights(),
          w_v: dense_weights(),
          d_in: pos_integer(),
          d_out: pos_integer(),
          seed: integer()
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
      `:seed` (optional, deterministic initialization).

  ## Returns

    * `%LlmScratch.SelfAttentionV2{}` with dense-initialized projections.
  """
  def new(d_in, d_out, opts)
      when is_integer(d_in) and d_in > 0 and is_integer(d_out) and d_out > 0 do
    seed = normalize_seed(Keyword.get(opts, :seed))

    w_q = init_dense_weights(d_in, d_out, seed, "q_proj")
    w_k = init_dense_weights(d_in, d_out, seed, "k_proj")
    w_v = init_dense_weights(d_in, d_out, seed, "v_proj")

    %__MODULE__{w_q: w_q, w_k: w_k, w_v: w_v, d_in: d_in, d_out: d_out, seed: seed}
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

  defp init_dense_weights(d_in, d_out, seed, layer_name) do
    model =
      Axon.input("input", shape: {nil, d_in})
      |> Axon.dense(d_out, use_bias: true, name: layer_name)

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

            if match?(%Nx.Tensor{}, kernel) and match?(%Nx.Tensor{}, bias) do
              layer
            else
              nil
            end

          _ ->
            nil
        end)

    kernel = layer_params && (Map.get(layer_params, "kernel") || Map.get(layer_params, :kernel))
    bias = layer_params && (Map.get(layer_params, "bias") || Map.get(layer_params, :bias))

    if match?(%Nx.Tensor{}, kernel) and match?(%Nx.Tensor{}, bias) do
      %{kernel: Nx.as_type(kernel, {:f, 32}), bias: Nx.as_type(bias, {:f, 32})}
    else
      raise ArgumentError,
            "could not extract dense kernel/bias params for layer #{inspect(layer_name)}"
    end
  end

  defp dense_project(inputs, %{kernel: kernel, bias: bias}) do
    Nx.add(Nx.dot(inputs, kernel), bias)
  end

  defp normalize_seed(nil), do: System.unique_integer([:positive])
  defp normalize_seed(seed) when is_integer(seed), do: seed

  defp normalize_seed(seed) do
    raise ArgumentError, "seed must be an integer or nil, got: #{inspect(seed)}"
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
