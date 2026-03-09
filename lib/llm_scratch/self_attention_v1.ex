defmodule LlmScratch.SelfAttentionV1 do
  @moduledoc """
  Simple self-attention module with trainable projection weights.

  API:

    * `new/2` and `new/3` - initialize `w_q`, `w_k`, and `w_v`
    * `forward/2` - compute full context vectors for all tokens
  """

  defstruct [:w_q, :w_k, :w_v, :d_in, :d_out, :seed]

  @type t :: %__MODULE__{
          w_q: Nx.Tensor.t(),
          w_k: Nx.Tensor.t(),
          w_v: Nx.Tensor.t(),
          d_in: pos_integer(),
          d_out: pos_integer(),
          seed: integer() | nil
        }

  @spec new(pos_integer(), pos_integer()) :: t()
  def new(d_in, d_out), do: new(d_in, d_out, [])

  @spec new(pos_integer(), pos_integer(), keyword()) :: t()
  def new(d_in, d_out, opts)
      when is_integer(d_in) and d_in > 0 and is_integer(d_out) and d_out > 0 do
    seed = Keyword.get(opts, :seed)

    key =
      case seed do
        nil -> LlmScratch.Random.manual_seed(System.unique_integer([:positive]))
        int when is_integer(int) -> LlmScratch.Random.manual_seed(int)
        other -> raise ArgumentError, "seed must be an integer or nil, got: #{inspect(other)}"
      end

    w_q = init_weight({d_in, d_out}, key, Keyword.get(opts, :w_q))
    w_k = init_weight({d_in, d_out}, key, Keyword.get(opts, :w_k))
    w_v = init_weight({d_in, d_out}, key, Keyword.get(opts, :w_v))

    %__MODULE__{w_q: w_q, w_k: w_k, w_v: w_v, d_in: d_in, d_out: d_out, seed: seed}
  end

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{} = sa, %Nx.Tensor{} = inputs) do
    validate_input_shape!(inputs, sa.d_in)

    q = Nx.dot(inputs, sa.w_q)
    k = Nx.dot(inputs, sa.w_k)
    v = Nx.dot(inputs, sa.w_v)
    LlmScratch.SelfAttentionCore.context_from_qkv(q, k, v, sa.d_out)
  end

  defp init_weight(expected_shape, _key, %Nx.Tensor{} = provided_weight) do
    if Nx.shape(provided_weight) != expected_shape do
      raise ArgumentError,
            "expected weight shape #{inspect(expected_shape)}, got: #{inspect(Nx.shape(provided_weight))}"
    end

    Nx.as_type(provided_weight, {:f, 32})
  end

  defp init_weight(expected_shape, key, nil) do
    Axon.Initializers.uniform(scale: 1.0).(expected_shape, {:f, 32}, key)
  end

  defp init_weight(_expected_shape, _key, other) do
    raise ArgumentError, "weight must be an Nx.Tensor or nil, got: #{inspect(other)}"
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
