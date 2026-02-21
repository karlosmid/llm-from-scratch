defmodule LlmScratch.EmbeddingNative do
  @moduledoc """
  Native embedding layer implemented with Nx and Axon.

  Provides a small API compatible with `LlmScratch.Embedding`:

    * `new/3`
    * `weight/1`
    * `forward/2`
    * `call/2`
  """

  defstruct [:weight, :vocab_size, :embedding_dim, :seed]

  @type t :: %__MODULE__{
          weight: Nx.Tensor.t(),
          vocab_size: pos_integer(),
          embedding_dim: pos_integer(),
          seed: integer() | nil
        }

  @spec new(pos_integer(), pos_integer(), keyword()) :: t()
  @doc """
  Creates a new embedding layer.

  ## Options

    * `:seed` - integer seed used to initialize weights deterministically.
    * `:weight` - optional pre-initialized weight tensor with shape
      `{vocab_size, embedding_dim}`.
  """
  def new(vocab_size, embedding_dim, opts \\ [])
      when is_integer(vocab_size) and vocab_size > 0 and is_integer(embedding_dim) and
             embedding_dim > 0 do
    seed = Keyword.get(opts, :seed)

    weight =
      case Keyword.get(opts, :weight) do
        %Nx.Tensor{} = provided_weight ->
          expected_shape = {vocab_size, embedding_dim}

          if Nx.shape(provided_weight) != expected_shape do
            raise ArgumentError,
                  "expected :weight shape #{inspect(expected_shape)}, got: #{inspect(Nx.shape(provided_weight))}"
          end

          Nx.as_type(provided_weight, {:f, 32})

        nil ->
          init_weight(vocab_size, embedding_dim, seed)
      end

    %__MODULE__{
      weight: weight,
      vocab_size: vocab_size,
      embedding_dim: embedding_dim,
      seed: seed
    }
  end

  @spec weight(t()) :: Nx.Tensor.t()
  @doc """
  Returns the embedding weight matrix.

  The returned tensor has shape `{vocab_size, embedding_dim}`.
  """
  def weight(%__MODULE__{weight: weight}), do: weight

  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Maps token ids to embedding vectors.

  `token_ids` is cast to `{:s, 64}` and used as row indices into the
  embedding matrix.
  """
  def forward(%__MODULE__{weight: weight}, token_ids) do
    token_ids
    |> Nx.as_type({:s, 64})
    |> then(&Nx.take(weight, &1, axis: 0))
  end

  @spec call(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  @doc """
  Alias for `forward/2`.
  """
  def call(embedding, token_ids), do: forward(embedding, token_ids)

  defp init_weight(vocab_size, embedding_dim, seed) do
    key =
      case seed do
        nil -> LlmScratch.Random.manual_seed(System.unique_integer([:positive]))
        int when is_integer(int) -> LlmScratch.Random.manual_seed(int)
        other -> raise ArgumentError, "seed must be an integer or nil, got: #{inspect(other)}"
      end

    Axon.Initializers.uniform(scale: 1.0).({vocab_size, embedding_dim}, {:f, 32}, key)
  end
end
