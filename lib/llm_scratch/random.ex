defmodule LlmScratch.Random do
  @moduledoc """
  Random number generation utilities, similar to PyTorch's `torch.manual_seed()`.

  Provides reproducible random number generation for neural network initialization.

  ## Example

      # PyTorch equivalent:
      # torch.manual_seed(42)
      
      key = LlmScratch.Random.manual_seed(42)
      # Use the key for random operations
  """

  @doc """
  Sets the random seed for reproducible initialization.

  Equivalent to PyTorch's `torch.manual_seed(seed)`.

  Returns a random key that can be used with Nx random functions.

  ## Examples

      iex> key1 = LlmScratch.Random.manual_seed(42)
      iex> key2 = LlmScratch.Random.manual_seed(42)
      # Same seed produces the same key
      iex> key1 == key2
      true
      
      iex> key1 = LlmScratch.Random.manual_seed(42)
      iex> key2 = LlmScratch.Random.manual_seed(43)
      # Different seeds produce different keys
      iex> key1 != key2
      true
  """
  def manual_seed(seed) when is_integer(seed) do
    Nx.Random.key(seed)
  end

  def manual_seed(seed) do
    raise ArgumentError, "seed must be an integer, got: #{inspect(seed)}"
  end

  @doc """
  Generates random uniform values in the range [min, max).

  Similar to PyTorch's `torch.rand()` or `torch.uniform()`.

  Returns `{new_key, tensor}` tuple where `new_key` is the updated random key.

  ## Examples

      iex> key = LlmScratch.Random.manual_seed(42)
      iex> {new_key, tensor} = LlmScratch.Random.uniform(key, 0.0, 1.0, shape: {2, 3})
      iex> Nx.shape(tensor)
      {2, 3}
  """
  def uniform(key, min, max, opts \\ []) do
    shape = Keyword.fetch!(opts, :shape)
    type = Keyword.get(opts, :type, {:f, 32})

    # Try to use Nx.Random.uniform if available
    try do
      if function_exported?(Nx.Random, :uniform, 3) do
        Nx.Random.uniform(key, min, max, shape: shape, type: type)
      else
        # Fallback: use the key directly with Nx functions
        # This might not work in all Nx versions, but we try
        raise "Nx.Random.uniform/3 not available in this Nx version"
      end
    rescue
      _ ->
        # If random functions aren't available, we can't generate random values
        # This is a limitation - the user would need to provide their own initialization
        raise """
        Random number generation not available in this Nx version.
        Please initialize embedding weights manually using the :weight option:

        embedding = LlmScratch.Embedding.new(vocab_size, embedding_dim, 
          weight: your_weight_tensor)
        """
    end
  end

  @doc """
  Generates random normal (Gaussian) values.

  Similar to PyTorch's `torch.randn()`.

  Returns `{new_key, tensor}` tuple where `new_key` is the updated random key.

  ## Examples

      iex> key = LlmScratch.Random.manual_seed(42)
      iex> {new_key, tensor} = LlmScratch.Random.normal(key, 0.0, 1.0, shape: {2, 3})
      iex> Nx.shape(tensor)
      {2, 3}
  """
  def normal(key, mean, stddev, opts \\ []) do
    shape = Keyword.fetch!(opts, :shape)
    type = Keyword.get(opts, :type, {:f, 32})

    # Try to use Nx.Random.normal if available
    try do
      if function_exported?(Nx.Random, :normal, 3) do
        Nx.Random.normal(key, mean, stddev, shape: shape, type: type)
      else
        raise "Nx.Random.normal/3 not available in this Nx version"
      end
    rescue
      _ ->
        raise """
        Random number generation not available in this Nx version.
        Please initialize embedding weights manually using the :weight option.
        """
    end
  end
end
