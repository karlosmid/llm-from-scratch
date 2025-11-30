defmodule LlmScratch.Embedding do
  @moduledoc """
  PyTorch embedding layer wrapper using Pythonx.

  This module wraps PyTorch's `torch.nn.Embedding` to ensure exact compatibility
  with PyTorch's behavior and initialization. All operations are delegated to PyTorch
  through Pythonx, ensuring identical results.

  ## Example
      
      # PyTorch equivalent:
      # embedding = torch.nn.Embedding(vocab_size=1000, embedding_dim=256)
      # embeddings = embedding(token_ids)
      
      embedding = LlmScratch.Embedding.new(1000, 256, seed: 42)
      token_ids = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      embeddings = LlmScratch.Embedding.forward(embedding, token_ids)
  """

  defstruct [:pytorch_embedding, :vocab_size, :embedding_dim, :seed]

  @doc """
  Creates a new embedding layer using PyTorch, similar to `torch.nn.Embedding(vocab_size, embedding_dim)`.

  This uses Pythonx to create a PyTorch embedding layer, ensuring exact compatibility.

  ## Options

    * `:seed` - Random seed for weight initialization (default: `nil`)

  ## Examples

      iex> embedding = LlmScratch.Embedding.new(1000, 256, seed: 42)
      iex> embedding.vocab_size
      1000
      iex> embedding.embedding_dim
      256
  """
  def new(vocab_size, embedding_dim, opts \\ []) do
    seed = Keyword.get(opts, :seed)

    # Create PyTorch embedding layer using Pythonx
    python_code = """
    import torch

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Create embedding layer
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

    # Return the embedding layer object (Pythonx will handle serialization)
    embedding_layer
    """

    globals = %{
      "vocab_size" => vocab_size,
      "embedding_dim" => embedding_dim,
      "seed" => seed
    }

    {pytorch_embedding, _new_globals} = Pythonx.eval(python_code, globals)

    %__MODULE__{
      pytorch_embedding: pytorch_embedding,
      vocab_size: vocab_size,
      embedding_dim: embedding_dim,
      seed: seed
    }
  end

  @doc """
  Returns the weight matrix of the embedding layer.

  Equivalent to PyTorch's `embedding_layer.weight`.

  The weight matrix has shape `{vocab_size, embedding_dim}` and contains
  the embedding vectors for all vocabulary tokens.

  ## Examples

      iex> embedding = LlmScratch.Embedding.new(10, 5, seed: 42)
      iex> weight = LlmScratch.Embedding.weight(embedding)
      iex> Nx.shape(weight)
      {10, 5}
  """
  def weight(%__MODULE__{pytorch_embedding: pytorch_embedding}) do
    # Get weight from PyTorch embedding layer
    python_code = """
    import numpy as np

    # Get weight and convert to numpy array, then to list
    weight = embedding_layer.weight.detach().numpy().tolist()
    weight
    """

    globals = %{"embedding_layer" => pytorch_embedding}

    {result_obj, _new_globals} = Pythonx.eval(python_code, globals)
    weight_list = Pythonx.decode(result_obj)

    # Convert Python list to Nx tensor
    Nx.tensor(weight_list, type: {:f, 32})
  end

  @doc """
  Forward pass: converts token IDs to embeddings.

  Equivalent to calling the embedding layer in PyTorch: `embedding(token_ids)`

  ## Examples

      iex> embedding = LlmScratch.Embedding.new(10, 5, seed: 42)
      iex> token_ids = Nx.tensor([[1, 2, 3]])
      iex> embeddings = LlmScratch.Embedding.forward(embedding, token_ids)
      iex> Nx.shape(embeddings)
      {1, 3, 5}
  """
  def forward(%__MODULE__{pytorch_embedding: pytorch_embedding}, token_ids) do
    # Convert Nx tensor to Python list
    token_ids_list = Nx.to_list(token_ids)

    # Run forward pass in PyTorch
    python_code = """
    import torch
    import numpy as np

    # Convert token_ids to PyTorch tensor
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

    # Forward pass
    embeddings = embedding_layer(token_ids_tensor)

    # Convert to numpy array, then to list
    embeddings_np = embeddings.detach().numpy()
    embeddings_list = embeddings_np.tolist()
    embeddings_list
    """

    globals = %{
      "embedding_layer" => pytorch_embedding,
      "token_ids" => token_ids_list
    }

    {result_obj, _new_globals} = Pythonx.eval(python_code, globals)
    embeddings_list = Pythonx.decode(result_obj)

    # Convert Python list to Nx tensor
    Nx.tensor(embeddings_list, type: {:f, 32})
  end

  @doc """
  Alias for `forward/2` to match PyTorch's callable behavior.

  ## Examples

      iex> embedding = LlmScratch.Embedding.new(10, 5, seed: 42)
      iex> token_ids = Nx.tensor([[1, 2, 3]])
      iex> embeddings = embedding.(token_ids)
      iex> Nx.shape(embeddings)
      {1, 3, 5}
  """
  def call(embedding, token_ids) do
    forward(embedding, token_ids)
  end

  # Make the struct callable like PyTorch
  defimpl Inspect do
    def inspect(embedding, _opts) do
      "LlmScratch.Embedding(vocab_size=#{embedding.vocab_size}, embedding_dim=#{embedding.embedding_dim}, seed=#{inspect(embedding.seed)})"
    end
  end
end
