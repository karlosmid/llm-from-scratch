defmodule LlmScratch.Tokenizer do
  @moduledoc """
  A module demonstrating tiktoken usage for text tokenization.
  """

  @doc """
  Encodes text using the GPT-4 tokenizer and returns the token count and tokens.
  """
  def encode_text(text) do
    # Encode the text to get token IDs
    {:ok, tokens} = Tiktoken.encode("gpt-4", text)

    # Get the token count
    token_count = length(tokens)

    %{
      text: text,
      tokens: tokens,
      token_count: token_count,
      encoding: "gpt-4"
    }
  end

  @doc """
  Decodes token IDs back to text.
  """
  def decode_tokens(tokens) do
    {:ok, text} = Tiktoken.decode("gpt-4", tokens)
    text
  end

  @doc """
  Counts tokens in text without returning the actual tokens.
  """
  def count_tokens(text) do
    {:ok, count} = Tiktoken.count_tokens("gpt-4", text)
    count
  end

  @doc """
  Demonstrates different encoding models available.
  """
  def available_models do
    [
      # GPT-4 models
      "gpt-4",
      # GPT-3.5-turbo
      "gpt-3.5-turbo",
      # GPT-3 models
      "text-davinci-003",
      # Codex models
      "code-davinci-002"
    ]
  end

  @doc """
  Simple example of encoding text with tiktoken.
  """
  def simple_encode_example do
    text = "Hello, world! This is a tiktoken example."

    # Method 1: Using encode directly
    {:ok, tokens} = Tiktoken.encode("gpt-4", text)

    # Method 2: Using count_tokens directly
    {:ok, token_count} = Tiktoken.count_tokens("gpt-4", text)

    %{
      original_text: text,
      tokens: tokens,
      token_count: token_count,
      decoded_text: decode_tokens(tokens)
    }
  end
end
