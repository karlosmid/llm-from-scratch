defmodule LlmScratch.SimpleTokenizerV1 do
  @moduledoc """
  A simple tokenizer that tokenizes text into words.
  """
  def vocab_from_file(filename) do
    {:ok, file_content} = File.read(filename)

    tokenize(file_content)
    |> MapSet.new()
    |> Enum.sort()
    |> Enum.with_index()
  end

  def encode(text, vocab) do
    tokenize(text)
    |> Enum.map(fn token ->
      Enum.find(vocab, fn {vocab_token, _} -> token == vocab_token end)
      |> case do
        {_, id} -> id
        nil -> raise "Token not found in vocab: #{token}"
      end
    end)
  end

  def decode(tokens, vocab) do
    text =
      Enum.map(tokens, fn token ->
        Enum.find(vocab, fn {_, id} -> id == token end)
        |> case do
          {token, _} -> token
          nil -> raise "Token not found in vocab: #{token}"
        end
      end)
      |> Enum.join(" ")

    Regex.replace(~r/\s+([,.?!"()\'])/, text, "\\1")
  end

  defp tokenize(text) do
    Regex.split(~r/([,.:;?_!"()\']|--|\s)/, text, include_captures: true, trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
  end
end
