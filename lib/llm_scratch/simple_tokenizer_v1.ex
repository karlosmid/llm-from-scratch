defmodule LlmScratch.SimpleTokenizerV1 do
  @moduledoc """
  A simple tokenizer that tokenizes text into words.
  """
  def vocab_from_file(filename, special_tokens \\ []) do
    {:ok, file_content} = File.read(filename)

    tokens =
      tokenize(file_content)
      |> MapSet.new()
      |> Enum.sort()

    (tokens ++ special_tokens)
    |> Enum.with_index()
  end

  def encode(text, vocab) do
    tokenize(text)
    |> Enum.map(fn token ->
      Enum.find(vocab, fn {vocab_token, _} -> token == vocab_token end)
      |> case do
        {_, id} -> id
        nil -> find_unknown_token(vocab, token)
      end
    end)
  end

  defp find_unknown_token(vocab, token) do
    Enum.find(vocab, fn {vocab_token, _} -> vocab_token == "<|unk|>" end)
    |> case do
      {_, id} -> id
      nil -> raise "Token not found in vocab: #{token}"
    end
  end

  def decode(ids, vocab) do
    text =
      Enum.map(ids, fn id ->
        Enum.find(vocab, fn {_, vocab_id} -> id == vocab_id end)
        |> case do
          {vocab_token, _} -> vocab_token
          nil -> raise "ID not found in vocab: #{id}"
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
