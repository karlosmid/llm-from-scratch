defmodule LlmScratch.InstructionsEvaluation do
  @moduledoc """
  Helpers for evaluating an instruction fine-tuned model on instruction records.
  """

  alias LlmScratch.{FineTuneDataLoader, OllamaUtils, TextGeneration, TextUtils}

  @default_output_path "instruction-data-with-response.json"
  @default_max_new_tokens 256
  @default_eos_id 50_256

  @type instruction_record :: FineTuneDataLoader.instruction_record()
  @type device :: nil | :default | atom() | tuple()

  @doc """
  Generates model responses for instruction records and writes them as JSON.

  Each input record is preserved and extended with a `"model_response"` key.
  The output mirrors the chapter 7 Python loop that writes
  `instruction-data-with-response.json`.

  ## Options

    * `:output_path` - JSON destination. Defaults to
      `"instruction-data-with-response.json"`.
    * `:max_new_tokens` - maximum generated tokens per record. Defaults to
      `256`.
    * `:context_size` - context length passed to `TextGeneration.generate/7`.
      Defaults to `model.cfg.context_length`.
    * `:eos_id` - end-of-text token id. Defaults to `50256`.

  Returns the enriched records after writing the file.

  ## Example

      test_data = [
        %{
          "instruction" => "Say hi.",
          "input" => "",
          "output" => "Hi."
        }
      ]

      LlmScratch.InstructionsEvaluation.write_responses!(
        test_data,
        model,
        "code-davinci-002",
        device,
        output_path: "instruction-data-with-response.json"
      )

  Example output:

      [
        %{
          "input" => "",
          "instruction" => "Say hi.",
          "model_response" => "Hi.",
          "output" => "Hi."
        }
      ]
  """
  @spec write_responses!([instruction_record()], struct(), String.t(), device(), keyword()) ::
          [instruction_record()]
  def write_responses!(test_data, model, tokenizer, device, opts \\ [])
      when is_list(test_data) and is_struct(model) and is_binary(tokenizer) and is_list(opts) do
    output_path = Keyword.get(opts, :output_path, @default_output_path)

    enriched_data = generate_responses(test_data, model, tokenizer, device, opts)

    output_path
    |> Path.dirname()
    |> File.mkdir_p!()

    {:ok, encoded} = Jason.encode(enriched_data, pretty: true)
    File.write!(output_path, encoded)

    enriched_data
  end

  @doc """
  Generates model responses for instruction records.

  Returns the original records with an added `"model_response"` field.

  ## Example

      test_data = [
        %{
          "instruction" => "Name one color.",
          "input" => "",
          "output" => "Blue."
        }
      ]

      LlmScratch.InstructionsEvaluation.generate_responses(
        test_data,
        model,
        "code-davinci-002",
        device
      )

  Example output:

      [
        %{
          "input" => "",
          "instruction" => "Name one color.",
          "model_response" => "Blue.",
          "output" => "Blue."
        }
      ]
  """
  @spec generate_responses([instruction_record()], struct(), String.t(), device(), keyword()) ::
          [instruction_record()]
  def generate_responses(test_data, model, tokenizer, device, opts \\ [])
      when is_list(test_data) and is_struct(model) and is_binary(tokenizer) and is_list(opts) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, @default_max_new_tokens)
    context_size = Keyword.get_lazy(opts, :context_size, fn -> model.cfg.context_length end)
    eos_id = Keyword.get(opts, :eos_id, @default_eos_id)

    Enum.with_index(test_data, fn entry, index ->
      if Keyword.get(opts, :progress, false) do
        IO.puts("Generating response #{index + 1}/#{length(test_data)}")
      end

      Map.put(
        entry,
        "model_response",
        generate_response(model, entry, tokenizer, device, max_new_tokens, context_size, eos_id)
      )
    end)
  end

  @doc """
  Scores generated model responses with a local Ollama model.

  `json_key` identifies the generated response field in each instruction
  record, for example `"model_response"`. Entries whose score cannot be
  parsed as an integer are skipped, matching the chapter 7 evaluation loop.
  """
  @spec generate_model_scores([instruction_record()], String.t(), String.t()) :: [integer()]
  def generate_model_scores(json_data, json_key, model \\ "llama3")
      when is_list(json_data) and is_binary(json_key) and is_binary(model) do
    Enum.reduce(json_data, [], fn entry, scores ->
      prompt =
        "Given the input `#{FineTuneDataLoader.format_input(entry)}` " <>
          "and correct output `#{entry["output"]}`, " <>
          "score the model response `#{entry[json_key]}`" <>
          " on a scale from 0 to 100, where 100 is the best score. " <>
          "Respond with the integer number only."

      score = OllamaUtils.query_model(prompt, model)

      try do
        [String.to_integer(String.trim(score)) | scores]
      rescue
        ArgumentError ->
          IO.puts("Could not convert score: #{score}")
          scores
      end
    end)
    |> Enum.reverse()
  end

  defp generate_response(model, entry, tokenizer, device, max_new_tokens, context_size, eos_id) do
    input_text = FineTuneDataLoader.format_input(entry)

    token_ids =
      TextGeneration.generate(
        model,
        input_text
        |> TextUtils.text_to_token_ids(tokenizer)
        |> maybe_transfer(device),
        max_new_tokens,
        context_size,
        0.0,
        nil,
        eos_id
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    token_ids
    |> TextUtils.token_ids_to_text(tokenizer)
    |> String.slice(String.length(input_text)..-1//1)
    |> String.replace("### Response:", "")
    |> String.trim()
  end

  defp maybe_transfer(tensor, nil), do: tensor
  defp maybe_transfer(tensor, :default), do: tensor
  defp maybe_transfer(tensor, device), do: Nx.backend_transfer(tensor, device)
end
