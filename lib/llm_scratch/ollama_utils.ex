defmodule LlmScratch.OllamaUtils do
  @moduledoc """
  Utilities for working with a local Ollama runtime.
  """

  @llama_server_process "ollama"
  @default_model "llama3"
  @default_chat_url "http://localhost:11434/api/chat"

  @doc """
  Returns whether Ollama's `llama-server` process is running.

  The check first uses `pgrep -x llama-server`, then falls back to scanning
  process command names with `ps` when `pgrep` is unavailable.
  """
  @spec ollama_running?() :: boolean()
  def ollama_running? do
    case System.find_executable("pgrep") do
      nil -> ps_process_running?(@llama_server_process)
      _pgrep -> pgrep_process_running?(@llama_server_process)
    end
  end

  @doc """
  Sends a prompt to Ollama's local chat API and returns the generated text.

  Ollama's standard local chat endpoint is
  `http://localhost:11434/api/chat`. The response is streamed as
  newline-delimited JSON; this function concatenates each
  `"message"."content"` chunk into one string.
  """
  @spec query_model(String.t(), String.t(), String.t()) :: String.t()
  def query_model(prompt, model \\ @default_model, url \\ @default_chat_url)
      when is_binary(prompt) and is_binary(model) and is_binary(url) do
    request_body = %{
      model: model,
      messages: [
        %{role: "user", content: prompt}
      ],
      options: %{
        seed: 123,
        temperature: 0,
        num_ctx: 2048
      }
    }

    response = Req.post!(url, json: request_body, receive_timeout: :infinity)

    case response.status do
      200 ->
        parse_chat_response(response.body)

      status ->
        raise "Ollama chat request failed with status #{status}: #{inspect(response.body)}"
    end
  end

  defp pgrep_process_running?(process_name) do
    case System.cmd("pgrep", ["-x", process_name], stderr_to_stdout: true) do
      {_output, 0} -> true
      {_output, _status} -> false
    end
  end

  defp ps_process_running?(process_name) do
    case System.cmd("ps", ["-axo", "comm"], stderr_to_stdout: true) do
      {output, 0} ->
        output
        |> String.split("\n", trim: true)
        |> Enum.any?(&(Path.basename(&1) == process_name))

      {_output, _status} ->
        false
    end
  end

  defp parse_chat_response(body) when is_binary(body) do
    body
    |> String.split("\n", trim: true)
    |> Enum.map(&decode_chat_chunk!/1)
    |> Enum.join()
  end

  defp decode_chat_chunk!(line) do
    line
    |> Jason.decode!()
    |> get_in(["message", "content"])
    |> Kernel.||("")
  end
end
