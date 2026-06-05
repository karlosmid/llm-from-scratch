defmodule LlmScratch.EMLXBackend do
  @compile {:no_warn_undefined, {EMLX, :__info__, 1}}
  @compile {:no_warn_undefined, {EMLX.Backend, :__info__, 1}}

  @moduledoc """
  Experimental EMLX backend selection helpers.

  This module is intentionally separate from the core training/model code. It
  lets tests or scripts opt into EMLX on Apple Silicon without changing the
  existing `llm_scratch` library modules.
  """

  @type context :: %{
          backend: term(),
          compiler: module(),
          previous_backend: term(),
          previous_defn_options: keyword(),
          selected: :emlx | :exla
        }

  @doc """
  Selects EMLX on the Apple GPU for the current process.

  ## Output

  Returns a context map containing the previous Nx backend and `Nx.Defn`
  options. Pass that map to `restore!/1` when the experiment is finished.
  """
  @spec apple_gpu!() :: context()
  def apple_gpu! do
    ensure_loaded!()

    previous_backend = Nx.default_backend()
    previous_defn_options = Nx.Defn.default_options()
    backend = {EMLX.Backend, device: :gpu}

    Nx.default_backend(backend)
    Nx.Defn.default_options(compiler: EMLX)

    %{
      backend: backend,
      compiler: EMLX,
      previous_backend: previous_backend,
      previous_defn_options: previous_defn_options,
      selected: :emlx
    }
  end

  @doc """
  Selects the best available training backend for the current process.

  ## Output

  Returns the same context shape as `apple_gpu!/0`. It selects EMLX on Apple GPU
  when supported, otherwise falls back to `EXLA.Backend`.
  """
  @spec apple_gpu_or_exla!() :: context()
  def apple_gpu_or_exla! do
    if supported?() do
      apple_gpu!()
    else
      previous_backend = Nx.default_backend()
      previous_defn_options = Nx.Defn.default_options()
      backend = EXLA.Backend

      Nx.default_backend(backend)

      %{
        backend: backend,
        compiler: Nx.Defn.default_options()[:compiler],
        previous_backend: previous_backend,
        previous_defn_options: previous_defn_options,
        selected: :exla
      }
    end
  end

  @doc """
  Restores backend settings returned by `apple_gpu!/0`.

  ## Input Parameters

    * `context` - map returned by `apple_gpu!/0`.

  ## Output

  Returns `:ok`.
  """
  @spec restore!(context()) :: :ok
  def restore!(context) do
    Nx.default_backend(context.previous_backend)
    Nx.Defn.default_options(context.previous_defn_options)
    :ok
  end

  @doc """
  Returns whether EMLX modules are available.

  ## Output

  Returns `true` when the EMLX dependency is compiled and loadable, otherwise
  `false`.
  """
  @spec available?() :: boolean()
  def available? do
    Code.ensure_loaded?(EMLX) and Code.ensure_loaded?(EMLX.Backend)
  end

  @doc """
  Returns whether the EMLX Apple GPU backend can run a small tensor operation.

  ## Output

  Returns `true` when EMLX is compiled, the backend can be selected, and a small
  tensor can round-trip through the EMLX backend. Returns `false` when EMLX is
  unavailable or unsupported on the current machine.
  """
  @spec supported?() :: boolean()
  def supported? do
    try do
      context = apple_gpu!()

      try do
        [1.0]
        |> Nx.tensor()
        |> Nx.backend_transfer(context.backend)
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Nx.to_flat_list()

        true
      after
        restore!(context)
      end
    rescue
      _ -> false
    catch
      _, _ -> false
    end
  end

  defp ensure_loaded! do
    unless available?() do
      raise """
      EMLX backend is not available.

      Run:

          mix deps.get
          mix compile
      """
    end
  end
end
