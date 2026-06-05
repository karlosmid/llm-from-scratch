defmodule LlmScratch.MixProject do
  use Mix.Project

  def project do
    [
      app: :llm_scratch,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {LlmScratch.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:tiktoken, "~> 0.4.1"},
      {:pythonx, "~> 0.4.0"},
      {:nx, "~> 0.6"},
      {:exla, "~> 0.6"},
      {:emlx, github: "elixir-nx/emlx", branch: "main", subdir: "emlx", only: [:dev, :test]},
      {:axon, "~> 0.6"},
      {:req, "~> 0.4.0"},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false}
    ]
  end
end
