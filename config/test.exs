import Config

# Print only warnings and errors during test
config :logger, level: :warning

# Configure Pythonx for tests
config :pythonx, :uv_init,
  pyproject_toml: """
  [project]
  name = "llm_scratch_test"
  version = "0.1.0"
  requires-python = ">=3.11,<3.13"
  dependencies = [
    "torch",
    "numpy",
    "tensorflow>=2.15.0",
    "tqdm>=4.66"
  ]
  """
