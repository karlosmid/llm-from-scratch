# This file is responsible for configuring your application
# and its dependencies with the aid of the Config module.
#
# This configuration file is loaded before any dependency and
# is restricted to this project.

# General application configuration
import Config

config :llm_scratch,
  ecto_repos: []

# Configures Elixir's Logger
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id]

# Force local build of tiktoken Rust NIF (instead of using precompiled binaries)
# This allows you to compile local changes to the Rust code
config :rustler_precompiled, :force_build, tiktoken: true

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"
