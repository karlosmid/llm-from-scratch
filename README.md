# LLM Scratch

A basic Elixir application with OTP support demonstrating GenServer patterns and supervision trees.

## Features

- **OTP Application**: Properly configured Elixir application with supervision tree
- **GenServer Worker**: Example worker process demonstrating OTP patterns
- **Supervision**: Automatic process supervision and restart capabilities
- **Configuration**: Environment-specific configuration files

## Project Structure

```
lib/
├── llm_scratch/
│   ├── application.ex    # OTP Application module with supervision tree
│   └── worker.ex         # GenServer worker process
config/
├── config.exs           # Main configuration
├── dev.exs              # Development configuration
├── prod.exs             # Production configuration
└── test.exs             # Test configuration
mix.exs                  # Project dependencies and configuration
```

## Getting Started

### Prerequisites

- Elixir 1.15 or later
- Mix build tool

### Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   mix deps.get
   ```

### Running the Application

Start the application in the foreground:
```bash
mix run --no-halt
```

Or start an interactive Elixir shell with the application loaded:
```bash
iex -S mix
```

## Usage Examples

Once the application is running, you can interact with the worker process:

```elixir
# Get the current state
LlmScratch.Worker.get_state(pid)

# Update the state
LlmScratch.Worker.update_state(pid, 42)

# Send an asynchronous message
LlmScratch.Worker.send_message(pid, "Hello from OTP!")

# Send a tick message (demonstrates handle_info)
send(pid, :tick)
```

## OTP Concepts Demonstrated

1. **Application Module**: `LlmScratch.Application` defines the supervision tree
2. **GenServer**: `LlmScratch.Worker` implements a stateful server process
3. **Supervision**: Automatic process monitoring and restart on failure
4. **Call/Cast/Info**: Different message handling patterns in GenServer
5. **Process Lifecycle**: Proper initialization and termination handling

## Development

### Running Tests

```bash
mix test
```

### Compiling

```bash
mix compile
```

### Code Formatting

```bash
mix format
```

## Configuration

The application uses environment-specific configuration files:
- `config/dev.exs` - Development settings
- `config/prod.exs` - Production settings  
- `config/test.exs` - Test settings

## License

This project is open source and available under the [MIT License](LICENSE).
