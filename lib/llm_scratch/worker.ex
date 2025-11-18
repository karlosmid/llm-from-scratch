defmodule LlmScratch.Worker do
  @moduledoc """
  A simple GenServer worker process that demonstrates OTP patterns.
  """

  use GenServer

  # Client API

  @doc """
  Starts the worker process.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, :ok, opts)
  end

  @doc """
  Gets the current state of the worker.
  """
  def get_state(pid) do
    GenServer.call(pid, :get_state)
  end

  @doc """
  Updates the worker's state with a new value.
  """
  def update_state(pid, value) do
    GenServer.call(pid, {:update_state, value})
  end

  @doc """
  Sends an asynchronous message to the worker.
  """
  def send_message(pid, message) do
    GenServer.cast(pid, {:message, message})
  end

  # Server callbacks

  @impl true
  def init(:ok) do
    state = %{
      value: 0,
      messages: [],
      started_at: DateTime.utc_now()
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call({:update_state, value}, _from, state) do
    new_state = %{state | value: value}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_cast({:message, message}, state) do
    new_messages = [message | state.messages]
    new_state = %{state | messages: new_messages}
    {:noreply, new_state}
  end

  @impl true
  def handle_info(:tick, state) do
    # This demonstrates handling of arbitrary messages
    new_value = state.value + 1
    new_state = %{state | value: new_value}
    {:noreply, new_state}
  end

  @impl true
  def terminate(reason, _state) do
    IO.puts("Worker terminating with reason: #{inspect(reason)}")
    :ok
  end
end
