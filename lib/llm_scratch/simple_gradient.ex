defmodule LlmScratch.SimpleGradient do
  @moduledoc """
  Small Nx gradient example matching the scalar PyTorch example from the book.

  Nx tensors do not store gradients in a mutable `.grad` field. The `backward/4`
  function returns the loss and gradients as values.
  """

  import Nx.Defn

  @doc """
  Computes binary cross entropy loss for a one-input logistic unit.

  ## Parameters

    * `w1` - trainable weight tensor.
    * `b` - trainable bias tensor.
    * `x1` - input feature tensor.
    * `y` - target label tensor.

  ## Returns

  A scalar tensor with the mean binary cross entropy between `sigmoid(x1 * w1 + b)`
  and `y`.
  """
  defn loss(w1, b, x1, y) do
    z = x1 * w1 + b
    a = Nx.sigmoid(z)

    Nx.mean(-(y * Nx.log(a) + (1 - y) * Nx.log(1 - a)))
  end

  @doc """
  Computes mean squared error loss.

  This mirrors `nn.MSELoss()` in the PyTorch examples. It is defined with
  `Nx.Defn`, so it can be called from other differentiable functions that use
  `grad/2` or `value_and_grad/2`.

  ## Parameters

    * `output` - predicted values.
    * `target` - expected values with a shape compatible with `output`.

  ## Returns

  A scalar tensor with the mean of `(output - target)^2`.

  ## Example

      iex> output = Nx.tensor([[1.0, 2.0]])
      iex> target = Nx.tensor([[0.0, 1.0]])
      iex> LlmScratch.SimpleGradient.mse_loss(output, target) |> Nx.to_number()
      1.0
  """
  defn mse_loss(output, target) do
    output
    |> Nx.subtract(target)
    |> Nx.pow(2)
    |> Nx.mean()
  end

  @doc """
  Computes the gradient of `loss/4` with respect to `w1`.

  ## Parameters

    * `w1` - trainable weight tensor to differentiate.
    * `b` - trainable bias tensor used to compute the loss.
    * `x1` - input feature tensor used to compute the loss.
    * `y` - target label tensor used to compute the loss.

  ## Returns

  A tensor with the same shape as `w1`, containing `dL/dw1`.
  """
  defn grad_w1(w1, b, x1, y) do
    grad(w1, fn w1 ->
      loss(w1, b, x1, y)
    end)
  end

  @doc """
  Computes the gradient of `loss/4` with respect to `b`.

  ## Parameters

    * `w1` - trainable weight tensor used to compute the loss.
    * `b` - trainable bias tensor to differentiate.
    * `x1` - input feature tensor used to compute the loss.
    * `y` - target label tensor used to compute the loss.

  ## Returns

  A tensor with the same shape as `b`, containing `dL/db`.
  """
  defn grad_b(w1, b, x1, y) do
    grad(b, fn b ->
      loss(w1, b, x1, y)
    end)
  end

  @doc """
  Computes loss and gradients for the trainable parameters `{w1, b}`.

  This is the Nx counterpart to calling `loss.backward()` in the PyTorch example.
  Instead of mutating `w1.grad` and `b.grad`, it returns the gradients.

  ## Parameters

    * `w1` - trainable weight tensor.
    * `b` - trainable bias tensor.
    * `x1` - input feature tensor used to compute the loss.
    * `y` - target label tensor used to compute the loss.

  ## Returns

  `{loss, {grad_w1, grad_b}}`, where `loss` is a scalar tensor and the gradient
  tensors match the shapes of `w1` and `b`.
  """
  defn backward(w1, b, x1, y) do
    value_and_grad({w1, b}, fn {w1, b} ->
      loss(w1, b, x1, y)
    end)
  end
end
