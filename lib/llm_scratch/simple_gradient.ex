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
