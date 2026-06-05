defmodule LlmScratch.EMLXBackendTest do
  use ExUnit.Case

  import LlmScratch.TestHelpers

  alias LlmScratch.{EMLXBackend, GPTModel, LossUtils, Training}

  defmodule DefnSmoke do
    import Nx.Defn

    defn matmul_plus_bias(left, right, bias) do
      left
      |> Nx.dot(right)
      |> Nx.add(bias)
    end
  end

  @tag :emlx
  test "selects EMLX Apple GPU backend when available" do
    if EMLXBackend.supported?() do
      context = EMLXBackend.apple_gpu!()

      try do
        result =
          [1.0, 2.0, 3.0]
          |> Nx.tensor()
          |> Nx.multiply(2.0)
          |> Nx.add(1.0)
          |> Nx.backend_transfer(Nx.BinaryBackend)
          |> Nx.to_flat_list()

        assert context.backend == {EMLX.Backend, device: :gpu}
        assert context.compiler == EMLX
        assert result == [3.0, 5.0, 7.0]
      after
        EMLXBackend.restore!(context)
      end
    else
      IO.puts("Skipping EMLX backend smoke test because EMLX is not supported.")
      assert true
    end
  end

  @tag :emlx
  test "runs Nx.Defn matrix operations on EMLX Apple GPU backend when available" do
    if EMLXBackend.supported?() do
      context = EMLXBackend.apple_gpu!()

      try do
        left = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = Nx.tensor([[2.0, 0.0], [1.0, 2.0]])
        bias = Nx.tensor([0.5, -0.5])

        result =
          left
          |> DefnSmoke.matmul_plus_bias(right, bias)
          |> Nx.backend_transfer(Nx.BinaryBackend)
          |> Nx.to_list()

        assert Nx.Defn.default_options()[:compiler] == EMLX
        assert result == [[4.5, 3.5], [10.5, 7.5]]
      after
        EMLXBackend.restore!(context)
      end
    else
      IO.puts("Skipping EMLX defn smoke test because EMLX is not supported.")
      assert true
    end
  end

  @tag :emlx
  test "runs tiny GPT forward loss and full gradients on EMLX Apple GPU backend when available" do
    if EMLXBackend.supported?() do
      cfg = tiny_gpt_config()
      model = GPTModel.new(cfg, seed: 123) |> Nx.backend_transfer(Nx.BinaryBackend)
      input_batch = tiny_input_batch()
      target_batch = tiny_target_batch()
      key = Nx.Random.key(123) |> Nx.backend_transfer(Nx.BinaryBackend)

      context = EMLXBackend.apple_gpu!()

      try do
        backend = context.backend
        model_emlx = Nx.backend_transfer(model, backend)
        input_batch_emlx = Nx.backend_transfer(input_batch, backend)
        target_batch_emlx = Nx.backend_transfer(target_batch, backend)

        logits = GPTModel.forward(model_emlx, input_batch_emlx)
        loss = LossUtils.cross_entropy_loss(logits, target_batch_emlx)

        {grad_loss, gradients, _key} =
          Training.loss_and_grad(model, input_batch, target_batch, key)

        assert Nx.shape(logits) == {2, 4, 32}
        assert %GPTModel{} = gradients

        loss_value = loss |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number()
        grad_loss_value = grad_loss |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number()

        assert is_float(loss_value)
        assert is_float(grad_loss_value)
        assert loss_value > 0.0
        assert grad_loss_value > 0.0
      after
        EMLXBackend.restore!(context)
      end
    else
      IO.puts("Skipping EMLX GPT smoke test because EMLX is not supported.")
      assert true
    end
  end
end
