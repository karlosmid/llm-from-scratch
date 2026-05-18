defmodule LlmFromScratch5Test do
  use ExUnit.Case

  alias LlmScratch.{GPTConfig, GPTModel, LossUtils, TextGeneration, TextUtils}

  test "5.1.1 generate_text_simple generates text from a GPT-124M start context" do
    # set EXLA for faster computing

    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    # we set context_length to 256 so we can train this model in this century on m3 chip

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 256,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.0,
      qkv_bias: false
    }

    start_context = "Every effort moves you"
    encoded_tensor = TextUtils.text_to_token_ids(start_context, "code-davinci-002")
    model = GPTModel.new(gpt_config_124m, seed: 123)

    out =
      TextGeneration.generate_text_simple(
        model,
        encoded_tensor,
        6,
        gpt_config_124m.context_length
      )

    output_tokens = Nx.to_flat_list(out)
    decoded_text = TextUtils.token_ids_to_text(out, "code-davinci-002")

    assert Nx.to_flat_list(encoded_tensor) == [6109, 3626, 6100, 345]
    assert Nx.shape(encoded_tensor) == {1, 4}
    assert output_tokens == [6109, 3626, 6100, 345, 16244, 42570, 46784, 9329, 30840, 27005]
    assert Nx.shape(out) == {1, 10}
    assert length(output_tokens) == 10
    assert decoded_text == "Every effort moves youdevelopclimategra Lind Healthy Dread"
  end

  test "5.1.2 calculating the text generation loss" do
    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    gpt_config_124m = %GPTConfig{
      vocab_size: 50_257,
      context_length: 256,
      emb_dim: 768,
      n_heads: 12,
      n_layers: 12,
      drop_rate: 0.0,
      qkv_bias: false
    }

    input_texts = ["every effort moves", "I really like"]
    target_texts = [" effort moves you", " really like chocolate"]

    inputs = TextUtils.texts_to_token_ids(input_texts, "code-davinci-002")
    targets = TextUtils.texts_to_token_ids(target_texts, "code-davinci-002")

    # initialize our gpt model
    model = GPTModel.new(gpt_config_124m, seed: 123)
    # predict next input token
    logits = GPTModel.forward(model, inputs)
    # using softmax get probability values (this is redundant step)
    probas = Axon.Activations.softmax(logits, axis: -1)
    # get token ids that have maximal logit values
    token_ids = Nx.argmax(probas, axis: -1, keep_axis: true)

    assert Nx.to_list(inputs) == [
             [16_833, 3626, 6100],
             [40, 1107, 588]
           ]

    assert Nx.to_list(targets) == [
             [3626, 6100, 345],
             [1107, 588, 11_311]
           ]

    assert Nx.shape(logits) == {2, 3, 50_257}
    assert Nx.shape(targets) == {2, 3}

    assert Nx.to_list(token_ids) == [
             [[40_524], [8520], [4436]],
             [[22_666], [2829], [5387]]
           ]

    # targets what model should predict
    assert TextUtils.token_ids_to_text(targets[0] |> Nx.new_axis(0), "code-davinci-002") ==
             " effort moves you"

    #what model actually predicted
    assert TextUtils.token_ids_to_text(
             token_ids[0] |> Nx.flatten() |> Nx.new_axis(0),
             "code-davinci-002"
           ) ==
             " Clarksonerved hospital"

    #probabilities that model generated for targets
    target_probas = LossUtils.target_token_probas(probas, targets)

    assert_close(
      target_probas,
      Nx.tensor([
        2.2307187e-5,
        2.2980759e-5,
        2.1722459e-5,
        2.1338150e-5,
        1.7680910e-5,
        1.4839508e-5
      ]),
      atol: 1.0e-10
    )

    #we calculate loss as difference what model actually predicted for target tokens
    #this loss should be minimized
    #we have big loss result, and that is expected as we did not train the model
    #goal is to have loss close to zero
    loss = LossUtils.cross_entropy_loss(logits, targets)
    assert_close(loss, Nx.tensor(10.824145), atol: 1.0e-6)
  end

  defp assert_close(actual, expected, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-6)

    assert Nx.all_close(actual, expected, atol: atol) |> Nx.to_number() == 1
  end
end
