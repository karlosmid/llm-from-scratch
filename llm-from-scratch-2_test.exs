defmodule LlmFromScratch2Test do
  use ExUnit.Case
  
  # Helper function to decode a list of tokens to their corresponding text pieces
    defp decode_token_pieces(model, tokens) do
      Enum.map(tokens, fn token ->
        {:ok, text_piece} = Tiktoken.decode(model, [token])
        text_piece
      end)
    end
  setup do
    model = "gpt-4"
    special_token = "<|endoftext|>"
    %{model: model, special_token: special_token}
  end

  test "encode and decode text with special token preserves original text", %{model: model, special_token: special_token} do
    test_text =
      "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

    # Encode the text with the special token allowed
    {:ok, encoded_tokens} = Tiktoken.encode(model, test_text, [special_token])

    # Assert on encoded tokens: must be a list of integers
    assert is_list(encoded_tokens)
    assert Enum.all?(encoded_tokens, &is_integer/1)

    # Assert on exact length of encoded_tokens
    assert length(encoded_tokens) == 20

    # Assert on exact content of encoded_tokens list
    expected_tokens = [
      9906,
      11,
      656,
      499,
      1093,
      15600,
      30,
      220,
      100_257,
      763,
      279,
      7160,
      32735,
      7317,
      2492,
      315,
      1063,
      16476,
      17826,
      13
    ]

    assert encoded_tokens == expected_tokens

    # Decode the tokens back to text
    {:ok, decoded_text} = Tiktoken.decode(model, encoded_tokens)

    # Compare that we got the same result
    assert decoded_text == test_text,
           "Decoded text does not match original.\nOriginal: #{inspect(test_text)}\nDecoded: #{inspect(decoded_text)}"
  end

  test "different tokens", %{model: model} do
    test_text = "do do"
    {:ok, encoded_tokens} = Tiktoken.encode(model, test_text)
    assert encoded_tokens == [3055, 656]
    

    # Use the helper function in the test
    decoded_pieces = decode_token_pieces(model, encoded_tokens)
    assert decoded_pieces == ["do", " do"]
    # Decode each token one by one and collect the decoded pieces
    decoded_pieces =
      Enum.map(encoded_tokens, fn token ->
        {:ok, text_piece} = Tiktoken.decode(model, [token])
        text_piece
      end)

    assert decoded_pieces == ["do", " do"]
  end

  test "endcoding with special tokens when not allowed does not raise an error", %{model: model, special_token: special_token} do
    # text contains a special token format "<|endoftext|>"
    test_text = "secret <|endoftext|> here"

    # Without allowing special token, it gets encoded as regular text (broken into multiple tokens)
    {:ok, encoded_without} = Tiktoken.encode(model, test_text)
    assert encoded_without == [21_107, 83_739, 8862, 728, 428, 91, 29, 1618]
    decoded_pieces = decode_token_pieces(model, encoded_without)
    assert decoded_pieces == ["secret", " <|", "endo", "ft", "ext", "|", ">", " here"]

    # With allowing special token, it gets encoded as a single special token
    {:ok, encoded_with} = Tiktoken.encode(model, test_text, [special_token])
    assert encoded_with == [21_107, 220, 100_257, 1618]

    # Verify the decoded pieces when special token is allowed
    decoded_pieces = decode_token_pieces(model, encoded_with)

    assert decoded_pieces == ["secret", " ", "<|endoftext|>", " here"]
  end
end
