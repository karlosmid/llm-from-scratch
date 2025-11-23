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

  test "the-verdict.txt character count and exact line 99 content" do
    # Read the file content as a single string
    filename = "the-verdict.txt"
    {:ok, file_content} = File.read(filename)

    # Assert on the number of characters in the file
    char_count = String.length(file_content)
    assert char_count == 20_479
    first_99_chars = String.slice(file_content, 0, 99)

    # Assert on content of line 99
    assert first_99_chars ==
             "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no "
  end

  test "split text on whitespace, keep the whitespace" do
    text = "Hello, world. This, is a test."
    result = Regex.split(~r{\s}, text, include_captures: true, trim: false)
    assert result == ["Hello,", " ", "world.", " ", "This,", " ", "is", " ", "a", " ", "test."]
  end

  test "split text on whitespace, commas, and periods, keep them except whitespace" do
    text = "Hello, world. This, is a test."
    result = Regex.split(~r{[,.]|\s}, text, include_captures: true, trim: true)

    assert result == [
             "Hello",
             ",",
             " ",
             "world",
             ".",
             " ",
             "This",
             ",",
             " ",
             "is",
             " ",
             "a",
             " ",
             "test",
             "."
           ]

    # Remove whitespaces from the result
    result_no_whitespace = Enum.reject(result, fn s -> s == " " end)

    assert result_no_whitespace == [
             "Hello",
             ",",
             "world",
             ".",
             "This",
             ",",
             "is",
             "a",
             "test",
             "."
           ]
  end

  test "split text on punctuation, keep them except whitespace" do
    text = "Hello, world. Is this-- a test?"
    result = Regex.split(~r{[,.:;?_!"()\']|--|\s}, text, include_captures: true, trim: true)
    # Remove whitespaces from the result
    result_no_whitespace = Enum.reject(result, fn s -> s == " " end)

    assert result_no_whitespace == [
             "Hello",
             ",",
             "world",
             ".",
             "Is",
             "this",
             "--",
             "a",
             "test",
             "?"
           ]
  end

  test "predprocess the verdict.txt file" do
    filename = "the-verdict.txt"
    {:ok, file_content} = File.read(filename)

    result =
      Regex.split(~r/([,.:;?_!"()\']|--|\s)/, file_content, include_captures: true, trim: true)

    # Remove whitespaces from the result - match Python's behavior: strip and filter empty
    predprocessed_text =
      result
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    assert Enum.slice(predprocessed_text, 0..29) == [
             "I",
             "HAD",
             "always",
             "thought",
             "Jack",
             "Gisburn",
             "rather",
             "a",
             "cheap",
             "genius",
             "--",
             "though",
             "a",
             "good",
             "fellow",
             "enough",
             "--",
             "so",
             "it",
             "was",
             "no",
             "great",
             "surprise",
             "to",
             "me",
             "to",
             "hear",
             "that",
             ",",
             "in"
           ]

    assert length(predprocessed_text) == 4690

    # Use Pythonx to read the file
    {result_obj, _globals} =
      Pythonx.eval(
        """
        import re
        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
        pattern = r'''([,.:;?_!"()']|--|\s)'''
        preprocessed = re.split(pattern, raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed
        """,
        %{}
      )

    preprocessed_text_python = Pythonx.decode(result_obj)
    assert preprocessed_text_python == predprocessed_text

    # Verify the file content starts correctly
    assert String.starts_with?(
             file_content,
             "I HAD always thought Jack Gisburn rather a cheap genius"
           )
  end

  test "token IDs" do
    filename = "the-verdict.txt"
    {:ok, file_content} = File.read(filename)

    result =
      Regex.split(~r/([,.:;?_!"()\']|--|\s)/, file_content, include_captures: true, trim: true)

    # Remove whitespaces from the result - match Python's behavior: strip and filter empty
    predprocessed_text =
      result
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    all_words = Enum.sort(MapSet.new(predprocessed_text))
    vocab_size = length(all_words)
    assert vocab_size == 1130

    vocab =
      Enum.with_index(all_words)

    assert Enum.slice(vocab, 0..50) == [
             {"!", 0},
             {"\"", 1},
             {"'", 2},
             {"(", 3},
             {")", 4},
             {",", 5},
             {"--", 6},
             {".", 7},
             {":", 8},
             {";", 9},
             {"?", 10},
             {"A", 11},
             {"Ah", 12},
             {"Among", 13},
             {"And", 14},
             {"Are", 15},
             {"Arrt", 16},
             {"As", 17},
             {"At", 18},
             {"Be", 19},
             {"Begin", 20},
             {"Burlington", 21},
             {"But", 22},
             {"By", 23},
             {"Carlo", 24},
             {"Chicago", 25},
             {"Claude", 26},
             {"Come", 27},
             {"Croft", 28},
             {"Destroyed", 29},
             {"Devonshire", 30},
             {"Don", 31},
             {"Dubarry", 32},
             {"Emperors", 33},
             {"Florence", 34},
             {"For", 35},
             {"Gallery", 36},
             {"Gideon", 37},
             {"Gisburn", 38},
             {"Gisburns", 39},
             {"Grafton", 40},
             {"Greek", 41},
             {"Grindle", 42},
             {"Grindles", 43},
             {"HAD", 44},
             {"Had", 45},
             {"Hang", 46},
             {"Has", 47},
             {"He", 48},
             {"Her", 49},
             {"Hermia", 50}
           ]
  end

  test "encode and decode text with simple tokenizer" do
    filename = "the-verdict.txt"
    vocab = LlmScratch.SimpleTokenizerV1.vocab_from_file(filename)

    text = """
    "It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride.
    """

    encoded_text = LlmScratch.SimpleTokenizerV1.encode(text, vocab)

    assert encoded_text == [
             1,
             56,
             2,
             850,
             988,
             602,
             533,
             746,
             5,
             1126,
             596,
             5,
             1,
             67,
             7,
             38,
             851,
             1108,
             754,
             793,
             7
           ]

    decoded_text = LlmScratch.SimpleTokenizerV1.decode(encoded_text, vocab)

    assert decoded_text ==
             "\" It' s the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
  end

  test "missing token in vocab raises an error" do
    vocab = LlmScratch.SimpleTokenizerV1.vocab_from_file("the-verdict.txt")
    text = "Hello, do you like tea. Is this-- a test?"

    assert_raise RuntimeError, "Token not found in vocab: Hello", fn ->
      LlmScratch.SimpleTokenizerV1.encode(text, vocab)
    end
  end

  test "encode and decode text with special token" do
    vocab =
      LlmScratch.SimpleTokenizerV1.vocab_from_file("the-verdict.txt", ["<|endoftext|>", "<|unk|>"])

    assert length(vocab) == 1132
    last_five = Enum.take(vocab, -5)

    assert last_five == [
             {"younger", 1127},
             {"your", 1128},
             {"yourself", 1129},
             {"<|endoftext|>", 1130},
             {"<|unk|>", 1131}
           ]

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = text1 <> " <|endoftext|> " <> text2

    assert text == "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."

    assert LlmScratch.SimpleTokenizerV1.encode(text, vocab) == [
             1131,
             5,
             355,
             1126,
             628,
             975,
             10,
             1130,
             55,
             988,
             956,
             984,
             722,
             988,
             1131,
             7
           ]

    assert text
           |> LlmScratch.SimpleTokenizerV1.encode(vocab)
           |> LlmScratch.SimpleTokenizerV1.decode(vocab) ==
             "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
  end

  test "encode and decode text with special token preserves original text", %{
    model: model,
    special_token: special_token
  } do
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

  test "endcoding with special tokens when not allowed does not raise an error", %{
    model: model,
    special_token: special_token
  } do
    # text contains a special token format "<|endoftext|>"
    test_text = "secret <|endoftext|> here"

    # Without allowing special token, it gets encoded as regular text (broken into multiple tokens)
    {:ok, encoded_without} = Tiktoken.encode(model, test_text)
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
