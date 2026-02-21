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
    url =
      "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    filename = "the-verdict.txt"
    %Req.Response{status: 200, body: body} = Req.get!(url)
    File.write!(filename, body)
    # Read the file content as a single string

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

  test "byte pair encoding using gpt2 tiktoken" do
    model = "code-davinci-002"

    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces\of someunknownPlace."

    {:ok, encoded_tokens} = Tiktoken.encode(model, text, ["<|endoftext|>"])

    assert encoded_tokens == [
             15496,
             11,
             466,
             345,
             588,
             8887,
             30,
             220,
             50256,
             554,
             262,
             4252,
             18250,
             8812,
             2114,
             1659,
             617,
             34680,
             27271,
             13
           ]

    assert Tiktoken.decode(model, encoded_tokens) == {:ok, text}
  end

  test "data sampling with sliding window" do
    {:ok, file_content} = File.read("the-verdict.txt")
    model = "code-davinci-002"
    {:ok, encoded_tokens} = Tiktoken.encode(model, file_content)

    assert length(encoded_tokens) == 5145
    encoded_last_50 = Enum.drop(encoded_tokens, 50)
    context_size = 4

    context_desired_pairs =
      for i <- 1..context_size do
        context = Enum.slice(encoded_last_50, 0..(i - 1))
        desired = Enum.at(encoded_last_50, i)
        {context, desired}
      end

    assert context_desired_pairs == [
             {[290], 4920},
             {[290, 4920], 2241},
             {[290, 4920, 2241], 287},
             {[290, 4920, 2241, 287], 257}
           ]

    decoded_context_desired_pairs =
      Enum.map(context_desired_pairs, fn {current_context, current_desired} ->
        {:ok, decoded_current_context} = Tiktoken.decode(model, current_context)
        {:ok, decoded_current_desired} = Tiktoken.decode(model, [current_desired])
        {decoded_current_context, decoded_current_desired}
      end)

    assert decoded_context_desired_pairs == [
             {" and", " established"},
             {" and established", " himself"},
             {" and established himself", " in"},
             {" and established himself in", " a"}
           ]
  end

  test "chunk dataset" do
    txt = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    model = "code-davinci-002"
    max_length = 10 - 1
    stride = 5
    chunks = LlmScratch.GptDatasetV1.chunk_dataset(txt, model, max_length, stride)
    assert length(chunks) == 2

    [input_chunks: input_chunks, target_chunks: target_chunks] = chunks

    assert input_chunks == [
             Nx.tensor([15496, 11, 466, 345, 588, 8887, 30, 220, 50256]),
             Nx.tensor([8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812]),
             Nx.tensor([262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271])
           ]

    assert target_chunks == [
             Nx.tensor([11, 466, 345, 588, 8887, 30, 220, 50256, 554]),
             Nx.tensor([30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114]),
             Nx.tensor([4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13])
           ]
  end

  test "gpt dataset v1" do
    {:ok, file_content} = File.read("the-verdict.txt")
    model = "code-davinci-002"

    dataloader =
      LlmScratch.GptDatasetV1.create_dataloader_v1(
        raw_text: file_content,
        batch_size: 1,
        max_length: 4,
        stride: 1,
        shuffle: false,
        drop_last: false,
        num_workers: 0
      )

    batch_0 = dataloader.stream |> Enum.at(0)
    [{input_chunk, target_chunk}] = batch_0
    input_decoded_chunk = decode_token_pieces(model, Nx.to_flat_list(input_chunk))
    target_decoded_chunk = decode_token_pieces(model, Nx.to_flat_list(target_chunk))
    assert batch_0 == [{Nx.tensor([40, 367, 2885, 1464]), Nx.tensor([367, 2885, 1464, 1807])}]

    assert [{input_decoded_chunk, target_decoded_chunk}] == [
             {["I", " H", "AD", " always"], [" H", "AD", " always", " thought"]}
           ]
  end

  test "gpt dataset v1, batch_size is 8" do
    {:ok, file_content} = File.read("the-verdict.txt")
    model = "code-davinci-002"

    dataloader =
      LlmScratch.GptDatasetV1.create_dataloader_v1(
        raw_text: file_content,
        batch_size: 8,
        max_length: 4,
        stride: 4,
        shuffle: false,
        drop_last: false,
        num_workers: 0
      )

    batch_0 = dataloader.stream |> Enum.at(0)

    decoded_batch_0 =
      Enum.map(batch_0, fn {input_chunk, target_chunk} ->
        {
          decode_token_pieces(model, Nx.to_flat_list(input_chunk)),
          decode_token_pieces(model, Nx.to_flat_list(target_chunk))
        }
      end)

    assert batch_0 == [
             {Nx.tensor([40, 367, 2885, 1464]), Nx.tensor([367, 2885, 1464, 1807])},
             {Nx.tensor([1807, 3619, 402, 271]), Nx.tensor([3619, 402, 271, 10899])},
             {Nx.tensor([10899, 2138, 257, 7026]), Nx.tensor([2138, 257, 7026, 15632])},
             {Nx.tensor([15632, 438, 2016, 257]), Nx.tensor([438, 2016, 257, 922])},
             {Nx.tensor([922, 5891, 1576, 438]), Nx.tensor([5891, 1576, 438, 568])},
             {Nx.tensor([568, 340, 373, 645]), Nx.tensor([340, 373, 645, 1049])},
             {Nx.tensor([1049, 5975, 284, 502]), Nx.tensor([5975, 284, 502, 284])},
             {Nx.tensor([284, 3285, 326, 11]), Nx.tensor([3285, 326, 11, 287])}
           ]

    assert decoded_batch_0 == [
             {["I", " H", "AD", " always"], [" H", "AD", " always", " thought"]},
             {[" thought", " Jack", " G", "is"], [" Jack", " G", "is", "burn"]},
             {["burn", " rather", " a", " cheap"], [" rather", " a", " cheap", " genius"]},
             {[" genius", "--", "though", " a"], ["--", "though", " a", " good"]},
             {[" good", " fellow", " enough", "--"], [" fellow", " enough", "--", "so"]},
             {["so", " it", " was", " no"], [" it", " was", " no", " great"]},
             {[" great", " surprise", " to", " me"], [" surprise", " to", " me", " to"]},
             {[" to", " hear", " that", ","], [" hear", " that", ",", " in"]}
           ]
  end

  test "data loader drop_last true drops trailing incomplete batch from tuple dataset" do
    dataset = [
      {Nx.tensor([40, 367, 2885, 1464]), Nx.tensor([367, 2885, 1464, 1807])},
      {Nx.tensor([1807, 3619, 402, 271]), Nx.tensor([3619, 402, 271, 10899])},
      {Nx.tensor([10899, 2138, 257, 7026]), Nx.tensor([2138, 257, 7026, 15632])},
      {Nx.tensor([15632, 438, 2016, 257]), Nx.tensor([438, 2016, 257, 922])},
      {Nx.tensor([922, 5891, 1576, 438]), Nx.tensor([5891, 1576, 438, 568])},
      {Nx.tensor([568, 340, 373, 645]), Nx.tensor([340, 373, 645, 1049])},
      {Nx.tensor([1049, 5975, 284, 502]), Nx.tensor([5975, 284, 502, 284])},
      {Nx.tensor([284, 3285, 326, 11]), Nx.tensor([3285, 326, 11, 287])}
    ]

    batch_size = 3

    dataloader_keep_last =
      LlmScratch.DataLoader.new(dataset, batch_size: batch_size, shuffle: false, drop_last: false)

    dataloader_drop_last =
      LlmScratch.DataLoader.new(dataset, batch_size: batch_size, shuffle: false, drop_last: true)

    assert Enum.take(dataloader_keep_last.stream, 3) == [
             [
               {Nx.tensor([40, 367, 2885, 1464]), Nx.tensor([367, 2885, 1464, 1807])},
               {Nx.tensor([1807, 3619, 402, 271]), Nx.tensor([3619, 402, 271, 10899])},
               {Nx.tensor([10899, 2138, 257, 7026]), Nx.tensor([2138, 257, 7026, 15632])}
             ],
             [
               {Nx.tensor([15632, 438, 2016, 257]), Nx.tensor([438, 2016, 257, 922])},
               {Nx.tensor([922, 5891, 1576, 438]), Nx.tensor([5891, 1576, 438, 568])},
               {Nx.tensor([568, 340, 373, 645]), Nx.tensor([340, 373, 645, 1049])}
             ],
             [
               {Nx.tensor([1049, 5975, 284, 502]), Nx.tensor([5975, 284, 502, 284])},
               {Nx.tensor([284, 3285, 326, 11]), Nx.tensor([3285, 326, 11, 287])}
             ]
           ]

    assert Enum.take(dataloader_drop_last.stream, 3) == [
             [
               {Nx.tensor([40, 367, 2885, 1464]), Nx.tensor([367, 2885, 1464, 1807])},
               {Nx.tensor([1807, 3619, 402, 271]), Nx.tensor([3619, 402, 271, 10899])},
               {Nx.tensor([10899, 2138, 257, 7026]), Nx.tensor([2138, 257, 7026, 15632])}
             ],
             [
               {Nx.tensor([15632, 438, 2016, 257]), Nx.tensor([438, 2016, 257, 922])},
               {Nx.tensor([922, 5891, 1576, 438]), Nx.tensor([5891, 1576, 438, 568])},
               {Nx.tensor([568, 340, 373, 645]), Nx.tensor([340, 373, 645, 1049])}
             ],
             [
               {Nx.tensor([40, 367, 2885, 1464]), Nx.tensor([367, 2885, 1464, 1807])},
               {Nx.tensor([1807, 3619, 402, 271]), Nx.tensor([3619, 402, 271, 10899])},
               {Nx.tensor([10899, 2138, 257, 7026]), Nx.tensor([2138, 257, 7026, 15632])}
             ]
           ]
  end

  test "PyTorch-style Embedding with manual_seed (torch.nn.Embedding equivalent)" do
    # PyTorch equivalent:
    # torch.manual_seed(123)
    # embedding = torch.nn.Embedding(vocab_size=6, embedding_dim=3)
    # token_ids = torch.tensor([[2, 3, 5, 1]])
    # embeddings = embedding(token_ids)

    vocab_size = 6
    embedding_dim = 3

    embedding = LlmScratch.Embedding.new(vocab_size, embedding_dim, seed: 123)

    # Verify the embedding struct
    assert embedding.vocab_size == vocab_size
    assert embedding.embedding_dim == embedding_dim

    # Test weight access (equivalent to embedding_layer.weight in PyTorch)
    weight = LlmScratch.Embedding.weight(embedding)

    # Expected PyTorch weights with seed=123, vocab_size=6, embedding_dim=3
    expected_weight =
      Nx.tensor(
        [
          [0.3373701572418213, -0.1777772158384323, -0.16895616054534912],
          [0.9177640080451965, 1.5809690952301025, 1.3010399341583252],
          [1.275301218032837, -0.20095309615135193, -0.16056379675865173],
          [-0.40148791670799255, 0.966571569442749, -1.1481444835662842],
          [-1.158868670463562, 0.32547101378440857, -0.6315054297447205],
          [-2.839993953704834, -0.7848533391952515, -1.4095723628997803]
        ],
        type: {:f, 32}
      )

    # Assert that weights match PyTorch exactly (using small tolerance for floating point)
    assert Nx.all_close(weight, expected_weight, atol: 1.0e-6),
           "Embedding weights should match PyTorch's weights exactly with seed=123"

    # Get specific row from tensor (row 3, which is index 3)
    # Method 1: Using Nx.slice/4 (recommended for tensors)
    row_3 = Nx.slice(weight, [3, 0], [1, embedding_dim]) |> Nx.squeeze(axes: [0])

    expected_row_3 =
      Nx.tensor([-0.40148791670799255, 0.966571569442749, -1.1481444835662842], type: {:f, 32})

    assert Nx.all_close(row_3, expected_row_3, atol: 1.0e-6)

    # Test forward pass: mapping input_ids to embeddings (equivalent to embedding_layer(input_ids))
    # PyTorch: embeddings = embedding_layer(input_ids)
    input_ids = Nx.tensor([[2, 3, 5, 1]], type: {:s, 64})
    embeddings = LlmScratch.Embedding.forward(embedding, input_ids)

    # Expected forward pass output with seed=123, input_ids=[[2, 3, 5, 1]]
    expected_forward =
      Nx.tensor(
        [
          [
            [1.275301218032837, -0.20095309615135193, -0.16056379675865173],
            [-0.40148791670799255, 0.966571569442749, -1.1481444835662842],
            [-2.839993953704834, -0.7848533391952515, -1.4095723628997803],
            [0.9177640080451965, 1.5809690952301025, 1.3010399341583252]
          ]
        ],
        type: {:f, 32}
      )

    # Verify embeddings match expected output exactly
    assert Nx.all_close(embeddings, expected_forward, atol: 1.0e-6),
           "Embeddings from forward pass should match expected values exactly"
  end

  test "Elixir-style EmbeddingNative with manual_seed (torch.nn.Embedding equivalent)" do
    vocab_size = 6
    embedding_dim = 3

    embedding = LlmScratch.EmbeddingNative.new(vocab_size, embedding_dim, seed: 123)

    assert embedding.vocab_size == vocab_size
    assert embedding.embedding_dim == embedding_dim

    weight = LlmScratch.EmbeddingNative.weight(embedding)

    expected_weight =
      Nx.tensor(
        [
          [0.3373701572418213, -0.1777772158384323, -0.16895616054534912],
          [0.9177640080451965, 1.5809690952301025, 1.3010399341583252],
          [1.275301218032837, -0.20095309615135193, -0.16056379675865173],
          [-0.40148791670799255, 0.966571569442749, -1.1481444835662842],
          [-1.158868670463562, 0.32547101378440857, -0.6315054297447205],
          [-2.839993953704834, -0.7848533391952515, -1.4095723628997803]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(weight, expected_weight, atol: 1.0e-6),
           "Embedding weights should match PyTorch's weights exactly with seed=123"

    row_3 = Nx.slice(weight, [3, 0], [1, embedding_dim])

    expected_row_3 =
      Nx.tensor([-0.40148791670799255, 0.966571569442749, -1.1481444835662842], type: {:f, 32})

    assert Nx.all_close(row_3, expected_row_3, atol: 1.0e-6)

    input_ids = Nx.tensor([[2, 3, 5, 1]], type: {:s, 64})
    embeddings = LlmScratch.EmbeddingNative.forward(embedding, input_ids)

    expected_forward =
      Nx.tensor(
        [
          [
            [1.275301218032837, -0.20095309615135193, -0.16056379675865173],
            [-0.40148791670799255, 0.966571569442749, -1.1481444835662842],
            [-2.839993953704834, -0.7848533391952515, -1.4095723628997803],
            [0.9177640080451965, 1.5809690952301025, 1.3010399341583252]
          ]
        ],
        type: {:f, 32}
      )

    assert Nx.all_close(embeddings, expected_forward, atol: 1.0e-6),
           "Embeddings from forward pass should match expected values exactly"
  end

  test "positional embedding" do
    previous_backend = Nx.default_backend()
    Nx.default_backend(EXLA.Backend)
    on_exit(fn -> Nx.default_backend(previous_backend) end)

    vocab_size = 50257
    embedding_dim = 256

    token_embeding_layer = LlmScratch.EmbeddingNative.new(vocab_size, embedding_dim, seed: 123)

    {:ok, file_content} = File.read("the-verdict.txt")

    dataloader =
      LlmScratch.GptDatasetV1.create_dataloader_v1(
        raw_text: file_content,
        batch_size: 8,
        max_length: 4,
        stride: 4,
        shuffle: false,
        drop_last: false,
        num_workers: 0
      )

    batch_0 = dataloader.stream |> Enum.at(0)
    inputs_list = Enum.map(batch_0, fn {input, _target} -> input end)

    # Stack list of tensors into a single tensor: [tensor1, tensor2, ...] -> tensor with shape [batch_size, seq_len]
    inputs = Nx.stack(inputs_list)

    expected_inputs =
      [
        [40, 367, 2885, 1464],
        [1807, 3619, 402, 271],
        [10899, 2138, 257, 7026],
        [15632, 438, 2016, 257],
        [922, 5891, 1576, 438],
        [568, 340, 373, 645],
        [1049, 5975, 284, 502],
        [284, 3285, 326, 11]
      ]
      |> Nx.tensor(type: {:s, 32})

    # Verify inputs match expected output exactly
    assert Nx.all_close(inputs, expected_inputs, atol: 1.0e-6),
           "Input embeddings should match expected values exactly"

    # Assert the size/shape of inputs is [8, 4]
    assert Nx.shape(inputs) == {8, 4}

    # Get token embeddings: shape [8, 4, 256]
    token_embeddings = LlmScratch.EmbeddingNative.forward(token_embeding_layer, inputs)
    assert Nx.shape(token_embeddings) == {8, 4, 256}

    # Create positional embedding layer: vocab_size=4 (positions 0,1,2,3), embedding_dim=256
    positional_embedding_layer = LlmScratch.EmbeddingNative.new(4, 256, seed: 123)
    positional_embedding_weights = LlmScratch.EmbeddingNative.weight(positional_embedding_layer)
    assert Nx.shape(positional_embedding_weights) == {4, 256}

    # Create positional indices: [0, 1, 2, 3] for each position in the sequence
    # Shape: [4] -> expand to [1, 4] -> broadcast to [8, 4]
    positional_indices = Nx.tensor([0, 1, 2, 3], type: {:s, 64})
    positional_indices_batch = Nx.broadcast(Nx.new_axis(positional_indices, 0), {8, 4})

    # Get positional embeddings: shape [8, 4, 256]
    positional_embeddings =
      LlmScratch.EmbeddingNative.forward(positional_embedding_layer, positional_indices_batch)

    assert Nx.shape(positional_embeddings) == {8, 4, 256}

    # Add token embeddings and positional embeddings: shape [8, 4, 256]
    embeddings_sum = Nx.add(token_embeddings, positional_embeddings)
    assert Nx.shape(embeddings_sum) == {8, 4, 256}
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
