# Tiktoken Encode Method Example
IO.puts("=== Tiktoken Encode Method Example ===")

# Example text
text = "Hello, world! This is a tiktoken example."
IO.puts("\nText to encode: '#{text}'")

# Method 1: Using Tiktoken.encode/2 directly
IO.puts("\n1. Using Tiktoken.encode/2:")
{:ok, tokens} = Tiktoken.encode("gpt-4", text)
IO.puts("   Tokens: #{inspect(tokens)}")
IO.puts("   Number of tokens: #{length(tokens)}")

# Method 2: Using Tiktoken.count_tokens/2
IO.puts("\n2. Using Tiktoken.count_tokens/2:")
{:ok, token_count} = Tiktoken.count_tokens("gpt-4", text)
IO.puts("   Token count: #{token_count}")

# Method 3: Decode tokens back to text
IO.puts("\n3. Decoding tokens back to text:")
{:ok, decoded_text} = Tiktoken.decode("gpt-4", tokens)
IO.puts("   Decoded text: '#{decoded_text}'")

# Method 4: Using encode_ordinary (without special tokens)
IO.puts("\n4. Using Tiktoken.encode_ordinary/2:")
{:ok, ordinary_tokens} = Tiktoken.encode_ordinary("gpt-4", text)
IO.puts("   Ordinary tokens: #{inspect(ordinary_tokens)}")

# Method 5: Compare different models
IO.puts("\n5. Comparing different models:")
models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

for model <- models do
  {:ok, tokens} = Tiktoken.encode(model, text)
  {:ok, count} = Tiktoken.count_tokens(model, text)
  IO.puts("   #{model}: #{count} tokens")
end

# Method 6: Test with longer text
IO.puts("\n6. Testing with longer text:")

long_text =
  "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes. It helps us understand how tokenization works with different text lengths and complexities."

{:ok, long_tokens} = Tiktoken.encode("gpt-4", long_text)
{:ok, long_count} = Tiktoken.count_tokens("gpt-4", long_text)

IO.puts("   Long text: #{String.length(long_text)} characters")
IO.puts("   Token count: #{long_count}")
IO.puts("   Tokens per character: #{Float.round(long_count / String.length(long_text), 3)}")

IO.puts("\n=== Example Complete ===")
