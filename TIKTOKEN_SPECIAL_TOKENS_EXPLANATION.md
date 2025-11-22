# Why Tiktoken Doesn't Report Errors for Special Tokens

## The Issue

When encoding text containing special tokens like `<|endoftext|>` without explicitly allowing them, the Elixir `tiktoken` library does **not** raise an error. Instead, it silently encodes the special token as regular text (breaking it into multiple tokens).

## Root Cause

The issue is in the Rust wrapper implementation. **All five encoding functions** have the same problem - they only take the first element of the tuple returned by `bpe.encode()`, discarding information about disallowed special tokens.

The affected functions are:
- `p50k_encode` (line 35-37)
- `p50k_edit_encode` (line 64-66)
- `r50k_encode` (line 93-95)
- `cl100k_encode` (line 122-124)
- `o200k_encode` (line 151-153)

Example from the code:

```rust
fn cl100k_encode(text: &str, allowed_special: Vec<&str>) -> Result<Vec<u32>, String> {
    let set = HashSet::from_iter(allowed_special.iter().cloned());
    Ok(CL100K_BASE.with(|bpe| bpe.encode(text, &set).0))  // <-- Only taking .0
}

fn p50k_encode(text: &str, allowed_special: Vec<&str>) -> Result<Vec<u32>, String> {
    let set = HashSet::from_iter(allowed_special.iter().cloned());
    Ok(P50K_BASE.with(|bpe| bpe.encode(text, &set).0))  // <-- Same issue
}

// ... and similarly for p50k_edit_encode, r50k_encode, o200k_encode
```

### What's Happening:

1. **The underlying `tiktoken-rs` library** (`bpe.encode()`) returns a **tuple**: `(Vec<u32>, usize)`
   - First element: The encoded token IDs
   - Second element: A `usize` value (appears to always be `1` in tests, not useful for detecting disallowed tokens)

2. **Investigation Results**: After testing, we found that:
   - The second element is **not** a `HashSet<String>` of disallowed tokens as initially expected
   - The second element is always `1` regardless of whether disallowed tokens are present
   - The tiktoken-rs API does **not** provide a way to detect disallowed special tokens through the `encode()` method

3. **Result**: The library silently encodes special tokens as regular text, and there is **no way to detect** that a special token was encountered but not allowed using the current API.

## Expected vs Actual Behavior

### Expected (from tiktoken-rs documentation):
- By default, `encode` should raise an error if special tokens are encountered but not allowed
- OR return information about disallowed special tokens

### Actual (current Elixir implementation):
- No error is raised
- Special tokens are encoded as regular text (broken into multiple tokens)
- No way to detect that a special token was encountered but not allowed

## Example

```elixir
# Text with special token
text = "secret <|endoftext|> here"

# Without allowing special token - NO ERROR, just encodes as regular text
{:ok, tokens} = Tiktoken.encode("gpt-4", text)
# Returns: [21107, 83739, 8862, 728, 428, 91, 29, 1618]
# The special token is broken into: " <|", "endo", "ft", "ext", "|", ">"

# With allowing special token - encodes as single special token
{:ok, tokens} = Tiktoken.encode("gpt-4", text, ["<|endoftext|>"])
# Returns: [21107, 220, 100257, 1618]
# Token 100257 is the special token <|endoftext|>
```

## Important: Python vs Rust Implementation Difference

**Python tiktoken CAN detect special tokens**, but **tiktoken-rs (Rust) cannot** in the same way.

### Python tiktoken Behavior:

Python's `tiktoken` library has both `allowed_special` and `disallowed_special` parameters:
- **By default**, Python tiktoken **raises an error** if special tokens are encountered but not in `allowed_special`
- You can set `disallowed_special=set()` to allow encoding special tokens as regular text without errors
- This allows Python tiktoken to **detect and report** disallowed special tokens

### tiktoken-rs (Rust) Behavior:

The Rust implementation (`tiktoken-rs`) behaves differently:
- The `bpe.encode()` method returns `(Vec<u32>, usize)` where:
  - First element: The encoded token IDs
  - Second element: A `usize` value (always `1` in our tests, not useful for detection)
- **It does NOT raise errors** when special tokens are encountered but not allowed
- **It does NOT provide information** about which special tokens were disallowed
- Special tokens are silently encoded as regular text

### Why This Limitation Exists:

The tiktoken-rs library (Rust implementation) appears to be designed differently from the Python version:
1. It always encodes special tokens as regular text when not allowed (no error)
2. It encodes them as special tokens when allowed
3. It does **not** provide feedback about which special tokens were encountered but not allowed

**This is a limitation of the tiktoken-rs (Rust) library, not the Elixir wrapper.** The Python version has this capability, but the Rust version does not expose it through the same API.

## Workarounds

Since detection of disallowed tokens is not possible with the current API, you can:

1. **Always explicitly allow special tokens** if you expect them in your text:
   ```elixir
   {:ok, tokens} = Tiktoken.encode("gpt-4", text, ["<|endoftext|>"])
   ```

2. **Compare token counts** - if a special token is encoded as regular text, you'll get more tokens:
   ```elixir
   # Without allowing: 8 tokens (special token broken up)
   # With allowing: 4 tokens (special token as single token)
   ```

3. **Manually pre-scan text** for special token patterns before encoding:
   ```elixir
   special_tokens = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
   found_tokens = Enum.filter(special_tokens, &String.contains?(text, &1))
   if Enum.any?(found_tokens), do: # handle accordingly
   ```

4. **Use `encode_ordinary`** if you want to ensure no special tokens are processed:
   ```elixir
   {:ok, tokens} = Tiktoken.encode_ordinary("gpt-4", text)
   ```

## Summary

**It is not possible to detect disallowed special tokens with tiktoken-rs (the Rust implementation used by the Elixir wrapper).** 

**Key Points:**
- ✅ **Python tiktoken** CAN detect special tokens (raises errors by default)
- ❌ **tiktoken-rs (Rust)** cannot detect special tokens (silently encodes as regular text)
- The Rust library silently encodes special tokens as regular text without providing any indication that special tokens were encountered
- This is a **difference between the Python and Rust implementations**, not a bug in the Elixir wrapper

If you need to detect disallowed special tokens, you would need to:
1. Use the Python tiktoken library directly
2. Implement manual pre-scanning of text for special token patterns
3. Compare token counts between `encode_ordinary` and `encode` to infer special token presence

