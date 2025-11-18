# Why Tiktoken Doesn't Report Errors for Special Tokens

## The Issue

When encoding text containing special tokens like `<|endoftext|>` without explicitly allowing them, the Elixir `tiktoken` library does **not** raise an error. Instead, it silently encodes the special token as regular text (breaking it into multiple tokens).

## Root Cause

The issue is in the Rust wrapper implementation. Looking at the code:

```rust
fn cl100k_encode(text: &str, allowed_special: Vec<&str>) -> Result<Vec<u32>, String> {
    let set = HashSet::from_iter(allowed_special.iter().cloned());
    Ok(CL100K_BASE.with(|bpe| bpe.encode(text, &set).0))  // <-- Only taking .0
}
```

### What's Happening:

1. **The underlying `tiktoken-rs` library** (`bpe.encode()`) returns a **tuple**: `(Vec<u32>, HashSet<String>)`
   - First element: The encoded token IDs
   - Second element: A set of special tokens that were encountered but **not** in `allowed_special`

2. **The Elixir wrapper** only takes the first element (`.0`), completely **discarding** the second element that contains information about disallowed special tokens.

3. **Result**: The library silently encodes special tokens as regular text instead of reporting that they were encountered but not allowed.

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

## Solution

To properly detect disallowed special tokens, the Elixir wrapper would need to:

1. **Capture the second element** of the tuple returned by `bpe.encode()`
2. **Return it** to the caller (either as part of the return value or by raising an error)
3. **Check if the set is non-empty** and handle accordingly

### Proposed Fix:

The Rust wrapper should be modified to:

```rust
fn cl100k_encode(text: &str, allowed_special: Vec<&str>) -> Result<(Vec<u32>, Vec<String>), String> {
    let set = HashSet::from_iter(allowed_special.iter().cloned());
    let (tokens, disallowed) = CL100K_BASE.with(|bpe| bpe.encode(text, &set));
    
    // Convert HashSet to Vec for Elixir
    let disallowed_vec: Vec<String> = disallowed.into_iter().collect();
    
    Ok((tokens, disallowed_vec))
}
```

Then the Elixir side could check if `disallowed` is non-empty and raise an error or return a warning.

## Current Workaround

Since the current implementation doesn't report errors, you need to:

1. **Always explicitly allow special tokens** if you expect them in your text
2. **Compare token counts** - if a special token is encoded as regular text, you'll get more tokens
3. **Manually check** if your text contains special token patterns before encoding

