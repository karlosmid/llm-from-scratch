# Nx Shape Rules For `Nx.add`

`Nx.add(a, b)` produces one output value for every output position:

```text
output[position] = value_from_a + value_from_b
```

The main question is which value from each input should be used at each output
position. Broadcasting answers that.

## Shape Checking Algorithm

For `Nx.add(a, b)`:

1. Get the shapes of `a` and `b`.
2. Align the shapes from the right.
3. Compare each aligned dimension pair.
4. A dimension pair is valid when:
   - the dimensions are equal
   - one dimension is `1`
   - one side is missing
5. A dimension pair is invalid when both dimensions exist, are different, and
   neither is `1`.
6. The output shape uses the larger dimension at each position.

Broadcasting can repeat values. It cannot remove values, shrink dimensions, or
guess which values should be dropped.

## Examples

Same shape:

```text
{3} + {3} -> {3}
```

Example:

```text
[1, 2, 3] + [10, 20, 30] = [11, 22, 33]
```

Scalar and vector:

```text
{3} + {} -> {3}
```

Example:

```text
[1, 2, 3] + 10 = [11, 12, 13]
```

Matrix and scalar:

```text
{2, 3} + {} -> {2, 3}
```

Example:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
+
10
=
[
  [11, 12, 13],
  [14, 15, 16]
]
```

Matrix and row vector:

```text
{2, 3} + {3} -> {2, 3}
```

Right-aligned:

```text
{2, 3}
{   3}
```

The `3` matches the last dimension, so the vector is reused for every row:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
+
[10, 20, 30]
=
[
  [11, 22, 33],
  [14, 25, 36]
]
```

For output position `{i, j}`:

```text
output[i, j] = a[i, j] + b[j]
```

Matrix and column vector:

```text
{2, 3} + {2, 1} -> {2, 3}
```

Right-aligned:

```text
{2, 3}
{2, 1}
```

The first dimension matches. The second dimension is `1`, so it stretches:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
+
[
  [10],
  [20]
]
=
[
  [11, 12, 13],
  [24, 25, 26]
]
```

For output position `{i, j}`:

```text
output[i, j] = a[i, j] + b[i, 0]
```

Invalid matrix and vector:

```text
{2, 3} + {2} -> error
```

Right-aligned:

```text
{2, 3}
{   2}
```

The last dimensions are `3` and `2`. They are different, and neither is `1`, so
Nx cannot broadcast them.

Conceptually:

```text
[1, 2, 3] + [10, 20]
```

There is no value to pair with `3`:

```text
1 + 10
2 + 20
3 + ?
```

Nx does not guess, so it raises an error.

## Useful Mental Model

For each output position:

1. If an input has the same dimension, use the same index.
2. If an input has dimension `1`, use index `0` for that dimension.
3. If an input is missing that dimension, reuse it across that dimension.
4. If dimensions conflict, the operation is invalid.

Broadcasting expands by repetition. Reductions and slicing shrink tensors.
