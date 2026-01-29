# fine_calc

A terminal-based calculator with a TUI interface, built in Rust using ncurses.

## Features

- Interactive TUI with bordered, color-coded display
- Calculation history with scrolling
- Arithmetic: `+`, `-`, `*`, `/`, `^` (exponentiation)
- Constants: `pi`, `e`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=` (return 0 or 1)
- Boolean: `&&`, `||`, `!` (truthy = nonzero)
- Unary negation: `-expr`
- Variables: assign with `x = expr`, use in later expressions
- Nested parentheses support
- Arrays: range `[n]`, literals `{x1, x2, ...}`, element-wise operations with broadcasting (including comparisons and booleans)
- Tuples: `(a, b, c)` creates a flat tuple; nested tuples flatten automatically; single-element tuples reduce to scalars
- Tuple arrays: `{(1,2), (3,4)}` or broadcasting `(scalar, array)` creates arrays of tuples
- Tuple decomposition: `(a, b) = (1, 2, 3)` assigns `a=1`, `b=(2,3)` (last var gets rest)
- Array indexing: `arr[i]` returns element at index; `arr[lambda]` filters with `(index, value...)` predicate
- Lambdas: define with `(|x| expr)` or multi-param `(|(x,y)| expr)`, call with `name(arg)`
- Lambda decomposition: `(|(a, rest)| ...)` binds first element to `a`, remaining to `rest` as tuple
- Tensor product: `x ** y` creates Cartesian product of arrays with flattened tuples

## Keybindings

| Key | Action |
|-----|--------|
| Enter | Evaluate expression |
| Up / Down | Navigate input history |
| Left / Right | Move cursor |
| Home / End | Jump to start / end of input |
| Page Up / Page Down | Scroll history |
| Backspace / Delete | Delete character |

## Building

Requires Rust and ncurses development libraries.

```sh
# Debian/Ubuntu
sudo apt install libncurses5-dev

# Fedora
sudo dnf install ncurses-devel
```

```sh
cargo build --release
```

## Installing

```sh
cargo install --path .
```

This installs `fine_calc` to your Cargo bin directory (usually `~/.cargo/bin`).

## Running

```sh
cargo run --release
# or, if installed:
fine_calc
```

## Examples

```
>> 2^10
1024
>> sin(pi/2)
1
>> cosh(0)
1
>> 3*4 + 2
14
>> x = 5
5
>> x * 2
10
>> radius = 3
3
>> 2 * pi * radius
18.849555921538759
>> [5]
{0, 1, 2, 3, 4}
>> {1, 2, 3} + 10
{11, 12, 13}
>> {1, 2, 3} * {4, 5, 6}
{4, 10, 18}
>> sin({0, pi})
{0, 0}
>> {1, 2, 3} + {10, 20}
{11, 22, 3}
>> 3 > 2
1
>> 1 == 1
1
>> {1, 2, -1} > 0
{1, 1, 0}
>> !0
1
>> 1 && 0
0
>> 1 || 0
1
>> 3 > 2 && 1 == 1
1
>> sq = (|x| x*x)
(|x| x*x)
>> sq(3)
9
>> sq({1, 2, 4})
{1, 4, 16}
>> inc = (|x| x+1)
(|x| x+1)
>> inc(10)
11
>> (1, 2, 3)
(1, 2, 3)
>> (1, (2, 3), 4)
(1, 2, 3, 4)
>> {(1, 2), (3, 4), (5, 6)}
{(1, 2), (3, 4), (5, 6)}
>> (10, {1, 2, 3})
{(10, 1), (10, 2), (10, 3)}
>> add = (|(x,y)| x+y)
(|(x,y)| x+y)
>> add((3, 4))
7
>> add({(1, 2), (3, 4), (5, 6)})
{3, 7, 11}
>> mul = (|(a,b)| a*b)
(|(a,b)| a*b)
>> mul((2, {1, 2, 3}))
{2, 4, 6}
>> arr = {10, 20, 30, 40, 50}
{10, 20, 30, 40, 50}
>> arr[2]
30
>> gt20 = (|(i,x)| x > 20)
(|(i,x)| x > 20)
>> arr[gt20]
{30, 40, 50}
>> [10][gt20]
{6, 7, 8, 9}
>> evenIdx = (|(i,x)| floor(i/2)*2 == i)
(|(i,x)| floor(i/2)*2 == i)
>> arr[evenIdx]
{10, 30, 50}
>> ta = {(1, 10), (2, 20), (3, 30)}
{(1, 10), (2, 20), (3, 30)}
>> ta[0]
(1, 10)
>> first_gt_1 = (|(i,x,y)| x > 1)
(|(i,x,y)| x > 1)
>> ta[first_gt_1]
{(2, 20), (3, 30)}
>> tri = {(1, 2, 3), (4, 5, 6), (7, 8, 9)}
{(1, 2, 3), (4, 5, 6), (7, 8, 9)}
>> getFirst = (|(a, rest)| a)
(|(a,rest)| a)
>> getFirst({(1, 2, 3), (4, 5, 6)})
{1, 4}
>> getRest = (|(a, rest)| rest)
(|(a,rest)| rest)
>> getRest((1, 2, 3))
(2, 3)
>> firstGt3 = (|(i, first, rest)| first > 3)
(|(i,first,rest)| first > 3)
>> tri[firstGt3]
{(4, 5, 6), (7, 8, 9)}
>> {1, 2} ** {10, 20, 30}
{(1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)}
>> [2] ** [3]
{(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
>> (1, 2) ** (3, 4)
(1, 2, 3, 4)
>> add = (|(x, y)| x + y)
(|(x,y)| x+y)
>> add({1, 2} ** {10, 20})
{11, 21, 12, 22}
```