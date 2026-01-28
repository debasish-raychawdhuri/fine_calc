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
```