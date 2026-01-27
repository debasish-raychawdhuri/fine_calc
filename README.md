# fine_calc

A terminal-based calculator with a TUI interface, built in Rust using ncurses.

## Features

- Interactive TUI with bordered, color-coded display
- Calculation history with scrolling
- Arithmetic: `+`, `-`, `*`, `/`, `^` (exponentiation)
- Constants: `pi`, `e`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Nested parentheses support

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
```