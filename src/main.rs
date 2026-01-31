use ncurses::*;
use std::collections::HashMap;

mod ast;
mod error;
mod eval;
mod value;

use ast::Span;
use error::{EvalError, format_error_indicator};
use eval::eval;
use value::Value;

lalrpop_util::lalrpop_mod!(
    #[allow(clippy::all)]
    expr
);

fn evaluate_expression(input: &str, vars: &HashMap<String, Value>) -> Result<Value, EvalError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(EvalError::new("Empty expression"));
    }

    // Parse to AST
    let ast = expr::ExprParser::new().parse(input).map_err(|e| {
        // Extract span from parse error if possible
        match &e {
            lalrpop_util::ParseError::InvalidToken { location } => {
                EvalError::with_span(format!("{}", e), Span::new(*location, location + 1))
            }
            lalrpop_util::ParseError::UnrecognizedToken {
                token: (start, _, end),
                ..
            } => EvalError::with_span(format!("{}", e), Span::new(*start, *end)),
            lalrpop_util::ParseError::ExtraToken {
                token: (start, _, end),
            } => EvalError::with_span(format!("{}", e), Span::new(*start, *end)),
            lalrpop_util::ParseError::UnrecognizedEof { location, .. } => {
                EvalError::with_span(format!("{}", e), Span::new(*location, location + 1))
            }
            _ => EvalError::new(format!("{}", e)),
        }
    })?;

    // Evaluate AST
    eval(&ast, vars)
}

/// Check if identifier is valid
fn is_valid_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
        && (s.starts_with('_') || s.chars().next().unwrap().is_alphabetic())
}

/// Parse a tuple pattern like (a, b, c) and return the variable names
fn parse_tuple_pattern(s: &str) -> Option<Vec<String>> {
    let s = s.trim();
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        let vars: Vec<String> = inner.split(',').map(|v| v.trim().to_string()).collect();
        if vars.len() >= 2 && vars.iter().all(|v| is_valid_ident(v)) {
            return Some(vars);
        }
    }
    None
}

/// Represents what kind of assignment we have
enum Assignment {
    /// Single variable assignment: x = expr
    Single(String),
    /// Tuple destructuring: (a, b, c) = expr
    Tuple(Vec<String>),
}

/// Parse the left-hand side of an assignment
fn parse_assignment(trimmed: &str) -> (Option<Assignment>, &str) {
    // Find a standalone '=' that is not part of ==, !=, >=, <=
    let mut assign_pos = None;
    let bytes = trimmed.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'=' {
            let prev = if i > 0 { bytes[i - 1] } else { 0 };
            let next = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
            if prev != b'!' && prev != b'>' && prev != b'<' && prev != b'=' && next != b'=' {
                assign_pos = Some(i);
                break;
            }
        }
    }

    if let Some(eq_pos) = assign_pos {
        let lhs = trimmed[..eq_pos].trim();
        let rhs = trimmed[eq_pos + 1..].trim();

        // Check for tuple pattern: (a, b, c) = expr
        if let Some(vars) = parse_tuple_pattern(lhs) {
            return (Some(Assignment::Tuple(vars)), rhs);
        }

        // Check for single variable: x = expr
        if is_valid_ident(lhs) {
            return (Some(Assignment::Single(lhs.to_string())), rhs);
        }
    }

    (None, trimmed)
}

/// Wrap a string into multiple lines of at most `width` characters.
/// Returns a Vec of lines that fit within the width.
fn wrap_line(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![s.to_string()];
    }
    let mut lines = Vec::new();
    let mut remaining = s;
    while !remaining.is_empty() {
        if remaining.len() <= width {
            lines.push(remaining.to_string());
            break;
        }
        // Take up to width characters
        let split_at = remaining
            .char_indices()
            .take(width)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(remaining.len());
        lines.push(remaining[..split_at].to_string());
        remaining = &remaining[split_at..];
    }
    lines
}
#[derive(PartialEq)]
enum LineType {
    Prompt,
    Result,
    Error,
    Indicator,
    Normal,
}

struct Line {
    content: String,
    line_type: LineType,
}

fn main() {
    initscr();
    cbreak();
    keypad(stdscr(), true);
    noecho();

    start_color();
    use_default_colors();
    // History area colors (dark grey background)
    init_pair(1, COLOR_GREEN, 236); // result
    init_pair(2, COLOR_RED, 236); // error
    init_pair(3, COLOR_CYAN, -1); // prompt (input row, default bg)
    init_pair(4, COLOR_WHITE, 236); // normal history text
    init_pair(5, COLOR_CYAN, 236); // prompt in history
    init_pair(6, COLOR_WHITE, 236); // separator line

    let mut input = String::new();
    let mut cursor: usize = 0;
    let mut history: Vec<Line> = Vec::new();
    let mut scroll_offset: usize = 0;
    let mut input_history: Vec<String> = Vec::new();
    let mut hist_idx: usize = 0;
    let mut saved_input = String::new();
    let mut variables: HashMap<String, Value> = HashMap::new();

    loop {
        let max_y = getmaxy(stdscr()) as usize;
        let max_x = getmaxx(stdscr()) as usize;
        let inner_w = if max_x > 2 { max_x - 2 } else { 1 };
        // Layout: top border, history rows, separator, input row, bottom border
        let visible_rows = if max_y > 4 { max_y - 4 } else { 1 };

        let max_offset = if history.len() > visible_rows {
            history.len() - visible_rows
        } else {
            0
        };
        if scroll_offset > max_offset {
            scroll_offset = max_offset;
        }

        clear();

        // Top border: ┌───┐
        mv(0, 0);
        addch(ACS_ULCORNER());
        for _ in 0..inner_w {
            addch(ACS_HLINE());
        }
        addch(ACS_URCORNER());

        // History rows (rows 1..=visible_rows)
        for row in 0..visible_rows {
            let y = (row + 1) as i32;
            mv(y, 0);
            addch(ACS_VLINE());

            let idx = scroll_offset + row;
            let (line_content, color_pair, line_len) = if idx < history.len() {
                let line = &history[idx];
                match line.line_type {
                    LineType::Result => (line.content.as_str(), 1, line.content.chars().count()),
                    LineType::Error => (line.content.as_str(), 2, line.content.chars().count()),
                    LineType::Indicator => (line.content.as_str(), 2, line.content.chars().count()),
                    LineType::Prompt => ("", 0, 0), // handled specially below
                    LineType::Normal => (line.content.as_str(), 4, line.content.chars().count()),
                }
            } else {
                ("", 4, 0)
            };

            if idx < history.len() && history[idx].line_type == LineType::Prompt {
                // Prompt line: ">> " in cyan, rest in white, all on grey background
                let line = &history[idx].content;
                attron(COLOR_PAIR(5));
                addstr(">> ");
                attroff(COLOR_PAIR(5));
                attron(COLOR_PAIR(4));
                let rest = &line[3..];
                addstr(rest);
                let used = 3 + rest.chars().count();
                // Pad remaining with spaces
                for _ in used..inner_w {
                    addch(' ' as u32);
                }
                attroff(COLOR_PAIR(4));
            } else {
                // Regular line: error, result, or empty
                attron(COLOR_PAIR(color_pair));
                addstr(line_content);
                // Pad remaining with spaces to maintain background
                for _ in line_len..inner_w {
                    addch(' ' as u32);
                }
                attroff(COLOR_PAIR(color_pair));
            }

            addch(ACS_VLINE());
        }

        // Middle separator: ├───┤
        let sep_y = (visible_rows + 1) as i32;
        mv(sep_y, 0);
        addch(ACS_LTEE());
        for _ in 0..inner_w {
            addch(ACS_HLINE());
        }
        addch(ACS_RTEE());

        // Input row
        let input_y = (visible_rows + 2) as i32;
        mv(input_y, 0);
        addch(ACS_VLINE());
        for _ in 0..inner_w {
            addch(' ' as u32);
        }
        addch(ACS_VLINE());

        mv(input_y, 1);
        attron(COLOR_PAIR(3));
        addstr(">> ");
        attroff(COLOR_PAIR(3));
        addstr(&input);

        // Bottom border: └───┘
        let bot_y = (visible_rows + 3) as i32;
        mv(bot_y, 0);
        addch(ACS_LLCORNER());
        for _ in 0..inner_w {
            addch(ACS_HLINE());
        }
        addch(ACS_LRCORNER());

        // Place cursor at correct position
        mv(input_y, (4 + cursor) as i32);
        refresh();

        let ch = getch();
        match ch {
            KEY_LEFT => {
                if cursor > 0 {
                    cursor -= 1;
                }
            }
            KEY_RIGHT => {
                if cursor < input.len() {
                    cursor += 1;
                }
            }
            KEY_UP => {
                // Navigate input history (older)
                if !input_history.is_empty() && hist_idx > 0 {
                    if hist_idx == input_history.len() {
                        saved_input = input.clone();
                    }
                    hist_idx -= 1;
                    input = input_history[hist_idx].clone();
                    cursor = input.len();
                }
            }
            KEY_DOWN => {
                // Navigate input history (newer)
                if hist_idx < input_history.len() {
                    hist_idx += 1;
                    if hist_idx == input_history.len() {
                        input = saved_input.clone();
                    } else {
                        input = input_history[hist_idx].clone();
                    }
                    cursor = input.len();
                }
            }
            KEY_PPAGE => {
                // Page Up: scroll viewport up
                scroll_offset = scroll_offset.saturating_sub(visible_rows);
            }
            KEY_NPAGE => {
                // Page Down: scroll viewport down
                scroll_offset = std::cmp::min(scroll_offset + visible_rows, max_offset);
            }
            KEY_HOME => {
                cursor = 0;
            }
            KEY_END => {
                cursor = input.len();
            }
            KEY_DC => {
                // Delete key: remove char at cursor
                if cursor < input.len() {
                    input.remove(cursor);
                }
            }
            KEY_BACKSPACE | 127 => {
                if cursor > 0 {
                    cursor -= 1;
                    input.remove(cursor);
                }
            }
            KEY_ENTER | 10 => {
                if !input.trim().is_empty() {
                    let trimmed = input.trim();
                    let (assignment, expr_str) = parse_assignment(trimmed);
                    let result = evaluate_expression(expr_str, &variables);
                    match result {
                        Ok(res) => {
                            // Wrap input line (account for ">> " prefix)
                            let input_line = format!(">> {}", input);
                            for line in wrap_line(&input_line, inner_w) {
                                history.push(Line {
                                    content: line,
                                    line_type: LineType::Prompt,
                                });
                            }
                            // Wrap result output
                            let result_str = format!("{}", &res);
                            for line in wrap_line(&result_str, inner_w) {
                                history.push(Line {
                                    content: line,
                                    line_type: LineType::Result,
                                });
                            }
                            match assignment {
                                Some(Assignment::Single(name)) => {
                                    variables.insert(name, res);
                                }
                                Some(Assignment::Tuple(names)) => {
                                    match res.decompose_tuple(names.len()) {
                                        Ok(values) => {
                                            for (name, val) in names.into_iter().zip(values) {
                                                variables.insert(name, val);
                                            }
                                        }
                                        Err(e) => {
                                            for line in wrap_line(&format!("Error: {}", e), inner_w)
                                            {
                                                history.push(Line {
                                                    content: line,
                                                    line_type: LineType::Error,
                                                });
                                            }
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                        Err(e) => {
                            let input_line = format!(">> {}", input);
                            for line in wrap_line(&input_line, inner_w) {
                                history.push(Line {
                                    content: line,
                                    line_type: LineType::Prompt,
                                });
                            }
                            // Add indicator line if we have span info
                            if let Some(span) = e.span {
                                // Calculate offset: expr_str starts at this position within input
                                let trim_offset = input.find(trimmed).unwrap_or(0);
                                let expr_offset = if assignment.is_some() {
                                    // There was an assignment, find where expr_str starts in trimmed
                                    trim_offset + trimmed.find(expr_str).unwrap_or(0)
                                } else {
                                    trim_offset
                                };
                                let adjusted_span =
                                    Span::new(span.start + expr_offset, span.end + expr_offset);
                                let indicator =
                                    format!("   {}", format_error_indicator(&input, adjusted_span));
                                for line in wrap_line(&indicator, inner_w) {
                                    history.push(Line {
                                        content: line,
                                        line_type: LineType::Indicator,
                                    });
                                }
                            }
                            let error_message = format!("Error: {}", e.message);
                            let error_lines = error_message.split('\n');
                            for error_line in error_lines {
                                for line in wrap_line(&error_line, inner_w) {
                                    history.push(Line {
                                        content: line,
                                        line_type: LineType::Error,
                                    });
                                }
                            }
                        }
                    }
                    input_history.push(input.clone());
                    input.clear();
                    cursor = 0;
                    hist_idx = input_history.len();
                    saved_input.clear();
                    scroll_offset = usize::MAX;
                }
            }
            _ => {
                if ch >= 32 && ch <= 126 {
                    let c = ch as u8 as char;
                    input.insert(cursor, c);
                    cursor += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn eval(expr: &str) -> Result<Value, String> {
        evaluate_expression(expr, &HashMap::new()).map_err(|e| e.message)
    }

    fn eval_with(expr: &str, vars: &HashMap<String, Value>) -> Result<Value, String> {
        evaluate_expression(expr, vars).map_err(|e| e.message)
    }

    fn scalar(v: &Value) -> f64 {
        v.as_scalar().expect("expected scalar")
    }

    fn array(v: &Value) -> Vec<f64> {
        match v {
            Value::Array(a) => a.clone(),
            _ => panic!("expected array"),
        }
    }

    fn tuple(v: &Value) -> Vec<f64> {
        match v {
            Value::Tuple(t) => t.clone(),
            _ => panic!("expected tuple"),
        }
    }

    fn tuple_array(v: &Value) -> (usize, Vec<f64>) {
        match v {
            Value::TupleArray { width, data } => (*width, data.clone()),
            _ => panic!("expected tuple array"),
        }
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn approx_arr(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx(*x, *y))
    }

    #[test]
    fn basic_arithmetic() {
        assert!(approx(scalar(&eval("2+3").unwrap()), 5.0));
        assert!(approx(scalar(&eval("10-4").unwrap()), 6.0));
        assert!(approx(scalar(&eval("3*5").unwrap()), 15.0));
        assert!(approx(scalar(&eval("8/2").unwrap()), 4.0));
    }

    #[test]
    fn exponentiation() {
        assert!(approx(scalar(&eval("2^10").unwrap()), 1024.0));
    }

    #[test]
    fn negative_result() {
        assert!(approx(scalar(&eval("3-5").unwrap()), -2.0));
    }

    #[test]
    fn decimal_numbers() {
        assert!(approx(scalar(&eval("1.5+2.5").unwrap()), 4.0));
        assert!(approx(scalar(&eval("0.1*10").unwrap()), 1.0));
    }

    #[test]
    fn constants() {
        assert!(approx(scalar(&eval("pi").unwrap()), std::f64::consts::PI));
        assert!(approx(scalar(&eval("e").unwrap()), std::f64::consts::E));
    }

    #[test]
    fn trig_functions() {
        assert!(approx(scalar(&eval("sin(0)").unwrap()), 0.0));
        assert!(approx(scalar(&eval("cos(0)").unwrap()), 1.0));
        assert!(approx(scalar(&eval("tan(0)").unwrap()), 0.0));
        assert!(approx(
            scalar(&eval("asin(1)").unwrap()),
            std::f64::consts::FRAC_PI_2
        ));
        assert!(approx(scalar(&eval("acos(1)").unwrap()), 0.0));
        assert!(approx(
            scalar(&eval("atan(1)").unwrap()),
            std::f64::consts::FRAC_PI_4
        ));
    }

    #[test]
    fn hyperbolic_functions() {
        assert!(approx(scalar(&eval("sinh(0)").unwrap()), 0.0));
        assert!(approx(scalar(&eval("cosh(0)").unwrap()), 1.0));
        assert!(approx(scalar(&eval("tanh(0)").unwrap()), 0.0));
        assert!(approx(scalar(&eval("asinh(0)").unwrap()), 0.0));
        assert!(approx(scalar(&eval("acosh(1)").unwrap()), 0.0));
        assert!(approx(scalar(&eval("atanh(0)").unwrap()), 0.0));
    }

    #[test]
    fn parentheses() {
        assert!(approx(scalar(&eval("(2+3)*4").unwrap()), 20.0));
        assert!(approx(scalar(&eval("((1+2)*(3+4))").unwrap()), 21.0));
    }

    #[test]
    fn nested_functions() {
        assert!(approx(scalar(&eval("sin(pi/2)").unwrap()), 1.0));
        assert!(approx(scalar(&eval("cos(2*pi)").unwrap()), 1.0));
    }

    #[test]
    fn variables() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), Value::Scalar(10.0));
        vars.insert("y".to_string(), Value::Scalar(3.0));
        assert!(approx(
            scalar(&evaluate_expression("x+y", &vars).unwrap()),
            13.0
        ));
        assert!(approx(
            scalar(&evaluate_expression("x*y", &vars).unwrap()),
            30.0
        ));
    }

    #[test]
    fn complex_expressions() {
        assert!(approx(scalar(&eval("2^3+sin(pi/2)*4-1").unwrap()), 11.0));
        assert!(approx(scalar(&eval("(2+3)*(4-1)/3").unwrap()), 5.0));
    }

    #[test]
    fn error_empty() {
        assert!(eval("").is_err());
    }

    #[test]
    fn error_unknown_function() {
        assert!(eval("foo(1)").is_err());
    }

    #[test]
    fn error_unknown_identifier() {
        assert!(eval("xyz").is_err());
    }

    #[test]
    fn operator_precedence() {
        assert!(approx(scalar(&eval("2+3*4").unwrap()), 14.0));
    }

    // Array tests

    #[test]
    fn range_array() {
        let v = eval("[5]").unwrap();
        assert!(approx_arr(&array(&v), &[0.0, 1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn explicit_array() {
        let v = eval("{1,2,3}").unwrap();
        assert!(approx_arr(&array(&v), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn array_scalar_add() {
        let v = eval("{1,2,3}+1").unwrap();
        assert!(approx_arr(&array(&v), &[2.0, 3.0, 4.0]));
    }

    #[test]
    fn array_array_mul() {
        let v = eval("{1,2,3}*{4,5,6}").unwrap();
        assert!(approx_arr(&array(&v), &[4.0, 10.0, 18.0]));
    }

    #[test]
    fn sin_array() {
        let v = eval("sin({0,pi})").unwrap();
        let a = array(&v);
        assert!(approx(a[0], 0.0));
        assert!(approx(a[1], 0.0));
    }

    #[test]
    fn zero_pad_behavior() {
        let v = eval("{1,2,3}+{10,20}").unwrap();
        assert!(approx_arr(&array(&v), &[11.0, 22.0, 3.0]));
    }

    // Boolean and comparison tests

    #[test]
    fn comparison_gt() {
        assert!(approx(scalar(&eval("3>2").unwrap()), 1.0));
        assert!(approx(scalar(&eval("2>3").unwrap()), 0.0));
    }

    #[test]
    fn comparison_lt() {
        assert!(approx(scalar(&eval("2<3").unwrap()), 1.0));
        assert!(approx(scalar(&eval("3<2").unwrap()), 0.0));
    }

    #[test]
    fn comparison_ge_le() {
        assert!(approx(scalar(&eval("3>=3").unwrap()), 1.0));
        assert!(approx(scalar(&eval("2>=3").unwrap()), 0.0));
        assert!(approx(scalar(&eval("3<=3").unwrap()), 1.0));
        assert!(approx(scalar(&eval("4<=3").unwrap()), 0.0));
    }

    #[test]
    fn comparison_eq_ne() {
        assert!(approx(scalar(&eval("1==1").unwrap()), 1.0));
        assert!(approx(scalar(&eval("1==2").unwrap()), 0.0));
        assert!(approx(scalar(&eval("1!=2").unwrap()), 1.0));
        assert!(approx(scalar(&eval("1!=1").unwrap()), 0.0));
    }

    #[test]
    fn boolean_and_or() {
        assert!(approx(scalar(&eval("1&&1").unwrap()), 1.0));
        assert!(approx(scalar(&eval("1&&0").unwrap()), 0.0));
        assert!(approx(scalar(&eval("0||1").unwrap()), 1.0));
        assert!(approx(scalar(&eval("0||0").unwrap()), 0.0));
    }

    #[test]
    fn boolean_not() {
        assert!(approx(scalar(&eval("!0").unwrap()), 1.0));
        assert!(approx(scalar(&eval("!1").unwrap()), 0.0));
        assert!(approx(scalar(&eval("!5").unwrap()), 0.0));
    }

    #[test]
    fn comparison_array_broadcast() {
        let v = eval("{1,2,-1}>0").unwrap();
        assert!(approx_arr(&array(&v), &[1.0, 1.0, 0.0]));
    }

    #[test]
    fn lambda_define_and_call() {
        let mut vars = HashMap::new();
        let lam = eval("|x|(x*x)").unwrap();
        vars.insert("sq".to_string(), lam);
        assert!(approx(
            scalar(&evaluate_expression("sq(3)", &vars).unwrap()),
            9.0
        ));
    }

    #[test]
    fn lambda_call_with_array() {
        let mut vars = HashMap::new();
        let lam = eval("|x|(x*x)").unwrap();
        vars.insert("sq".to_string(), lam);
        let v = evaluate_expression("sq({1,2,4})", &vars).unwrap();
        assert!(approx_arr(&array(&v), &[1.0, 4.0, 16.0]));
    }

    #[test]
    fn lambda_literal_parse() {
        let mut vars = HashMap::new();
        let lam = evaluate_expression("|x|(x+1)", &vars).unwrap();
        vars.insert("inc".to_string(), lam);
        assert!(approx(
            scalar(&evaluate_expression("inc(5)", &vars).unwrap()),
            6.0
        ));
    }

    #[test]
    fn compound_boolean() {
        // (3>2) && (1==1) => 1 && 1 => 1
        assert!(approx(scalar(&eval("3>2&&1==1").unwrap()), 1.0));
        // (1>2) || (3<4) => 0 || 1 => 1
        assert!(approx(scalar(&eval("1>2||3<4").unwrap()), 1.0));
    }

    // Tuple tests

    #[test]
    fn tuple_creation() {
        let v = eval("(1, 2, 3)").unwrap();
        assert!(approx_arr(&tuple(&v), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn tuple_two_elements() {
        let v = eval("(5, 10)").unwrap();
        assert!(approx_arr(&tuple(&v), &[5.0, 10.0]));
    }

    #[test]
    fn tuple_with_expressions() {
        let v = eval("(1+1, 2*3, 10-3)").unwrap();
        assert!(approx_arr(&tuple(&v), &[2.0, 6.0, 7.0]));
    }

    #[test]
    fn tuple_flattening() {
        // ((1,2), 3) should flatten to (1, 2, 3)
        let v = eval("((1, 2), 3)").unwrap();
        assert!(approx_arr(&tuple(&v), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn tuple_nested_flattening() {
        // (1, (2, 3), 4) should flatten to (1, 2, 3, 4)
        let v = eval("(1, (2, 3), 4)").unwrap();
        assert!(approx_arr(&tuple(&v), &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn tuple_array_from_literal() {
        // {(1,2), (3,4)} creates a TupleArray
        let v = eval("{(1,2), (3,4)}").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn tuple_array_broadcast() {
        // (x, {1,2,3}) where x=5 should create TupleArray {(5,1), (5,2), (5,3)}
        let v = eval("(5, {1, 2, 3})").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0]));
    }

    #[test]
    fn tuple_array_two_arrays() {
        // ({1,2}, {10,20}) creates TupleArray {(1,10), (2,20)}
        let v = eval("({1, 2}, {10, 20})").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[1.0, 10.0, 2.0, 20.0]));
    }

    #[test]
    fn lambda_multi_param_parse() {
        let v = eval("|x,y|(x+y)").unwrap();
        match v {
            Value::Lambda { params, .. } => {
                assert_eq!(params, vec!["x", "y"]);
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn lambda_multi_param_call() {
        let mut vars = HashMap::new();
        let lam = eval("|x,y|(x+y)").unwrap();
        vars.insert("add".to_string(), lam);
        let result = eval_with("add((3, 4))", &vars).unwrap();
        assert!(approx(scalar(&result), 7.0));
    }

    #[test]
    fn lambda_multi_param_call_product() {
        let mut vars = HashMap::new();
        let lam = eval("|a,b|(a*b)").unwrap();
        vars.insert("mul".to_string(), lam);
        let result = eval_with("mul((5, 6))", &vars).unwrap();
        assert!(approx(scalar(&result), 30.0));
    }

    #[test]
    fn lambda_over_tuple_array() {
        let mut vars = HashMap::new();
        let lam = eval("|x,y|(x+y)").unwrap();
        vars.insert("add".to_string(), lam);
        // add applied to {(1,2), (3,4), (5,6)} should return {3, 7, 11}
        let result = eval_with("add({(1,2), (3,4), (5,6)})", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[3.0, 7.0, 11.0]));
    }

    #[test]
    fn lambda_over_tuple_array_broadcast() {
        let mut vars = HashMap::new();
        let lam = eval("|x,y|(x*y)").unwrap();
        vars.insert("mul".to_string(), lam);
        // mul applied to (2, {1,2,3}) => TupleArray {(2,1), (2,2), (2,3)} => {2, 4, 6}
        let result = eval_with("mul((2, {1, 2, 3}))", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[2.0, 4.0, 6.0]));
    }

    #[test]
    fn tuple_display() {
        let v = eval("(1, 2, 3)").unwrap();
        assert_eq!(format!("{}", v), "(1, 2, 3)");
    }

    #[test]
    fn tuple_array_display() {
        let v = eval("{(1, 2), (3, 4)}").unwrap();
        assert_eq!(format!("{}", v), "{(1, 2), (3, 4)}");
    }

    #[test]
    fn lambda_single_display() {
        let v = eval("|x|x+1").unwrap();
        assert_eq!(format!("{}", v), "|x|(x + 1)");
    }

    #[test]
    fn lambda_multi_display() {
        let v = eval("|a,b|a+b").unwrap();
        assert_eq!(format!("{}", v), "|a, b|(a + b)");
    }

    #[test]
    fn tuple_arithmetic_error() {
        // Arithmetic on tuples should return an error, not panic
        let result = eval("(1, 2) + 1");
        assert!(result.is_err());
    }

    #[test]
    fn grouping_paren_still_works() {
        // Single element in parens is grouping, not tuple
        assert!(approx(scalar(&eval("(2+3)").unwrap()), 5.0));
        assert!(approx(scalar(&eval("(((5)))").unwrap()), 5.0));
    }

    #[test]
    fn lambda_wrong_tuple_length_error() {
        let mut vars = HashMap::new();
        let lam = eval("|x,y|(x+y)").unwrap();
        vars.insert("f".to_string(), lam);
        // Calling with wrong tuple length should error
        let result = eval_with("f((1, 2, 3))", &vars);
        assert!(result.is_err());
    }

    #[test]
    fn three_param_lambda() {
        let mut vars = HashMap::new();
        let lam = eval("|a,b,c|(a+b+c)").unwrap();
        vars.insert("sum3".to_string(), lam);
        let result = eval_with("sum3((1, 2, 3))", &vars).unwrap();
        assert!(approx(scalar(&result), 6.0));
    }

    // Single-element tuple reduction tests

    #[test]
    fn single_element_tuple_reduces_to_scalar() {
        // A tuple with one element should become a scalar
        // We can test this via decomposition: (a, b) = (1, 2) then b should be scalar 2
        let t = Value::Tuple(vec![1.0, 2.0]);
        let decomposed = t.decompose_tuple(2).unwrap();
        assert!(approx(scalar(&decomposed[0]), 1.0));
        assert!(approx(scalar(&decomposed[1]), 2.0));
    }

    #[test]
    fn decompose_tuple_rest_is_tuple() {
        // (a, b) = (1, 2, 3) => a=1, b=(2,3)
        let t = Value::Tuple(vec![1.0, 2.0, 3.0]);
        let decomposed = t.decompose_tuple(2).unwrap();
        assert!(approx(scalar(&decomposed[0]), 1.0));
        assert!(approx_arr(&tuple(&decomposed[1]), &[2.0, 3.0]));
    }

    #[test]
    fn decompose_tuple_three_vars() {
        // (a, b, c) = (1, 2, 3, 4) => a=1, b=2, c=(3,4)
        let t = Value::Tuple(vec![1.0, 2.0, 3.0, 4.0]);
        let decomposed = t.decompose_tuple(3).unwrap();
        assert!(approx(scalar(&decomposed[0]), 1.0));
        assert!(approx(scalar(&decomposed[1]), 2.0));
        assert!(approx_arr(&tuple(&decomposed[2]), &[3.0, 4.0]));
    }

    #[test]
    fn decompose_tuple_exact_match() {
        // (a, b, c) = (1, 2, 3) => a=1, b=2, c=3
        let t = Value::Tuple(vec![1.0, 2.0, 3.0]);
        let decomposed = t.decompose_tuple(3).unwrap();
        assert!(approx(scalar(&decomposed[0]), 1.0));
        assert!(approx(scalar(&decomposed[1]), 2.0));
        assert!(approx(scalar(&decomposed[2]), 3.0));
    }

    #[test]
    fn decompose_tuple_not_enough_elements_error() {
        let t = Value::Tuple(vec![1.0]);
        assert!(t.decompose_tuple(2).is_err());
    }

    // Array indexing tests

    #[test]
    fn array_index_scalar() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![10.0, 20.0, 30.0]));
        let result = eval_with("arr[0]", &vars).unwrap();
        assert!(approx(scalar(&result), 10.0));
        let result = eval_with("arr[2]", &vars).unwrap();
        assert!(approx(scalar(&result), 30.0));
    }

    #[test]
    fn array_index_expression() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![10.0, 20.0, 30.0]));
        let result = eval_with("arr[1+1]", &vars).unwrap();
        assert!(approx(scalar(&result), 30.0));
    }

    #[test]
    fn array_index_out_of_bounds() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![10.0, 20.0, 30.0]));
        let result = eval_with("arr[5]", &vars);
        assert!(result.is_err());
    }

    // Array filter/comprehension tests

    #[test]
    fn array_filter_lambda_two_params() {
        let mut vars = HashMap::new();
        vars.insert(
            "arr".to_string(),
            Value::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        // Filter elements > 2: arr[pred] where pred = |i,x|(x > 2)
        let lam = eval("|i,x|(x > 2)").unwrap();
        vars.insert("pred".to_string(), lam);
        let result = eval_with("arr[pred]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[3.0, 4.0, 5.0]));
    }

    #[test]
    fn array_filter_even_indices() {
        let mut vars = HashMap::new();
        vars.insert(
            "arr".to_string(),
            Value::Array(vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        );
        // Filter even indices: i % 2 == 0
        // We can use floor(i/2)*2 == i to check evenness
        let lam = eval("|i,x|(floor(i/2)*2 == i)").unwrap();
        vars.insert("even_idx".to_string(), lam);
        let result = eval_with("arr[even_idx]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[10.0, 30.0, 50.0]));
    }

    #[test]
    fn array_filter_positive() {
        let mut vars = HashMap::new();
        vars.insert(
            "arr".to_string(),
            Value::Array(vec![-2.0, -1.0, 0.0, 1.0, 2.0]),
        );
        let lam = eval("|i,x|(x > 0)").unwrap();
        vars.insert("pos".to_string(), lam);
        let result = eval_with("arr[pos]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[1.0, 2.0]));
    }

    #[test]
    fn array_filter_empty_result() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![1.0, 2.0, 3.0]));
        let lam = eval("|i,x|(x > 100)").unwrap();
        vars.insert("never".to_string(), lam);
        let result = eval_with("arr[never]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[]));
    }

    #[test]
    fn array_filter_all_pass() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![1.0, 2.0, 3.0]));
        let lam = eval("|i,x|(1)").unwrap();
        vars.insert("always".to_string(), lam);
        let result = eval_with("arr[always]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn range_then_filter() {
        // [10][gt5] where gt5 = |i,x|(x > 5) should give {6, 7, 8, 9}
        let mut vars = HashMap::new();
        let lam = eval("|i,x|(x > 5)").unwrap();
        vars.insert("gt5".to_string(), lam);
        let result = eval_with("[10][gt5]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[6.0, 7.0, 8.0, 9.0]));
    }

    #[test]
    fn tuple_array_index() {
        let mut vars = HashMap::new();
        let ta = Value::TupleArray {
            width: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        vars.insert("ta".to_string(), ta);
        // ta[0] should give (1, 2)
        let result = eval_with("ta[0]", &vars).unwrap();
        assert!(approx_arr(&tuple(&result), &[1.0, 2.0]));
        // ta[2] should give (5, 6)
        let result = eval_with("ta[2]", &vars).unwrap();
        assert!(approx_arr(&tuple(&result), &[5.0, 6.0]));
    }

    #[test]
    fn tuple_array_filter() {
        let mut vars = HashMap::new();
        // {(1,10), (2,20), (3,30)} filter where first element > 1
        let ta = Value::TupleArray {
            width: 2,
            data: vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        };
        vars.insert("ta".to_string(), ta);
        // Filter: keep tuples where x > 1 (x is the first element of the tuple)
        let lam = eval("|i,x,y|(x > 1)").unwrap();
        vars.insert("f".to_string(), lam);
        let result = eval_with("ta[f]", &vars).unwrap();
        let (w, d) = tuple_array(&result);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[2.0, 20.0, 3.0, 30.0]));
    }

    #[test]
    fn chained_indexing() {
        // Test chained indexing: create array, then filter, then index
        let mut vars = HashMap::new();
        vars.insert(
            "arr".to_string(),
            Value::Array(vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        );
        let lam = eval("|i,x|(x > 20)").unwrap();
        vars.insert("gt20".to_string(), lam);
        // arr[gt20][0] should give 30
        let result = eval_with("arr[gt20][0]", &vars).unwrap();
        assert!(approx(scalar(&result), 30.0));
    }

    #[test]
    fn filter_with_precision_tolerance() {
        // Test that values very close to zero are treated as falsy
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![1.0, 2.0, 3.0]));
        // A tiny value below the 1e-10 threshold should be falsy
        let lam = eval("|i,x|(0.00000000001)").unwrap(); // 1e-11, below threshold
        vars.insert("tiny".to_string(), lam);
        let result = eval_with("arr[tiny]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[])); // All filtered out
    }

    // Parse assignment tests

    #[test]
    fn parse_tuple_pattern_valid() {
        assert_eq!(
            parse_tuple_pattern("(a, b)"),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            parse_tuple_pattern("(x, y, z)"),
            Some(vec!["x".to_string(), "y".to_string(), "z".to_string()])
        );
    }

    #[test]
    fn parse_tuple_pattern_invalid() {
        assert_eq!(parse_tuple_pattern("(a)"), None); // single element
        assert_eq!(parse_tuple_pattern("a, b"), None); // no parens
        assert_eq!(parse_tuple_pattern("(1, 2)"), None); // numbers not idents
    }

    // Lambda decomposition tests

    #[test]
    fn lambda_decomposition_call() {
        // f((1, 2, 3)) with |a, rest|(a) should bind a=1, rest=(2,3)
        let mut vars = HashMap::new();
        let lam = eval("|a, rest|(a)").unwrap();
        vars.insert("first".to_string(), lam);
        let result = eval_with("first((1, 2, 3))", &vars).unwrap();
        assert!(approx(scalar(&result), 1.0));
    }

    #[test]
    fn lambda_decomposition_rest_is_tuple() {
        // Access rest as tuple - rest should be (2, 3)
        let mut vars = HashMap::new();
        let lam = eval("|a, rest|(rest)").unwrap();
        vars.insert("tail".to_string(), lam);
        let result = eval_with("tail((1, 2, 3))", &vars).unwrap();
        assert!(approx_arr(&tuple(&result), &[2.0, 3.0]));
    }

    #[test]
    fn lambda_decomposition_exact_match() {
        // |a, b|(a+b) with (1, 2) - exact match, b gets scalar
        let mut vars = HashMap::new();
        let lam = eval("|a, b|(a + b)").unwrap();
        vars.insert("add".to_string(), lam);
        let result = eval_with("add((1, 2))", &vars).unwrap();
        assert!(approx(scalar(&result), 3.0));
    }

    #[test]
    fn filter_lambda_decomposition() {
        // TupleArray of width 3, filter with |i, rest|(...)
        // rest should be the tuple (a, b, c)
        let mut vars = HashMap::new();
        let ta = Value::TupleArray {
            width: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        vars.insert("ta".to_string(), ta);
        // Filter where index == 0
        let lam = eval("|i, rest|(i == 0)").unwrap();
        vars.insert("first_only".to_string(), lam);
        let result = eval_with("ta[first_only]", &vars).unwrap();
        let (w, d) = tuple_array(&result);
        assert_eq!(w, 3);
        assert!(approx_arr(&d, &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn filter_lambda_decomposition_two_params() {
        // Array filter with |i, x|(i == 1) - x is the element
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![10.0, 20.0, 30.0]));
        let lam = eval("|i, x|(i == 1)").unwrap();
        vars.insert("second".to_string(), lam);
        let result = eval_with("arr[second]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[20.0]));
    }

    #[test]
    fn tuple_array_lambda_decomposition() {
        // Map over TupleArray with decomposition
        // {(1,2,3), (4,5,6)} with |a, rest|(a) should give {1, 4}
        let mut vars = HashMap::new();
        let ta = Value::TupleArray {
            width: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        vars.insert("ta".to_string(), ta);
        let lam = eval("|a, rest|(a)").unwrap();
        vars.insert("first_elem".to_string(), lam);
        let result = eval_with("first_elem(ta)", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[1.0, 4.0]));
    }

    #[test]
    fn filter_three_dim_tuple_array_by_index() {
        // This is the user's example: x[|i,y|(i==0)] on 3D tuple array
        let mut vars = HashMap::new();
        let ta = Value::TupleArray {
            width: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        vars.insert("x".to_string(), ta);
        let lam = eval("|i, y|(i == 0)").unwrap();
        vars.insert("f".to_string(), lam);
        let result = eval_with("x[f]", &vars).unwrap();
        let (w, d) = tuple_array(&result);
        assert_eq!(w, 3);
        assert!(approx_arr(&d, &[1.0, 2.0, 3.0])); // First element only
    }

    // Tensor product tests

    #[test]
    fn tensor_product_scalars() {
        // 1 ** 2 = (1, 2)
        let v = eval("1 ** 2").unwrap();
        assert!(approx_arr(&tuple(&v), &[1.0, 2.0]));
    }

    #[test]
    fn tensor_product_scalar_array() {
        // 5 ** {1, 2, 3} = {(5, 1), (5, 2), (5, 3)}
        let v = eval("5 ** {1, 2, 3}").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0]));
    }

    #[test]
    fn tensor_product_array_scalar() {
        // {1, 2} ** 10 = {(1, 10), (2, 10)}
        let v = eval("{1, 2} ** 10").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[1.0, 10.0, 2.0, 10.0]));
    }

    #[test]
    fn tensor_product_arrays() {
        // {1, 2} ** {10, 20, 30} = {(1,10), (1,20), (1,30), (2,10), (2,20), (2,30)}
        let v = eval("{1, 2} ** {10, 20, 30}").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(
            &d,
            &[
                1.0, 10.0, 1.0, 20.0, 1.0, 30.0, 2.0, 10.0, 2.0, 20.0, 2.0, 30.0
            ]
        ));
    }

    #[test]
    fn tensor_product_tuple_scalar() {
        // (1, 2) ** 3 = (1, 2, 3)
        let v = eval("(1, 2) ** 3").unwrap();
        assert!(approx_arr(&tuple(&v), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn tensor_product_tuples() {
        // (1, 2) ** (3, 4) = (1, 2, 3, 4)
        let v = eval("(1, 2) ** (3, 4)").unwrap();
        assert!(approx_arr(&tuple(&v), &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn tensor_product_tuple_array() {
        // (1, 2) ** {10, 20} = {(1, 2, 10), (1, 2, 20)}
        let v = eval("(1, 2) ** {10, 20}").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 3);
        assert!(approx_arr(&d, &[1.0, 2.0, 10.0, 1.0, 2.0, 20.0]));
    }

    #[test]
    fn tensor_product_array_tuple() {
        // {1, 2} ** (10, 20) = {(1, 10, 20), (2, 10, 20)}
        let v = eval("{1, 2} ** (10, 20)").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 3);
        assert!(approx_arr(&d, &[1.0, 10.0, 20.0, 2.0, 10.0, 20.0]));
    }

    #[test]
    fn tensor_product_tuple_arrays() {
        // {(1, 2), (3, 4)} ** {10, 20} = {(1,2,10), (1,2,20), (3,4,10), (3,4,20)}
        let v = eval("{(1, 2), (3, 4)} ** {10, 20}").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 3);
        assert!(approx_arr(
            &d,
            &[
                1.0, 2.0, 10.0, 1.0, 2.0, 20.0, 3.0, 4.0, 10.0, 3.0, 4.0, 20.0
            ]
        ));
    }

    #[test]
    fn tensor_product_ranges() {
        // [2] ** [3] = {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)}
        let v = eval("[2] ** [3]").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(
            &d,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0]
        ));
    }

    #[test]
    fn tensor_product_empty_array() {
        // {} ** {1, 2} = {}
        let mut vars = HashMap::new();
        vars.insert("empty".to_string(), Value::Array(vec![]));
        let result = eval_with("empty ** {1, 2}", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[]));
    }

    #[test]
    fn tensor_product_chained() {
        // [2] ** [2] ** [2] = 8 elements
        let v = eval("[2] ** [2] ** [2]").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 3);
        // Should have 2*2*2 = 8 tuples, each with 3 elements
        assert_eq!(d.len(), 24);
    }

    #[test]
    fn tensor_product_with_lambda() {
        // Create tensor product, then apply lambda
        let mut vars = HashMap::new();
        let lam = eval("|x, y|(x + y)").unwrap();
        vars.insert("add".to_string(), lam);
        // {1, 2} ** {10, 20} = {(1,10), (1,20), (2,10), (2,20)}
        // add on that = {11, 21, 12, 22}
        let result = eval_with("add({1, 2} ** {10, 20})", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[11.0, 21.0, 12.0, 22.0]));
    }

    // Line wrapping tests

    #[test]
    fn wrap_line_short() {
        // Short line that fits
        let lines = wrap_line("hello", 10);
        assert_eq!(lines, vec!["hello"]);
    }

    #[test]
    fn wrap_line_exact() {
        // Line exactly at width
        let lines = wrap_line("hello", 5);
        assert_eq!(lines, vec!["hello"]);
    }

    #[test]
    fn wrap_line_long() {
        // Line needs wrapping
        let lines = wrap_line("hello world", 5);
        assert_eq!(lines, vec!["hello", " worl", "d"]);
    }

    #[test]
    fn wrap_line_long_result() {
        // Simulating a long result like a tensor product
        let long = "{(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}";
        let lines = wrap_line(long, 20);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "{(0, 0), (0, 1), (0,");
        assert_eq!(lines[1], " 2), (1, 0), (1, 1),");
        assert_eq!(lines[2], " (1, 2)}");
    }

    #[test]
    fn wrap_line_empty() {
        let lines = wrap_line("", 10);
        assert_eq!(lines, Vec::<String>::new());
    }

    // Inline lambda tests

    #[test]
    fn inline_lambda_single_param() {
        // Single param lambda used in function call (not filter)
        let mut vars = HashMap::new();
        vars.insert("sq".to_string(), eval("|x|(x*x)").unwrap());
        let v = eval_with("sq(5)", &vars).unwrap();
        assert!(approx(scalar(&v), 25.0));
    }

    #[test]
    fn inline_lambda_multi_param() {
        // Multi-param lambda inline
        let v = eval("{1, 2, 3, 4, 5}[|i,x|(x > 2)]").unwrap();
        assert!(approx_arr(&array(&v), &[3.0, 4.0, 5.0]));
    }

    #[test]
    fn inline_lambda_tensor_filter() {
        // The user's example: ([10]**[10])[|i,x,y|(x*y > 10)]
        let v = eval("([3]**[3])[|i,x,y|(x*y > 1)]").unwrap();
        // 3x3 = pairs (0,0)..(2,2), filter where x*y > 1
        // (1,2), (2,1), (2,2) have products 2, 2, 4 > 1
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[1.0, 2.0, 2.0, 1.0, 2.0, 2.0]));
    }

    #[test]
    fn inline_lambda_as_value() {
        // Lambda literal evaluates to a lambda value
        let v = eval("|x|(x + 1)").unwrap();
        match v {
            Value::Lambda { params, .. } => {
                assert_eq!(params, vec!["x"]);
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn user_example_tensor_filter() {
        // The exact example the user asked about (with new syntax)
        let v = eval("([10]**[10])[|i,x,y|(x*y>10)]").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(d.len() > 0);
        for chunk in d.chunks(2) {
            assert!(chunk[0] * chunk[1] > 10.0);
        }
    }

    #[test]
    fn lambda_filter() {
        // |x|(body) syntax in filter
        let v = eval("{1, 2, 3, 4, 5}[|i,x|(x > 2)]").unwrap();
        assert!(approx_arr(&array(&v), &[3.0, 4.0, 5.0]));
    }

    #[test]
    fn lambda_tensor_filter() {
        // Lambda filter on tensor product
        let v = eval("([10]**[10])[|i,x,y|(x*y>10)]").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(d.len() > 0);
        for chunk in d.chunks(2) {
            assert!(chunk[0] * chunk[1] > 10.0);
        }
    }

    #[test]
    fn lambda_as_value() {
        // Lambda at top level
        let v = eval("|x|(x + 1)").unwrap();
        match v {
            Value::Lambda { params, .. } => {
                assert_eq!(params, vec!["x"]);
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn lambda_multi_param() {
        // Multi-param lambda
        let v = eval("|x,y|(x + y)").unwrap();
        match v {
            Value::Lambda { params, .. } => {
                assert_eq!(params, vec!["x", "y"]);
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn logical_or_not_confused_with_lambda() {
        // || should still be logical or, not lambda
        let v = eval("1 || 0").unwrap();
        assert!(approx(scalar(&v), 1.0));
    }

    // AST-based lambda tests

    #[test]
    fn ast_lambda_nested_arithmetic() {
        // Lambda body with complex arithmetic
        let mut vars = HashMap::new();
        let lam = eval("|x|(x*x + 2*x + 1)").unwrap();
        vars.insert("f".to_string(), lam);
        // f(3) = 9 + 6 + 1 = 16
        let result = eval_with("f(3)", &vars).unwrap();
        assert!(approx(scalar(&result), 16.0));
    }

    #[test]
    fn ast_lambda_with_builtin() {
        // Lambda using builtin function
        let mut vars = HashMap::new();
        let lam = eval("|x|(sqrt(x))").unwrap();
        vars.insert("sq".to_string(), lam);
        let result = eval_with("sq(16)", &vars).unwrap();
        assert!(approx(scalar(&result), 4.0));
    }

    #[test]
    fn ast_lambda_comparison_in_body() {
        // Lambda with comparison operators
        let mut vars = HashMap::new();
        let lam = eval("|x|(x > 5)").unwrap();
        vars.insert("gt5".to_string(), lam);
        assert!(approx(scalar(&eval_with("gt5(10)", &vars).unwrap()), 1.0));
        assert!(approx(scalar(&eval_with("gt5(3)", &vars).unwrap()), 0.0));
    }

    #[test]
    fn ast_lambda_boolean_ops() {
        // Lambda with boolean operators
        let mut vars = HashMap::new();
        let lam = eval("|x|(x > 0 && x < 10)").unwrap();
        vars.insert("inrange".to_string(), lam);
        assert!(approx(
            scalar(&eval_with("inrange(5)", &vars).unwrap()),
            1.0
        ));
        assert!(approx(
            scalar(&eval_with("inrange(-1)", &vars).unwrap()),
            0.0
        ));
        assert!(approx(
            scalar(&eval_with("inrange(15)", &vars).unwrap()),
            0.0
        ));
    }

    #[test]
    fn ast_lambda_inline_in_filter() {
        // Inline lambda directly in filter expression (needs i,x pattern)
        let v = eval("{1, 2, 3, 4, 5}[|i,x|(x > 3)]").unwrap();
        assert!(approx_arr(&array(&v), &[4.0, 5.0]));
    }

    #[test]
    fn ast_lambda_inline_multi_param_filter() {
        // Inline multi-param lambda in filter
        let v = eval("{10, 20, 30}[|i,x|(i == 1)]").unwrap();
        assert!(approx_arr(&array(&v), &[20.0]));
    }

    #[test]
    fn ast_lambda_power_in_body() {
        // Lambda with power operator
        let mut vars = HashMap::new();
        let lam = eval("|x|(x^3)").unwrap();
        vars.insert("cube".to_string(), lam);
        let result = eval_with("cube(2)", &vars).unwrap();
        assert!(approx(scalar(&result), 8.0));
    }

    #[test]
    fn ast_lambda_negation() {
        // Lambda with unary negation
        let mut vars = HashMap::new();
        let lam = eval("|x|(-x)").unwrap();
        vars.insert("neg".to_string(), lam);
        let result = eval_with("neg(5)", &vars).unwrap();
        assert!(approx(scalar(&result), -5.0));
    }

    #[test]
    fn ast_lambda_not_operator() {
        // Lambda with logical not
        let mut vars = HashMap::new();
        let lam = eval("|x|(!(x > 5))").unwrap();
        vars.insert("le5".to_string(), lam);
        assert!(approx(scalar(&eval_with("le5(3)", &vars).unwrap()), 1.0));
        assert!(approx(scalar(&eval_with("le5(10)", &vars).unwrap()), 0.0));
    }

    #[test]
    fn ast_tensor_product_filter_inline() {
        // The original user example with new syntax
        let v = eval("([5]**[5])[|i,x,y|(x*y > 6)]").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        // All pairs where product > 6: (2,4), (3,3), (3,4), (4,2), (4,3), (4,4)
        for chunk in d.chunks(2) {
            assert!(chunk[0] * chunk[1] > 6.0);
        }
    }

    #[test]
    fn ast_lambda_with_constants() {
        // Lambda using pi and e constants
        let mut vars = HashMap::new();
        let lam = eval("|x|(x * pi)").unwrap();
        vars.insert("mult_pi".to_string(), lam);
        let result = eval_with("mult_pi(2)", &vars).unwrap();
        assert!(approx(scalar(&result), 2.0 * std::f64::consts::PI));
    }

    #[test]
    fn ast_lambda_array_input() {
        // Lambda applied to array
        let mut vars = HashMap::new();
        let lam = eval("|x|(x * 2)").unwrap();
        vars.insert("double".to_string(), lam);
        let result = eval_with("double({1, 2, 3})", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[2.0, 4.0, 6.0]));
    }

    #[test]
    fn ast_lambda_tuple_sum() {
        // Multi-param lambda summing tuple elements
        let mut vars = HashMap::new();
        let lam = eval("|a,b,c|(a + b + c)").unwrap();
        vars.insert("sum3".to_string(), lam);
        let result = eval_with("sum3((10, 20, 30))", &vars).unwrap();
        assert!(approx(scalar(&result), 60.0));
    }

    #[test]
    fn ast_lambda_tuple_product() {
        // Multi-param lambda multiplying tuple elements
        let mut vars = HashMap::new();
        let lam = eval("|x,y|(x * y)").unwrap();
        vars.insert("prod".to_string(), lam);
        let result = eval_with("prod((7, 8))", &vars).unwrap();
        assert!(approx(scalar(&result), 56.0));
    }

    #[test]
    fn ast_lambda_complex_filter_condition() {
        // Complex filter: keep elements where (i + x) is even
        let v = eval("{10, 11, 12, 13, 14}[|i,x|(floor((i+x)/2)*2 == i+x)]").unwrap();
        // i=0,x=10: 10 even -> keep
        // i=1,x=11: 12 even -> keep
        // i=2,x=12: 14 even -> keep
        // i=3,x=13: 16 even -> keep
        // i=4,x=14: 18 even -> keep
        assert!(approx_arr(&array(&v), &[10.0, 11.0, 12.0, 13.0, 14.0]));
    }

    #[test]
    fn ast_define_and_use_lambda() {
        // Define lambda in vars and use it
        let mut vars = HashMap::new();
        let lam = eval("|n|(n * (n + 1) / 2)").unwrap();
        vars.insert("triangular".to_string(), lam);
        // triangular(5) = 5*6/2 = 15
        let result = eval_with("triangular(5)", &vars).unwrap();
        assert!(approx(scalar(&result), 15.0));
    }

    // Fold/reduce tests

    #[test]
    fn fold_sum() {
        // Sum of {1,2,3,4,5} = 15
        let v = eval("{1,2,3,4,5} @ 0 {|a,b|(a+b)}").unwrap();
        assert!(approx(scalar(&v), 15.0));
    }

    #[test]
    fn fold_product() {
        // Product of {1,2,3,4} = 24
        let v = eval("{1,2,3,4} @ 1 {|a,b|(a*b)}").unwrap();
        assert!(approx(scalar(&v), 24.0));
    }

    #[test]
    fn fold_max() {
        // Max of {3,1,4,1,5,9,2,6} starting from 0
        let v = eval("{3,1,4,1,5,9,2,6} @ 0 {|max,x|(max * (max > x) + x * (x >= max))}").unwrap();
        assert!(approx(scalar(&v), 9.0));
    }

    #[test]
    fn fold_count() {
        // Count elements > 3
        let v = eval("{1,2,3,4,5,6} @ 0 {|cnt,x|(cnt + (x > 3))}").unwrap();
        assert!(approx(scalar(&v), 3.0)); // 4, 5, 6
    }

    #[test]
    fn fold_range_sum() {
        // Sum of [10] = 0+1+2+...+9 = 45
        let v = eval("[10] @ 0 {|a,b|(a+b)}").unwrap();
        assert!(approx(scalar(&v), 45.0));
    }

    #[test]
    fn fold_factorial() {
        // 5! = 1*2*3*4*5 = 120 using [5]+1 = {1,2,3,4,5}
        // We need to use range and add 1 to each
        let v = eval("([5]+1) @ 1 {|a,b|(a*b)}").unwrap();
        assert!(approx(scalar(&v), 120.0));
    }

    #[test]
    fn fold_with_tuple_accumulator() {
        // Keep running sum and count: start with (0, 0), return (sum, count)
        // {1,2,3} -> (0,0) -> (1,1) -> (3,2) -> (6,3)
        // Need double parens: outer for lambda body, inner for tuple
        let v = eval("{1,2,3} @ (0,0) {|sum,cnt,x|((sum+x, cnt+1))}").unwrap();
        assert!(approx_arr(&tuple(&v), &[6.0, 3.0]));
    }

    #[test]
    fn fold_tuple_array() {
        // Sum x*y for each tuple in tuple array
        // {(1,2), (3,4), (5,6)} -> 1*2 + 3*4 + 5*6 = 2 + 12 + 30 = 44
        let v = eval("{(1,2), (3,4), (5,6)} @ 0 {|acc,x,y|(acc + x*y)}").unwrap();
        assert!(approx(scalar(&v), 44.0));
    }

    #[test]
    fn fold_empty_array() {
        // Fold on empty array returns init
        let mut vars = HashMap::new();
        vars.insert("empty".to_string(), Value::Array(vec![]));
        let v = eval_with("empty @ 42 {|a,b|(a+b)}", &vars).unwrap();
        assert!(approx(scalar(&v), 42.0));
    }

    #[test]
    fn fold_single_element() {
        // Fold on single element: init combined with that element
        let v = eval("{5} @ 10 {|a,b|(a+b)}").unwrap();
        assert!(approx(scalar(&v), 15.0));
    }

    // Lambda grammar tests - lambdas should only appear in specific places

    #[test]
    fn lambda_not_in_arithmetic() {
        // -|x|x should not parse - lambda can't be operand of unary minus
        assert!(eval("-|x|x").is_err());
    }

    #[test]
    fn lambda_not_multipliable() {
        // 2 * |x|x should not parse - lambda can't be RHS of multiplication
        assert!(eval("2 * |x|x").is_err());
    }

    #[test]
    fn lambda_no_parens() {
        // Lambda without parens around body should work
        let mut vars = HashMap::new();
        let lam = eval("|x|x*x").unwrap();
        vars.insert("sq".to_string(), lam);
        assert!(approx(scalar(&eval_with("sq(3)", &vars).unwrap()), 9.0));
    }

    #[test]
    fn lambda_no_parens_multi_param() {
        // Multi-param lambda without parens
        let mut vars = HashMap::new();
        let lam = eval("|x,y|x+y").unwrap();
        vars.insert("add".to_string(), lam);
        assert!(approx(scalar(&eval_with("add((2, 3))", &vars).unwrap()), 5.0));
    }

    #[test]
    fn lambda_body_greedy() {
        // Lambda body should consume entire expression
        let mut vars = HashMap::new();
        let lam = eval("|x|x + 1").unwrap();
        vars.insert("inc".to_string(), lam);
        assert!(approx(scalar(&eval_with("inc(5)", &vars).unwrap()), 6.0));
    }

    #[test]
    fn lambda_standalone_ok() {
        // Lambda at top level should work (for assignment)
        let v = eval("|x|x*x").unwrap();
        match v {
            Value::Lambda { params, .. } => assert_eq!(params, vec!["x"]),
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn lambda_in_parens_ok() {
        // Lambda wrapped in parens should work
        let v = eval("(|x|x+1)").unwrap();
        match v {
            Value::Lambda { params, .. } => assert_eq!(params, vec!["x"]),
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn lambda_in_filter_ok() {
        // Lambda in array filter should work (no parens needed)
        let v = eval("{1,2,3,4,5}[|i,x|x > 2]").unwrap();
        assert!(approx_arr(&array(&v), &[3.0, 4.0, 5.0]));
    }

    #[test]
    fn lambda_in_fold_ok() {
        // Lambda in fold should work (no parens needed)
        let v = eval("{1,2,3} @ 0 {|a,b|a+b}").unwrap();
        assert!(approx(scalar(&v), 6.0));
    }

    #[test]
    fn nested_lambda_ok() {
        // Nested lambda (currying) - needs space or parens to avoid || being parsed as OR
        let v = eval("|x| |y|x+y").unwrap();
        match v {
            Value::Lambda { params, .. } => assert_eq!(params, vec!["x"]),
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn nested_lambda_with_parens() {
        // Nested lambda with explicit parens also works
        let v = eval("|x|(|y|x+y)").unwrap();
        match v {
            Value::Lambda { params, .. } => assert_eq!(params, vec!["x"]),
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn lambda_with_boolean_body() {
        // Lambda body can include boolean operators
        let mut vars = HashMap::new();
        let lam = eval("|x|x > 0 && x < 10").unwrap();
        vars.insert("inrange".to_string(), lam);
        assert!(approx(scalar(&eval_with("inrange(5)", &vars).unwrap()), 1.0));
        assert!(approx(scalar(&eval_with("inrange(15)", &vars).unwrap()), 0.0));
    }
}
