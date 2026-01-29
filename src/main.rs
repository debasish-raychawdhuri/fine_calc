use ncurses::*;
use std::collections::HashMap;

mod value;
use value::Value;

lalrpop_util::lalrpop_mod!(#[allow(clippy::all)] expr);

/// Try to parse a lambda literal from a string starting at position `start`.
/// Returns Some((Lambda, end_position)) if successful, None otherwise.
fn parse_lambda_at(s: &str, start: usize) -> Option<(Value, usize)> {
    let rest = &s[start..];
    if !rest.starts_with("(|") {
        return None;
    }

    let inner_start = start + 2; // after "(|"
    let inner = &s[inner_start..];

    // Find the second | that ends params
    let pipe_pos = inner.find('|')?;
    let param_part = inner[..pipe_pos].trim();

    // Find matching ) for the lambda - need to track paren depth
    let body_start = inner_start + pipe_pos + 1;
    let mut depth = 1; // we're inside one (
    let mut end_pos = body_start;
    for (i, c) in s[body_start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end_pos = body_start + i;
                    break;
                }
            }
            _ => {}
        }
    }

    if depth != 0 {
        return None; // unbalanced
    }

    let body = s[body_start..end_pos].trim();
    if body.is_empty() {
        return None;
    }

    // Parse params
    let params = if param_part.starts_with('(') && param_part.ends_with(')') {
        // Multi-param: (x,y,z)
        let params_str = &param_part[1..param_part.len() - 1];
        let params: Vec<String> = params_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let valid = params.iter().all(|p| {
            !p.is_empty()
                && p.chars().all(|c| c.is_alphanumeric() || c == '_')
                && (p.starts_with('_') || p.chars().next().unwrap().is_alphabetic())
        });
        if !valid || params.is_empty() {
            return None;
        }
        params
    } else {
        // Single param: x
        if param_part.is_empty()
            || !param_part.chars().all(|c| c.is_alphanumeric() || c == '_')
            || (!param_part.starts_with('_') && !param_part.chars().next().unwrap().is_alphabetic())
        {
            return None;
        }
        vec![param_part.to_string()]
    };

    Some((Value::Lambda { params, body: body.to_string() }, end_pos + 1))
}

/// Extract all lambda literals from input, replacing them with placeholder variables.
/// Returns (modified_input, map of placeholder -> Lambda).
fn extract_lambdas(input: &str) -> (String, HashMap<String, Value>) {
    let mut result = String::new();
    let mut lambdas = HashMap::new();
    let mut i = 0;
    let mut lambda_count = 0;

    while i < input.len() {
        if input[i..].starts_with("(|") {
            if let Some((lambda, end)) = parse_lambda_at(input, i) {
                let placeholder = format!("__lambda{}__", lambda_count);
                lambda_count += 1;
                lambdas.insert(placeholder.clone(), lambda);
                result.push_str(&placeholder);
                i = end;
                continue;
            }
        }
        result.push(input[i..].chars().next().unwrap());
        i += input[i..].chars().next().unwrap().len_utf8();
    }

    (result, lambdas)
}

fn evaluate_expression(input: &str, vars: &HashMap<String, Value>) -> Result<Value, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Empty expression".to_string());
    }

    // Extract inline lambdas and replace with placeholders
    let (processed_input, extracted_lambdas) = extract_lambdas(input);

    // Merge extracted lambdas with existing vars
    let mut merged_vars = vars.clone();
    for (name, lambda) in extracted_lambdas {
        merged_vars.insert(name, lambda);
    }

    expr::TopExprParser::new()
        .parse(&merged_vars, &processed_input)
        .map_err(|e| match e {
            lalrpop_util::ParseError::User { error } => error.to_string(),
            other => format!("{}", other),
        })
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
        let split_at = remaining.char_indices()
            .take(width)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(remaining.len());
        lines.push(remaining[..split_at].to_string());
        remaining = &remaining[split_at..];
    }
    lines
}

fn main() {
    initscr();
    cbreak();
    keypad(stdscr(), true);
    noecho();

    start_color();
    use_default_colors();
    // History area colors (dark grey background)
    init_pair(1, COLOR_GREEN, 236);   // result
    init_pair(2, COLOR_RED, 236);     // error
    init_pair(3, COLOR_CYAN, -1);     // prompt (input row, default bg)
    init_pair(4, COLOR_WHITE, 236);   // normal history text
    init_pair(5, COLOR_CYAN, 236);    // prompt in history
    init_pair(6, COLOR_WHITE, 236);   // separator line

    let mut input = String::new();
    let mut cursor: usize = 0;
    let mut history: Vec<String> = Vec::new();
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

            // Fill with dark grey background
            attron(COLOR_PAIR(4));
            for _ in 0..inner_w {
                addch(' ' as u32);
            }
            attroff(COLOR_PAIR(4));

            addch(ACS_VLINE());

            let idx = scroll_offset + row;
            if idx >= history.len() {
                continue;
            }
            let line = &history[idx];
            mv(y, 1);
            if line.starts_with("Error:") {
                attron(COLOR_PAIR(2));
                addstr(line);
                attroff(COLOR_PAIR(2));
            } else if line.starts_with(">> ") {
                attron(COLOR_PAIR(5));
                addstr(">> ");
                attroff(COLOR_PAIR(5));
                attron(COLOR_PAIR(4));
                addstr(&line[3..]);
                attroff(COLOR_PAIR(4));
            } else {
                attron(COLOR_PAIR(1));
                addstr(line);
                attroff(COLOR_PAIR(1));
            }
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
                                history.push(line);
                            }
                            // Wrap result output
                            let result_str = format!("{}", &res);
                            for line in wrap_line(&result_str, inner_w) {
                                history.push(line);
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
                                            for line in wrap_line(&format!("Error: {}", e), inner_w) {
                                                history.push(line);
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
                                history.push(line);
                            }
                            for line in wrap_line(&format!("Error: {}", e), inner_w) {
                                history.push(line);
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
        evaluate_expression(expr, &HashMap::new())
    }

    fn eval_with(expr: &str, vars: &HashMap<String, Value>) -> Result<Value, String> {
        evaluate_expression(expr, vars)
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
        assert!(approx(scalar(&eval("asin(1)").unwrap()), std::f64::consts::FRAC_PI_2));
        assert!(approx(scalar(&eval("acos(1)").unwrap()), 0.0));
        assert!(approx(scalar(&eval("atan(1)").unwrap()), std::f64::consts::FRAC_PI_4));
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
        assert!(approx(scalar(&evaluate_expression("x+y", &vars).unwrap()), 13.0));
        assert!(approx(scalar(&evaluate_expression("x*y", &vars).unwrap()), 30.0));
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
        let lam = Value::Lambda { params: vec!["x".to_string()], body: "x*x".to_string() };
        vars.insert("sq".to_string(), lam);
        assert!(approx(scalar(&evaluate_expression("sq(3)", &vars).unwrap()), 9.0));
    }

    #[test]
    fn lambda_call_with_array() {
        let mut vars = HashMap::new();
        let lam = Value::Lambda { params: vec!["x".to_string()], body: "x*x".to_string() };
        vars.insert("sq".to_string(), lam);
        let v = evaluate_expression("sq({1,2,4})", &vars).unwrap();
        assert!(approx_arr(&array(&v), &[1.0, 4.0, 16.0]));
    }

    #[test]
    fn lambda_literal_parse() {
        let mut vars = HashMap::new();
        let lam = evaluate_expression("(|x| x+1)", &vars).unwrap();
        vars.insert("inc".to_string(), lam);
        assert!(approx(scalar(&evaluate_expression("inc(5)", &vars).unwrap()), 6.0));
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
        let v = eval("(|(x,y)| x+y)").unwrap();
        match v {
            Value::Lambda { params, body } => {
                assert_eq!(params, vec!["x", "y"]);
                assert_eq!(body, "x+y");
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn lambda_multi_param_call() {
        let mut vars = HashMap::new();
        let lam = eval("(|(x,y)| x+y)").unwrap();
        vars.insert("add".to_string(), lam);
        let result = eval_with("add((3, 4))", &vars).unwrap();
        assert!(approx(scalar(&result), 7.0));
    }

    #[test]
    fn lambda_multi_param_call_product() {
        let mut vars = HashMap::new();
        let lam = eval("(|(a,b)| a*b)").unwrap();
        vars.insert("mul".to_string(), lam);
        let result = eval_with("mul((5, 6))", &vars).unwrap();
        assert!(approx(scalar(&result), 30.0));
    }

    #[test]
    fn lambda_over_tuple_array() {
        let mut vars = HashMap::new();
        let lam = eval("(|(x,y)| x+y)").unwrap();
        vars.insert("add".to_string(), lam);
        // add applied to {(1,2), (3,4), (5,6)} should return {3, 7, 11}
        let result = eval_with("add({(1,2), (3,4), (5,6)})", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[3.0, 7.0, 11.0]));
    }

    #[test]
    fn lambda_over_tuple_array_broadcast() {
        let mut vars = HashMap::new();
        let lam = eval("(|(x,y)| x*y)").unwrap();
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
        let v = eval("(|x| x+1)").unwrap();
        assert_eq!(format!("{}", v), "(|x| x+1)");
    }

    #[test]
    fn lambda_multi_display() {
        let v = eval("(|(a,b)| a+b)").unwrap();
        assert_eq!(format!("{}", v), "(|(a,b)| a+b)");
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
        let lam = eval("(|(x,y)| x+y)").unwrap();
        vars.insert("f".to_string(), lam);
        // Calling with wrong tuple length should error
        let result = eval_with("f((1, 2, 3))", &vars);
        assert!(result.is_err());
    }

    #[test]
    fn three_param_lambda() {
        let mut vars = HashMap::new();
        let lam = eval("(|(a,b,c)| a+b+c)").unwrap();
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
        vars.insert("arr".to_string(), Value::Array(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        // Filter elements > 2: arr[pred] where pred = (|(i,x)| x > 2)
        let lam = eval("(|(i,x)| x > 2)").unwrap();
        vars.insert("pred".to_string(), lam);
        let result = eval_with("arr[pred]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[3.0, 4.0, 5.0]));
    }

    #[test]
    fn array_filter_even_indices() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![10.0, 20.0, 30.0, 40.0, 50.0]));
        // Filter even indices: i % 2 == 0
        // We can use floor(i/2)*2 == i to check evenness
        let lam = eval("(|(i,x)| floor(i/2)*2 == i)").unwrap();
        vars.insert("even_idx".to_string(), lam);
        let result = eval_with("arr[even_idx]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[10.0, 30.0, 50.0]));
    }

    #[test]
    fn array_filter_positive() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![-2.0, -1.0, 0.0, 1.0, 2.0]));
        let lam = eval("(|(i,x)| x > 0)").unwrap();
        vars.insert("pos".to_string(), lam);
        let result = eval_with("arr[pos]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[1.0, 2.0]));
    }

    #[test]
    fn array_filter_empty_result() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![1.0, 2.0, 3.0]));
        let lam = eval("(|(i,x)| x > 100)").unwrap();
        vars.insert("never".to_string(), lam);
        let result = eval_with("arr[never]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[]));
    }

    #[test]
    fn array_filter_all_pass() {
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![1.0, 2.0, 3.0]));
        let lam = eval("(|(i,x)| 1)").unwrap();
        vars.insert("always".to_string(), lam);
        let result = eval_with("arr[always]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn range_then_filter() {
        // [10][gt5] where gt5 = (|(i,x)| x > 5) should give {6, 7, 8, 9}
        let mut vars = HashMap::new();
        let lam = eval("(|(i,x)| x > 5)").unwrap();
        vars.insert("gt5".to_string(), lam);
        let result = eval_with("[10][gt5]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[6.0, 7.0, 8.0, 9.0]));
    }

    #[test]
    fn tuple_array_index() {
        let mut vars = HashMap::new();
        let ta = Value::TupleArray { width: 2, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
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
        let ta = Value::TupleArray { width: 2, data: vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0] };
        vars.insert("ta".to_string(), ta);
        // Filter: keep tuples where x > 1 (x is the first element of the tuple)
        let lam = eval("(|(i,x,y)| x > 1)").unwrap();
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
        vars.insert("arr".to_string(), Value::Array(vec![10.0, 20.0, 30.0, 40.0, 50.0]));
        let lam = eval("(|(i,x)| x > 20)").unwrap();
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
        let lam = eval("(|(i,x)| 0.00000000001)").unwrap(); // 1e-11, below threshold
        vars.insert("tiny".to_string(), lam);
        let result = eval_with("arr[tiny]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[])); // All filtered out
    }

    // Parse assignment tests

    #[test]
    fn parse_tuple_pattern_valid() {
        assert_eq!(parse_tuple_pattern("(a, b)"), Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(parse_tuple_pattern("(x, y, z)"), Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]));
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
        // f((1, 2, 3)) with (|(a, rest)| ...) should bind a=1, rest=(2,3)
        let mut vars = HashMap::new();
        let lam = eval("(|(a, rest)| a)").unwrap();
        vars.insert("first".to_string(), lam);
        let result = eval_with("first((1, 2, 3))", &vars).unwrap();
        assert!(approx(scalar(&result), 1.0));
    }

    #[test]
    fn lambda_decomposition_rest_is_tuple() {
        // Access rest as tuple - rest should be (2, 3)
        let mut vars = HashMap::new();
        let lam = eval("(|(a, rest)| rest)").unwrap();
        vars.insert("tail".to_string(), lam);
        let result = eval_with("tail((1, 2, 3))", &vars).unwrap();
        assert!(approx_arr(&tuple(&result), &[2.0, 3.0]));
    }

    #[test]
    fn lambda_decomposition_exact_match() {
        // (|(a, b)| a+b) with (1, 2) - exact match, b gets scalar
        let mut vars = HashMap::new();
        let lam = eval("(|(a, b)| a + b)").unwrap();
        vars.insert("add".to_string(), lam);
        let result = eval_with("add((1, 2))", &vars).unwrap();
        assert!(approx(scalar(&result), 3.0));
    }

    #[test]
    fn filter_lambda_decomposition() {
        // TupleArray of width 3, filter with (|(i, rest)| ...)
        // rest should be the tuple (a, b, c)
        let mut vars = HashMap::new();
        let ta = Value::TupleArray { width: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] };
        vars.insert("ta".to_string(), ta);
        // Filter where index == 0
        let lam = eval("(|(i, rest)| i == 0)").unwrap();
        vars.insert("first_only".to_string(), lam);
        let result = eval_with("ta[first_only]", &vars).unwrap();
        let (w, d) = tuple_array(&result);
        assert_eq!(w, 3);
        assert!(approx_arr(&d, &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn filter_lambda_decomposition_two_params() {
        // Array filter with (|(i, x)| ...) - x is the element
        let mut vars = HashMap::new();
        vars.insert("arr".to_string(), Value::Array(vec![10.0, 20.0, 30.0]));
        let lam = eval("(|(i, x)| i == 1)").unwrap();
        vars.insert("second".to_string(), lam);
        let result = eval_with("arr[second]", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[20.0]));
    }

    #[test]
    fn tuple_array_lambda_decomposition() {
        // Map over TupleArray with decomposition
        // {(1,2,3), (4,5,6)} with (|(a, rest)| a) should give {1, 4}
        let mut vars = HashMap::new();
        let ta = Value::TupleArray { width: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        vars.insert("ta".to_string(), ta);
        let lam = eval("(|(a, rest)| a)").unwrap();
        vars.insert("first_elem".to_string(), lam);
        let result = eval_with("first_elem(ta)", &vars).unwrap();
        assert!(approx_arr(&array(&result), &[1.0, 4.0]));
    }

    #[test]
    fn filter_three_dim_tuple_array_by_index() {
        // This is the user's example: x[(|(i,y)| i==0)] on 3D tuple array
        let mut vars = HashMap::new();
        let ta = Value::TupleArray { width: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] };
        vars.insert("x".to_string(), ta);
        let lam = eval("(|(i, y)| i == 0)").unwrap();
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
        assert!(approx_arr(&d, &[1.0, 10.0, 1.0, 20.0, 1.0, 30.0, 2.0, 10.0, 2.0, 20.0, 2.0, 30.0]));
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
        assert!(approx_arr(&d, &[1.0, 2.0, 10.0, 1.0, 2.0, 20.0, 3.0, 4.0, 10.0, 3.0, 4.0, 20.0]));
    }

    #[test]
    fn tensor_product_ranges() {
        // [2] ** [3] = {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)}
        let v = eval("[2] ** [3]").unwrap();
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0]));
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
        let lam = eval("(|(x, y)| x + y)").unwrap();
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
        vars.insert("sq".to_string(), eval("(|x| x*x)").unwrap());
        let v = eval_with("sq(5)", &vars).unwrap();
        assert!(approx(scalar(&v), 25.0));
    }

    #[test]
    fn inline_lambda_multi_param() {
        // Multi-param lambda inline
        let v = eval("{1, 2, 3, 4, 5}[(|(i,x)| x > 2)]").unwrap();
        assert!(approx_arr(&array(&v), &[3.0, 4.0, 5.0]));
    }

    #[test]
    fn inline_lambda_tensor_filter() {
        // The user's example: ([10]**[10])[(|(i,x,y)| x*y > 10)]
        let v = eval("([3]**[3])[(|(i,x,y)| x*y > 1)]").unwrap();
        // 3x3 = pairs (0,0)..(2,2), filter where x*y > 1
        // (1,2), (2,1), (2,2) have products 2, 2, 4 > 1
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        assert!(approx_arr(&d, &[1.0, 2.0, 2.0, 1.0, 2.0, 2.0]));
    }

    #[test]
    fn inline_lambda_as_value() {
        // Lambda literal evaluates to a lambda value
        let v = eval("(|x| x + 1)").unwrap();
        match v {
            Value::Lambda { params, body } => {
                assert_eq!(params, vec!["x"]);
                assert_eq!(body, "x + 1");
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn user_example_tensor_filter() {
        // The exact example the user asked about
        let v = eval("([10]**[10])[(|(i,x,y)|x*y>10)]").unwrap();
        // Should filter pairs where x*y > 10
        // Pairs like (2,6), (3,4), (4,3), (6,2), etc.
        let (w, d) = tuple_array(&v);
        assert_eq!(w, 2);
        // Just check that we got some results and they satisfy x*y > 10
        assert!(d.len() > 0);
        for chunk in d.chunks(2) {
            assert!(chunk[0] * chunk[1] > 10.0);
        }
    }
}
