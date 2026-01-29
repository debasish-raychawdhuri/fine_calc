use ncurses::*;
use std::collections::HashMap;

mod value;
use value::Value;

lalrpop_util::lalrpop_mod!(#[allow(clippy::all)] expr);

fn evaluate_expression(input: &str, vars: &HashMap<String, Value>) -> Result<Value, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Empty expression".to_string());
    }
    // Detect lambda literal: (|param| body) or (|(x,y)| body)
    if input.starts_with("(|") && input.ends_with(')') {
        let inner = &input[2..input.len() - 1]; // strip "(|" and ")"
        if let Some(pipe_pos) = inner.find('|') {
            let param_part = inner[..pipe_pos].trim();
            let body = inner[pipe_pos + 1..].trim();
            if !body.is_empty() {
                // Check if it's multi-param: (x,y,z) or single: x
                if param_part.starts_with('(') && param_part.ends_with(')') {
                    // Multi-param: (|(x,y)| body)
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
                    if valid && !params.is_empty() {
                        return Ok(Value::Lambda {
                            params,
                            body: body.to_string(),
                        });
                    }
                } else {
                    // Single param: (|x| body)
                    let param = param_part;
                    if !param.is_empty()
                        && param.chars().all(|c| c.is_alphanumeric() || c == '_')
                        && (param.starts_with('_') || param.chars().next().unwrap().is_alphabetic())
                    {
                        return Ok(Value::Lambda {
                            params: vec![param.to_string()],
                            body: body.to_string(),
                        });
                    }
                }
            }
        }
    }
    expr::TopExprParser::new()
        .parse(vars, input)
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
                            history.push(format!(">> {}", input));
                            history.push(format!("{}", &res));
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
                                            history.push(format!("Error: {}", e));
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                        Err(e) => {
                            history.push(format!(">> {}", input));
                            history.push(format!("Error: {}", e));
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
}
