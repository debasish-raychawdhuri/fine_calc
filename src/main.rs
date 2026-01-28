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
    expr::TopExprParser::new()
        .parse(vars, input)
        .map_err(|e| match e {
            lalrpop_util::ParseError::User { error } => error.to_string(),
            other => format!("{}", other),
        })
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
                    let (var_name, expr_str) = {
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
                            if !lhs.is_empty() && lhs.chars().all(|c| c.is_alphanumeric() || c == '_') && (lhs.starts_with('_') || lhs.chars().next().unwrap().is_alphabetic()) {
                                (Some(lhs.to_string()), trimmed[eq_pos + 1..].trim())
                            } else {
                                (None, trimmed)
                            }
                        } else {
                            (None, trimmed)
                        }
                    };
                    let result = evaluate_expression(expr_str, &variables);
                    match result {
                        Ok(res) => {
                            history.push(format!(">> {}", input));
                            history.push(format!("{}", &res));
                            if let Some(name) = var_name {
                                variables.insert(name, res);
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

    fn scalar(v: &Value) -> f64 {
        v.as_scalar().expect("expected scalar")
    }

    fn array(v: &Value) -> Vec<f64> {
        match v {
            Value::Array(a) => a.clone(),
            _ => panic!("expected array"),
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
    fn compound_boolean() {
        // (3>2) && (1==1) => 1 && 1 => 1
        assert!(approx(scalar(&eval("3>2&&1==1").unwrap()), 1.0));
        // (1>2) || (3<4) => 0 || 1 => 1
        assert!(approx(scalar(&eval("1>2||3<4").unwrap()), 1.0));
    }
}
