use ncurses::*;

fn evaluate_expression(expr: &str) -> Result<f64, String> {
    let expr = expr.trim();
    if expr.is_empty() {
        return Err("Empty expression".to_string());
    }

    if let Ok(num) = expr.parse::<f64>() {
        return Ok(num);
    }

    let mut stack = Vec::new();
    let mut num = 0.0;
    let mut sign = '+';
    let mut i = 0;
    let chars: Vec<char> = expr.chars().collect();

    while i < chars.len() {
        let c = chars[i];
        if c.is_digit(10) || c == '.' {
            let mut num_str = String::new();
            while i < chars.len() && (chars[i].is_digit(10) || chars[i] == '.') {
                num_str.push(chars[i]);
                i += 1;
            }
            num = num_str.parse::<f64>().unwrap();
            continue;
        }

        if c.is_alphabetic() {
            let mut func_name = String::new();
            while i < chars.len() && chars[i].is_alphabetic() {
                func_name.push(chars[i]);
                i += 1;
            }
            // Expect '(' after function name
            if i < chars.len() && chars[i] == '(' {
                let mut paren_expr = String::new();
                let mut balance = 1;
                i += 1;
                while i < chars.len() && balance > 0 {
                    if chars[i] == '(' {
                        balance += 1;
                    } else if chars[i] == ')' {
                        balance -= 1;
                    }
                    if balance > 0 {
                        paren_expr.push(chars[i]);
                    }
                    i += 1;
                }
                let arg = evaluate_expression(&paren_expr)?;
                num = match func_name.as_str() {
                    "sin" => arg.sin(),
                    "cos" => arg.cos(),
                    "tan" => arg.tan(),
                    "asin" => arg.asin(),
                    "acos" => arg.acos(),
                    "atan" => arg.atan(),
                    "sinh" => arg.sinh(),
                    "cosh" => arg.cosh(),
                    "tanh" => arg.tanh(),
                    "asinh" => arg.asinh(),
                    "acosh" => arg.acosh(),
                    "atanh" => arg.atanh(),
                    _ => return Err(format!("Unknown function: {}", func_name)),
                };
                continue;
            } else {
                num = match func_name.as_str() {
                    "pi" => std::f64::consts::PI,
                    "e" => std::f64::consts::E,
                    _ => return Err(format!("Unknown identifier: {}", func_name)),
                };
                continue;
            }
        }

        if c == '(' {
            let mut paren_expr = String::new();
            let mut balance = 1;
            i += 1;
            while i < chars.len() && balance > 0 {
                if chars[i] == '(' {
                    balance += 1;
                } else if chars[i] == ')' {
                    balance -= 1;
                }
                if balance > 0 {
                    paren_expr.push(chars[i]);
                }
                i += 1;
            }
            num = evaluate_expression(&paren_expr)?;
            continue;
        }

        if !c.is_whitespace() {
            match sign {
                '+' => stack.push(num),
                '-' => stack.push(-num),
                '*' => {
                    let top = stack.pop().unwrap_or(0.0);
                    stack.push(top * num);
                }
                '/' => {
                    let top = stack.pop().unwrap_or(0.0);
                    stack.push(top / num);
                }
                '^' => {
                    let top = stack.pop().unwrap_or(0.0);
                    stack.push(top.powf(num));
                }
                _ => return Err("Invalid operator".to_string()),
            }
            sign = c;
            num = 0.0;
        }
        i += 1;
    }

    match sign {
        '+' => stack.push(num),
        '-' => stack.push(-num),
        '*' => {
            let top = stack.pop().unwrap_or(0.0);
            stack.push(top * num);
        }
        '/' => {
            let top = stack.pop().unwrap_or(0.0);
            stack.push(top / num);
        }
        '^' => {
            let top = stack.pop().unwrap_or(0.0);
            stack.push(top.powf(num));
        }
        _ => return Err("Invalid operator".to_string()),
    }

    Ok(stack.iter().sum())
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
            if line.starts_with("Result:") {
                attron(COLOR_PAIR(1));
                addstr(line);
                attroff(COLOR_PAIR(1));
            } else if line.starts_with("Error:") {
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
                attron(COLOR_PAIR(4));
                addstr(line);
                attroff(COLOR_PAIR(4));
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
                    let result = evaluate_expression(&input);
                    match result {
                        Ok(res) => {
                            history.push(format!(">> {}", input));
                            history.push(format!("Result: {}", res));
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
