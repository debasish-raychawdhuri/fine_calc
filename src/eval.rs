use std::collections::HashMap;
use crate::ast::{Expr, BinOp, UnaryOp};
use crate::value::Value;

/// Evaluate an expression AST with the given variable bindings
pub fn eval(expr: &Expr, vars: &HashMap<String, Value>) -> Result<Value, String> {
    match expr {
        Expr::Scalar(v) => Ok(Value::Scalar(*v)),

        Expr::Ident(name) => {
            match name.as_str() {
                "pi" => Ok(Value::Scalar(std::f64::consts::PI)),
                "e" => Ok(Value::Scalar(std::f64::consts::E)),
                _ => vars.get(name)
                    .cloned()
                    .ok_or_else(|| format!("Unknown identifier: {}", name))
            }
        }

        Expr::BinOp(left, op, right) => {
            let l = eval(left, vars)?;
            let r = eval(right, vars)?;
            match op {
                BinOp::Add => l.add(r).map_err(|e| e.to_string()),
                BinOp::Sub => l.sub(r).map_err(|e| e.to_string()),
                BinOp::Mul => l.mul(r).map_err(|e| e.to_string()),
                BinOp::Div => l.div(r).map_err(|e| e.to_string()),
                BinOp::Pow => l.powf(r).map_err(|e| e.to_string()),
                BinOp::Gt => l.gt(r).map_err(|e| e.to_string()),
                BinOp::Lt => l.lt(r).map_err(|e| e.to_string()),
                BinOp::Ge => l.ge(r).map_err(|e| e.to_string()),
                BinOp::Le => l.le(r).map_err(|e| e.to_string()),
                BinOp::Eq => l.eq_val(r).map_err(|e| e.to_string()),
                BinOp::Ne => l.ne_val(r).map_err(|e| e.to_string()),
                BinOp::And => l.and(r).map_err(|e| e.to_string()),
                BinOp::Or => l.or(r).map_err(|e| e.to_string()),
                BinOp::TensorProduct => l.tensor_product(r).map_err(|e| e.to_string()),
            }
        }

        Expr::UnaryOp(op, inner) => {
            let v = eval(inner, vars)?;
            match op {
                UnaryOp::Neg => v.neg().map_err(|e| e.to_string()),
                UnaryOp::Not => v.not().map_err(|e| e.to_string()),
            }
        }

        Expr::BuiltinCall(name, arg) => {
            let v = eval(arg, vars)?;
            let f: fn(f64) -> f64 = match name.as_str() {
                "sin" => f64::sin,
                "cos" => f64::cos,
                "tan" => f64::tan,
                "asin" => f64::asin,
                "acos" => f64::acos,
                "atan" => f64::atan,
                "sinh" => f64::sinh,
                "cosh" => f64::cosh,
                "tanh" => f64::tanh,
                "asinh" => f64::asinh,
                "acosh" => f64::acosh,
                "atanh" => f64::atanh,
                "sqrt" => f64::sqrt,
                "abs" => f64::abs,
                "ln" => f64::ln,
                "log2" => f64::log2,
                "log10" => f64::log10,
                "exp" => f64::exp,
                "floor" => f64::floor,
                "ceil" => f64::ceil,
                "round" => f64::round,
                _ => return Err(format!("Unknown builtin function: {}", name)),
            };
            v.apply_fn(f).map_err(|e| e.to_string())
        }

        Expr::Call(name, arg) => {
            let lambda = vars.get(name)
                .ok_or_else(|| format!("Unknown function: {}", name))?;

            match lambda {
                Value::Lambda { params, body } => {
                    let arg_val = eval(arg, vars)?;
                    call_lambda(params, body, arg_val, vars)
                }
                _ => Err(format!("{} is not a function", name))
            }
        }

        Expr::Array(elems) => {
            let values: Result<Vec<Value>, _> = elems.iter()
                .map(|e| eval(e, vars))
                .collect();
            let values = values?;

            // Check if all scalars -> Array, otherwise -> TupleArray
            let all_scalars = values.iter().all(|v| matches!(v, Value::Scalar(_)));
            if all_scalars {
                let scalars: Vec<f64> = values.iter()
                    .map(|v| v.as_scalar().unwrap())
                    .collect();
                Ok(Value::Array(scalars))
            } else {
                // Check widths for TupleArray
                let mut width: Option<usize> = None;
                for v in &values {
                    let w = match v {
                        Value::Scalar(_) => 1,
                        Value::Tuple(t) => t.len(),
                        Value::Array(_) => return Err("Cannot mix arrays in array literal".to_string()),
                        Value::TupleArray { .. } => return Err("Cannot nest tuple arrays".to_string()),
                        Value::Lambda { .. } => return Err("Cannot include lambda in array".to_string()),
                    };
                    match width {
                        None => width = Some(w),
                        Some(prev) if prev != w => return Err("Tuple array elements must have same width".to_string()),
                        _ => {}
                    }
                }
                let w = width.unwrap_or(1);
                let mut data = Vec::with_capacity(values.len() * w);
                for v in values {
                    match v {
                        Value::Scalar(x) => data.push(x),
                        Value::Tuple(t) => data.extend(t),
                        _ => {}
                    }
                }
                Ok(Value::TupleArray { width: w, data })
            }
        }

        Expr::Tuple(elems) => {
            let values: Result<Vec<Value>, _> = elems.iter()
                .map(|e| eval(e, vars))
                .collect();
            Value::make_tuple(values?).map_err(|e| e.to_string())
        }

        Expr::Range(inner) => {
            let v = eval(inner, vars)?;
            v.range().map_err(|e| e.to_string())
        }

        Expr::Index(base_expr, idx_expr) => {
            let base = eval(base_expr, vars)?;
            let idx = eval(idx_expr, vars)?;

            match idx {
                Value::Scalar(i) => {
                    let index = i as usize;
                    base.get_at(index)
                        .ok_or_else(|| "Index out of bounds".to_string())
                }
                Value::Lambda { params, body } => {
                    filter_with_lambda(&base, &params, &body, vars)
                }
                _ => Err("Index must be scalar or lambda".to_string())
            }
        }

        Expr::Lambda { params, body } => {
            Ok(Value::Lambda {
                params: params.clone(),
                body: body.clone(),
            })
        }

        Expr::Fold { array, init, lambda } => {
            let arr_val = eval(array, vars)?;
            let init_val = eval(init, vars)?;
            let lambda_val = eval(lambda, vars)?;

            match lambda_val {
                Value::Lambda { params, body } => {
                    fold_with_lambda(&arr_val, init_val, &params, &body, vars)
                }
                _ => Err("Fold requires a lambda".to_string())
            }
        }
    }
}

/// Call a lambda with an argument
fn call_lambda(
    params: &[String],
    body: &Expr,
    arg: Value,
    vars: &HashMap<String, Value>
) -> Result<Value, String> {
    if params.len() == 1 {
        // Single param: bind directly
        let mut inner = vars.clone();
        inner.insert(params[0].clone(), arg);
        eval(body, &inner)
    } else {
        // Multi-param: expects Tuple or TupleArray
        match arg {
            Value::Tuple(ref t) => {
                let bindings = Value::decompose_elements(t, params.len())
                    .map_err(|e| e.to_string())?;
                let mut inner = vars.clone();
                for (p, v) in params.iter().zip(bindings) {
                    inner.insert(p.clone(), v);
                }
                eval(body, &inner)
            }
            Value::TupleArray { width, ref data } => {
                let count = data.len() / width;
                let mut results = Vec::with_capacity(count);
                for i in 0..count {
                    let tuple_elems: Vec<f64> = data[i * width..(i + 1) * width].to_vec();
                    let bindings = Value::decompose_elements(&tuple_elems, params.len())
                        .map_err(|e| e.to_string())?;
                    let mut inner = vars.clone();
                    for (p, v) in params.iter().zip(bindings) {
                        inner.insert(p.clone(), v);
                    }
                    let res = eval(body, &inner)?;
                    match res.as_scalar() {
                        Some(v) => results.push(v),
                        None => return Err("Lambda must return scalar when mapping over tuple array".to_string()),
                    }
                }
                Ok(Value::Array(results))
            }
            _ => Err("Multi-param lambda requires tuple or tuple array argument".to_string())
        }
    }
}

/// Filter an array or tuple array with a lambda predicate
fn filter_with_lambda(
    base: &Value,
    params: &[String],
    body: &Expr,
    vars: &HashMap<String, Value>
) -> Result<Value, String> {
    match base {
        Value::Array(arr) => {
            let mut result = Vec::new();
            for (i, &elem) in arr.iter().enumerate() {
                let tuple_elems = vec![i as f64, elem];
                let bindings = Value::decompose_elements(&tuple_elems, params.len())
                    .map_err(|e| e.to_string())?;
                let mut inner = vars.clone();
                for (p, v) in params.iter().zip(bindings) {
                    inner.insert(p.clone(), v);
                }
                let res = eval(body, &inner)?;
                if let Some(v) = res.as_scalar() {
                    if Value::is_truthy(v) {
                        result.push(elem);
                    }
                } else {
                    return Err("Filter lambda must return scalar".to_string());
                }
            }
            Ok(Value::Array(result))
        }
        Value::TupleArray { width, data } => {
            let count = data.len() / width;
            let mut result_data = Vec::new();
            for i in 0..count {
                let mut tuple_elems = vec![i as f64];
                for j in 0..*width {
                    tuple_elems.push(data[i * width + j]);
                }
                let bindings = Value::decompose_elements(&tuple_elems, params.len())
                    .map_err(|e| e.to_string())?;
                let mut inner = vars.clone();
                for (p, v) in params.iter().zip(bindings) {
                    inner.insert(p.clone(), v);
                }
                let res = eval(body, &inner)?;
                if let Some(v) = res.as_scalar() {
                    if Value::is_truthy(v) {
                        for j in 0..*width {
                            result_data.push(data[i * width + j]);
                        }
                    }
                } else {
                    return Err("Filter lambda must return scalar".to_string());
                }
            }
            if *width == 1 {
                Ok(Value::Array(result_data))
            } else {
                Ok(Value::TupleArray { width: *width, data: result_data })
            }
        }
        _ => Err("Can only filter arrays or tuple arrays".to_string())
    }
}

/// Fold/reduce an array with a lambda: array(init){|acc,elem|(body)}
fn fold_with_lambda(
    arr: &Value,
    init: Value,
    params: &[String],
    body: &Expr,
    vars: &HashMap<String, Value>
) -> Result<Value, String> {
    match arr {
        Value::Array(elements) => {
            let mut acc = init;
            for &elem in elements {
                // Build tuple (acc, elem) for lambda binding
                let arg = build_fold_arg(&acc, Value::Scalar(elem))?;
                acc = call_lambda_for_fold(params, body, arg, vars)?;
            }
            Ok(acc)
        }
        Value::TupleArray { width, data } => {
            let count = data.len() / width;
            let mut acc = init;
            for i in 0..count {
                let elem = if *width == 1 {
                    Value::Scalar(data[i])
                } else {
                    Value::Tuple(data[i * width..(i + 1) * width].to_vec())
                };
                let arg = build_fold_arg(&acc, elem)?;
                acc = call_lambda_for_fold(params, body, arg, vars)?;
            }
            Ok(acc)
        }
        _ => Err("Fold requires an array or tuple array".to_string())
    }
}

/// Build the argument tuple for fold lambda by flattening (acc, elem)
fn build_fold_arg(acc: &Value, elem: Value) -> Result<Value, String> {
    let mut flat = Vec::new();

    // Flatten acc
    match acc {
        Value::Scalar(v) => flat.push(*v),
        Value::Tuple(t) => flat.extend(t),
        _ => return Err("Accumulator must be scalar or tuple".to_string()),
    }

    // Flatten elem
    match elem {
        Value::Scalar(v) => flat.push(v),
        Value::Tuple(t) => flat.extend(t),
        _ => return Err("Array element must be scalar or tuple".to_string()),
    }

    Ok(Value::Tuple(flat))
}

/// Call lambda for fold - always uses tuple decomposition
fn call_lambda_for_fold(
    params: &[String],
    body: &Expr,
    arg: Value,
    vars: &HashMap<String, Value>
) -> Result<Value, String> {
    match arg {
        Value::Tuple(ref t) => {
            let bindings = Value::decompose_elements(t, params.len())
                .map_err(|e| e.to_string())?;
            let mut inner = vars.clone();
            for (p, v) in params.iter().zip(bindings) {
                inner.insert(p.clone(), v);
            }
            eval(body, &inner)
        }
        _ => Err("Fold argument must be tuple".to_string())
    }
}
