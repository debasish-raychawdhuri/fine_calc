use std::collections::HashMap;
use crate::ast::{Expr, BinOp, UnaryOp, SpannedExpr, Span};
use crate::value::Value;
use crate::error::EvalError;

/// Evaluate an expression AST with the given variable bindings
pub fn eval(expr: &SpannedExpr, vars: &HashMap<String, Value>) -> Result<Value, EvalError> {
    let span = expr.span;
    match &expr.node {
        Expr::Scalar(v) => Ok(Value::Scalar(*v)),

        Expr::Ident(name) => {
            match name.as_str() {
                "pi" => Ok(Value::Scalar(std::f64::consts::PI)),
                "e" => Ok(Value::Scalar(std::f64::consts::E)),
                _ => vars.get(name)
                    .cloned()
                    .ok_or_else(|| EvalError::with_span(format!("Unknown identifier: {}", name), span))
            }
        }

        Expr::BinOp(left, op, right) => {
            let l = eval(left, vars)?;
            let r = eval(right, vars)?;
            let result = match op {
                BinOp::Add => l.add(r),
                BinOp::Sub => l.sub(r),
                BinOp::Mul => l.mul(r),
                BinOp::Div => l.div(r),
                BinOp::Pow => l.powf(r),
                BinOp::Gt => l.gt(r),
                BinOp::Lt => l.lt(r),
                BinOp::Ge => l.ge(r),
                BinOp::Le => l.le(r),
                BinOp::Eq => l.eq_val(r),
                BinOp::Ne => l.ne_val(r),
                BinOp::And => l.and(r),
                BinOp::Or => l.or(r),
                BinOp::TensorProduct => l.tensor_product(r),
            };
            result.map_err(|e| EvalError::with_span(e.to_string(), span))
        }

        Expr::UnaryOp(op, inner) => {
            let v = eval(inner, vars)?;
            let result = match op {
                UnaryOp::Neg => v.neg(),
                UnaryOp::Not => v.not(),
            };
            result.map_err(|e| EvalError::with_span(e.to_string(), span))
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
                _ => return Err(EvalError::with_span(format!("Unknown builtin function: {}", name), span)),
            };
            v.apply_fn(f).map_err(|e| EvalError::with_span(e.to_string(), span))
        }

        Expr::Call(name, arg) => {
            let lambda = vars.get(name)
                .ok_or_else(|| EvalError::with_span(format!("Unknown function: {}", name), span))?;

            match lambda {
                Value::Lambda { params, body } => {
                    let arg_val = eval(arg, vars)?;
                    call_lambda(params, body, arg_val, vars, span)
                }
                _ => Err(EvalError::with_span(format!("{} is not a function", name), span))
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
                        Value::Array(_) => return Err(EvalError::with_span("Cannot mix arrays in array literal".to_string(), span)),
                        Value::TupleArray { .. } => return Err(EvalError::with_span("Cannot nest tuple arrays".to_string(), span)),
                        Value::Lambda { .. } => return Err(EvalError::with_span("Cannot include lambda in array".to_string(), span)),
                    };
                    match width {
                        None => width = Some(w),
                        Some(prev) if prev != w => return Err(EvalError::with_span("Tuple array elements must have same width".to_string(), span)),
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
            Value::make_tuple(values?).map_err(|e| EvalError::with_span(e.to_string(), span))
        }

        Expr::Range(inner) => {
            let v = eval(inner, vars)?;
            v.range().map_err(|e| EvalError::with_span(e.to_string(), span))
        }

        Expr::Index(base_expr, idx_expr) => {
            let base = eval(base_expr, vars)?;
            let idx = eval(idx_expr, vars)?;

            match idx {
                Value::Scalar(i) => {
                    let index = i as usize;
                    base.get_at(index)
                        .ok_or_else(|| EvalError::with_span("Index out of bounds".to_string(), idx_expr.span))
                }
                Value::Lambda { params, body } => {
                    filter_with_lambda(&base, &params, &body, vars, span)
                }
                _ => Err(EvalError::with_span("Index must be scalar or lambda".to_string(), idx_expr.span))
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
                    fold_with_lambda(&arr_val, init_val, &params, &body, vars, span)
                }
                _ => Err(EvalError::with_span("Fold requires a lambda".to_string(), lambda.span))
            }
        }
    }
}

/// Call a lambda with an argument
fn call_lambda(
    params: &[String],
    body: &SpannedExpr,
    arg: Value,
    vars: &HashMap<String, Value>,
    span: Span,
) -> Result<Value, EvalError> {
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
                    .map_err(|e| EvalError::with_span(e.to_string(), span))?;
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
                        .map_err(|e| EvalError::with_span(e.to_string(), span))?;
                    let mut inner = vars.clone();
                    for (p, v) in params.iter().zip(bindings) {
                        inner.insert(p.clone(), v);
                    }
                    let res = eval(body, &inner)?;
                    match res.as_scalar() {
                        Some(v) => results.push(v),
                        None => return Err(EvalError::with_span("Lambda must return scalar when mapping over tuple array".to_string(), span)),
                    }
                }
                Ok(Value::Array(results))
            }
            _ => Err(EvalError::with_span("Multi-param lambda requires tuple or tuple array argument".to_string(), span))
        }
    }
}

/// Filter an array or tuple array with a lambda predicate
fn filter_with_lambda(
    base: &Value,
    params: &[String],
    body: &SpannedExpr,
    vars: &HashMap<String, Value>,
    span: Span,
) -> Result<Value, EvalError> {
    match base {
        Value::Array(arr) => {
            let mut result = Vec::new();
            for (i, &elem) in arr.iter().enumerate() {
                let tuple_elems = vec![i as f64, elem];
                let bindings = Value::decompose_elements(&tuple_elems, params.len())
                    .map_err(|e| EvalError::with_span(e.to_string(), span))?;
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
                    return Err(EvalError::with_span("Filter lambda must return scalar".to_string(), span));
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
                    .map_err(|e| EvalError::with_span(e.to_string(), span))?;
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
                    return Err(EvalError::with_span("Filter lambda must return scalar".to_string(), span));
                }
            }
            if *width == 1 {
                Ok(Value::Array(result_data))
            } else {
                Ok(Value::TupleArray { width: *width, data: result_data })
            }
        }
        _ => Err(EvalError::with_span("Can only filter arrays or tuple arrays".to_string(), span))
    }
}

/// Fold/reduce an array with a lambda: array(init){|acc,elem|(body)}
fn fold_with_lambda(
    arr: &Value,
    init: Value,
    params: &[String],
    body: &SpannedExpr,
    vars: &HashMap<String, Value>,
    span: Span,
) -> Result<Value, EvalError> {
    match arr {
        Value::Array(elements) => {
            let mut acc = init;
            for &elem in elements {
                // Build tuple (acc, elem) for lambda binding
                let arg = build_fold_arg(&acc, Value::Scalar(elem), span)?;
                acc = call_lambda_for_fold(params, body, arg, vars, span)?;
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
                let arg = build_fold_arg(&acc, elem, span)?;
                acc = call_lambda_for_fold(params, body, arg, vars, span)?;
            }
            Ok(acc)
        }
        _ => Err(EvalError::with_span("Fold requires an array or tuple array".to_string(), span))
    }
}

/// Build the argument tuple for fold lambda by flattening (acc, elem)
fn build_fold_arg(acc: &Value, elem: Value, span: Span) -> Result<Value, EvalError> {
    let mut flat = Vec::new();

    // Flatten acc
    match acc {
        Value::Scalar(v) => flat.push(*v),
        Value::Tuple(t) => flat.extend(t),
        _ => return Err(EvalError::with_span("Accumulator must be scalar or tuple".to_string(), span)),
    }

    // Flatten elem
    match elem {
        Value::Scalar(v) => flat.push(v),
        Value::Tuple(t) => flat.extend(t),
        _ => return Err(EvalError::with_span("Array element must be scalar or tuple".to_string(), span)),
    }

    Ok(Value::Tuple(flat))
}

/// Call lambda for fold - always uses tuple decomposition
fn call_lambda_for_fold(
    params: &[String],
    body: &SpannedExpr,
    arg: Value,
    vars: &HashMap<String, Value>,
    span: Span,
) -> Result<Value, EvalError> {
    match arg {
        Value::Tuple(ref t) => {
            let bindings = Value::decompose_elements(t, params.len())
                .map_err(|e| EvalError::with_span(e.to_string(), span))?;
            let mut inner = vars.clone();
            for (p, v) in params.iter().zip(bindings) {
                inner.insert(p.clone(), v);
            }
            eval(body, &inner)
        }
        _ => Err(EvalError::with_span("Fold argument must be tuple".to_string(), span))
    }
}
