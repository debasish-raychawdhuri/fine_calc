use std::fmt;

/// Tolerance for comparing floating point values to zero
pub const ZERO_TOLERANCE: f64 = 1e-10;

#[derive(Debug, Clone)]
pub enum Value {
    Scalar(f64),
    Array(Vec<f64>),
    Tuple(Vec<f64>),
    TupleArray { width: usize, data: Vec<f64> },
    Lambda { params: Vec<String>, body: String },
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Scalar(v) => write!(f, "{}", v),
            Value::Array(elems) => {
                write!(f, "{{")?;
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "}}")
            }
            Value::Tuple(elems) => {
                write!(f, "(")?;
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            Value::TupleArray { width, data } => {
                let count = data.len() / width;
                write!(f, "{{")?;
                for i in 0..count {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "(")?;
                    for j in 0..*width {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", data[i * width + j])?;
                    }
                    write!(f, ")")?;
                }
                write!(f, "}}")
            }
            Value::Lambda { params, body } => {
                if params.len() == 1 {
                    write!(f, "(|{}| {})", params[0], body)
                } else {
                    write!(f, "(|({})| {})", params.join(","), body)
                }
            }
        }
    }
}

impl Value {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    /// Check if a value is truthy (non-zero with tolerance)
    pub fn is_truthy(v: f64) -> bool {
        v.abs() >= ZERO_TOLERANCE
    }

    /// Get element at index from Array (returns Scalar) or TupleArray (returns Tuple or Scalar if width=1)
    pub fn get_at(&self, idx: usize) -> Option<Value> {
        match self {
            Value::Array(a) => a.get(idx).map(|&v| Value::Scalar(v)),
            Value::TupleArray { width, data } => {
                let count = data.len() / width;
                if idx < count {
                    let tuple_data: Vec<f64> = data[idx * width..(idx + 1) * width].to_vec();
                    Some(Self::tuple_or_scalar(tuple_data))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Convert a vec to Tuple or Scalar if single element
    fn tuple_or_scalar(v: Vec<f64>) -> Value {
        if v.len() == 1 {
            Value::Scalar(v[0])
        } else {
            Value::Tuple(v)
        }
    }

    /// Build a tuple (or tuple array with broadcasting) from a list of values.
    /// Single-element tuples are reduced to scalars.
    pub fn make_tuple(elems: Vec<Value>) -> Result<Value, &'static str> {
        // Check if any element is array-like
        let has_array = elems.iter().any(|e| matches!(e, Value::Array(_) | Value::TupleArray { .. }));

        if !has_array {
            // All Scalar/Tuple → flatten into single Tuple
            let mut flat = Vec::new();
            for e in &elems {
                match e {
                    Value::Scalar(v) => flat.push(*v),
                    Value::Tuple(t) => flat.extend(t),
                    _ => return Err("Cannot include lambda in tuple"),
                }
            }
            // Reduce single-element tuple to scalar
            Ok(Self::tuple_or_scalar(flat))
        } else {
            // Determine max array length and tuple width
            let mut max_len: usize = 0;
            let mut total_width: usize = 0;
            for e in &elems {
                match e {
                    Value::Scalar(_) => { total_width += 1; }
                    Value::Tuple(t) => { total_width += t.len(); }
                    Value::Array(a) => { max_len = max_len.max(a.len()); total_width += 1; }
                    Value::TupleArray { width, data } => {
                        let count = data.len() / width;
                        max_len = max_len.max(count);
                        total_width += width;
                    }
                    Value::Lambda { .. } => return Err("Cannot include lambda in tuple"),
                }
            }
            let mut data = Vec::with_capacity(max_len * total_width);
            for i in 0..max_len {
                for e in &elems {
                    match e {
                        Value::Scalar(v) => data.push(*v),
                        Value::Tuple(t) => data.extend(t),
                        Value::Array(a) => {
                            data.push(if i < a.len() { a[i] } else { 0.0 });
                        }
                        Value::TupleArray { width, data: td } => {
                            let count = td.len() / width;
                            let idx = if i < count { i } else { count - 1 };
                            data.extend_from_slice(&td[idx * width..(idx + 1) * width]);
                        }
                        _ => {}
                    }
                }
            }
            // If width is 1, return Array instead of TupleArray
            if total_width == 1 {
                Ok(Value::Array(data))
            } else {
                Ok(Value::TupleArray { width: total_width, data })
            }
        }
    }

    /// Decompose a tuple for destructuring assignment.
    /// Given n variable names, first n-1 get one element each, last gets the rest.
    /// Returns Vec of Values to assign to each variable.
    pub fn decompose_tuple(&self, n: usize) -> Result<Vec<Value>, &'static str> {
        match self {
            Value::Tuple(t) => {
                Self::decompose_elements(t, n)
            }
            Value::Scalar(v) => {
                if n == 1 {
                    Ok(vec![Value::Scalar(*v)])
                } else {
                    Err("Cannot destructure scalar into multiple variables")
                }
            }
            _ => Err("Can only destructure tuples"),
        }
    }

    /// Decompose a slice of elements into n values with tuple decomposition semantics.
    /// First n-1 values get one element each, last value gets the rest.
    pub fn decompose_elements(elements: &[f64], n: usize) -> Result<Vec<Value>, &'static str> {
        if elements.len() < n {
            return Err("Not enough elements for destructuring");
        }
        let mut result = Vec::with_capacity(n);
        // First n-1 variables get one element each
        for i in 0..n - 1 {
            result.push(Value::Scalar(elements[i]));
        }
        // Last variable gets the rest
        let rest: Vec<f64> = elements[n - 1..].to_vec();
        result.push(Self::tuple_or_scalar(rest));
        Ok(result)
    }

    fn broadcast_op(self, other: Value, f: fn(f64, f64) -> f64) -> Result<Value, &'static str> {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(f(a, b))),
            (Value::Scalar(a), Value::Array(b)) => {
                Ok(Value::Array(b.iter().map(|&x| f(a, x)).collect()))
            }
            (Value::Array(a), Value::Scalar(b)) => {
                Ok(Value::Array(a.iter().map(|&x| f(x, b)).collect()))
            }
            (Value::Array(a), Value::Array(b)) => {
                let len = a.len().max(b.len());
                Ok(Value::Array(
                    (0..len)
                        .map(|i| {
                            let av = if i < a.len() { a[i] } else { 0.0 };
                            let bv = if i < b.len() { b[i] } else { 0.0 };
                            f(av, bv)
                        })
                        .collect(),
                ))
            }
            (Value::Tuple(_), _) | (_, Value::Tuple(_)) => Err("Cannot perform arithmetic on tuples"),
            (Value::TupleArray { .. }, _) | (_, Value::TupleArray { .. }) => Err("Cannot perform arithmetic on tuple arrays"),
            (Value::Lambda { .. }, _) | (_, Value::Lambda { .. }) => Err("Cannot perform arithmetic on lambdas"),
        }
    }

    pub fn add(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| a + b)
    }

    pub fn sub(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| a - b)
    }

    pub fn mul(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| a * b)
    }

    pub fn div(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| a / b)
    }

    pub fn powf(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| a.powf(b))
    }

    pub fn apply_fn(self, f: fn(f64) -> f64) -> Result<Value, &'static str> {
        match self {
            Value::Scalar(v) => Ok(Value::Scalar(f(v))),
            Value::Array(elems) => Ok(Value::Array(elems.iter().map(|&x| f(x)).collect())),
            Value::Tuple(_) => Err("Cannot apply numeric function to tuple"),
            Value::TupleArray { .. } => Err("Cannot apply numeric function to tuple array"),
            Value::Lambda { .. } => Err("Cannot apply numeric function to lambda"),
        }
    }

    pub fn gt(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    pub fn lt(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    pub fn ge(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    pub fn le(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }

    pub fn eq_val(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if (a - b).abs() < ZERO_TOLERANCE { 1.0 } else { 0.0 })
    }

    pub fn ne_val(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if (a - b).abs() >= ZERO_TOLERANCE { 1.0 } else { 0.0 })
    }

    pub fn and(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if Self::is_truthy(a) && Self::is_truthy(b) { 1.0 } else { 0.0 })
    }

    pub fn or(self, other: Value) -> Result<Value, &'static str> {
        self.broadcast_op(other, |a, b| if Self::is_truthy(a) || Self::is_truthy(b) { 1.0 } else { 0.0 })
    }

    pub fn not(self) -> Result<Value, &'static str> {
        self.apply_fn(|v| if Self::is_truthy(v) { 0.0 } else { 1.0 })
    }

    pub fn neg(self) -> Result<Value, &'static str> {
        self.apply_fn(|v| -v)
    }

    pub fn range(self) -> Result<Value, &'static str> {
        match self {
            Value::Scalar(v) => {
                let n = v as i64;
                if n < 0 {
                    return Err("Range requires non-negative integer");
                }
                Ok(Value::Array((0..n).map(|i| i as f64).collect()))
            }
            _ => Err("Range requires a scalar argument"),
        }
    }
}
