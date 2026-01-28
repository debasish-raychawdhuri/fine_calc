use std::fmt;

#[derive(Debug, Clone)]
pub enum Value {
    Scalar(f64),
    Array(Vec<f64>),
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

    fn broadcast_op(self, other: Value, f: fn(f64, f64) -> f64) -> Value {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(f(a, b)),
            (Value::Scalar(a), Value::Array(b)) => {
                Value::Array(b.iter().map(|&x| f(a, x)).collect())
            }
            (Value::Array(a), Value::Scalar(b)) => {
                Value::Array(a.iter().map(|&x| f(x, b)).collect())
            }
            (Value::Array(a), Value::Array(b)) => {
                let len = a.len().max(b.len());
                Value::Array(
                    (0..len)
                        .map(|i| {
                            let av = if i < a.len() { a[i] } else { 0.0 };
                            let bv = if i < b.len() { b[i] } else { 0.0 };
                            f(av, bv)
                        })
                        .collect(),
                )
            }
        }
    }

    pub fn add(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| a + b)
    }

    pub fn sub(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| a - b)
    }

    pub fn mul(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| a * b)
    }

    pub fn div(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| a / b)
    }

    pub fn powf(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| a.powf(b))
    }

    pub fn apply_fn(self, f: fn(f64) -> f64) -> Value {
        match self {
            Value::Scalar(v) => Value::Scalar(f(v)),
            Value::Array(elems) => Value::Array(elems.iter().map(|&x| f(x)).collect()),
        }
    }

    pub fn gt(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    pub fn lt(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    pub fn ge(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    pub fn le(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }

    pub fn eq_val(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if (a - b).abs() < 1e-10 { 1.0 } else { 0.0 })
    }

    pub fn ne_val(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if (a - b).abs() >= 1e-10 { 1.0 } else { 0.0 })
    }

    pub fn and(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 })
    }

    pub fn or(self, other: Value) -> Value {
        self.broadcast_op(other, |a, b| if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 })
    }

    pub fn not(self) -> Value {
        self.apply_fn(|v| if v == 0.0 { 1.0 } else { 0.0 })
    }

    pub fn neg(self) -> Value {
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
