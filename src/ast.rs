use std::fmt;

/// Binary operators
#[derive(Debug, Clone)]
pub enum BinOp {
    Add, Sub, Mul, Div, Pow,
    Gt, Lt, Ge, Le, Eq, Ne,
    And, Or,
    TensorProduct,
}

/// Unary operators
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Abstract Syntax Tree for expressions
#[derive(Debug, Clone)]
pub enum Expr {
    /// Numeric literal
    Scalar(f64),
    /// Variable reference
    Ident(String),
    /// Binary operation
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    /// Unary operation
    UnaryOp(UnaryOp, Box<Expr>),
    /// Built-in function call: sin, cos, etc.
    BuiltinCall(String, Box<Expr>),
    /// User-defined function/lambda call
    Call(String, Box<Expr>),
    /// Array literal: {1, 2, 3}
    Array(Vec<Expr>),
    /// Tuple literal: (1, 2, 3)
    Tuple(Vec<Expr>),
    /// Range: [n]
    Range(Box<Expr>),
    /// Indexing/filtering: arr[idx] or arr[lambda]
    Index(Box<Expr>, Box<Expr>),
    /// Lambda definition: |x| body or |(x,y)| body
    Lambda { params: Vec<String>, body: Box<Expr> },
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Scalar(v) => write!(f, "{}", v),
            Expr::Ident(name) => write!(f, "{}", name),
            Expr::BinOp(l, op, r) => {
                let op_str = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    BinOp::Pow => "^",
                    BinOp::Gt => ">",
                    BinOp::Lt => "<",
                    BinOp::Ge => ">=",
                    BinOp::Le => "<=",
                    BinOp::Eq => "==",
                    BinOp::Ne => "!=",
                    BinOp::And => "&&",
                    BinOp::Or => "||",
                    BinOp::TensorProduct => "**",
                };
                write!(f, "({} {} {})", l, op_str, r)
            }
            Expr::UnaryOp(op, e) => {
                let op_str = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "!",
                };
                write!(f, "{}{}", op_str, e)
            }
            Expr::BuiltinCall(name, arg) => write!(f, "{}({})", name, arg),
            Expr::Call(name, arg) => write!(f, "{}({})", name, arg),
            Expr::Array(elems) => {
                write!(f, "{{")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "}}")
            }
            Expr::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Expr::Range(e) => write!(f, "[{}]", e),
            Expr::Index(base, idx) => write!(f, "{}[{}]", base, idx),
            Expr::Lambda { params, body } => {
                if params.len() == 1 {
                    write!(f, "|{}| {}", params[0], body)
                } else {
                    write!(f, "|({})| {}", params.join(", "), body)
                }
            }
        }
    }
}
