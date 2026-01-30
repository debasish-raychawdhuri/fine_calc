use crate::ast::Span;

/// Evaluation error with optional source location
#[derive(Debug, Clone)]
pub struct EvalError {
    pub message: String,
    pub span: Option<Span>,
}

impl EvalError {
    pub fn new(message: impl Into<String>) -> Self {
        EvalError {
            message: message.into(),
            span: None,
        }
    }

    pub fn with_span(message: impl Into<String>, span: Span) -> Self {
        EvalError {
            message: message.into(),
            span: Some(span),
        }
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for EvalError {}

/// Format an error indicator line showing where the error occurred.
/// Returns underscores for positions outside the span, ^ at start/end, - in between.
///
/// For input `1 + foo * 2` with span `{start: 4, end: 7}` (for "foo"):
/// ```text
/// ____^--^____
/// ```
pub fn format_error_indicator(input: &str, span: Span) -> String {
    let len = input.len();

    // Handle empty input
    if len == 0 {
        return "^".to_string();
    }

    // If span starts at or past the end, show marker at end
    if span.start >= len {
        let mut result = "_".repeat(len);
        result.push('^');
        return result;
    }

    let mut result = String::with_capacity(len);
    let start = span.start;
    let end = span.end.min(len).max(start + 1);
    let span_len = end - start;

    for i in 0..len {
        if i < start {
            result.push('_');
        } else if i == start {
            result.push('^');
        } else if i < end - 1 {
            result.push('-');
        } else if i == end - 1 && span_len > 1 {
            result.push('^');
        } else {
            result.push('_');
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indicator_single_char() {
        // Single character span
        let indicator = format_error_indicator("1 + x", Span::new(4, 5));
        assert_eq!(indicator, "____^");
    }

    #[test]
    fn indicator_multi_char() {
        // Multi-character span for "foo"
        let indicator = format_error_indicator("1 + foo * 2", Span::new(4, 7));
        assert_eq!(indicator, "____^-^____");
    }

    #[test]
    fn indicator_at_start() {
        // Span at the beginning
        let indicator = format_error_indicator("xyz + 1", Span::new(0, 3));
        assert_eq!(indicator, "^-^____");
    }

    #[test]
    fn indicator_at_end() {
        // Span at the end
        let indicator = format_error_indicator("1 + bar", Span::new(4, 7));
        assert_eq!(indicator, "____^-^");
    }

    #[test]
    fn indicator_two_chars() {
        // Two character span
        let indicator = format_error_indicator("ab + cd", Span::new(5, 7));
        assert_eq!(indicator, "_____^^");
    }

    #[test]
    fn indicator_at_eof() {
        // Span at end of input (e.g., unexpected EOF)
        let indicator = format_error_indicator("foo", Span::new(3, 4));
        assert_eq!(indicator, "___^");
    }

    #[test]
    fn indicator_past_end() {
        // Span past end of input
        let indicator = format_error_indicator("foo", Span::new(10, 15));
        assert_eq!(indicator, "___^");
    }

    #[test]
    fn indicator_empty_input() {
        // Empty input
        let indicator = format_error_indicator("", Span::new(0, 1));
        assert_eq!(indicator, "^");
    }
}
