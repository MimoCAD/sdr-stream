//! Dependency-free JSON: parse and pretty-serialize, ASCII output,
//! insertion-ordered objects, malformed input refused rather than
//! guessed. Lives in the stream crate so the sidecar projection and its
//! readers (receiver, server, browser, phone) share one implementation
//! with zero dependencies; `no_std` + `alloc`.
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Insertion-ordered so output is stable and diff-friendly (a
    /// hashmap would reorder keys run to run).
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// A member of an object by key, or `None` for non-objects / absent.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    /// Integer accessors: our numbers are all integers well under 2^53,
    /// so f64 holds them exactly (Hz ≤ ~1 GHz, offsets ± tens of MHz).
    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().map(|n| round(n) as u64)
    }
    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| round(n) as i64)
    }

    /// Serialize with two-space indentation and a trailing newline —
    /// human-readable and diff-stable, which is the whole point of
    /// leaving CSV behind.
    pub fn to_pretty(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, 0);
        s.push('\n');
        s
    }

    fn write(&self, out: &mut String, depth: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if fract(*n) == 0.0 && n.abs() < 9e15 {
                    out.push_str(&(*n as i64).to_string());
                } else {
                    out.push_str(&n.to_string());
                }
            }
            Json::Str(s) => write_str(out, s),
            Json::Arr(a) if a.is_empty() => out.push_str("[]"),
            Json::Arr(a) => {
                out.push_str("[\n");
                for (i, v) in a.iter().enumerate() {
                    indent(out, depth + 1);
                    v.write(out, depth + 1);
                    out.push_str(if i + 1 < a.len() { ",\n" } else { "\n" });
                }
                indent(out, depth);
                out.push(']');
            }
            Json::Obj(kv) if kv.is_empty() => out.push_str("{}"),
            Json::Obj(kv) => {
                out.push_str("{\n");
                for (i, (k, v)) in kv.iter().enumerate() {
                    indent(out, depth + 1);
                    write_str(out, k);
                    out.push_str(": ");
                    v.write(out, depth + 1);
                    out.push_str(if i + 1 < kv.len() { ",\n" } else { "\n" });
                }
                indent(out, depth);
                out.push('}');
            }
        }
    }

    /// Parse a complete JSON document; `None` on any malformation or
    /// trailing garbage.
    pub fn parse(text: &str) -> Option<Json> {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        (p.i == p.b.len()).then_some(v)
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn value(&mut self) -> Option<Json> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Json::Str),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'n' => self.lit("null", Json::Null),
            _ => self.number(),
        }
    }
    fn lit(&mut self, word: &str, v: Json) -> Option<Json> {
        self.b[self.i..].starts_with(word.as_bytes()).then(|| {
            self.i += word.len();
            v
        })
    }
    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.i += 1;
        }
        core::str::from_utf8(&self.b[start..self.i]).ok()?.parse().ok().map(Json::Num)
    }
    fn string(&mut self) -> Option<String> {
        self.i += 1; // opening quote
        let mut s = String::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => return Some(s),
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    s.push(match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        _ => return None, // \uXXXX etc. unused in our files
                    });
                }
                // ASCII content only, as documented.
                c if c < 0x80 => s.push(c as char),
                _ => return None,
            }
        }
    }
    fn array(&mut self) -> Option<Json> {
        self.i += 1; // '['
        let mut a = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Some(Json::Arr(a));
        }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Some(Json::Arr(a));
                }
                _ => return None,
            }
        }
    }
    fn object(&mut self) -> Option<Json> {
        self.i += 1; // '{'
        let mut kv = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Some(Json::Obj(kv));
        }
        loop {
            self.ws();
            if self.peek()? != b'"' {
                return None;
            }
            let k = self.string()?;
            self.ws();
            if self.peek()? != b':' {
                return None;
            }
            self.i += 1;
            self.ws();
            kv.push((k, self.value()?));
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Some(Json::Obj(kv));
                }
                _ => return None,
            }
        }
    }
}

/// Builders that keep the site-store serialization readable.
pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}
pub fn num<T: Into<f64>>(n: T) -> Json {
    Json::Num(n.into())
}
pub fn str(s: impl Into<String>) -> Json {
    Json::Str(s.into())
}

/// Truncate toward zero without libm (`no_std`): exact below 2^63.
fn trunc(n: f64) -> f64 {
    if n.abs() < 9.2e18 { (n as i64) as f64 } else { n }
}

fn fract(n: f64) -> f64 {
    n - trunc(n)
}

/// Round half away from zero without libm.
fn round(n: f64) -> f64 {
    if n >= 0.0 { trunc(n + 0.5) } else { trunc(n - 0.5) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_nested_document() {
        let doc = obj(vec![
            ("system", obj(vec![("WACN", str("ABCDE")), ("SYSID", str("123"))])),
            (
                "sites",
                obj(vec![(
                    "3C3",
                    obj(vec![
                        ("RFSS", num(1u32)),
                        ("controls", Json::Arr(vec![num(772_918_750f64)])),
                        ("channels", Json::Arr(vec![])),
                    ]),
                )]),
            ),
        ]);
        let text = doc.to_pretty();
        let back = Json::parse(&text).expect("parses its own output");
        assert_eq!(back, doc, "structure survives the round trip");
        // Spot-check accessors and that integers print without a dot.
        assert_eq!(
            back.get("sites").and_then(|s| s.get("3C3")).and_then(|s| s.get("RFSS")).and_then(Json::as_u64),
            Some(1)
        );
        assert!(text.contains("772918750"), "big integers stay integers: {text}");
        assert!(!text.contains("772918750.0"));
    }

    #[test]
    fn parses_the_users_hand_written_shape_and_rejects_garbage() {
        let hand = r#"{
            "system": { "WACN": "ABCDE", "SYSID": "123" },
            "sites": {
                "3C3": {
                    "RFSS": 1, "siteId": 3,
                    "controls": [772918750],
                    "iden": [{ "idx": 0, "hz": 851006250, "offset": -45000000, "slots": 1 }],
                    "channels": [{ "hz": 770206250, "type": "energy" }]
                }
            }
        }"#;
        let v = Json::parse(hand).expect("valid");
        let site = v.get("sites").and_then(|s| s.get("3C3")).expect("site");
        assert_eq!(site.get("siteId").and_then(Json::as_u64), Some(3));
        let iden0 = &site.get("iden").and_then(Json::as_array).unwrap()[0];
        assert_eq!(iden0.get("offset").and_then(Json::as_i64), Some(-45_000_000));
        assert_eq!(
            site.get("channels").and_then(Json::as_array).unwrap()[0].get("type").and_then(Json::as_str),
            Some("energy")
        );
        // Malformation is refused, not guessed.
        assert!(Json::parse("{ \"a\": }").is_none());
        assert!(Json::parse("[1, 2").is_none());
        assert!(Json::parse("{} trailing").is_none());
    }

    #[test]
    fn strings_escape_and_unescape() {
        let v = str("a\"b\\c\nd");
        let back = Json::parse(&Json::Arr(vec![v.clone()]).to_pretty()).unwrap();
        assert_eq!(back.as_array().unwrap()[0], v);
    }
}
