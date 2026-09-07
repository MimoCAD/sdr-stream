//! The call's metadata as ONE JSON document — the projection of an
//! `.sdr` file ([`Sidecar::from_records`]) or the `MSDR` chunk inside a
//! conventional WAV ([`Sidecar::to_json`] is what the recorder embeds,
//! [`Sidecar::parse`] reads it back). Nothing on disk or on the wire is
//! a sidecar file any more (since 2026-09-06 the file is the
//! record); this is what a server, a browser or a phone derives from
//! the bytes it holds, with this crate, and what a database row is
//! built from. Versioned — `"version": 1` is the first key.
//!
//! Field conventions, pinned by the golden tests below:
//! - Identity fields the mode doesn't have are `null` (a P25 call has
//!   no tone; a conventional keyup has no NAC) — the key is always
//!   present so a reader never guesses the schema from the mode.
//! - `tone`/`alphatag`/`mdc` appear only on conventional calls.
//! - MDC `unit_id` is the ANI's HEX-DIGIT STRING ("0042" = unit 4-2-3,
//!   the data says what it means, no conversions).
//! - dB values carry one decimal; `offset_hz` is a whole number.
//! - `wav_rate` is the archived PCM rate of a conventional WAV and 0
//!   for anything digital (there is no PCM; the client decodes).
use crate::json::Json;
use crate::{
    HDR_FLAG_SEEDED, Mode, OFFSET_UNKNOWN, Record, TYP_DMR_VOICE, TYP_LDU, TYP_P2_VCH,
};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub struct Sidecar {
    pub start_ms: i64,
    /// Milliseconds of the call. For a conventional WAV this is the
    /// ARCHIVED audio (post-trim) and matches the file; for a digital
    /// call the trailer's call length.
    pub duration_ms: u64,
    pub hz: u64,
    /// The filename's mode token: "F" / "A" / "T0" / "T1" / "D1" / "D2".
    pub mode: String,
    pub emergency: bool,
    pub encrypted: bool,
    /// The archive system dir: `{WACN}{SYSID}` or the FCC callsign.
    pub system: String,
    pub nac: Option<u16>,
    pub tg: Option<u32>,
    /// The P25 OTA (over-the-air) / DMR talker alias at close.
    pub alias: Option<String>,
    /// Conventional squelch token ("D466N") — `None` on digital.
    pub tone: Option<String>,
    /// Channel / talkgroup label — `None` when unknown.
    pub alphatag: Option<String>,
    /// P25/DMR src as decimal digits / MDC ANI as hex digits / "0".
    pub src: String,
    /// Peak signal-over-floor — the filename's `{dbfs}`.
    pub delta_db: f32,
    /// The absolutes behind the delta, measured at the peak-delta
    /// instant (so `signal - delta = floor` exactly).
    pub signal_db: Option<f32>,
    pub floor_db: Option<f32>,
    /// Mean carrier frequency offset — transmitter-drift health.
    pub offset_hz: Option<i32>,
    /// Every MDC-1200 packet heard in the keyup: (op, arg, unit_id).
    /// `None` on digital (key omitted); an empty Vec prints `[]`.
    pub mdc: Option<Vec<(u8, u8, u16)>>,
    pub wav_rate: u32,
}

/// Minimal JSON string escape — quotes, backslashes, control bytes
/// (alphatags and aliases are config- or air-supplied text).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

impl Sidecar {
    pub fn to_json(&self) -> String {
        let opt_str = |v: &Option<String>| match v {
            Some(s) => format!("\"{}\"", json_escape(s)),
            None => "null".into(),
        };
        let opt_f1 = |v: Option<f32>| match v {
            Some(x) => format!("{x:.1}"),
            None => "null".into(),
        };
        let mut s = format!(
            "{{\"version\":1,\"start_ms\":{},\"duration_ms\":{},\"hz\":{},\
             \"mode\":\"{}\",\"emergency\":{},\"encrypted\":{},\"system\":\"{}\",\
             \"nac\":{},\"tg\":{},\"alias\":{}",
            self.start_ms,
            self.duration_ms,
            self.hz,
            json_escape(&self.mode),
            self.emergency,
            self.encrypted,
            json_escape(&self.system),
            match self.nac {
                Some(n) => format!("\"{n:03X}\""),
                None => "null".into(),
            },
            match self.tg {
                Some(t) => t.to_string(),
                None => "null".into(),
            },
            opt_str(&self.alias),
        );
        if let Some(tone) = &self.tone {
            s.push_str(&format!(",\"tone\":\"{}\"", json_escape(tone)));
        }
        if let Some(tag) = &self.alphatag {
            s.push_str(&format!(",\"alphatag\":\"{}\"", json_escape(tag)));
        }
        s.push_str(&format!(
            ",\"src\":\"{}\",\"delta_db\":{:.1},\"signal_db\":{},\"floor_db\":{},\
             \"offset_hz\":{}",
            json_escape(&self.src),
            self.delta_db,
            opt_f1(self.signal_db),
            opt_f1(self.floor_db),
            match self.offset_hz {
                Some(v) => v.to_string(),
                None => "null".into(),
            },
        ));
        if let Some(mdc) = &self.mdc {
            s.push_str(",\"mdc\":[");
            for (i, (op, arg, unit)) in mdc.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&format!("{{\"op\":{op},\"arg\":{arg},\"unit_id\":\"{unit:04X}\"}}"));
            }
            s.push(']');
        }
        s.push_str(&format!(",\"wav_rate\":{}}}", self.wav_rate));
        s
    }

    /// Read a sidecar document back (an `MSDR` chunk). `None` on any
    /// shape this version does not know — malformed input is refused,
    /// never guessed.
    pub fn parse(text: &str) -> Option<Sidecar> {
        let j = Json::parse(text)?;
        if j.get("version")?.as_u64()? != 1 {
            return None;
        }
        let s = |k: &str| -> Option<String> { j.get(k).and_then(|v| v.as_str()).map(String::from) };
        let f = |k: &str| -> Option<f32> { j.get(k).and_then(|v| v.as_f64()).map(|x| x as f32) };
        let mdc = match j.get("mdc") {
            None => None,
            Some(v) => {
                let mut out = Vec::new();
                for p in v.as_array()? {
                    let op = p.get("op")?.as_u64()? as u8;
                    let arg = p.get("arg")?.as_u64()? as u8;
                    let unit = u16::from_str_radix(p.get("unit_id")?.as_str()?, 16).ok()?;
                    out.push((op, arg, unit));
                }
                Some(out)
            }
        };
        Some(Sidecar {
            start_ms: j.get("start_ms")?.as_i64()?,
            duration_ms: j.get("duration_ms")?.as_u64()?,
            hz: j.get("hz")?.as_u64()?,
            mode: s("mode")?,
            emergency: matches!(j.get("emergency"), Some(Json::Bool(true))),
            encrypted: matches!(j.get("encrypted"), Some(Json::Bool(true))),
            system: s("system")?,
            nac: s("nac").and_then(|n| u16::from_str_radix(&n, 16).ok()),
            tg: j.get("tg").and_then(|v| v.as_u64()).map(|t| t as u32),
            alias: s("alias"),
            tone: s("tone"),
            alphatag: s("alphatag"),
            src: s("src")?,
            delta_db: f("delta_db")?,
            signal_db: f("signal_db"),
            floor_db: f("floor_db"),
            offset_hz: j.get("offset_hz").and_then(|v| v.as_i64()).map(|v| v as i32),
            mdc,
            wav_rate: j.get("wav_rate")?.as_u64()? as u32,
        })
    }

    /// THE PROJECTION: a digital call's metadata from its records — the
    /// CallHeader (identity at grant, the carrier, the site), the
    /// CallTrailer (the close-time facts) and the ESS (encryption), plus
    /// the voice records' clock when the trailer is missing (a crash-cut
    /// file: the duration is the last voice record's epoch plus one
    /// unit, the identity the header's seed, the verdict the ESS).
    /// `None` without a CallHeader.
    pub fn from_records(recs: &[Record]) -> Option<Sidecar> {
        let mut header = None;
        let mut trailer = None;
        let mut ess = false;
        let mut last_voice_us = 0u64;
        let mut unit_ms = 0u64;
        for r in recs {
            match r {
                Record::Header(h) => header = Some(h),
                Record::Trailer(t) => trailer = Some(t),
                Record::Ess(_) => ess = true,
                Record::Raw(raw) => match raw.typ {
                    TYP_LDU => {
                        last_voice_us = raw.head.epoch_us;
                        unit_ms = 180;
                    }
                    TYP_P2_VCH | TYP_DMR_VOICE => {
                        last_voice_us = raw.head.epoch_us;
                        unit_ms = 360;
                    }
                    _ => {}
                },
            }
        }
        let h = header?;
        let mode = Mode::from_u8(h.mode);
        let start_ms = (h.head.epoch_us / 1000) as i64;
        let seeded = h.head.flags & HDR_FLAG_SEEDED != 0;
        let opt = |s: &str| if s.is_empty() { None } else { Some(String::from(s)) };
        let (duration_ms, tg, src, encrypted, emergency, delta_db, signal_db, floor_db, offset_hz, alias, alphatag) =
            match trailer {
                Some(t) => (
                    t.duration_ms as u64,
                    t.tg,
                    t.src,
                    t.encrypted(),
                    t.emergency(),
                    t.delta_db,
                    (!t.signal_db.is_nan()).then_some(t.signal_db),
                    (!t.floor_db.is_nan()).then_some(t.floor_db),
                    (t.offset_hz != OFFSET_UNKNOWN).then_some(t.offset_hz),
                    opt(&t.alias),
                    opt(&t.alphatag),
                ),
                None => (
                    last_voice_us.saturating_sub(h.head.epoch_us) / 1000 + unit_ms,
                    if seeded { h.tg } else { 0 },
                    if seeded { h.src } else { 0 },
                    ess,
                    false,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            };
        Some(Sidecar {
            start_ms,
            duration_ms,
            hz: h.head.hz as u64,
            mode: mode.map(|m| m.token(h.slot)).unwrap_or_else(|| format!("?{}", h.mode)),
            emergency,
            encrypted,
            system: h.system.clone(),
            nac: (mode != Some(Mode::Analog)).then_some(h.head.nac),
            tg: (tg != 0).then_some(tg),
            alias,
            tone: None,
            alphatag,
            src: src.to_string(),
            delta_db,
            signal_db,
            floor_db,
            offset_hz,
            mdc: None,
            wav_rate: 0,
        })
    }

    /// The sidecar as an insertion-ordered [`Json`] tree, for callers
    /// that compose it into a larger document.
    pub fn to_json_tree(&self) -> Json {
        Json::parse(&self.to_json()).unwrap_or(Json::Null)
    }
}

impl core::fmt::Display for Sidecar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CallHeader, CallTrailer, EssRecord, MsgHead, TRL_FLAG_EMERGENCY, TRL_FLAG_ENCRYPTED, records};
    use alloc::vec;

    /// The P25 sidecar, byte-exact — the NOTES anchor call's metadata.
    /// nac/tg/alias filled, tone/alphatag/mdc keys ABSENT, absolutes
    /// null until the P25 path plumbs them.
    #[test]
    fn sidecar_p25_golden() {
        let s = Sidecar {
            start_ms: 1786503850696,
            duration_ms: 4250,
            hz: 851012500,
            mode: "F".into(),
            emergency: false,
            encrypted: false,
            system: "ABCDE123".into(),
            nac: Some(0x293),
            tg: Some(100),
            alias: Some("UNIT1".into()),
            tone: None,
            alphatag: None,
            src: "1234567".into(),
            delta_db: 50.4,
            signal_db: None,
            floor_db: None,
            offset_hz: None,
            mdc: None,
            wav_rate: 0,
        };
        let text = "{\"version\":1,\"start_ms\":1786503850696,\"duration_ms\":4250,\
             \"hz\":851012500,\"mode\":\"F\",\"emergency\":false,\"encrypted\":false,\
             \"system\":\"ABCDE123\",\"nac\":\"293\",\"tg\":100,\"alias\":\"UNIT1\",\
             \"src\":\"1234567\",\"delta_db\":50.4,\"signal_db\":null,\"floor_db\":null,\
             \"offset_hz\":null,\"wav_rate\":0}";
        assert_eq!(s.to_json(), text);
        assert_eq!(Sidecar::parse(text).unwrap(), s, "parse is the inverse of to_json");
    }

    /// The conventional sidecar, byte-exact — a conventional keyup with an MDC-1200 ANI:
    /// tone/alphatag/mdc present, unit_id as HEX DIGITS (0042 = unit
    /// 4-2, no conversions), nac/tg/alias null.
    #[test]
    fn sidecar_conventional_golden() {
        let s = Sidecar {
            start_ms: 1786503850696,
            duration_ms: 9470,
            hz: 460012500,
            mode: "A".into(),
            emergency: false,
            encrypted: false,
            system: "KA1ABC".into(),
            nac: None,
            tg: None,
            alias: None,
            tone: Some("D466N".into()),
            alphatag: Some("Ops 1".into()),
            src: "0042".into(),
            delta_db: 56.0,
            signal_db: Some(-22.0),
            floor_db: Some(-78.0),
            offset_hz: Some(120),
            mdc: Some(vec![(0x01, 0x80, 0x0042)]),
            wav_rate: 16_000,
        };
        let text = "{\"version\":1,\"start_ms\":1786503850696,\"duration_ms\":9470,\
             \"hz\":460012500,\"mode\":\"A\",\"emergency\":false,\"encrypted\":false,\
             \"system\":\"KA1ABC\",\"nac\":null,\"tg\":null,\"alias\":null,\
             \"tone\":\"D466N\",\"alphatag\":\"Ops 1\",\"src\":\"0042\",\
             \"delta_db\":56.0,\"signal_db\":-22.0,\"floor_db\":-78.0,\"offset_hz\":120,\
             \"mdc\":[{\"op\":1,\"arg\":128,\"unit_id\":\"0042\"}],\"wav_rate\":16000}";
        assert_eq!(s.to_json(), text);
        assert_eq!(Sidecar::parse(text).unwrap(), s);
        assert!(Sidecar::parse("{\"version\":2}").is_none(), "a foreign version is refused");
        assert!(Sidecar::parse("{\"version\":1,\"start_ms\":1}").is_none(), "a missing field is refused");
    }

    /// Air- and config-supplied text is escaped, never trusted.
    #[test]
    fn sidecar_escapes_text() {
        let mut s = Sidecar {
            start_ms: 0,
            duration_ms: 0,
            hz: 0,
            mode: "A".into(),
            emergency: false,
            encrypted: false,
            system: "K\"B".into(),
            nac: None,
            tg: None,
            alias: None,
            tone: Some("CSQ".into()),
            alphatag: Some("a\\b\"c".into()),
            src: "0".into(),
            delta_db: 0.0,
            signal_db: None,
            floor_db: None,
            offset_hz: None,
            mdc: Some(vec![]),
            wav_rate: 16_000,
        };
        let j = s.to_json();
        assert!(j.contains("\"system\":\"K\\\"B\""));
        assert!(j.contains("\"alphatag\":\"a\\\\b\\\"c\""));
        assert!(j.contains("\"mdc\":[]"), "empty packet list prints []: {j}");
        s.alias = Some("x\ny".into());
        assert!(s.to_json().contains("\"alias\":\"x\\u000ay\""));
    }

    fn head(seq: u32, epoch_us: u64) -> MsgHead {
        MsgHead { seq, epoch_us, hz: 851_012_500, nac: 0x293, flags: 0, site: 13 }
    }

    /// A digital call's sidecar IS its records: header + trailer give
    /// the whole document; without a trailer the voice clock and the
    /// ESS stand in.
    #[test]
    fn projection_from_records() {
        let start = 1_786_503_850_696_000u64;
        let mut buf = Vec::new();
        CallHeader {
            head: MsgHead { flags: HDR_FLAG_SEEDED, ..head(0, start) },
            mode: Mode::P25Fdma as u8,
            slot: 0xFF,
            tg: 100,
            src: 1_234_567,
            wacn: 0xABCDE,
            sysid: 0x123,
            system: "ABCDE123".into(),
        }
        .encode_into(&mut buf);
        for k in 1..=3u32 {
            crate::push_head(&mut buf, TYP_LDU, 40, &head(k, start + k as u64 * 180_000));
            buf.extend_from_slice(&[0; 8]);
        }
        let cut = buf.len();
        EssRecord { head: head(4, start), ess: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA, 0x00, 0x2A] }.encode_into(&mut buf);
        CallTrailer {
            head: MsgHead { flags: TRL_FLAG_ENCRYPTED | TRL_FLAG_EMERGENCY, ..head(5, start + 4_250_000) },
            duration_ms: 4250,
            tg: 100,
            src: 1_234_567,
            delta_db: 50.4,
            signal_db: f32::NAN,
            floor_db: f32::NAN,
            offset_hz: OFFSET_UNKNOWN,
            alias: "UNIT1".into(),
            alphatag: String::new(),
        }
        .encode_into(&mut buf);
        let recs: Vec<Record> = records(&buf).collect();
        let s = Sidecar::from_records(&recs).unwrap();
        assert_eq!(
            s.to_json(),
            "{\"version\":1,\"start_ms\":1786503850696,\"duration_ms\":4250,\
             \"hz\":851012500,\"mode\":\"F\",\"emergency\":true,\"encrypted\":true,\
             \"system\":\"ABCDE123\",\"nac\":\"293\",\"tg\":100,\"alias\":\"UNIT1\",\
             \"src\":\"1234567\",\"delta_db\":50.4,\"signal_db\":null,\"floor_db\":null,\
             \"offset_hz\":null,\"wav_rate\":0}"
        );
        // Crash-cut before the ESS and trailer: the header's seed names
        // it, the voice clock sizes it, nothing claims encryption.
        let cut: Vec<Record> = records(&buf[..cut]).collect();
        let c = Sidecar::from_records(&cut).unwrap();
        assert_eq!((c.duration_ms, c.tg, c.src.as_str(), c.encrypted, c.emergency), (720, Some(100), "1234567", false, false));
        // A DMR header names its slot in the token.
        let mut d = Vec::new();
        CallHeader {
            head: MsgHead { nac: 1, ..head(0, start) },
            mode: Mode::Dmr as u8,
            slot: 1,
            tg: 0,
            src: 0,
            wacn: 0,
            sysid: 0,
            system: "BAOFENG".into(),
        }
        .encode_into(&mut d);
        let d: Vec<Record> = records(&d).collect();
        let s = Sidecar::from_records(&d).unwrap();
        assert_eq!((s.mode.as_str(), s.nac, s.tg, s.duration_ms), ("D2", Some(1), None, 0));
        assert!(Sidecar::from_records(&[]).is_none());
    }
}
