//! The archive filename grammar — the same on
//! the receiver's disk and in the server's store (a server names what
//! it stores from the record's own metadata, never from a client's
//! filename):
//!
//! ```text
//! {out}/{system}/YYYY/MM/DD/HH:MM:SS.mmm-{tg}-{src}-{flags}-{nac}-{hz}-{dbfs}   (UTC)
//! audio/ABCDE123/2026/08/12/03:04:10.696-00100-01234567-F-293-851012500-50.sdr
//! audio/KA1ABC/2026/08/12/03:04:10.696-D466N-0042-A-000-460012500-22.wav
//! ```
//!
//! `system` is `{WACN}{SYSID}` for P25 and the FCC CALLSIGN for
//! conventional (licensee-unique where a shared frequency is not);
//! `tg`/`src`/`nac` arrive PRE-FORMATTED — P25 `{:05}` / `{:08}` /
//! `{:03X}`, DMR the same, conventional the tone token / the MDC ANI
//! hex digits or `"0"` / `"000"` — so ONE builder serves every mode
//! and a reader keeps a single dash-split parse. `{dbfs}` is the peak
//! signal-over-floor (SNR-like, NOT absolute dBFS): uncalibrated
//! dongles compare best-copy by SNR, not level. Date buckets keep any
//! directory under a day of calls.
use crate::sidecar::Sidecar;
use alloc::format;
use alloc::string::{String, ToString};

/// The traffic-character flag token: mode token (`F`, `A`, `T0`/`T1`,
/// `D1`/`D2`), then `C` for emergency, then `E` for encrypted.
/// `ls *-*C*` = every emergency, `*-*E.*` = every ciphertext.
pub fn flags_token(mode: &str, emergency: bool, encrypted: bool) -> String {
    let mut t = mode.to_string();
    if emergency {
        t.push('C');
    }
    if encrypted {
        t.push('E');
    }
    t
}

/// `YYYY/MM/DD/HH:MM:SS.mmm` in UTC from Unix milliseconds — portable
/// civil math (no localtime, no DST hole), sub-second kept.
pub fn fmt_utc_path(millis: i64) -> String {
    let epoch = millis.div_euclid(1000);
    let ms = millis.rem_euclid(1000);
    let (days, secs) = (epoch.div_euclid(86_400), epoch.rem_euclid(86_400));
    let z = days + 719_468; // days from 0000-03-01 to 1970-01-01
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}/{m:02}/{d:02}/{:02}:{:02}:{:02}.{ms:03}", secs / 3600, secs / 60 % 60, secs % 60)
}

/// The archive path, without extension.
#[allow(clippy::too_many_arguments)]
pub fn wav_base(
    out_dir: &str,
    system_dir: &str,
    start_ms: i64,
    tg: &str,
    src: &str,
    flags: &str,
    nac: &str,
    hz: u64,
    peak_delta_db: f32,
) -> String {
    let sys = if system_dir.is_empty() { "unknown" } else { system_dir };
    format!("{out_dir}/{sys}/{}-{tg}-{src}-{flags}-{nac}-{hz}-{peak_delta_db:.0}", fmt_utc_path(start_ms))
}

/// The archive path a sidecar names, without extension — what a server
/// stores a received `.sdr` or `.wav` under. Digital modes format tg
/// and src as the recorder does (`{:05}` / `{:08}`, the NAC as three
/// hex digits); conventional carries the tone token in the tg slot,
/// the ANI in src, and `000` for the NAC.
pub fn base_for(out_dir: &str, s: &Sidecar) -> String {
    let flags = flags_token(&s.mode, s.emergency, s.encrypted);
    let (tg, src, nac) = match &s.tone {
        Some(tone) => (tone.clone(), s.src.clone(), String::from("000")),
        None => (
            format!("{:05}", s.tg.unwrap_or(0)),
            format!("{:08}", s.src.parse::<u32>().unwrap_or(0)),
            format!("{:03X}", s.nac.unwrap_or(0)),
        ),
    };
    wav_base(out_dir, &s.system, s.start_ms, &tg, &src, &flags, &nac, s.hz, s.delta_db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_token_truth_table() {
        assert_eq!(flags_token("F", false, false), "F");
        assert_eq!(flags_token("F", true, false), "FC");
        assert_eq!(flags_token("F", false, true), "FE");
        assert_eq!(flags_token("F", true, true), "FCE");
        assert_eq!(flags_token("A", false, false), "A");
        assert_eq!(flags_token("T0", false, false), "T0");
        assert_eq!(flags_token("T1", false, true), "T1E");
        assert_eq!(flags_token("T0", true, true), "T0CE");
        assert_eq!(flags_token("D2", true, false), "D2C");
    }

    /// The NOTES.md archive anchor, byte-exact — the recorder's output
    /// grammar must never drift.
    #[test]
    fn wav_base_p25_anchor() {
        let start_ms = 1786503850696i64; // 2026-08-12T03:04:10.696Z
        let base = wav_base(
            "audio",
            "ABCDE123",
            start_ms,
            &format!("{:05}", 100),
            &format!("{:08}", 1234567),
            &flags_token("F", false, false),
            &format!("{:03X}", 0x293),
            851012500,
            50.4,
        );
        assert_eq!(base, "audio/ABCDE123/2026/08/12/03:04:10.696-00100-01234567-F-293-851012500-50");
    }

    /// The conventional shape: callsign dir, tone token in the tg slot,
    /// src 0, flags A, literal 000 in the nac slot — same field count as
    /// P25: one dash-split parse serves both modes.
    #[test]
    fn wav_base_conventional_shape() {
        let start_ms = 1786503850696i64;
        let base = wav_base("audio", "KA1ABC", start_ms, "D466N", "0", &flags_token("A", false, false), "000", 460012500, 22.0);
        assert_eq!(base, "audio/KA1ABC/2026/08/12/03:04:10.696-D466N-0-A-000-460012500-22");
    }

    /// An empty system dir still buckets (the P25 pre-identity case).
    #[test]
    fn wav_base_unknown_system() {
        let base = wav_base("out", "", 0, "00000", "00000000", "F", "000", 851000000, 0.0);
        assert!(base.starts_with("out/unknown/1970/01/01/"));
    }

    /// A sidecar names its own archive path, both modes.
    #[test]
    fn base_for_both_modes() {
        let p25 = Sidecar::parse(
            "{\"version\":1,\"start_ms\":1786503850696,\"duration_ms\":4250,\"hz\":851012500,\
             \"mode\":\"F\",\"emergency\":false,\"encrypted\":true,\"system\":\"ABCDE123\",\
             \"nac\":\"293\",\"tg\":100,\"alias\":null,\"src\":\"1234567\",\"delta_db\":50.4,\
             \"signal_db\":null,\"floor_db\":null,\"offset_hz\":null,\"wav_rate\":0}",
        )
        .unwrap();
        assert_eq!(base_for("audio", &p25), "audio/ABCDE123/2026/08/12/03:04:10.696-00100-01234567-FE-293-851012500-50");
        let conv = Sidecar::parse(
            "{\"version\":1,\"start_ms\":1786503850696,\"duration_ms\":9470,\"hz\":460012500,\
             \"mode\":\"A\",\"emergency\":true,\"encrypted\":false,\"system\":\"KA1ABC\",\
             \"nac\":null,\"tg\":null,\"alias\":null,\"tone\":\"D466N\",\"alphatag\":\"x\",\
             \"src\":\"0042\",\"delta_db\":56.0,\"signal_db\":-22.0,\"floor_db\":-78.0,\
             \"offset_hz\":120,\"mdc\":[],\"wav_rate\":16000}",
        )
        .unwrap();
        assert_eq!(base_for("audio", &conv), "audio/KA1ABC/2026/08/12/03:04:10.696-D466N-0042-AC-000-460012500-56");
    }
}
