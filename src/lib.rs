//! DigitalStream — one binary record format that is simultaneously the
//! `.sdr` archive file, the WebSocket binary message, and the WebRTC
//! datagram. One `encode_into`, one [`parse`], every transport; the file
//! is byte-for-byte a recorded stream.
//!
//! THE FILE IS THE RECORD (since 2026-09-06): a digital call
//! travels as `.sdr` alone, end to end — receiver, server, browser,
//! phone — and every piece of metadata a user interface or a database
//! wants is a PROJECTION of the file ([`sidecar::Sidecar::from_records`]),
//! computed by this one crate on whichever side holds the bytes. The
//! server never decodes: the voice frames stay in the digital domain
//! (twelve times smaller than PCM), and an encrypted call's ciphertext
//! rides through untouched for a client that holds the key. That is why
//! this crate has ZERO dependencies, builds `no_std` + `alloc`, and owns
//! every layout the receiver writes.
//!
//! Framing follows the `trudp` convention with its proven forward-
//! compatibility rules: `len` is AUTHORITATIVE (readers advance `len`
//! with no per-type knowledge), tail bytes beyond a known layout are
//! ignored, unknown `typ`s are skipped whole. Every record opens with
//! the magic and repeats hz/nac/epoch, so any record is understandable
//! in isolation — mid-call join, datagram loss, and file truncation all
//! degrade to "start from any record". A truncated file is readable up
//! to the cut with no header to patch.
//!
//! Every record opens with the same 32-octet head (all multi-octet
//! fields little-endian), laid out so that every field sits on its
//! natural boundary and a file mapped at an aligned base reads natively:
//!
//! ```text
//!  off  size  field
//!   0     4   magic     's' 'd' 'r' 0
//!   4     1   ver       format version, 1
//!   5     1   typ       the record registry below
//!   6     2   len       total record size in OCTETS, head included,
//!                       a multiple of 8 — AUTHORITATIVE
//!   8     8   epoch_us  µs since the Unix epoch, UTC, the receiver's clock
//!  16     4   hz        the carrier
//!  20     4   seq       per-channel voice-clock ordinal
//!  24     2   nac       P25 Network Access Code, or the DMR colour code
//!  26     2   flags     per-typ bits; undefined bits write 0, readers ignore
//!  28     4   site      the receiver's MimoCAD SITE_ID (which of OUR
//!                       receivers heard it; 0 = unknown)
//! ```
//!
//! - `typ` — 1 CallHeader · 2 Ldu (P25 Phase 1, [`p25`]) · 3 CallTrailer
//!   · 4 Ess · 5 P2Vch (P25 Phase 2, [`p25`]) · 6..=10 DMR ([`dmr`]);
//!   others reserved.
//! - `ver` — 1. A reader refuses any other value the way it refuses a
//!   cut: the format before this one (magic `MS`, a 24-octet head, a
//!   one-octet `len/4`) was v0 and was never deployed; there is no v0
//!   reader.
//! - `seq` — per-channel voice-clock ordinal (header = 0, voice from
//!   1; concealed units consume numbers, so a jump of k in 2..=4 means
//!   (k−1) concealed units and > 4 a new talker).
//!
//! THIS CRATE parses the head, the mode-neutral records
//! ([`CallHeader`], [`CallTrailer`], [`EssRecord`]) and the body LAYOUTS
//! of every mode ([`p25`], [`dmr`]); the codec crates add the
//! conversions between those layouts and their decoders. Unknown typs
//! come back as [`Raw`] records — body bytes plus head — and are
//! re-emitted verbatim.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

pub mod dmr;
pub mod json;
pub mod name;
pub mod p25;
pub mod riff;
pub mod sidecar;

/// Every record opens with these four bytes.
pub const MAGIC: [u8; 4] = *b"sdr\0";

/// The format version this crate writes and the only one it reads.
pub const VERSION: u8 = 1;

/// The common head is always 32 bytes.
pub const HEAD_BYTES: usize = 32;

/// Records are padded to a multiple of this, so every head — and every
/// naturally aligned field behind it — stays aligned in a mapped file.
pub const RECORD_ALIGN: usize = 8;

/// The largest record the two-byte `len` can describe, rounded down to
/// the record alignment.
pub const MAX_RECORD_BYTES: usize = 65_528;

pub const TYP_CALL_HEADER: u8 = 1;
pub const TYP_LDU: u8 = 2;
pub const TYP_CALL_TRAILER: u8 = 3;
pub const TYP_ESS: u8 = 4;
pub const TYP_P2_VCH: u8 = 5;
pub const TYP_DMR_VOICE: u8 = 6;
pub const TYP_DMR_LC: u8 = 7;
pub const TYP_DMR_PACKET: u8 = 8;
pub const TYP_DMR_FIX: u8 = 9;
pub const TYP_DMR_ALIAS: u8 = 10;

// CallHeader flags.
/// A sibling `.wav` exists (dual-write acceptance mode).
pub const HDR_FLAG_DUAL_WRITE: u16 = 1 << 0;
/// tg/src hold a real grant-seeded identity; clear = placeholders, trust
/// the Link Control inside the frames or the trailer.
pub const HDR_FLAG_SEEDED: u16 = 1 << 1;

// Ldu flags (bodies in [`p25`]; the bits are registry, so they live here).
/// This is an LDU2 (else LDU1).
pub const LDU_FLAG_LDU2: u16 = 1 << 0;
/// NID inferred from cadence (flywheel), not decoded.
pub const LDU_FLAG_FLYWHEEL: u16 = 1 << 1;
/// DUID forced from the LDU1/LDU2 alternation grid after a NID reject.
pub const LDU_FLAG_FORCED: u16 = 1 << 2;
/// The live RS decode of the extra (LC/ESS) succeeded. AUTHORITATIVE:
/// when clear, the extra inside `body` is untrusted even though its
/// re-encoded parity checks out.
pub const LDU_FLAG_RS_OK: u16 = 1 << 3;

// CallTrailer flags.
pub const TRL_FLAG_ENCRYPTED: u16 = 1 << 0;
pub const TRL_FLAG_EMERGENCY: u16 = 1 << 1;

// P2VchFrame flags (bodies in [`p25`]).
/// Encryption was CONFIRMED (the Phase 2 EssConfirm discipline) by
/// this superframe.
pub const P2V_FLAG_ENCRYPTED: u16 = 1 << 0;
/// The assembler could not vouch for every placement in this
/// superframe (inferred location, misplaced voice, or a partial
/// flush) — see `phase2_vch::VchSuperframe::cadence_ok`.
pub const P2V_FLAG_CADENCE: u16 = 1 << 1;
/// The stored phase (and therefore the dibits it implies) is
/// DESCRAMBLED — the BBAE network scramble has been removed and the
/// frames are AMBE+2 codewords. Clear means the record holds exactly
/// what came off the air, which for an outbound 4V/2V burst means
/// SCRAMBLED — recordable, not vocodable.
pub const P2V_FLAG_DESCRAMBLED: u16 = 1 << 2;
/// A fragment's location was INFERRED (both I-ISCH failed → flywheel).
pub const P2V_FLAG_LOC_INFERRED: u16 = 1 << 3;
/// A voice burst contradicted its cadence position and was dropped,
/// or a decoded Channel Number contradicted slot parity.
pub const P2V_FLAG_MISPLACED: u16 = 1 << 4;
/// The superframe was closed early (tainted flush or teardown).
pub const P2V_FLAG_PARTIAL: u16 = 1 << 5;

/// The CallHeader `mode` values. 0 is deliberately invalid so zeroed
/// memory can never parse as a real header. The filename letter is
/// DERIVED from this at presentation, never stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// P25 Phase 1 FDMA voice.
    P25Fdma = 1,
    /// Analog FM conventional.
    Analog = 2,
    /// P25 Phase 2 TDMA voice — one P2Vch record per LCH per 360 ms
    /// superframe; `CallHeader.slot` carries the LCH.
    P25Tdma = 3,
    /// DMR (ETSI TS 102 361) — one [`dmr::DmrVoice`] record per slot
    /// per 360 ms voice superframe; `CallHeader.slot` carries the TDMA
    /// slot (0/1) and the head's `nac` the colour code.
    Dmr = 4,
}

impl Mode {
    pub fn from_u8(v: u8) -> Option<Mode> {
        Some(match v {
            1 => Mode::P25Fdma,
            2 => Mode::Analog,
            3 => Mode::P25Tdma,
            4 => Mode::Dmr,
            _ => return None,
        })
    }

    /// The archive filename's mode letter (the slot digit, where the
    /// mode has one, is appended by the recorder: `T0`/`T1`, `D1`/`D2`).
    pub fn letter(self) -> &'static str {
        match self {
            Mode::P25Fdma => "F",
            Mode::Analog => "A",
            Mode::P25Tdma => "T",
            Mode::Dmr => "D",
        }
    }

    /// The filename's mode token WITH the slot digit, as the recorder
    /// names files: `F`, `A`, `T0`/`T1` (the LCH), `D1`/`D2` (the TDMA
    /// slot, one-based).
    pub fn token(self, slot: u8) -> String {
        match self {
            Mode::P25Tdma => alloc::format!("T{}", slot),
            Mode::Dmr => alloc::format!("D{}", slot as u16 + 1),
            m => String::from(m.letter()),
        }
    }
}

/// The common head, minus magic/ver/typ/len (which are the codec's
/// business: typ is the record variant, len is computed at encode).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MsgHead {
    pub seq: u32,
    pub epoch_us: u64,
    pub hz: u32,
    pub nac: u16,
    pub flags: u16,
    /// The receiver's MimoCAD SITE_ID; 0 when not configured.
    pub site: u32,
}

/// Once per call, first record: the identity known at grant time plus
/// the receiver-side facts no voice frame ever carries (WACN/SYSID come
/// from a P25 control channel's Network Status Broadcast; DMR leaves
/// them 0). Body:
///
/// ```text
///  32  1  mode      32+1  1  slot     34  2  sysid
///  36  4  tg        40    4  src      44  4  wacn
///  48  n  system    length-prefixed string, then pad to 8
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallHeader {
    pub head: MsgHead,
    /// Raw wire value; interpret via [`Mode::from_u8`] (unknown values
    /// survive a parse→re-encode round trip).
    pub mode: u8,
    /// 0xFF unless the mode has slots; then the slot.
    pub slot: u8,
    /// Grant-seeded talkgroup — valid iff [`HDR_FLAG_SEEDED`]. 32 bits:
    /// a DMR group id is 24.
    pub tg: u32,
    /// Grant-seeded source radio (24-bit unit ID) — valid iff seeded.
    pub src: u32,
    /// 20-bit Wide Area Communications Network ID (P25).
    pub wacn: u32,
    /// 12-bit system ID (P25).
    pub sysid: u16,
    /// Archive directory label: WACN+SYSID hex for P25 ("ABCDE123"),
    /// callsign for conventional ("KA1ABC"), the frequency+type dir
    /// for a system nothing claims. Truncated to 255 bytes.
    pub system: String,
}

/// Once per call, last record: the close-time facts. A truncated file
/// missing its trailer falls back to the CallHeader identity and the
/// last voice record's epoch. Body:
///
/// ```text
///  32  4  duration_ms   36  4  tg          40  4  src
///  44  4  delta_db f32  48  4  signal_db   52  4  floor_db
///  56  4  offset_hz i32 60  n  alias str   ..  n  alphatag str, pad to 8
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CallTrailer {
    pub head: MsgHead,
    /// Call length on the PCM clock.
    pub duration_ms: u32,
    /// Final talkgroup, LC-refined; overrides the header's seed.
    pub tg: u32,
    /// Final source radio, LC-refined.
    pub src: u32,
    /// Peak signal above the floor over the call (demod-derived).
    pub delta_db: f32,
    /// The absolutes behind the delta, measured at the peak-delta
    /// instant (`signal_db − delta_db = floor_db` exactly); NaN when
    /// the path does not measure them.
    pub signal_db: f32,
    pub floor_db: f32,
    /// Mean carrier frequency offset — transmitter-drift health;
    /// [`OFFSET_UNKNOWN`] when not measured.
    pub offset_hz: i32,
    /// Config enrichment; empty when unknown. Truncated to 255 bytes.
    pub alias: String,
    pub alphatag: String,
}

/// `CallTrailer::offset_hz` when the offset was not measured.
pub const OFFSET_UNKNOWN: i32 = i32::MIN;

/// Once per encrypted call, at the first RS-ok ESS: the raw 12-byte
/// LDU2 extra verbatim — MI (9 B, the crypto IV), ALGID (1 B), KID
/// (2 B). Redundant with the LDU2 bodies by design: it survives even
/// when ciphertext frames are elided, and names the algorithm without
/// walking the stream. Always 48 bytes (body at 32..44, pad to 48).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EssRecord {
    pub head: MsgHead,
    pub ess: [u8; 12],
}

impl EssRecord {
    pub const BYTES: usize = 48;

    /// MI, ALGID, KID unpacked from the wire order (MI first).
    pub fn mi(&self) -> [u8; 9] {
        let mut mi = [0u8; 9];
        mi.copy_from_slice(&self.ess[..9]);
        mi
    }
    pub fn algid(&self) -> u8 {
        self.ess[9]
    }
    pub fn kid(&self) -> u16 {
        u16::from_be_bytes([self.ess[10], self.ess[11]])
    }
}

/// A record whose body this module does not interpret: the whole record
/// as declared by its `len`, head included, so a body codec parses with
/// the same offsets it encodes with (body fields start at
/// [`HEAD_BYTES`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raw {
    pub typ: u8,
    pub head: MsgHead,
    /// The record bytes, head included, exactly `len` long.
    pub bytes: Vec<u8>,
}

/// One parsed record. [`Record::Raw`] carries every typ the head parser
/// does not own — the mode bodies ([`p25`], [`dmr`] parse them from the
/// raw) and any unknown one from a future sender, which a reader skips
/// (its `len` was honored) and keeps going.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Header(CallHeader),
    Trailer(CallTrailer),
    Ess(EssRecord),
    Raw(Raw),
}

impl Record {
    pub fn head(&self) -> &MsgHead {
        match self {
            Record::Header(r) => &r.head,
            Record::Trailer(r) => &r.head,
            Record::Ess(r) => &r.head,
            Record::Raw(r) => &r.head,
        }
    }

    pub fn typ(&self) -> u8 {
        match self {
            Record::Header(_) => TYP_CALL_HEADER,
            Record::Trailer(_) => TYP_CALL_TRAILER,
            Record::Ess(_) => TYP_ESS,
            Record::Raw(r) => r.typ,
        }
    }

    /// Transport-blind encode-append. A raw record re-emits its bytes.
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Record::Header(r) => r.encode_into(out),
            Record::Trailer(r) => r.encode_into(out),
            Record::Ess(r) => r.encode_into(out),
            Record::Raw(r) => out.extend_from_slice(&r.bytes),
        }
    }
}

/// Round up to the record alignment.
///
/// (These small readers and helpers are `#[inline]` on purpose: every
/// consumer of this crate — the codec crates, the receiver, a server, a
/// WASM build — calls them across the crate boundary, where a body is
/// only available to the optimizer when exported like this or under
/// LTO; the automatic small-function export is a heuristic.)
#[inline]
pub fn pad8(n: usize) -> usize {
    n.div_ceil(RECORD_ALIGN) * RECORD_ALIGN
}

/// Append a record head. `total` is the whole record's byte length,
/// which must be a multiple of [`RECORD_ALIGN`] within
/// [`HEAD_BYTES`]..=[`MAX_RECORD_BYTES`].
pub fn push_head(out: &mut Vec<u8>, typ: u8, total: usize, h: &MsgHead) {
    debug_assert!(
        total % RECORD_ALIGN == 0 && (HEAD_BYTES..=MAX_RECORD_BYTES).contains(&total),
        "record length {total}"
    );
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(typ);
    out.extend_from_slice(&(total as u16).to_le_bytes());
    out.extend_from_slice(&h.epoch_us.to_le_bytes());
    out.extend_from_slice(&h.hz.to_le_bytes());
    out.extend_from_slice(&h.seq.to_le_bytes());
    out.extend_from_slice(&h.nac.to_le_bytes());
    out.extend_from_slice(&h.flags.to_le_bytes());
    out.extend_from_slice(&h.site.to_le_bytes());
}

/// Append a length-prefixed string, silently truncated to 255 bytes.
pub fn push_str(out: &mut Vec<u8>, s: &str) {
    let b = &s.as_bytes()[..s.len().min(255)];
    out.push(b.len() as u8);
    out.extend_from_slice(b);
}

pub fn pad_to(out: &mut Vec<u8>, target: usize) {
    while out.len() < target {
        out.push(0);
    }
}

impl CallHeader {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let sys = self.system.len().min(255);
        let total = pad8(HEAD_BYTES + 16 + 1 + sys);
        let base = out.len();
        push_head(out, TYP_CALL_HEADER, total, &self.head);
        out.push(self.mode);
        out.push(self.slot);
        out.extend_from_slice(&self.sysid.to_le_bytes());
        out.extend_from_slice(&self.tg.to_le_bytes());
        out.extend_from_slice(&self.src.to_le_bytes());
        out.extend_from_slice(&self.wacn.to_le_bytes());
        push_str(out, &self.system);
        pad_to(out, base + total);
    }

    fn parse(b: &[u8], head: MsgHead) -> Option<CallHeader> {
        const B: usize = HEAD_BYTES;
        if b.len() < B + 17 {
            return None;
        }
        let (system, _) = take_str(b, B + 16)?;
        Some(CallHeader {
            head,
            mode: b[B],
            slot: b[B + 1],
            sysid: u16le(b, B + 2),
            tg: u32le(b, B + 4),
            src: u32le(b, B + 8),
            wacn: u32le(b, B + 12),
            system,
        })
    }
}

impl CallTrailer {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let total = pad8(
            HEAD_BYTES + 28 + 1 + self.alias.len().min(255) + 1 + self.alphatag.len().min(255),
        );
        let base = out.len();
        push_head(out, TYP_CALL_TRAILER, total, &self.head);
        out.extend_from_slice(&self.duration_ms.to_le_bytes());
        out.extend_from_slice(&self.tg.to_le_bytes());
        out.extend_from_slice(&self.src.to_le_bytes());
        out.extend_from_slice(&self.delta_db.to_le_bytes());
        out.extend_from_slice(&self.signal_db.to_le_bytes());
        out.extend_from_slice(&self.floor_db.to_le_bytes());
        out.extend_from_slice(&self.offset_hz.to_le_bytes());
        push_str(out, &self.alias);
        push_str(out, &self.alphatag);
        pad_to(out, base + total);
    }

    fn parse(b: &[u8], head: MsgHead) -> Option<CallTrailer> {
        const B: usize = HEAD_BYTES;
        if b.len() < B + 30 {
            return None;
        }
        let (alias, used) = take_str(b, B + 28)?;
        let (alphatag, _) = take_str(b, B + 28 + used)?;
        Some(CallTrailer {
            head,
            duration_ms: u32le(b, B),
            tg: u32le(b, B + 4),
            src: u32le(b, B + 8),
            delta_db: f32le(b, B + 12),
            signal_db: f32le(b, B + 16),
            floor_db: f32le(b, B + 20),
            offset_hz: i32le(b, B + 24),
            alias,
            alphatag,
        })
    }

    pub fn encrypted(&self) -> bool {
        self.head.flags & TRL_FLAG_ENCRYPTED != 0
    }

    pub fn emergency(&self) -> bool {
        self.head.flags & TRL_FLAG_EMERGENCY != 0
    }
}

impl EssRecord {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_ESS, Self::BYTES, &self.head);
        out.extend_from_slice(&self.ess);
        pad_to(out, base + Self::BYTES);
    }

    fn parse(b: &[u8], head: MsgHead) -> Option<EssRecord> {
        if b.len() < HEAD_BYTES + 12 {
            return None;
        }
        let mut ess = [0u8; 12];
        ess.copy_from_slice(&b[HEAD_BYTES..HEAD_BYTES + 12]);
        Some(EssRecord { head, ess })
    }
}

#[inline]
pub fn u16le(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

#[inline]
pub fn u32le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

#[inline]
pub fn i32le(b: &[u8], i: usize) -> i32 {
    i32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

#[inline]
pub fn f32le(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

#[inline]
pub fn u64le(b: &[u8], i: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[i..i + 8]);
    u64::from_le_bytes(w)
}

/// Length-prefixed string at `at`, bounded by the record (`b` is already
/// sliced to the declared length, so the bound IS the declared length).
/// Returns the string and the WIRE bytes consumed (1 + n — the decoded
/// String's byte length can differ under lossy UTF-8 replacement).
pub fn take_str(b: &[u8], at: usize) -> Option<(String, usize)> {
    let n = *b.get(at)? as usize;
    let s = b.get(at + 1..at + 1 + n)?;
    Some((String::from_utf8_lossy(s).into_owned(), 1 + n))
}

/// Parse just the head. `None` when the magic, version, length, or
/// alignment disagree with the bytes — a streaming reader stops here.
pub fn parse_head(b: &[u8]) -> Option<(u8, MsgHead, usize)> {
    if b.len() < HEAD_BYTES || b[..4] != MAGIC || b[4] != VERSION {
        return None;
    }
    let typ = b[5];
    let total = u16le(b, 6) as usize;
    if total < HEAD_BYTES || total > b.len() || total % RECORD_ALIGN != 0 {
        return None;
    }
    let head = MsgHead {
        epoch_us: u64le(b, 8),
        hz: u32le(b, 16),
        seq: u32le(b, 20),
        nac: u16le(b, 24),
        flags: u16le(b, 26),
        site: u32le(b, 28),
    };
    Some((typ, head, total))
}

/// Parse one record from the front of `b`. Returns the record and the
/// bytes consumed (the declared length — trailing buffer is the next
/// record). `None` means cut or garbage: a streaming reader stops here;
/// at most one partial record is ever lost to a crash.
pub fn parse(b: &[u8]) -> Option<(Record, usize)> {
    let (typ, head, total) = parse_head(b)?;
    // Slice to the declared length: `len` is authoritative, and known
    // typs whose payload can't fit inside it are garbage, not records.
    let rec = &b[..total];
    let record = match typ {
        TYP_CALL_HEADER => Record::Header(CallHeader::parse(rec, head)?),
        TYP_CALL_TRAILER => Record::Trailer(CallTrailer::parse(rec, head)?),
        TYP_ESS => Record::Ess(EssRecord::parse(rec, head)?),
        _ => Record::Raw(Raw { typ, head, bytes: rec.to_vec() }),
    };
    Some((record, total))
}

/// Iterate the records in a buffer (an `.sdr` file, a coalesced network
/// read). Stops at the first cut or non-record byte — truncation-safe.
pub fn records(buf: &[u8]) -> Records<'_> {
    Records { buf, pos: 0 }
}

pub struct Records<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Iterator for Records<'_> {
    type Item = Record;

    fn next(&mut self) -> Option<Record> {
        let (rec, used) = parse(&self.buf[self.pos..])?;
        self.pos += used;
        Some(rec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head() -> MsgHead {
        MsgHead { seq: 0, epoch_us: 0x1122334455667788, hz: 851_012_500, nac: 0x293, flags: 0, site: 13 }
    }

    /// The CallHeader, byte for byte: the v1 head (magic, ver, typ, len
    /// in octets, epoch at 8, hz/seq, nac/flags/site) and the body.
    #[test]
    fn golden_call_header() {
        let h = CallHeader {
            head: MsgHead { flags: HDR_FLAG_SEEDED, ..head() },
            mode: Mode::P25Fdma as u8,
            slot: 0xFF,
            tg: 100,
            src: 1_234_567,
            wacn: 0xABCDE,
            sysid: 0x123,
            system: "ABCDE123".into(),
        };
        let mut out = Vec::new();
        h.encode_into(&mut out);
        #[rustfmt::skip]
        let expect: [u8; 64] = [
            b's', b'd', b'r', 0x00,
            0x01, 0x01, 0x40, 0x00,
            0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
            0x94, 0x6B, 0xB9, 0x32,
            0x00, 0x00, 0x00, 0x00,
            0x93, 0x02, 0x02, 0x00,
            0x0D, 0x00, 0x00, 0x00,
            0x01, 0xFF, 0x23, 0x01,
            0x64, 0x00, 0x00, 0x00,
            0x87, 0xD6, 0x12, 0x00,
            0xDE, 0xBC, 0x0A, 0x00,
            0x08, b'A', b'B', b'C', b'D', b'E', b'1', b'2', b'3',
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(out, expect);
        let (rec, used) = parse(&out).unwrap();
        assert_eq!(used, 64);
        assert_eq!(rec, Record::Header(h));
    }

    #[test]
    fn mode_registry_round_trips_and_tokens() {
        for m in [Mode::P25Fdma, Mode::Analog, Mode::P25Tdma, Mode::Dmr] {
            assert_eq!(Mode::from_u8(m as u8), Some(m));
        }
        assert_eq!(Mode::from_u8(0), None, "zeroed memory never parses as a mode");
        assert_eq!(Mode::Dmr.letter(), "D");
        assert_eq!(Mode::Dmr.token(1), "D2");
        assert_eq!(Mode::P25Tdma.token(0), "T0");
        assert_eq!(Mode::P25Fdma.token(0xFF), "F");
        assert_eq!(Mode::Analog.token(0xFF), "A");
    }

    #[test]
    fn golden_ess() {
        let e = EssRecord { head: MsgHead { seq: 7, ..head() }, ess: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0x84, 0x12, 0x34] };
        let mut out = Vec::new();
        e.encode_into(&mut out);
        assert_eq!(out.len(), 48);
        assert_eq!(&out[..8], &[b's', b'd', b'r', 0, 1, TYP_ESS, 48, 0]);
        assert_eq!(&out[32..44], &e.ess);
        let (rec, used) = parse(&out).unwrap();
        assert_eq!(used, 48);
        match rec {
            Record::Ess(p) => {
                assert_eq!(p, e);
                assert_eq!(p.mi(), [9, 8, 7, 6, 5, 4, 3, 2, 1]);
                assert_eq!(p.algid(), 0x84);
                assert_eq!(p.kid(), 0x1234);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn golden_trailer() {
        let t = CallTrailer {
            head: MsgHead { seq: 42, flags: TRL_FLAG_ENCRYPTED, ..head() },
            duration_ms: 4_260,
            tg: 100,
            src: 1_234_567,
            delta_db: 21.5,
            signal_db: -21.4,
            floor_db: -42.9,
            offset_hz: 139,
            alias: "DISP01".into(),
            alphatag: "SCPD Disp".into(),
        };
        let mut out = Vec::new();
        t.encode_into(&mut out);
        assert_eq!(out.len(), 80);
        assert_eq!(u16le(&out, 6), 80);
        assert_eq!(u32le(&out, 32), 4_260);
        assert_eq!(f32le(&out, 44), 21.5);
        assert_eq!(f32le(&out, 52), -42.9);
        assert_eq!(i32le(&out, 56), 139);
        assert_eq!(&out[61..67], b"DISP01");
        assert_eq!(&out[68..77], b"SCPD Disp");
        let (rec, used) = parse(&out).unwrap();
        assert_eq!(used, 80);
        match rec {
            Record::Trailer(p) => {
                assert!(p.encrypted() && !p.emergency());
                assert_eq!(p, t);
            }
            other => panic!("{other:?}"),
        }
        // Unknown absolutes are NaN and survive the wire as NaN.
        let u = CallTrailer { signal_db: f32::NAN, floor_db: f32::NAN, offset_hz: OFFSET_UNKNOWN, ..t };
        let mut out = Vec::new();
        u.encode_into(&mut out);
        match parse(&out).unwrap().0 {
            Record::Trailer(p) => {
                assert!(p.signal_db.is_nan() && p.floor_db.is_nan());
                assert_eq!(p.offset_hz, OFFSET_UNKNOWN);
            }
            other => panic!("{other:?}"),
        }
    }

    /// Unknown typ from a future sender: handed back RAW with its bytes,
    /// stream continues. And a codec-owned typ (an LDU, 2) comes back
    /// the same way — the head parser never guesses at a body.
    #[test]
    fn unowned_typs_come_back_raw_and_the_stream_continues() {
        let mut buf = Vec::new();
        push_head(&mut buf, 200, 40, &MsgHead { seq: 5, ..head() });
        buf.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        push_head(&mut buf, TYP_LDU, 40, &MsgHead { seq: 6, ..head() });
        buf.extend_from_slice(&[9; 8]);
        EssRecord { head: head(), ess: [0; 12] }.encode_into(&mut buf);
        let all: Vec<Record> = records(&buf).collect();
        assert_eq!(all.len(), 3);
        match &all[0] {
            Record::Raw(r) => {
                assert_eq!((r.typ, r.head.seq, r.bytes.len()), (200, 5, 40));
                assert_eq!(&r.bytes[32..], &[1, 2, 3, 4, 5, 6, 7, 8]);
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(&all[1], Record::Raw(r) if r.typ == TYP_LDU));
        assert!(matches!(all[2], Record::Ess(_)));
        // Re-encoding a raw record emits its bytes verbatim.
        let mut again = Vec::new();
        all[0].encode_into(&mut again);
        assert_eq!(again, buf[..40]);
    }

    #[test]
    fn stream_stops_at_a_cut_and_ignores_trailing_garbage() {
        let mut buf = Vec::new();
        EssRecord { head: head(), ess: [1; 12] }.encode_into(&mut buf);
        EssRecord { head: head(), ess: [2; 12] }.encode_into(&mut buf);
        assert_eq!(records(&buf).count(), 2);
        assert_eq!(records(&buf[..buf.len() - 10]).count(), 1, "a cut loses at most one record");
        let mut junk = buf.clone();
        junk.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0]);
        assert_eq!(records(&junk).count(), 2);
    }

    /// A later minor revision with tail fields we don't know: len is
    /// authoritative, the tail is ignored, parsing succeeds.
    #[test]
    fn longer_len_tail_ignored() {
        let e = EssRecord { head: head(), ess: [1; 12] };
        let mut buf = Vec::new();
        e.encode_into(&mut buf);
        buf[6] += 8;
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0, 0, 0, 0]);
        let (rec, used) = parse(&buf).unwrap();
        assert_eq!(used, 56);
        assert_eq!(rec, Record::Ess(e));
    }

    /// What is refused: the v0 magic, a foreign version, a misaligned
    /// length, and the `MC` referee packet.
    #[test]
    fn foreign_heads_rejected() {
        let e = EssRecord { head: head(), ess: [1; 12] };
        let mut buf = Vec::new();
        e.encode_into(&mut buf);
        let mut v0 = buf.clone();
        v0[..4].copy_from_slice(&[b'M', b'S', 4, 9]);
        assert!(parse(&v0).is_none(), "v0 files are not read");
        let mut v2 = buf.clone();
        v2[4] = 2;
        assert!(parse(&v2).is_none(), "a version this crate does not know");
        let mut odd = buf.clone();
        odd[6] = 44;
        assert!(parse(&odd).is_none(), "records are 8-aligned");
        let mut mc = [0u8; 36];
        mc[..4].copy_from_slice(&[b'M', b'C', 8, 9]);
        assert!(parse(&mc).is_none());
        assert!(parse(&buf[..20]).is_none(), "a cut head");
    }
}
