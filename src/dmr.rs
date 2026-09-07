//! DMR record bodies (ETSI TS 102 361): the archive's second digital
//! mode, defined here as plain byte layouts because no codec crate owns
//! them — `dsp::dmr` decodes the air, `p25::ambe` speaks the voice, and
//! this crate is where the wire format of what they found lives.
//!
//! Every DMR record's head carries the colour code in `nac` and the
//! TDMA slot (0/1) in its body; a call is one slot of one carrier.
//!
//! - [`DmrVoice`] (typ 6) — one 360 ms voice superframe: 6 bursts ×
//!   3 AMBE+2 half-rate frames of 72 bits, stored as the 72 air bits
//!   packed 8/byte (9 octets a frame, 162 in all) with a burst-valid
//!   mask; missing bursts are holes that replay as concealment, never
//!   fabricated audio. Same vocoder as P25 Phase 2 (the interleave IS
//!   Annex S), so `p25::ambe::decode_frame` takes each frame's dibits.
//! - [`DmrLc`] (typ 7) — a Full Link Control word: the RS-verified
//!   voice header or terminator, or the checksum-verified embedded LC
//!   assembled across bursts B–E. Nine octets, FLCO/FID first.
//! - [`DmrPacket`] (typ 8) — a reassembled data packet: the 12-octet
//!   data header and the user data (pad octets stripped), with the
//!   message CRC-32 verdict in the flags.
//! - [`DmrFix`] (typ 9) — a position parsed from an RMC sentence the
//!   packet layer carried, with the raw sentence beside it.
//! - [`DmrAlias`] (typ 10) — a talker alias assembled from the
//!   embedded-LC alias header and blocks.

use alloc::string::String;
use alloc::vec::Vec;

use crate::{
    HEAD_BYTES, MAX_RECORD_BYTES, MsgHead, Raw, TYP_DMR_ALIAS, TYP_DMR_FIX, TYP_DMR_LC,
    TYP_DMR_PACKET, TYP_DMR_VOICE, i32le, pad8, pad_to, push_head, push_str, take_str, u16le,
    u32le,
};

// DmrVoice flags.
/// The superframe's bursts carried a Privacy Indicator (encrypted).
pub const DV_FLAG_ENCRYPTED: u16 = 1 << 0;
/// One or more bursts were BACKFILLED from a later anchor (a missed
/// burst-A sync) rather than carved forward.
pub const DV_FLAG_BACKFILL: u16 = 1 << 1;

// DmrLc kinds.
pub const LC_KIND_HEADER: u8 = 1;
pub const LC_KIND_TERMINATOR: u8 = 2;
pub const LC_KIND_EMBEDDED: u8 = 3;

// DmrPacket flags.
pub const PKT_FLAG_CRC_OK: u16 = 1 << 0;
/// The user data was longer than a record can hold and was cut.
pub const PKT_FLAG_TRUNCATED: u16 = 1 << 1;

// DmrFix flags.
/// The sentence's status was `A` (active).
pub const FIX_FLAG_VALID: u16 = 1 << 0;
/// The sentence's own `*hh` checksum verified (reported, never
/// enforced — the message CRC-32 already vouched for the bytes).
pub const FIX_FLAG_NMEA_CHECKSUM_OK: u16 = 1 << 1;

/// Frames per superframe: 6 bursts × 3.
pub const VOICE_FRAMES: usize = 18;
/// A 72-bit AMBE+2 frame packed 8 bits per octet.
pub const FRAME_OCTETS: usize = 9;

/// One voice superframe on one slot. Always 200 bytes on the wire:
///
/// ```text
/// offset  size  field
///    0     32   head          (typ = 6, len = 200; nac = colour code)
///   32      1   slot          0 or 1
///   33      1   burst_valid   bits 0..6: lattice slot i carried a burst
///   34      1   slots         lattice slots this record covers (1..=6);
///                             slots ≥ this are past the call's end, not holes
///   35      1   pad
///   36    162   frames        18 × 9 octets, lattice order, 3 per slot
///  198      2   pad
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmrVoice {
    pub head: MsgHead,
    pub slot: u8,
    pub burst_valid: u8,
    /// Lattice slots this record covers (1..=6). A superframe is 6;
    /// a call's last record may cover fewer — the rest are not holes.
    pub slots: u8,
    /// The 72 air bits of each frame, MSB-first, 9 octets a frame.
    pub frames: [[u8; FRAME_OCTETS]; VOICE_FRAMES],
}

impl DmrVoice {
    pub const BYTES: usize = 200;

    /// Pack a frame's 72 air bits (one per element, 0/1) into 9 octets.
    pub fn pack_frame(bits: &[u8; 72]) -> [u8; FRAME_OCTETS] {
        let mut out = [0u8; FRAME_OCTETS];
        for (i, &b) in bits.iter().enumerate() {
            out[i / 8] |= (b & 1) << (7 - i % 8);
        }
        out
    }

    /// Unpack back to the 72 bits `p25::ambe::voice_frame_dibits` takes.
    pub fn unpack_frame(octets: &[u8; FRAME_OCTETS]) -> [u8; 72] {
        let mut out = [0u8; 72];
        for (i, b) in out.iter_mut().enumerate() {
            *b = octets[i / 8] >> (7 - i % 8) & 1;
        }
        out
    }

    pub fn burst_present(&self, burst: usize) -> bool {
        self.burst_valid >> burst & 1 == 1
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_DMR_VOICE, Self::BYTES, &self.head);
        out.push(self.slot);
        out.push(self.burst_valid);
        out.push(self.slots);
        out.push(0);
        for f in &self.frames {
            out.extend_from_slice(f);
        }
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<DmrVoice> {
        if r.typ != TYP_DMR_VOICE || r.bytes.len() < Self::BYTES {
            return None;
        }
        let b = &r.bytes;
        let mut frames = [[0u8; FRAME_OCTETS]; VOICE_FRAMES];
        const B: usize = HEAD_BYTES;
        for (i, f) in frames.iter_mut().enumerate() {
            f.copy_from_slice(&b[B + 4 + 9 * i..B + 4 + 9 * i + 9]);
        }
        Some(DmrVoice { head: r.head, slot: b[B], burst_valid: b[B + 1], slots: b[B + 2], frames })
    }
}

/// A Full Link Control word. Always 48 bytes on the wire: head, kind,
/// slot, the 9 LC octets, pad.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmrLc {
    pub head: MsgHead,
    /// [`LC_KIND_HEADER`] / [`LC_KIND_TERMINATOR`] / [`LC_KIND_EMBEDDED`].
    pub kind: u8,
    pub slot: u8,
    /// The 72-bit LC as decoded: octet 0 = PF·R·FLCO(6), octet 1 = FID,
    /// octets 2..9 the FLCO-specific fields (Grp/UU voice: service
    /// options, group/target, source).
    pub lc: [u8; 9],
}

impl DmrLc {
    pub const BYTES: usize = 48;

    pub fn flco(&self) -> u8 {
        self.lc[0] & 0x3F
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_DMR_LC, Self::BYTES, &self.head);
        out.push(self.kind);
        out.push(self.slot);
        out.extend_from_slice(&self.lc);
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<DmrLc> {
        if r.typ != TYP_DMR_LC || r.bytes.len() < Self::BYTES {
            return None;
        }
        let b = &r.bytes;
        const B: usize = HEAD_BYTES;
        let mut lc = [0u8; 9];
        lc.copy_from_slice(&b[B + 2..B + 11]);
        Some(DmrLc { head: r.head, kind: b[B], slot: b[B + 1], lc })
    }
}

/// A reassembled data packet. Variable length:
///
/// ```text
/// offset  size  field
///    0     32   head          (typ = 8; flags: CRC_OK, TRUNCATED)
///   32      1   slot
///   33      1   pad
///   34      2   user_len      u16 LE, octets that follow the header
///   36     12   header        the CRC-valid data header, raw
///   48      n   user_data     pad octets already stripped
///          pad  to a multiple of 8
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmrPacket {
    pub head: MsgHead,
    pub slot: u8,
    pub header: [u8; 12],
    pub user_data: Vec<u8>,
}

impl DmrPacket {
    /// The most user data one record can carry.
    pub const MAX_USER: usize = MAX_RECORD_BYTES - 48;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let n = self.user_data.len().min(Self::MAX_USER);
        let total = pad8(HEAD_BYTES + 16 + n);
        let base = out.len();
        let mut head = self.head;
        if n < self.user_data.len() {
            head.flags |= PKT_FLAG_TRUNCATED;
        }
        push_head(out, TYP_DMR_PACKET, total, &head);
        out.push(self.slot);
        out.push(0);
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&self.header);
        out.extend_from_slice(&self.user_data[..n]);
        pad_to(out, base + total);
    }

    pub fn from_raw(r: &Raw) -> Option<DmrPacket> {
        const B: usize = HEAD_BYTES;
        if r.typ != TYP_DMR_PACKET || r.bytes.len() < B + 16 {
            return None;
        }
        let b = &r.bytes;
        let n = u16le(b, B + 2) as usize;
        let mut header = [0u8; 12];
        header.copy_from_slice(&b[B + 4..B + 16]);
        let user_data = b.get(B + 16..B + 16 + n)?.to_vec();
        Some(DmrPacket { head: r.head, slot: b[B], header, user_data })
    }
}

/// A position fix. Variable length:
///
/// ```text
/// offset  size  field
///    0     32   head          (typ = 9; flags: VALID, NMEA_CHECKSUM_OK)
///   32      1   slot
///   33      1   pad
///   34      2   course_cdeg   u16 LE, centidegrees
///   28      4   lat_udeg      i32 LE, microdegrees, north positive
///   32      4   lon_udeg      i32 LE, microdegrees, east positive
///   36      4   utc_ms        u32 LE, ms since midnight UTC
///   40      4   speed_mknots  u32 LE, milli-knots
///   44      2   year          u16 LE
///   46      1   month
///   47      1   day
///   48      4   src           u32 LE, the reporting radio (LLID)
///   52    1+n   sentence      length-prefixed raw NMEA
///          pad  to a multiple of 4
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmrFix {
    pub head: MsgHead,
    pub slot: u8,
    pub lat_udeg: i32,
    pub lon_udeg: i32,
    pub utc_ms: u32,
    pub speed_mknots: u32,
    pub course_cdeg: u16,
    pub date: (u16, u8, u8),
    pub src: u32,
    pub sentence: String,
}

impl DmrFix {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let total = pad8(HEAD_BYTES + 28 + 1 + self.sentence.len().min(255));
        let base = out.len();
        push_head(out, TYP_DMR_FIX, total, &self.head);
        out.push(self.slot);
        out.push(0);
        out.extend_from_slice(&self.course_cdeg.to_le_bytes());
        out.extend_from_slice(&self.lat_udeg.to_le_bytes());
        out.extend_from_slice(&self.lon_udeg.to_le_bytes());
        out.extend_from_slice(&self.utc_ms.to_le_bytes());
        out.extend_from_slice(&self.speed_mknots.to_le_bytes());
        out.extend_from_slice(&self.date.0.to_le_bytes());
        out.push(self.date.1);
        out.push(self.date.2);
        out.extend_from_slice(&self.src.to_le_bytes());
        push_str(out, &self.sentence);
        pad_to(out, base + total);
    }

    pub fn from_raw(r: &Raw) -> Option<DmrFix> {
        const B: usize = HEAD_BYTES;
        if r.typ != TYP_DMR_FIX || r.bytes.len() < B + 29 {
            return None;
        }
        let b = &r.bytes;
        let (sentence, _) = take_str(b, B + 28)?;
        Some(DmrFix {
            head: r.head,
            slot: b[B],
            course_cdeg: u16le(b, B + 2),
            lat_udeg: i32le(b, B + 4),
            lon_udeg: i32le(b, B + 8),
            utc_ms: u32le(b, B + 12),
            speed_mknots: u32le(b, B + 16),
            date: (u16le(b, B + 20), b[B + 22], b[B + 23]),
            src: u32le(b, B + 24),
            sentence,
        })
    }
}

/// A talker alias. Variable length: head, slot, format (Table 7.25:
/// 0 = 7-bit, 1 = ISO 8-bit, 2 = UTF-8, 3 = UTF-16BE), source LLID,
/// then the alias as a length-prefixed UTF-8 string, padded to 8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmrAlias {
    pub head: MsgHead,
    pub slot: u8,
    pub format: u8,
    pub src: u32,
    pub alias: String,
}

impl DmrAlias {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let total = pad8(HEAD_BYTES + 8 + 1 + self.alias.len().min(255));
        let base = out.len();
        push_head(out, TYP_DMR_ALIAS, total, &self.head);
        out.push(self.slot);
        out.push(self.format);
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&self.src.to_le_bytes());
        push_str(out, &self.alias);
        pad_to(out, base + total);
    }

    pub fn from_raw(r: &Raw) -> Option<DmrAlias> {
        const B: usize = HEAD_BYTES;
        if r.typ != TYP_DMR_ALIAS || r.bytes.len() < B + 9 {
            return None;
        }
        let b = &r.bytes;
        let (alias, _) = take_str(b, B + 8)?;
        Some(DmrAlias { head: r.head, slot: b[B], format: b[B + 1], src: u32le(b, B + 4), alias })
    }
}

/// The DMR records this module knows, from a [`Raw`] record.
#[derive(Clone, Debug, PartialEq)]
pub enum DmrRecord {
    Voice(DmrVoice),
    Lc(DmrLc),
    Packet(DmrPacket),
    Fix(DmrFix),
    Alias(DmrAlias),
}

pub fn parse_dmr(r: &Raw) -> Option<DmrRecord> {
    Some(match r.typ {
        TYP_DMR_VOICE => DmrRecord::Voice(DmrVoice::from_raw(r)?),
        TYP_DMR_LC => DmrRecord::Lc(DmrLc::from_raw(r)?),
        TYP_DMR_PACKET => DmrRecord::Packet(DmrPacket::from_raw(r)?),
        TYP_DMR_FIX => DmrRecord::Fix(DmrFix::from_raw(r)?),
        TYP_DMR_ALIAS => DmrRecord::Alias(DmrAlias::from_raw(r)?),
        _ => return None,
    })
}

/// (Moved here from `dsp::dmr` 2026-09-07: a record's meaning belongs
/// with its layout, so a reader of a [`DmrLc`] record needs no
/// signal-processing crate to name the talker.)
/// What a Full LC payload MEANS, for the standard-feature FLCOs (Full
/// Link Control Opcodes) of TS 102 361-2 Table B.1. Applies to the
/// header/terminator LC and the embedded LC alike — same nine octets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LcInfo {
    /// FLCO 000000 — who is talking to which talkgroup (table 7.1).
    GroupVoice { tg: u32, src: u32 },
    /// FLCO 000011 — unit-to-unit call (table 7.2).
    UnitVoice { dst: u32, src: u32 },
    /// FLCO 001000 — inband position (table 7.3): 3-bit error bucket
    /// (7 = no fix), then microdegrees so `no_std` needs no floats —
    /// longitude in steps of 360/2²⁵ °, latitude 180/2²⁴ °, both
    /// two's complement (§7.2.16/17).
    GpsInfo { err: u8, lon_udeg: i64, lat_udeg: i64 },
    /// FLCO 000100 — Talker Alias header (table 7.4): data format
    /// (Table 7.25: 0 = 7-bit ASCII, 1 = ISO 8-bit, 2 = UTF-8,
    /// 3 = UTF-16BE), total length in code units, and the header's
    /// 48 octet-aligned data bits (the 49th, its MSB, is reserved for
    /// the 8/16-bit formats).
    TalkerAliasHeader { format: u8, len: u8, data: [u8; 6] },
    /// FLCO 000101..000111 — Talker Alias blocks 1..3 (table 7.5),
    /// seven more octets of alias each.
    TalkerAliasBlock { n: u8, data: [u8; 7] },
    /// Any other FLCO/FID combination — the bytes are the caller's to
    /// interpret.
    Other,
}

/// Interpret nine RS- or checksum-verified LC octets per TS 102 361-2.
/// Only standard-feature messages (FID 0) are named; a manufacturer
/// FID is `Other` — the CRC may vouch for the bits, not the layout.
pub fn lc_info(lc: &[u8; 9]) -> LcInfo {
    let a = |i: usize| u32::from(lc[i]) << 16 | u32::from(lc[i + 1]) << 8 | u32::from(lc[i + 2]);
    if lc[1] != 0 {
        return LcInfo::Other;
    }
    match lc[0] & 0x3F {
        0b000000 => LcInfo::GroupVoice { tg: a(3), src: a(6) },
        0b000011 => LcInfo::UnitVoice { dst: a(3), src: a(6) },
        0b001000 => {
            // reserved(4) · err(3) · lon(25) · lat(24) across the 56
            // payload bits of octets 2..9.
            let bits: u64 = lc[2..9].iter().fold(0, |acc, &b| acc << 8 | u64::from(b));
            let err = (bits >> 49 & 0x7) as u8;
            let lon_raw = (bits >> 24 & 0x1FF_FFFF) as i64;
            let lat_raw = (bits & 0xFF_FFFF) as i64;
            let lon = if lon_raw >= 1 << 24 { lon_raw - (1 << 25) } else { lon_raw };
            let lat = if lat_raw >= 1 << 23 { lat_raw - (1 << 24) } else { lat_raw };
            LcInfo::GpsInfo {
                err,
                lon_udeg: lon * 360_000_000 >> 25,
                lat_udeg: lat * 180_000_000 >> 24,
            }
        }
        0b000100 => {
            let mut data = [0u8; 6];
            data.copy_from_slice(&lc[3..9]);
            LcInfo::TalkerAliasHeader { format: lc[2] >> 6, len: lc[2] >> 1 & 0x1F, data }
        }
        op @ 0b000101..=0b000111 => {
            let mut data = [0u8; 7];
            data.copy_from_slice(&lc[2..9]);
            LcInfo::TalkerAliasBlock { n: op as u8 - 0b000100, data }
        }
        _ => LcInfo::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Record, parse, records};

    fn head() -> MsgHead {
        MsgHead { seq: 3, epoch_us: 0x1122334455667788, hz: 446_500_000, nac: 9, flags: 0, site: 0 }
    }

    fn raw(buf: &[u8]) -> Raw {
        match parse(buf).unwrap().0 {
            Record::Raw(r) => r,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn voice_superframe_pins_its_layout_and_round_trips_frames() {
        let mut frames = [[0u8; 9]; 18];
        let mut bits = [0u8; 72];
        for (i, b) in bits.iter_mut().enumerate() {
            *b = ((i * 7 + 3) % 5 == 0) as u8;
        }
        frames[0] = DmrVoice::pack_frame(&bits);
        assert_eq!(DmrVoice::unpack_frame(&frames[0]), bits, "pack/unpack is a bijection");
        frames[17] = [0xFF; 9];
        let v = DmrVoice { head: head(), slot: 1, burst_valid: 0b11_1101, slots: 6, frames };
        let mut out = Vec::new();
        v.encode_into(&mut out);
        assert_eq!(out.len(), DmrVoice::BYTES);
        assert_eq!(&out[..8], &[b's', b'd', b'r', 0, 1, TYP_DMR_VOICE, 200, 0]);
        assert_eq!(u16le(&out, 24), 9, "the colour code rides nac");
        assert_eq!((out[32], out[33], out[34]), (1, 0b11_1101, 6));
        assert_eq!(&out[36..45], &frames[0]);
        assert_eq!(&out[189..198], &[0xFF; 9]);
        let back = DmrVoice::from_raw(&raw(&out)).expect("parses");
        assert_eq!(back, v);
        assert!(back.burst_present(0) && !back.burst_present(1) && back.burst_present(5));
    }

    #[test]
    fn lc_packet_fix_and_alias_round_trip() {
        let lc = DmrLc { head: head(), kind: LC_KIND_HEADER, slot: 0, lc: [0x00, 0x00, 0x20, 0x00, 0x00, 0x63, 0x31, 0x25, 0xE2] };
        let mut out = Vec::new();
        lc.encode_into(&mut out);
        assert_eq!(out.len(), DmrLc::BYTES);
        assert_eq!(DmrLc::from_raw(&raw(&out)).unwrap(), lc);
        assert_eq!(lc.flco(), 0);

        let pkt = DmrPacket {
            head: MsgHead { flags: PKT_FLAG_CRC_OK, ..head() },
            slot: 0,
            header: [0x82, 0x38, 0x00, 0x00, 0x63, 0x31, 0x25, 0xE2, 0x88, 0x00, 0x34, 0x5C],
            user_data: b"\x00\x01\x00\x00\x00\x27\x7E\x27\x7E$GPRMC,032228.000,A*67\r\n\0".to_vec(),
        };
        let mut out = Vec::new();
        pkt.encode_into(&mut out);
        assert_eq!(out.len() % 8, 0);
        assert_eq!(u16le(&out, 34), pkt.user_data.len() as u16);
        assert_eq!(DmrPacket::from_raw(&raw(&out)).unwrap(), pkt);
        // Oversized user data is cut and flagged, never a malformed record.
        let big = DmrPacket { user_data: vec![7u8; 70_000], ..pkt.clone() };
        let mut out = Vec::new();
        big.encode_into(&mut out);
        assert!(out.len() <= MAX_RECORD_BYTES);
        let back = raw(&out);
        assert_ne!(back.head.flags & PKT_FLAG_TRUNCATED, 0);
        assert_eq!(DmrPacket::from_raw(&back).unwrap().user_data.len(), DmrPacket::MAX_USER);

        let fix = DmrFix {
            head: MsgHead { flags: FIX_FLAG_VALID, ..head() },
            slot: 0,
            lat_udeg: 40_850_350,
            lon_udeg: -73_197_190,
            utc_ms: 12_148_000,
            speed_mknots: 0,
            course_cdeg: 18_015,
            date: (2026, 8, 27),
            src: 7_654_321,
            sentence: "$GPRMC,032228.000,A,4051.02098,N,07311.83141,W,0.00,180.15,270826,,,A*67".into(),
        };
        let mut out = Vec::new();
        fix.encode_into(&mut out);
        assert_eq!(out.len() % 8, 0);
        assert_eq!(i32le(&out, 40), -73_197_190);
        assert_eq!(DmrFix::from_raw(&raw(&out)).unwrap(), fix);

        let alias = DmrAlias { head: head(), slot: 1, format: 1, src: 7_654_321, alias: "KE2HLF".into() };
        let mut out = Vec::new();
        alias.encode_into(&mut out);
        assert_eq!(DmrAlias::from_raw(&raw(&out)).unwrap(), alias);

        // A stream of all of them iterates through the generic reader
        // and parse_dmr names each.
        let mut buf = Vec::new();
        lc.encode_into(&mut buf);
        pkt.encode_into(&mut buf);
        fix.encode_into(&mut buf);
        alias.encode_into(&mut buf);
        let kinds: Vec<&str> = records(&buf)
            .map(|r| match r {
                Record::Raw(r) => match parse_dmr(&r) {
                    Some(DmrRecord::Lc(_)) => "lc",
                    Some(DmrRecord::Packet(_)) => "packet",
                    Some(DmrRecord::Fix(_)) => "fix",
                    Some(DmrRecord::Alias(_)) => "alias",
                    Some(DmrRecord::Voice(_)) => "voice",
                    None => "?",
                },
                _ => "generic",
            })
            .collect();
        assert_eq!(kinds, ["lc", "packet", "fix", "alias"]);
        // The wrong typ never parses as another record.
        let mut wrong = raw(&out);
        wrong.typ = TYP_DMR_VOICE;
        assert!(DmrVoice::from_raw(&wrong).is_none());
    }
}
