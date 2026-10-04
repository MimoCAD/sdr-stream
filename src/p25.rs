//! P25 record body LAYOUTS: the Phase 1 LDU (Logical link Data Unit) (typ 2) and the Phase 2
//! VCH superframe (typ 5). This module is bytes only — what the
//! receiver writes and any reader parses without a vocoder. The
//! conversions between these layouts and the live decode (`fec::
//! DecodedLdu`, `phase2_vch::VchSuperframe`) live in the p25 codec
//! crate, which is the only place a Golay or an IMBE (Improved Multi-Band Excitation) decoder exists.
//!
//! ```text
//! Ldu (typ 2), 240 octets:
//!  32  2    slot_valid   bit i = voice slot i carried a Golay-valid frame
//!  34  9    errors       per-slot Golay correction count (0xFF = invalid)
//!  43  1    pad
//!  44  196  body         the corrected on-air LDU body, dibits packed
//!                        4 per octet MSB-first (784 dibits). Records
//!                        written before 2026-09-12 carry 194 octets
//!                        (776 dibits — the LSD (Low Speed Data) laid out as 8 dibits, so
//!                        voice slot 8 was misaligned) and two pad zeros
//!                        where the last 8 dibits now live: same record
//!                        size, slot 8 of those files is garbage either way.
//!
//! P1Voice (typ 16), 56 octets — one voice frame of an LDU, sent on the
//! live feed the moment its 72 dibits are in (2026-10-04), so a listener
//! waits 20 ms for a frame instead of 180 ms for its LDU. `head.seq` is
//! its LDU's (the Ldu record's) seq and `head.flags` carries that LDU's
//! LDU_FLAG_LDU2; `head.epoch_us` is the frame's own first symbol.
//!  32  1    slot         the voice slot in its LDU (0..8)
//!  33  1    errors       Golay+Hamming corrections (0xFF = invalid)
//!  34  2    pad
//!  36  18   dibits       the corrected on-air voice frame, 72 dibits
//!                        packed 4 per octet MSB-first (the Ldu body's
//!                        packing)
//!  54  2    pad
//!
//! P2Voice (typ 17), 184 octets — one Phase 2 voice burst's frames, sent
//! on the live feed the moment the burst is filed into its superframe
//! (2026-10-04): a listener waits for a burst, not for the 360 ms
//! superframe. `head.seq` is its superframe's (the P2Vch record's) seq,
//! `head.flags` that superframe's P2V_FLAG_DESCRAMBLED; `head.epoch_us`
//! is the burst's own air time.
//!  32  1    slot         the LCH (0/1)
//!  33  1    first        the first frame's place in the superframe (0..17)
//!  34  1    count        frames carried: 4 (a 4V) or 2 (the 2V)
//!  35  1    pad
//!  36  144  phase        `count` frames × 36 symbols, the measured phase
//!                        (the P2Vch record's encoding); unused frames zero
//! 180  4    pad
//!
//! P2Vch (typ 5), 696 octets:
//!  32  1    slot         the LCH (0/1)
//!  33  3    pad
//!  36  4    frame_valid  bit i = frame i assembled
//!  40  648  phase        18 frames × 36 symbols, the MEASURED symbol
//!                        phase (u8 half-step ring; dibit = top two bits)
//! 688  1    flush_cause
//! 689  2    gap_fragments
//! 691  5    pad
//! ```
use crate::{
    HEAD_BYTES, LDU_FLAG_LDU2, LDU_FLAG_RS_OK, MsgHead, Raw, TYP_LDU, TYP_P1_VOICE, TYP_P2_VCH, TYP_P2_VOICE, pad_to,
    push_head, u16le, u32le,
};
use alloc::vec::Vec;

/// One P25 Phase 1 LDU: the corrected on-air body plus the live decode
/// verdicts re-encoding would launder (slot validity, error counts, the
/// RS-ok flag in the head).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LduFrame {
    pub head: MsgHead,
    pub slot_valid: u16,
    pub errors: [u8; 9],
    pub body: [u8; 196],
}

impl LduFrame {
    /// An `errors` entry for a slot that did not decode (no count).
    pub const SLOT_INVALID: u8 = 0xFF;

    /// Wire size of every Ldu record.
    pub const BYTES: usize = 240;
    pub const BODY_OCTETS: usize = 196;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_LDU, Self::BYTES, &self.head);
        out.extend_from_slice(&self.slot_valid.to_le_bytes());
        out.extend_from_slice(&self.errors);
        out.push(0);
        out.extend_from_slice(&self.body);
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<LduFrame> {
        const B: usize = HEAD_BYTES;
        let (b, head) = (&r.bytes[..], r.head);
        if r.typ != TYP_LDU || b.len() < Self::BYTES {
            return None;
        }
        let mut errors = [0u8; 9];
        errors.copy_from_slice(&b[B + 2..B + 11]);
        let mut body = [0u8; 196];
        body.copy_from_slice(&b[B + 12..B + 12 + 196]);
        Some(LduFrame { head, slot_valid: u16le(b, B), errors, body })
    }

    pub fn ldu1(&self) -> bool {
        self.head.flags & LDU_FLAG_LDU2 == 0
    }

    pub fn extra_rs_ok(&self) -> bool {
        self.head.flags & LDU_FLAG_RS_OK != 0
    }
}

/// One voice frame of a P25 Phase 1 LDU, on its own (the live feed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P1VoiceFrame {
    pub head: MsgHead,
    pub slot: u8,
    pub errors: u8,
    pub dibits: [u8; 18],
}

impl P1VoiceFrame {
    /// An `errors` value for a frame that did not decode.
    pub const INVALID: u8 = 0xFF;

    /// Wire size of every P1Voice record.
    pub const BYTES: usize = 56;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_P1_VOICE, Self::BYTES, &self.head);
        out.push(self.slot);
        out.push(self.errors);
        out.extend_from_slice(&[0; 2]);
        out.extend_from_slice(&self.dibits);
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<P1VoiceFrame> {
        const B: usize = HEAD_BYTES;
        let (b, head) = (&r.bytes[..], r.head);
        if r.typ != TYP_P1_VOICE || b.len() < Self::BYTES || b[B] > 8 {
            return None;
        }
        let mut dibits = [0u8; 18];
        dibits.copy_from_slice(&b[B + 4..B + 22]);
        Some(P1VoiceFrame { head, slot: b[B], errors: b[B + 1], dibits })
    }

    pub fn ldu1(&self) -> bool {
        self.head.flags & LDU_FLAG_LDU2 == 0
    }

    /// The frame's place in the call: its LDU's seq × 9 + its slot.
    pub fn position(&self) -> u64 {
        self.head.seq as u64 * 9 + self.slot as u64
    }
}

/// One Phase 2 voice burst's frames, on their own (the live feed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P2VoiceBurst {
    pub head: MsgHead,
    pub slot: u8,
    pub first: u8,
    /// The frames, 4 or 2 of them, measured phase as in [`P2VchFrame`].
    pub phase: Vec<[u8; 36]>,
}

impl P2VoiceBurst {
    /// Wire size of every P2Voice record.
    pub const BYTES: usize = 184;
    /// The most frames one burst carries (a 4V).
    pub const MAX_FRAMES: usize = 4;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_P2_VOICE, Self::BYTES, &self.head);
        let n = self.phase.len().min(Self::MAX_FRAMES);
        out.extend_from_slice(&[self.slot, self.first, n as u8, 0]);
        for frame in &self.phase[..n] {
            out.extend_from_slice(frame);
        }
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<P2VoiceBurst> {
        const B: usize = HEAD_BYTES;
        let b = &r.bytes[..];
        if r.typ != TYP_P2_VOICE || b.len() < Self::BYTES {
            return None;
        }
        let (first, n) = (b[B + 1], b[B + 2] as usize);
        if n == 0 || n > Self::MAX_FRAMES || first as usize + n > 18 {
            return None;
        }
        let phase = (0..n)
            .map(|k| {
                let mut f = [0u8; 36];
                f.copy_from_slice(&b[B + 4 + 36 * k..B + 4 + 36 * k + 36]);
                f
            })
            .collect();
        Some(P2VoiceBurst { head: r.head, slot: b[B], first, phase })
    }

    /// The first frame's place in the call: its superframe's seq × 18 +
    /// its place in the superframe.
    pub fn position(&self) -> u64 {
        self.head.seq as u64 * 18 + self.first as u64
    }
}

/// One P25 Phase 2 VCH superframe on one LCH: the measured symbol phase
/// of 18 frames, the assembler's validity mask, and its flush verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P2VchFrame {
    pub head: MsgHead,
    pub slot: u8,
    pub frame_valid: u32,
    pub phase: [[u8; 36]; 18],
    pub flush_cause: u8,
    pub gap_fragments: u16,
}

impl P2VchFrame {
    /// `frame_valid` with all 18 voice frames of the superframe present.
    pub const ALL_FRAMES_VALID: u32 = (1 << 18) - 1;

    /// Wire size of every P2 VCH record.
    pub const BYTES: usize = 696;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_P2_VCH, Self::BYTES, &self.head);
        out.push(self.slot);
        out.extend_from_slice(&[0; 3]);
        out.extend_from_slice(&self.frame_valid.to_le_bytes());
        for frame in &self.phase {
            out.extend_from_slice(frame);
        }
        out.push(self.flush_cause);
        out.extend_from_slice(&self.gap_fragments.to_le_bytes());
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<P2VchFrame> {
        const B: usize = HEAD_BYTES;
        let (b, head) = (&r.bytes[..], r.head);
        if r.typ != TYP_P2_VCH || b.len() < Self::BYTES {
            return None;
        }
        let mut phase = [[0u8; 36]; 18];
        for (i, frame) in phase.iter_mut().enumerate() {
            frame.copy_from_slice(&b[B + 8 + 36 * i..B + 8 + 36 * i + 36]);
        }
        Some(P2VchFrame {
            head,
            slot: b[B],
            frame_valid: u32le(b, B + 4),
            phase,
            flush_cause: b[B + 656],
            gap_fragments: u16le(b, B + 657),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Record, parse};

    fn head() -> MsgHead {
        MsgHead { seq: 0, epoch_us: 0x1122334455667788, hz: 851_012_500, nac: 0x293, flags: 0, site: 0 }
    }

    fn raw(buf: &[u8]) -> Raw {
        match parse(buf).unwrap().0 {
            Record::Raw(r) => r,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn ldu_layout_pinned() {
        let mut body = [0u8; 196];
        for (i, b) in body.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let f = LduFrame {
            head: MsgHead { seq: 3, flags: LDU_FLAG_LDU2 | LDU_FLAG_RS_OK, ..head() },
            slot_valid: 0x1EF,
            errors: [0, 1, 2, 3, 0xFF, 5, 6, 7, 8],
            body,
        };
        let mut out = Vec::new();
        f.encode_into(&mut out);
        assert_eq!(out.len(), LduFrame::BYTES);
        assert_eq!(u16le(&out, 6), 240);
        assert_eq!(u16le(&out, 32), 0x1EF);
        assert_eq!(&out[34..43], &f.errors);
        assert_eq!(out[43], 0);
        assert_eq!(&out[44..240], &body[..]);
        let back = LduFrame::from_raw(&raw(&out)).unwrap();
        assert_eq!(back, f);
        assert!(!back.ldu1() && back.extra_rs_ok());
    }

    #[test]
    fn p1_voice_layout_pinned() {
        let mut dibits = [0u8; 18];
        for (i, b) in dibits.iter_mut().enumerate() {
            *b = 0xA0 + i as u8;
        }
        let f = P1VoiceFrame { head: MsgHead { seq: 41, flags: LDU_FLAG_LDU2, ..head() }, slot: 8, errors: 3, dibits };
        let mut out = Vec::new();
        f.encode_into(&mut out);
        assert_eq!(out.len(), P1VoiceFrame::BYTES);
        assert_eq!(out[5], TYP_P1_VOICE);
        assert_eq!(u16le(&out, 6), 56);
        assert_eq!((out[32], out[33]), (8, 3));
        assert_eq!(&out[34..36], &[0, 0]);
        assert_eq!(&out[36..54], &dibits[..]);
        assert_eq!(&out[54..56], &[0, 0]);
        let back = P1VoiceFrame::from_raw(&raw(&out)).unwrap();
        assert_eq!(back, f);
        assert!(!back.ldu1());
        assert_eq!(back.position(), 41 * 9 + 8);
        // A slot past the LDU's nine never parses.
        out[32] = 9;
        assert!(P1VoiceFrame::from_raw(&raw(&out)).is_none());
    }

    #[test]
    fn p2_voice_layout_pinned() {
        let phase = alloc::vec![[0x11u8; 36], [0x22; 36]];
        let f = P2VoiceBurst { head: MsgHead { seq: 9, flags: crate::P2V_FLAG_DESCRAMBLED, ..head() }, slot: 1, first: 16, phase };
        let mut out = Vec::new();
        f.encode_into(&mut out);
        assert_eq!(out.len(), P2VoiceBurst::BYTES);
        assert_eq!(out[5], TYP_P2_VOICE);
        assert_eq!(u16le(&out, 6), 184);
        assert_eq!(&out[32..36], &[1, 16, 2, 0]);
        assert_eq!(&out[36..72], &[0x11; 36]);
        assert_eq!(&out[72..108], &[0x22; 36]);
        assert!(out[108..].iter().all(|&b| b == 0));
        let back = P2VoiceBurst::from_raw(&raw(&out)).unwrap();
        assert_eq!(back, f);
        assert_eq!(back.position(), 9 * 18 + 16);
        // A burst running past the superframe's 18 frames never parses.
        out[33] = 17;
        assert!(P2VoiceBurst::from_raw(&raw(&out)).is_none());
    }

    #[test]
    fn p2_vch_layout_pinned() {
        let mut phase = [[0u8; 36]; 18];
        phase[0][..4].copy_from_slice(&[160, 224, 96, 32]);
        phase[17][35] = 0x7F;
        let f = P2VchFrame {
            head: MsgHead { seq: 5, flags: crate::P2V_FLAG_DESCRAMBLED, ..head() },
            slot: 1,
            frame_valid: (1 << 18) - 1 & !(1 << 6),
            phase,
            flush_cause: 2,
            gap_fragments: 7,
        };
        let mut out = Vec::new();
        f.encode_into(&mut out);
        assert_eq!(out.len(), P2VchFrame::BYTES);
        assert_eq!(u16le(&out, 6), 696);
        assert_eq!(out[32], 1);
        assert_eq!(&out[33..36], &[0, 0, 0]);
        assert_eq!(u32le(&out, 36), f.frame_valid);
        assert_eq!(&out[40..44], &[160, 224, 96, 32]);
        assert_eq!(out[687], 0x7F);
        assert_eq!(out[688], 2);
        assert_eq!(u16le(&out, 689), 7);
        assert_eq!(P2VchFrame::from_raw(&raw(&out)).unwrap(), f);
        // The wrong typ never parses as this record.
        let mut wrong = raw(&out);
        wrong.typ = TYP_LDU;
        assert!(P2VchFrame::from_raw(&wrong).is_none());
        assert!(LduFrame::from_raw(&wrong).is_some(), "an LDU-typed 696-byte record parses as an LDU (len is authoritative)");
    }
}
