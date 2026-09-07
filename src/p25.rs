//! P25 record body LAYOUTS: the Phase 1 LDU (typ 2) and the Phase 2
//! VCH superframe (typ 5). This module is bytes only — what the
//! receiver writes and any reader parses without a vocoder. The
//! conversions between these layouts and the live decode (`fec::
//! DecodedLdu`, `phase2_vch::VchSuperframe`) live in the p25 codec
//! crate, which is the only place a Golay or an IMBE decoder exists.
//!
//! ```text
//! Ldu (typ 2), 240 octets:
//!  32  2    slot_valid   bit i = voice slot i carried a Golay-valid frame
//!  34  9    errors       per-slot Golay correction count (0xFF = invalid)
//!  43  1    pad
//!  44  194  body         the corrected on-air LDU body, dibits packed
//!                        4 per octet MSB-first (776 dibits)
//! 238  2    pad
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
    HEAD_BYTES, LDU_FLAG_LDU2, LDU_FLAG_RS_OK, MsgHead, Raw, TYP_LDU, TYP_P2_VCH, pad_to, push_head, u16le, u32le,
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
    pub body: [u8; 194],
}

impl LduFrame {
    /// Wire size of every Ldu record.
    pub const BYTES: usize = 240;
    pub const BODY_OCTETS: usize = 194;

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
        let mut body = [0u8; 194];
        body.copy_from_slice(&b[B + 12..B + 12 + 194]);
        Some(LduFrame { head, slot_valid: u16le(b, B), errors, body })
    }

    pub fn ldu1(&self) -> bool {
        self.head.flags & LDU_FLAG_LDU2 == 0
    }

    pub fn extra_rs_ok(&self) -> bool {
        self.head.flags & LDU_FLAG_RS_OK != 0
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
        let mut body = [0u8; 194];
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
        assert_eq!(&out[44..238], &body[..]);
        assert_eq!(&out[238..240], &[0, 0]);
        let back = LduFrame::from_raw(&raw(&out)).unwrap();
        assert_eq!(back, f);
        assert!(!back.ldu1() && back.extra_rs_ok());
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
