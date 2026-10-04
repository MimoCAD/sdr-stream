//! P25 control channel record body LAYOUTS: the live feed of a
//! control channel's signalling, as the octets a P25 radio decodes off
//! the air. This module is bytes only — no TSBK (Trunking Signaling
//! Block) is interpreted here; the p25 codec crate parses the octets
//! (TIA-102.AABC-E opcodes, TIA-102.BAAA-B data blocks) on whichever
//! side holds them.
//!
//! A feed DATAGRAM is records only, in this order:
//!
//! ```text
//! [Site] [Tsbk | Mbt]* [Mac]
//! ```
//!
//! - [`SiteRecord`] names the P25 site exactly as the site names itself
//!   on the air — WACN and SYSID from the Network Status Broadcast, RFSS
//!   ID and Site ID from the RFSS Status Broadcast — the four values a
//!   subscriber selects on. A control channel whose identity has not
//!   been decoded yet sends nothing.
//! - [`TsbkRecord`] / [`MbtRecord`] carry the decoded blocks.
//! - [`MacRecord`] closes the datagram: HMAC-SHA256 (Hash-based Message
//!   Authentication Code over SHA-256) under the receiver's site key
//!   over every octet before it. The server checks it, then forwards
//!   the datagram WITHOUT it; this crate has no dependencies, so the
//!   HMAC itself is the caller's ([`split_signed`] finds the parts).
//!
//! ```text
//! Site (typ 11), 48 octets:
//!  32  4  wacn    20-bit Wide Area Communications Network ID
//!  36  2  sysid   12-bit System ID
//!  38  1  rfss    RFSS (RF Subsystem) ID
//!  39  1  site    Site ID within the RFSS
//!  40  4  source  which of the receiver's sources heard it
//!  44  4  pad
//!  head: hz = the control channel, seq = this source's datagram
//!  ordinal (the server's replay guard), epoch_us = sent, site = the
//!  receiver's MimoCAD SITE_ID.
//!
//! Tsbk (typ 12), 48 octets:
//!  32  12  tsbk   the 12 octets out of the ½-rate trellis, in
//!                 transmission order (LB/P/opcode first, CRC-16 last);
//!                 only CRC-valid blocks are sent
//!  44  1   fec    trellis corrections
//!  45  3   pad
//!
//! Mbt (typ 13), 48 + n octets, padded to 8:
//!  32  1   blocks        data blocks after the header
//!  33  1   block_octets  12 (½ rate: unconfirmed data, AMBT) or 18 (¾
//!                        rate: confirmed — serial, CRC-9, 16 data)
//!  34  1   header_fec    the header block's trellis corrections
//!  35  1   pad
//!  36  12  header        the header block (CRC-16 last)
//!  48  n   data          blocks × block_octets, as decoded
//!
//! Mac (typ 14), 64 octets:
//!  32  32  tag    HMAC-SHA256(site key, the datagram before this record)
//! ```
//!
//! Tsbk and Mbt set [`CC_FLAG_INBOUND`] when the blocks were heard on
//! an uplink (an ISP (Inbound Signaling Packet) or an inbound data
//! packet); their head's `hz` is the frequency they were heard on and
//! `seq` the block ordinal on that frequency, so a gap shows loss.
use crate::{
    CC_FLAG_INBOUND, HEAD_BYTES, MsgHead, RECORD_ALIGN, Raw, TYP_CC_MAC, TYP_CC_MBT, TYP_CC_SITE, TYP_CC_TSBK, pad8,
    pad_to, parse_head, push_head, u16le, u32le,
};
use alloc::vec::Vec;

/// The site a feed datagram speaks for, and which receiver heard it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiteRecord {
    pub head: MsgHead,
    pub wacn: u32,
    pub sysid: u16,
    pub rfss: u8,
    pub site: u8,
    pub source: u32,
}

impl SiteRecord {
    pub const BYTES: usize = 48;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_CC_SITE, Self::BYTES, &self.head);
        out.extend_from_slice(&self.wacn.to_le_bytes());
        out.extend_from_slice(&self.sysid.to_le_bytes());
        out.push(self.rfss);
        out.push(self.site);
        out.extend_from_slice(&self.source.to_le_bytes());
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<SiteRecord> {
        const B: usize = HEAD_BYTES;
        let b = &r.bytes[..];
        if r.typ != TYP_CC_SITE || b.len() < Self::BYTES {
            return None;
        }
        Some(SiteRecord {
            head: r.head,
            wacn: u32le(b, B),
            sysid: u16le(b, B + 4),
            rfss: b[B + 6],
            site: b[B + 7],
            source: u32le(b, B + 8),
        })
    }
}

/// One CRC-valid TSBK as decoded off the air.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TsbkRecord {
    pub head: MsgHead,
    pub tsbk: [u8; 12],
    pub fec: u8,
}

impl TsbkRecord {
    pub const BYTES: usize = 48;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let base = out.len();
        push_head(out, TYP_CC_TSBK, Self::BYTES, &self.head);
        out.extend_from_slice(&self.tsbk);
        out.push(self.fec);
        pad_to(out, base + Self::BYTES);
    }

    pub fn from_raw(r: &Raw) -> Option<TsbkRecord> {
        const B: usize = HEAD_BYTES;
        let b = &r.bytes[..];
        if r.typ != TYP_CC_TSBK || b.len() < Self::BYTES {
            return None;
        }
        let mut tsbk = [0u8; 12];
        tsbk.copy_from_slice(&b[B..B + 12]);
        Some(TsbkRecord { head: r.head, tsbk, fec: b[B + 12] })
    }

    /// Heard on an uplink: an ISP, not an OSP.
    pub fn inbound(&self) -> bool {
        self.head.flags & CC_FLAG_INBOUND != 0
    }
}

/// One multi-block packet — an AMBT (Alternate Multiple Block
/// Trunking) message or a data PDU (Protocol Data Unit) — as decoded:
/// the header block and every data block, each as its trellis gave it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MbtRecord {
    pub head: MsgHead,
    pub header_fec: u8,
    pub header: [u8; 12],
    /// 12 for ½-rate blocks, 18 for ¾-rate (confirmed) blocks.
    pub block_octets: u8,
    /// `blocks × block_octets` octets.
    pub data: Vec<u8>,
}

impl MbtRecord {
    /// A ½-rate block: unconfirmed data, AMBT.
    pub const HALF_RATE_OCTETS: u8 = 12;
    /// A ¾-rate block: confirmed data (serial, CRC-9, 16 data octets).
    pub const THREE_QUARTER_RATE_OCTETS: u8 = 18;
    /// The fixed part; the data blocks follow.
    pub const FIXED_BYTES: usize = 48;

    pub fn blocks(&self) -> usize {
        if self.block_octets == 0 { 0 } else { self.data.len() / self.block_octets as usize }
    }

    pub fn total_bytes(&self) -> usize {
        pad8(Self::FIXED_BYTES + self.data.len())
    }

    /// The `i`th data block's octets.
    pub fn block(&self, i: usize) -> Option<&[u8]> {
        let n = self.block_octets as usize;
        self.data.get(i * n..(i + 1) * n)
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let total = self.total_bytes();
        let base = out.len();
        push_head(out, TYP_CC_MBT, total, &self.head);
        out.push(self.blocks() as u8);
        out.push(self.block_octets);
        out.push(self.header_fec);
        out.push(0);
        out.extend_from_slice(&self.header);
        out.extend_from_slice(&self.data);
        pad_to(out, base + total);
    }

    pub fn from_raw(r: &Raw) -> Option<MbtRecord> {
        const B: usize = HEAD_BYTES;
        let b = &r.bytes[..];
        if r.typ != TYP_CC_MBT || b.len() < Self::FIXED_BYTES {
            return None;
        }
        let (blocks, block_octets) = (b[B] as usize, b[B + 1]);
        let n = blocks * block_octets as usize;
        let data = b.get(Self::FIXED_BYTES..Self::FIXED_BYTES + n)?.to_vec();
        let mut header = [0u8; 12];
        header.copy_from_slice(&b[B + 4..B + 16]);
        Some(MbtRecord { head: r.head, header_fec: b[B + 2], header, block_octets, data })
    }

    /// Heard on an uplink.
    pub fn inbound(&self) -> bool {
        self.head.flags & CC_FLAG_INBOUND != 0
    }
}

/// The datagram's authentication tag; always its last record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MacRecord {
    pub head: MsgHead,
    pub tag: [u8; 32],
}

impl MacRecord {
    pub const BYTES: usize = 64;

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        push_head(out, TYP_CC_MAC, Self::BYTES, &self.head);
        out.extend_from_slice(&self.tag);
    }

    pub fn from_raw(r: &Raw) -> Option<MacRecord> {
        let b = &r.bytes[..];
        if r.typ != TYP_CC_MAC || b.len() < Self::BYTES {
            return None;
        }
        let mut tag = [0u8; 32];
        tag.copy_from_slice(&b[HEAD_BYTES..HEAD_BYTES + 32]);
        Some(MacRecord { head: r.head, tag })
    }
}

/// A feed datagram cut into what was signed and the tag over it: the
/// last [`MacRecord::BYTES`] octets must be a Mac record, and what
/// precedes it must open with a Site record. `None` for anything else
/// — the server drops it unread.
pub fn split_signed(datagram: &[u8]) -> Option<(&[u8], [u8; 32])> {
    let cut = datagram.len().checked_sub(MacRecord::BYTES)?;
    if cut < SiteRecord::BYTES || cut % RECORD_ALIGN != 0 {
        return None;
    }
    let (signed, mac) = datagram.split_at(cut);
    let (typ, _, len) = parse_head(mac)?;
    if typ != TYP_CC_MAC || len != MacRecord::BYTES {
        return None;
    }
    let (typ, _, len) = parse_head(signed)?;
    if typ != TYP_CC_SITE || len != SiteRecord::BYTES {
        return None;
    }
    let mut tag = [0u8; 32];
    tag.copy_from_slice(&mac[HEAD_BYTES..]);
    Some((signed, tag))
}

/// The Site record a feed datagram (signed or forwarded) opens with.
pub fn site_of(datagram: &[u8]) -> Option<SiteRecord> {
    match crate::parse(datagram)?.0 {
        crate::Record::Raw(r) => SiteRecord::from_raw(&r),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Record, parse, records, u64le};

    fn head() -> MsgHead {
        MsgHead { seq: 7, epoch_us: 0x1122334455667788, hz: 851_012_500, nac: 0x123, flags: 0, site: 99 }
    }

    fn raw(buf: &[u8]) -> Raw {
        match parse(buf).unwrap().0 {
            Record::Raw(r) => r,
            other => panic!("{other:?}"),
        }
    }

    fn site() -> SiteRecord {
        SiteRecord { head: head(), wacn: 0xABCDE, sysid: 0x123, rfss: 2, site: 5, source: 3 }
    }

    #[test]
    fn site_layout_pinned() {
        let s = site();
        let mut out = Vec::new();
        s.encode_into(&mut out);
        assert_eq!(out.len(), SiteRecord::BYTES);
        assert_eq!(out[5], TYP_CC_SITE);
        assert_eq!(u16le(&out, 6), 48);
        assert_eq!(u64le(&out, 8), head().epoch_us);
        assert_eq!(u32le(&out, 16), 851_012_500);
        assert_eq!(u32le(&out, 28), 99);
        assert_eq!(u32le(&out, 32), 0xABCDE);
        assert_eq!(u16le(&out, 36), 0x123);
        assert_eq!((out[38], out[39]), (2, 5));
        assert_eq!(u32le(&out, 40), 3);
        assert_eq!(&out[44..48], &[0; 4]);
        assert_eq!(SiteRecord::from_raw(&raw(&out)).unwrap(), s);
    }

    #[test]
    fn tsbk_layout_pinned() {
        let t = TsbkRecord {
            head: MsgHead { flags: CC_FLAG_INBOUND, ..head() },
            tsbk: [0x80, 0x00, 0x00, 1, 2, 3, 4, 5, 6, 7, 0xAA, 0x55],
            fec: 4,
        };
        let mut out = Vec::new();
        t.encode_into(&mut out);
        assert_eq!(out.len(), TsbkRecord::BYTES);
        assert_eq!(out[5], TYP_CC_TSBK);
        assert_eq!(&out[32..44], &t.tsbk);
        assert_eq!(out[44], 4);
        assert_eq!(&out[45..48], &[0; 3]);
        let back = TsbkRecord::from_raw(&raw(&out)).unwrap();
        assert_eq!(back, t);
        assert!(back.inbound());
    }

    #[test]
    fn mbt_layout_pinned() {
        // Two ¾-rate blocks: 48 + 36 = 84 → padded to 88.
        let data: Vec<u8> = (0..36).collect();
        let m = MbtRecord {
            head: head(),
            header_fec: 1,
            header: [0x15, 0x02, 0, 0x11, 0x22, 0x33, 0x82, 0x06, 0x18, 0, 0xAB, 0xCD],
            block_octets: MbtRecord::THREE_QUARTER_RATE_OCTETS,
            data: data.clone(),
        };
        let mut out = Vec::new();
        m.encode_into(&mut out);
        assert_eq!(out.len(), 88);
        assert_eq!(u16le(&out, 6), 88);
        assert_eq!((out[32], out[33], out[34], out[35]), (2, 18, 1, 0));
        assert_eq!(&out[36..48], &m.header);
        assert_eq!(&out[48..84], &data[..]);
        assert_eq!(&out[84..88], &[0; 4]);
        let back = MbtRecord::from_raw(&raw(&out)).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.blocks(), 2);
        assert_eq!(back.block(1).unwrap(), &data[18..36]);
        assert!(back.block(2).is_none() && !back.inbound());
        // A header-only packet (the AMBT/PDU header with no blocks yet).
        let bare = MbtRecord { data: Vec::new(), block_octets: MbtRecord::HALF_RATE_OCTETS, ..m };
        let mut out = Vec::new();
        bare.encode_into(&mut out);
        assert_eq!(out.len(), MbtRecord::FIXED_BYTES);
        assert_eq!(MbtRecord::from_raw(&raw(&out)).unwrap(), bare);
    }

    /// A datagram end to end: Site, two records, Mac — the records walk
    /// in order, the split hands back exactly the signed octets, and the
    /// forwarded form (Mac stripped) still opens with its Site.
    #[test]
    fn datagram_splits_and_walks() {
        let mut d = Vec::new();
        site().encode_into(&mut d);
        TsbkRecord { head: head(), tsbk: [0x3B; 12], fec: 0 }.encode_into(&mut d);
        MbtRecord { head: head(), header_fec: 0, header: [1; 12], block_octets: 12, data: alloc::vec![2; 12] }
            .encode_into(&mut d);
        let signed_len = d.len();
        MacRecord { head: head(), tag: [0xEE; 32] }.encode_into(&mut d);
        assert_eq!(d.len(), signed_len + MacRecord::BYTES);

        let (signed, tag) = split_signed(&d).unwrap();
        assert_eq!(signed, &d[..signed_len]);
        assert_eq!(tag, [0xEE; 32]);
        let typs: Vec<u8> = records(&d).map(|r| r.typ()).collect();
        assert_eq!(typs, [TYP_CC_SITE, TYP_CC_TSBK, TYP_CC_MBT, TYP_CC_MAC]);
        assert_eq!(site_of(signed).unwrap(), site());

        // Anything else is refused: no Mac, no Site first, a cut.
        assert!(split_signed(signed).is_none());
        assert!(split_signed(&d[SiteRecord::BYTES..]).is_none());
        assert!(split_signed(&d[..d.len() - 8]).is_none());
        assert!(split_signed(&[]).is_none());
    }
}
