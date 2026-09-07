//! The RIFF/WAVE chunk walk a conventional call's metadata rides on:
//! the recorder embeds the sidecar JSON as an `MSDR` chunk after `data`
//! (players skip chunk ids they do not know, so the file plays
//! everywhere), and any reader — server, browser, phone — pulls it out
//! with [`chunk`] and [`crate::sidecar::Sidecar::parse`]. No audio
//! decoding, no dependency: bytes in, a slice out.

/// The chunk id the sidecar rides in.
pub const MSDR: [u8; 4] = *b"MSDR";

/// Find chunk `id` in a RIFF/WAVE byte buffer and return its payload.
/// Walks chunks by their declared sizes (the same rule that keeps
/// players from rendering the metadata as audio); `None` when the
/// buffer isn't RIFF/WAVE or has no such chunk.
pub fn chunk<'a>(bytes: &'a [u8], id: &[u8; 4]) -> Option<&'a [u8]> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut i = 12usize;
    while i + 8 <= bytes.len() {
        let n = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        if &bytes[i..i + 4] == id {
            return bytes.get(i + 8..i + 8 + n);
        }
        i += 8 + n + (n % 2);
    }
    None
}

/// The sidecar JSON text inside a conventional WAV, if any.
pub fn sidecar_text(bytes: &[u8]) -> Option<&str> {
    core::str::from_utf8(chunk(bytes, &MSDR)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn walks_by_declared_size_and_pads_odd_chunks() {
        let mut f = Vec::new();
        f.extend_from_slice(b"RIFF");
        f.extend_from_slice(&0u32.to_le_bytes());
        f.extend_from_slice(b"WAVE");
        f.extend_from_slice(b"fmt ");
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(&[1, 2, 3, 0]); // odd chunk, one pad
        f.extend_from_slice(b"data");
        f.extend_from_slice(&2u32.to_le_bytes());
        f.extend_from_slice(&[9, 9]);
        f.extend_from_slice(b"MSDR");
        f.extend_from_slice(&5u32.to_le_bytes());
        f.extend_from_slice(b"{\"a\":1}"[..5].as_ref());
        f.push(0);
        assert_eq!(chunk(&f, b"fmt "), Some(&[1u8, 2, 3][..]));
        assert_eq!(chunk(&f, b"data"), Some(&[9u8, 9][..]));
        assert_eq!(sidecar_text(&f), Some("{\"a\":"));
        assert_eq!(chunk(&f, b"LIST"), None);
        assert_eq!(chunk(b"RIFX", b"data"), None);
    }
}
