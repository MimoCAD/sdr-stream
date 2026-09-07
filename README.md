# sdr-stream

DigitalStream — the record container behind MimoSDR's `.sdr` call
files, its WebSocket messages and its future WebRTC datagrams: one
framing for all three. `no_std` + `alloc`, zero dependencies,
permanently: a wire format needs nothing but bytes.

- Framing (format v1): a 32-byte head — magic `sdr\0`, version, type,
  length in bytes, epoch µs, Hz, sequence, NAC, flags, site — then the
  body; readers advance by length, ignore tails, keep unknown types raw.
- The mode-neutral records: `CallHeader`, `Ess` (encryption sync),
  `CallTrailer`.
- The P25 layouts (`p25`: LDU, Phase 2 VCH) and the DMR layouts (`dmr`:
  voice, Full LC + `lc_info`, packet, fix, alias).
- The sidecar document (`sidecar`): a projection of a call's records,
  or the `MSDR` chunk of an analog WAV (`riff`).
- The filename grammar (`name`), so a server names what it stores from
  the record, never from a client's filename.
- Dependency-free JSON (`json`).

The full specification is the crate documentation in `src/lib.rs`.

This repository is a tree export of `crates/stream` in the private
`mimosdr` repository, published as one commit per release; the same
bytes also ship inside github.com/MimoCAD/sdr-decode. Consumers pin a
git `rev`:

```toml
stream = { package = "sdr-stream", git = "https://github.com/MimoCAD/sdr-stream.git", rev = "…" }
```

License: MIT OR Apache-2.0.
