# pcm

[![Cargo Test](https://github.com/quietscroll/pcm/actions/workflows/cargo-test.yml/badge.svg)](https://github.com/quietscroll/pcm/actions/workflows/cargo-test.yml)

L16 mono PCM newtype with duration, speed-change, and optional base64 serde.

## Overview

`PCM` wraps a `Vec<u8>` of raw 16-bit little-endian mono samples at 24 kHz.
The type is intentionally lightweight: no resampling, no channel mixing — just
the bytes and the operations that are safe to run without additional context.

```rust
use pcm::{PCM, PCM_SAMPLE_RATE_HZ};

// One second of silence.
let one_second = PCM::from(vec![0u8; PCM_SAMPLE_RATE_HZ as usize * 2]);
assert_eq!(one_second.duration().unwrap(), time::Duration::seconds(1));

// 40 % faster — output is ≈ 0.71 s (linear interpolation, no aliasing).
let faster = one_second.speed_up(1.4);
assert!(faster.duration().unwrap() < time::Duration::seconds(1));
```

## Features

| feature | what it adds |
|---------|-------------|
| `serde` | `PCM` serialises as a base64-encoded string; enables `pcm::b64` and `pcm::b64_option` helper modules for explicit `#[serde(with = "...")]` use |

```toml
[dependencies]
pcm = { version = "0.1", features = ["serde"] }
```

With `serde` enabled, `PCM` round-trips through JSON as a base64 string:

```rust
use pcm::PCM;

let pcm = PCM::from(vec![0u8, 128, 0, 127]);
let json = serde_json::to_string(&pcm).unwrap();  // "\"AIA/\""  (base64)
let back: PCM = serde_json::from_str(&json).unwrap();
assert_eq!(pcm, back);
```

## API highlights

| item | description |
|------|-------------|
| `PCM_SAMPLE_RATE_HZ` | 24 000 — the assumed sample rate for all duration calculations |
| `PCM::duration()` | playback duration at 24 kHz; returns `Err` for odd-length buffers |
| `PCM::speed_up(f)` | linear-interpolation decimation (1.4 = 40 % faster) |
| `PCM::slow_down(f)` | linear-interpolation upsampling (1.2 = 20 % slower) |
| `PCM::to_b64()` | base64-encode the raw bytes (`serde` feature) |
| `PCM::from_b64(s)` | decode a base64 string into a `PCM` buffer (`serde` feature) |
| `PCM::i16_samples` | convert the raw byte buffer into a vector of i16 samples |

## License

MIT
