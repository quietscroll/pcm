//! L16 mono PCM newtype with duration, speed-change, and optional base64 serde.
//!
//! # Overview
//!
//! [`PCM`] wraps a `Vec<u8>` of raw 16-bit little-endian mono samples at
//! [`PCM_SAMPLE_RATE_HZ`] (24 kHz). The type is intentionally lightweight: no
//! resampling, no channel mixing, just the bytes and the operations that are
//! safe to run on them without additional context.
//!
//! ```
//! use pcm::{PCM, PCM_SAMPLE_RATE_HZ};
//!
//! // Build one second of silence.
//! let one_second = PCM::from(vec![0u8; PCM_SAMPLE_RATE_HZ as usize * 2]);
//! assert_eq!(one_second.duration().unwrap(), time::Duration::seconds(1));
//!
//! // 40 % faster — output is ≈ 0.71 s.
//! let faster = one_second.speed_up(1.4);
//! assert!(faster.duration().unwrap() < time::Duration::seconds(1));
//! ```
//!
//! # Features
//!
//! | feature | effect |
//! |---------|--------|
//! | `serde` | [`PCM`] serialises as a base64 string; enables [`b64`] and [`b64_option`] helper modules |

#![deny(missing_docs, unreachable_pub)]

use std::ops::Deref;

use time::Duration;

/// Sample rate assumed by all [`PCM`] operations, in Hz (24 000).
pub const PCM_SAMPLE_RATE_HZ: u16 = 24_000;

/// Errors that can arise from PCM operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The byte buffer has an odd length. L16 mono uses 2 bytes per sample,
    /// so an odd byte count cannot represent valid PCM data.
    #[error("PCM byte length must be even (L16 mono: 2 bytes per sample)")]
    ByteLengthNotEven,
}

/// Raw L16 mono PCM audio data (little-endian i16 samples at `PCM_SAMPLE_RATE_HZ`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PCM(Vec<u8>);

impl PCM {
    /// Wrap raw PCM bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Consume the wrapper and return the inner byte vector.
    pub fn into_inner(self) -> Vec<u8> {
        self.0
    }

    /// Number of bytes in this PCM buffer.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Compute the playback duration of this L16 mono buffer at
    /// [`PCM_SAMPLE_RATE_HZ`].
    ///
    /// Returns [`Error::ByteLengthNotEven`] when the buffer length is odd —
    /// L16 mono is always two bytes per sample, so an odd byte count is malformed.
    pub fn duration(&self) -> Result<Duration, Error> {
        if !self.len().is_multiple_of(2) {
            return Err(Error::ByteLengthNotEven);
        }
        let samples = self.len() as u64 / 2;
        let secs = samples as f64 / PCM_SAMPLE_RATE_HZ as f64;
        Ok(Duration::seconds_f64(secs))
    }

    /// Speed up this L16 mono PCM buffer by `speed` (e.g. 1.4 = 40 % faster).
    ///
    /// Uses linear interpolation so adjacent samples are blended rather than
    /// skipped, avoiding the harshness of nearest-neighbour decimation.
    /// Output length ≈ `self.len() / speed` bytes (always even).
    ///
    /// # Panics
    ///
    /// Panics if `speed` is not positive.
    pub fn speed_up(&self, speed: f32) -> PCM {
        assert!(speed > 0.0, "speed must be positive");
        if (speed - 1.0).abs() < f32::EPSILON {
            return self.clone();
        }

        let samples = self.i16_samples();
        let n = samples.len();
        if n == 0 {
            return PCM::default();
        }

        let out_len = ((n as f32) / speed).ceil() as usize;
        let mut out = Vec::with_capacity(out_len * 2);

        for i in 0..out_len {
            let pos = i as f32 * speed;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f32;
            let sample = samples[lo] as f32 * (1.0 - frac) + samples[hi] as f32 * frac;
            out.extend_from_slice(&(sample.round() as i16).to_le_bytes());
        }

        PCM::from(out)
    }

    /// Slow down this L16 mono PCM buffer by `factor` (e.g. 1.2 = 20 % slower).
    ///
    /// Inserts interpolated samples between input positions to stretch the buffer.
    /// Output length ≈ `self.len() * factor` bytes (always even).
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not positive.
    pub fn slow_down(&self, factor: f32) -> PCM {
        assert!(factor > 0.0, "factor must be positive");
        if (factor - 1.0).abs() < f32::EPSILON {
            return self.clone();
        }

        let samples = self.i16_samples();
        let n = samples.len();
        if n == 0 {
            return PCM::default();
        }

        let out_len = ((n as f32) * factor).ceil() as usize;
        let mut out = Vec::with_capacity(out_len * 2);

        for i in 0..out_len {
            let pos = i as f32 / factor;
            let lo = (pos.floor() as usize).min(n - 1);
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f32;
            let sample = samples[lo] as f32 * (1.0 - frac) + samples[hi] as f32 * frac;
            out.extend_from_slice(&(sample.round() as i16).to_le_bytes());
        }

        PCM::from(out)
    }

    fn i16_samples(&self) -> Vec<i16> {
        self.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

#[cfg(feature = "serde")]
impl PCM {
    /// Encode this PCM buffer as a base64 string (STANDARD alphabet).
    pub fn to_b64(&self) -> String {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        STANDARD.encode(&self.0)
    }

    /// Decode a base64 string (STANDARD alphabet) into a PCM buffer.
    ///
    /// Returns [`base64::DecodeError`] when the input is not valid base64.
    pub fn from_b64(s: &str) -> Result<Self, base64::DecodeError> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        Ok(Self(STANDARD.decode(s)?))
    }
}

/// Serde helpers for serialising [`PCM`] as a base64 string.
///
/// Use `#[serde(with = "pcm::b64")]` on a `PCM` field.
#[cfg(feature = "serde")]
pub mod b64 {
    use super::PCM;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    /// Serialize `PCM` as a base64 string.
    pub fn serialize<S: Serializer>(pcm: &PCM, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&pcm.to_b64())
    }

    /// Deserialize `PCM` from a base64 string.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<PCM, D::Error> {
        let raw = String::deserialize(d)?;
        PCM::from_b64(&raw).map_err(D::Error::custom)
    }
}

/// Serde helpers for serialising `Option<`[`PCM`]`>` as a nullable base64 string.
///
/// Use `#[serde(with = "pcm::b64_option")]` on an `Option<PCM>` field.
#[cfg(feature = "serde")]
pub mod b64_option {
    use super::PCM;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    /// Serialize `Option<PCM>` as a base64 string or `null`.
    pub fn serialize<S: Serializer>(opt: &Option<PCM>, s: S) -> Result<S::Ok, S::Error> {
        match opt {
            Some(pcm) => s.serialize_str(&pcm.to_b64()),
            None => s.serialize_none(),
        }
    }

    /// Deserialize `Option<PCM>` from a base64 string or `null`.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<PCM>, D::Error> {
        Option::<String>::deserialize(d)?
            .map(|raw| PCM::from_b64(&raw).map_err(D::Error::custom))
            .transpose()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for PCM {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_b64())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PCM {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let raw = <String as serde::Deserialize>::deserialize(d)?;
        PCM::from_b64(&raw).map_err(D::Error::custom)
    }
}

impl Deref for PCM {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for PCM {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<Vec<u8>> for PCM {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl From<PCM> for Vec<u8> {
    fn from(pcm: PCM) -> Self {
        pcm.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_pcm(hz: f32, seconds: f32) -> PCM {
        use std::f32::consts::PI;
        let n = (PCM_SAMPLE_RATE_HZ as f32 * seconds) as usize;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / PCM_SAMPLE_RATE_HZ as f32;
            let s = (2.0 * PI * hz * t).sin();
            let sample = (s * i16::MAX as f32) as i16;
            out.extend_from_slice(&sample.to_le_bytes());
        }
        PCM::from(out)
    }

    #[test]
    fn empty_buffer_is_zero_duration() {
        assert_eq!(PCM::default().duration().unwrap(), Duration::ZERO);
    }

    #[test]
    fn odd_length_is_rejected() {
        assert!(matches!(
            PCM::from(vec![0, 0, 0]).duration(),
            Err(Error::ByteLengthNotEven)
        ));
    }

    #[test]
    fn duration_matches_sample_count_at_configured_rate() {
        let one_second = PCM::from(vec![0u8; PCM_SAMPLE_RATE_HZ as usize * 2]);
        assert_eq!(one_second.duration().unwrap(), Duration::seconds(1));
    }

    #[test]
    fn identity_at_speed_one() {
        let pcm = sine_pcm(440.0, 0.1);
        assert_eq!(pcm.speed_up(1.0), pcm);
    }

    #[test]
    fn empty_buffer_speed_up_returns_empty() {
        assert_eq!(PCM::default().speed_up(2.0), PCM::default());
    }

    #[test]
    fn double_speed_halves_duration() {
        let two_sec = PCM::from(vec![0u8; PCM_SAMPLE_RATE_HZ as usize * 2 * 2]);
        let faster = two_sec.speed_up(2.0);
        let expected = (two_sec.len() / 2) as f32 / 2.0;
        let actual = faster.len() / 2;
        assert!(
            (actual as f32 - expected).abs() <= 2.0,
            "expected ~{expected} samples, got {actual}"
        );
    }

    #[test]
    fn speed_up_output_is_always_even_byte_length() {
        for speed in [1.1f32, 1.2, 1.4, 1.5, 2.0] {
            let pcm = sine_pcm(220.0, 0.05);
            let out = pcm.speed_up(speed);
            assert_eq!(
                out.len() % 2,
                0,
                "odd byte length at speed {speed}: {} bytes",
                out.len()
            );
        }
    }

    #[test]
    fn speed_up_preserves_dc_silence() {
        let silence = PCM::from(vec![0u8; PCM_SAMPLE_RATE_HZ as usize * 2]);
        let faster = silence.speed_up(1.4);
        assert!(
            faster.iter().all(|&b| b == 0),
            "silence introduced non-zero bytes"
        );
    }

    #[test]
    fn slow_down_identity_at_factor_one() {
        let pcm = sine_pcm(440.0, 0.1);
        assert_eq!(pcm.slow_down(1.0), pcm);
    }

    #[test]
    fn empty_buffer_slow_down_returns_empty() {
        assert_eq!(PCM::default().slow_down(2.0), PCM::default());
    }

    #[test]
    fn slow_down_doubles_duration() {
        let one_sec = PCM::from(vec![0u8; PCM_SAMPLE_RATE_HZ as usize * 2]);
        let slower = one_sec.slow_down(2.0);
        let expected = (one_sec.len() / 2) as f32 * 2.0;
        let actual = slower.len() / 2;
        assert!(
            (actual as f32 - expected).abs() <= 2.0,
            "expected ~{expected} samples, got {actual}"
        );
    }

    #[test]
    fn slow_down_output_is_even_byte_length() {
        for factor in [1.1f32, 1.2, 1.4, 1.5, 2.0] {
            let pcm = sine_pcm(220.0, 0.05);
            let out = pcm.slow_down(factor);
            assert_eq!(
                out.len() % 2,
                0,
                "odd byte length at factor {factor}: {} bytes",
                out.len()
            );
        }
    }

    #[test]
    fn slow_down_preserves_dc_silence() {
        let silence = PCM::from(vec![0u8; PCM_SAMPLE_RATE_HZ as usize * 2]);
        let slower = silence.slow_down(1.2);
        assert!(
            slower.iter().all(|&b| b == 0),
            "silence introduced non-zero bytes"
        );
    }

    #[test]
    fn slow_down_is_approximate_inverse_of_speed_up() {
        let pcm = sine_pcm(440.0, 0.5);
        let original_len = pcm.len();
        let faster = pcm.speed_up(1.2);
        let restored = faster.slow_down(1.2);
        assert!(
            (restored.len() as i64 - original_len as i64).abs() <= 4,
            "round-trip length mismatch: original={original_len}, restored={}",
            restored.len()
        );
    }
}
