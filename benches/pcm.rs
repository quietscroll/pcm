use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pcm::{PCM, PCM_SAMPLE_RATE_HZ};

fn sine_pcm(hz: f32, seconds: u32) -> PCM {
    use std::f32::consts::PI;

    let n = PCM_SAMPLE_RATE_HZ as usize * seconds as usize;
    let mut out = Vec::with_capacity(n * 2);

    for i in 0..n {
        let t = i as f32 / PCM_SAMPLE_RATE_HZ as f32;
        let sample = ((2.0 * PI * hz * t).sin() * i16::MAX as f32) as i16;
        out.extend_from_slice(&sample.to_le_bytes());
    }

    PCM::from(out)
}

fn pcm_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcm_operations");

    for seconds in [1_u32, 10, 60] {
        let pcm = sine_pcm(440.0, seconds);
        let samples = PCM_SAMPLE_RATE_HZ as u64 * seconds as u64;

        group.bench_with_input(BenchmarkId::new("duration", samples), &pcm, |b, pcm| {
            b.iter(|| black_box(pcm).duration())
        });

        group.bench_with_input(
            BenchmarkId::new("as_i16_samples", samples),
            &pcm,
            |b, pcm| b.iter(|| black_box(pcm).as_i16_samples()),
        );

        group.bench_with_input(BenchmarkId::new("i16_samples", samples), &pcm, |b, pcm| {
            b.iter(|| black_box(pcm).i16_samples())
        });

        group.bench_with_input(BenchmarkId::new("speed_up_1_4", samples), &pcm, |b, pcm| {
            b.iter(|| black_box(pcm).speed_up(black_box(1.4)))
        });

        group.bench_with_input(
            BenchmarkId::new("slow_down_1_4", samples),
            &pcm,
            |b, pcm| b.iter(|| black_box(pcm).slow_down(black_box(1.4))),
        );
    }

    group.finish();
}

criterion_group!(benches, pcm_operations);
criterion_main!(benches);
