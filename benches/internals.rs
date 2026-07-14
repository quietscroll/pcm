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

fn internal_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcm_internals");

    for seconds in [1_u32, 10, 60] {
        let pcm = sine_pcm(440.0, seconds);
        let mut odd = pcm.clone().into_inner();
        odd.push(0);
        let odd = PCM::from(odd);
        let samples = PCM_SAMPLE_RATE_HZ as u64 * seconds as u64;

        group.bench_with_input(
            BenchmarkId::new("i16_sample_view_len_borrowed", samples),
            &pcm,
            |b, pcm| b.iter(|| black_box(pcm).bench_i16_sample_view_len()),
        );

        group.bench_with_input(
            BenchmarkId::new("i16_sample_view_len_owned", samples),
            &odd,
            |b, pcm| b.iter(|| black_box(pcm).bench_i16_sample_view_len()),
        );
    }

    group.bench_function("write_i16x8_le", |b| {
        b.iter(|| pcm::bench_write_i16x8_le(black_box([1, -2, 3, -4, 5, -6, 7, -8])))
    });

    group.finish();
}

criterion_group!(benches, internal_operations);
criterion_main!(benches);
