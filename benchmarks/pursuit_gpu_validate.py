"""Compile/run the exact browser WGSL through native WebGPU; optional wgpu runtime.

bun benchmarks/pursuit_gpu_fixture.ts fixture.json /tmp/glyph-gpu-benchmark
uv run --no-project --with wgpu python benchmarks/pursuit_gpu_validate.py

Native batch measurements are not browser or end-to-end search speedups.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import wgpu
from wgpu.utils import get_default_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/tmp/glyph-gpu-benchmark"))
    args = parser.parse_args()
    root = args.input
    metadata = json.loads((root / "metadata.json").read_text())
    device = get_default_device()
    shader = device.create_shader_module(code=(root / "shader.wgsl").read_text())
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": shader, "entry_point": "main"}
    )
    buffers = []
    for name, usage in (
        ("params.bin", wgpu.BufferUsage.UNIFORM),
        ("source.f32", wgpu.BufferUsage.STORAGE),
        ("pixels.f32", wgpu.BufferUsage.STORAGE),
        ("masks.f32", wgpu.BufferUsage.STORAGE),
        ("candidates.u32", wgpu.BufferUsage.STORAGE),
    ):
        buffers.append(
            device.create_buffer_with_data(
                data=(root / name).read_bytes(), usage=usage | wgpu.BufferUsage.COPY_DST
            )
        )
    output = device.create_buffer(
        size=metadata["candidates"] * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    buffers.append(output)
    coverage = device.create_buffer_with_data(
        data=(root / "coverage.f32").read_bytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    buffers.append(coverage)
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": i, "resource": {"buffer": buffer}}
            for i, buffer in enumerate(buffers)
        ],
    )
    pixel_upload = (root / "pixels.f32").read_bytes()
    candidate_upload = (root / "candidates.u32").read_bytes()
    coverage_upload = (root / "coverage.f32").read_bytes()
    elapsed = []
    for _ in range(6):
        started = time.perf_counter()
        # Include per-batch uploads/readback; static target/masks stay resident.
        device.queue.write_buffer(buffers[2], 0, pixel_upload)
        device.queue.write_buffer(buffers[4], 0, candidate_upload)
        device.queue.write_buffer(buffers[6], 0, coverage_upload)
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, bind_group)
        compute.dispatch_workgroups((metadata["candidates"] + 63) // 64)
        compute.end()
        device.queue.submit([encoder.finish()])
        scores = np.frombuffer(
            device.queue.read_buffer(output), dtype=np.float32
        ).copy()
        elapsed.append((time.perf_counter() - started) * 1000)
    reference = np.fromfile(root / "reference.f64", dtype=np.float64)
    error = np.abs(scores - reference)
    if not np.isfinite(scores).all() or np.any(
        error > 5e-6 + np.abs(reference) * 0.002
    ):
        raise AssertionError(
            f"GPU/CPU candidate score mismatch; max error {error.max()}"
        )
    report = {
        **metadata,
        "adapter": dict(device.adapter.info),
        "wgpu_version": wgpu.__version__,
        "gpu_batch_ms_median_warm": float(np.median(elapsed[1:])),
        "gpu_first_batch_ms": elapsed[0],
        "gpu_samples_ms": elapsed,
        "max_absolute_score_error": float(error.max()),
        "mean_absolute_score_error": float(error.mean()),
        "scope": "One real candidate batch, native Metal/Vulkan WebGPU; includes uploads/readback, excludes pipeline creation and CPU final verification",
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    for buffer in buffers:
        buffer.destroy()
    device.destroy()


if __name__ == "__main__":
    main()
