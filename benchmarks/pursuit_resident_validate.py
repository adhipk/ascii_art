"""Execute the browser's resident GPU command plan through native WebGPU.

Optional runtime: uv run --no-project --with wgpu==0.32.0 --with numpy python
benchmarks/pursuit_resident_validate.py --input /tmp/pursuit-resident
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import wgpu
from wgpu.utils import get_default_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/tmp/pursuit-resident"))
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    root = args.input
    plan = json.loads((root / "plan.json").read_text())
    started = time.perf_counter()
    device = get_default_device()
    module = device.create_shader_module(code=(root / "shader.wgsl").read_text())
    entries = [
        {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": "uniform" if i == 0 else "read-only-storage" if i in (3, 4) else "storage"}}
        for i in range(8)
    ]
    layout = device.create_bind_group_layout(entries=entries)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[layout])
    names = sorted({d["entry"] for key in ("initialize", "iteration", "refinement", "flush") for d in plan[key]})
    pipelines = {name: device.create_compute_pipeline(layout=pipeline_layout, compute={"module": module, "entry_point": name}) for name in names}
    compile_ms = (time.perf_counter() - started) * 1000
    if args.compile_only:
        print(json.dumps({"pipelines": len(pipelines), "compileMs": compile_ms, "adapter": dict(device.adapter.info)}))
        return
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    initial_params = (root / "params.bin").read_bytes()
    buffers = [
        device.create_buffer_with_data(data=initial_params, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
        device.create_buffer_with_data(data=(root / "reference.f32").read_bytes(), usage=usage),
        device.create_buffer(size=(root / "reference.f32").stat().st_size, usage=usage),
        device.create_buffer_with_data(data=(root / "masks.f32").read_bytes(), usage=usage),
        device.create_buffer_with_data(data=(root / "kernels.bin").read_bytes(), usage=usage),
        device.create_buffer(size=plan["maxCandidates"] * 32, usage=usage),
        device.create_buffer(size=plan["workBytes"], usage=usage),
        device.create_buffer(size=plan["historyBytes"], usage=usage),
    ]
    group = device.create_bind_group(layout=layout, entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(buffers)])
    readbacks = 0

    def submit(commands, times=1):
        encoder = device.create_command_encoder()
        for _ in range(times):
            for command in commands:
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(pipelines[command["entry"]])
                compute.set_bind_group(0, group)
                compute.dispatch_workgroups(command["x"], command.get("y", 1))
                compute.end()
        device.queue.submit([encoder.finish()])

    def header():
        nonlocal readbacks
        readbacks += 1
        raw = bytes(device.queue.read_buffer(buffers[6], 0, 80))
        return np.frombuffer(raw, np.uint32).copy(), np.frombuffer(raw, np.float32).copy()

    def set_params(stage, round_index, mode=0, identity=0):
        data = bytearray(initial_params)
        u = np.frombuffer(data, np.uint32)
        u[5:9] = [stage, round_index, mode, identity]
        np.frombuffer(data, np.float32)[14] = minimum_gain
        device.queue.write_buffer(buffers[0], 0, data)
        device.queue.write_buffer(buffers[6], 0, np.array([0], np.uint32))

    def glyphs(count):
        return np.frombuffer(device.queue.read_buffer(buffers[7], 0, max(32, count * 32)), np.uint32).reshape(-1, 8)[:count].copy()

    exact_error = 0.0
    exact_file = root / "exact-reference.json"
    if exact_file.exists():
        submit(plan["initialize"])
        expected = json.loads(exact_file.read_text())
        candidates = np.zeros((len(expected), 8), np.uint32)
        for i, case in enumerate(expected):
            candidates[i, :3] = [case["x"], case["y"], case["kernel"]]
        device.queue.write_buffer(buffers[6], 28, np.array([len(expected)], np.uint32))
        device.queue.write_buffer(buffers[6], 208, candidates)
        first = next(i for i, d in enumerate(plan["iteration"]) if d["entry"] == "patch_0")
        last = next(i for i, d in enumerate(plan["iteration"]) if d["entry"] == "choose_best")
        submit(plan["iteration"][first:last+1])
        actual = np.frombuffer(device.queue.read_buffer(buffers[6], 208, len(expected)*32), np.float32).reshape(-1,8)
        reference_scores = np.array([case["color"]+[case["gain"]] for case in expected])
        exact_error = float(np.max(np.abs(actual[:,4:8]-reference_scores)))
        if not np.allclose(actual[:,4:8], reference_scores, atol=3e-6, rtol=2e-4):
            raise AssertionError(f"Exact GPU fitter differs from CPU: {actual[:,4:8]} vs {reference_scores}")
        device.queue.write_buffer(buffers[6], 0, bytes(112))
    run_start = time.perf_counter()
    submit(plan["initialize"])
    u, f = header()
    stages = []
    for stage_index, stage in enumerate(plan["stages"]):
        if not stage["count"]:
            stages.append({"size": stage["size"], "glyphs": 0, "converged": True})
            continue
        minimum_gain = max(1e-9, float(f[14]) * plan.get("coarseThreshold", .002)) if stage["size"] >= plan["stages"][0]["size"] / 2 else 1e-9
        round_count = plan["coarseRounds"] if stage["size"] >= plan["stages"][0]["size"] / 2 else 1
        for round_index in range(round_count):
            set_params(stage_index, round_index)
            refined = False
            for guard in range(260):
                submit(plan["iteration"], 16)
                submit(plan["flush"])
                u, f = header()
                if u[0] == 3:
                    raise AssertionError("GPU event capacity exceeded")
                if u[0] == 0:
                    continue
                if not refined:
                    refined = True
                    current = glyphs(int(u[4]))
                    eligible = [int(g[3]) for g in current if plan["kernels"][int(g[2])]["size"] == stage["size"]]
                    chosen = [eligible[(i * len(eligible) // min(8, len(eligible)) + round_index) % len(eligible)] for i in range(min(8, len(eligible)))]
                    old_events = int(u[5])
                    for identity in chosen:
                        set_params(stage_index, round_index, 1, identity)
                        submit(plan["refinement"])
                    u, f = header()
                    set_params(stage_index, round_index)
                    if int(u[5]) > old_events and int(u[4]) < plan["budget"]:
                        continue
                break
            else:
                raise AssertionError("Resident search failed to terminate")
            if int(u[4]) >= plan["budget"]:
                break
        stages.append({"size": stage["size"], "glyphs": int(u[4]), "searchRounds": round_index+1, "converged": int(u[4]) < plan["budget"]})
        if int(u[4]) >= plan["budget"]:
            break
    submit(plan["flush"])
    u, f = header()
    run_ms = (time.perf_counter() - run_start) * 1000
    event_count = int(u[5])
    event_raw = bytes(device.queue.read_buffer(buffers[7], 65536, event_count * 64))
    events_u = np.frombuffer(event_raw, np.uint32).reshape(-1, 16).copy()
    events_f = np.frombuffer(event_raw, np.float32).reshape(-1, 16).copy()
    canvas = np.frombuffer(device.queue.read_buffer(buffers[2], 0, plan["width"] * plan["height"] * 16), np.float32).reshape(-1, 4)[:, :3].copy()
    current = []
    energies = []
    for eu, ef in zip(events_u, events_f):
        energy = float(ef[11])
        if not np.isfinite(energy):
            raise AssertionError("Nonfinite GPU energy")
        if energies and energy >= energies[-1] + 2e-6:
            raise AssertionError(f"Energy increased: {energies[-1]} -> {energy}")
        energies.append(energy)
        if int(eu[3]) == 3:
            continue
        identity = int(eu[4])
        if int(eu[3]) == 2:
            current = [g for g in current if g["identity"] != identity]
        current.append({"x": int(eu[0]), "y": int(eu[1]), "kernel": int(eu[2]), "identity": identity, "color": ef[8:11].tolist()})
    reference = np.fromfile(root / "reference.f32", np.float32).reshape(-1, 4)[:plan["width"] * plan["height"], :3]
    assert len(current) == int(u[4])
    validation_started = time.perf_counter()
    meta = np.fromfile(root / "kernels.bin", np.uint32).reshape(-1,8)
    masks = np.fromfile(root / "masks.f32", np.float32)
    paper = np.frombuffer(initial_params, np.float32)[16:19]
    replay = np.broadcast_to(paper, (plan["height"],plan["width"],3)).copy()
    for glyph in current:
        size, offset = map(int,meta[glyph["kernel"],:2]);alpha=masks[offset:offset+size*size].reshape(size,size,1)
        region=replay[glyph["y"]:glyph["y"]+size,glyph["x"]:glyph["x"]+size]
        region += alpha*(np.array(glyph["color"],np.float32)-region)
    replay_error = float(np.max(np.abs(replay.reshape(-1,3)-canvas)))
    if replay_error > 3e-6: raise AssertionError(f"GPU canvas differs from replayed strokes: {replay_error}")
    residual=(reference-canvas).reshape(plan["height"],plan["width"],3)
    cpu_energy=0.0
    edge_weight=float(np.frombuffer(initial_params,np.float32)[12])
    for level in plan["levels"]:
        pix=float(np.mean(residual.astype(np.float64)**2))
        lum=residual @ np.array([.299,.587,.114],np.float32)
        edge=float((np.sum(np.diff(lum,axis=1)**2)+np.sum(np.diff(lum,axis=0)**2))/(lum.size))
        cpu_energy += level["weight"]*(pix+edge_weight*edge)
        padded=np.pad(residual,((1,1),(1,1),(0,0)),mode="edge")
        reduced=np.zeros(((residual.shape[0]+1)//2,(residual.shape[1]+1)//2,3),np.float32)
        for dy in range(3):
            for dx in range(3):
                reduced += (2 if dy==1 else 1)*(2 if dx==1 else 1)/16*padded[dy:dy+residual.shape[0]:2,dx:dx+residual.shape[1]:2]
        residual=reduced
    if abs(cpu_energy-energies[-1])>3e-6:raise AssertionError(f"GPU objective differs from CPU: {energies[-1]} vs {cpu_energy}")
    validation_ms=(time.perf_counter()-validation_started)*1000
    report = {"adapter": dict(device.adapter.info), "wgpuVersion": wgpu.__version__, "compileMs": compile_ms, "runMs": run_ms, "readbacks": readbacks, "exactMaxError": exact_error,
              "glyphs": len(current), "replayMaxError":replay_error,"cpuEnergy":cpu_energy,"validationMs":validation_ms,"events": event_count, "startEnergy": energies[0], "finalEnergy": energies[-1], "stages": stages,
              "scope": "Complete resident native GPU search including uploads, stage control, refinement and compact status readbacks; pipeline compilation reported separately"}
    (root / "result.json").write_text(json.dumps({"report": report, "glyphs": current, "energies": energies}, indent=2) + "\n")
    (root / "events.bin").write_bytes(event_raw)
    (root / "thumbs.bin").write_bytes(bytes(device.queue.read_buffer(buffers[7], 327680, event_count * plan["thumbWords"] * 4)))
    canvas.astype(np.float32).tofile(root / "canvas.f32")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
