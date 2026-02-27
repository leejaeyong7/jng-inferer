from time import perf_counter


class StageProfiler:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.start_time = perf_counter()

    def add(self, stage, seconds):
        if seconds < 0:
            return
        self.totals[stage] = self.totals.get(stage, 0.0) + float(seconds)
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def snapshot(self):
        wall_seconds = max(perf_counter() - self.start_time, 1e-9)
        return {
            "wall_seconds": wall_seconds,
            "totals": dict(self.totals),
            "counts": dict(self.counts),
        }


def format_stage_report(stage_profiler, total_frames=None):
    if stage_profiler is None:
        return ""
    snap = stage_profiler.snapshot()
    totals = snap["totals"]
    counts = snap["counts"]
    wall_seconds = snap["wall_seconds"]
    wall_fps = None
    if total_frames is not None and wall_seconds > 0:
        wall_fps = float(total_frames) / wall_seconds

    lines = ["Performance profile:"]
    if total_frames is not None and wall_fps is not None:
        lines.append(
            f"  frames={total_frames}, wall={wall_seconds:.3f}s, throughput={wall_fps:.2f} fps"
        )
    else:
        lines.append(f"  wall={wall_seconds:.3f}s")

    preferred_order = ["decode", "det", "pose", "feature", "score", "write"]
    known = [k for k in preferred_order if k in totals]
    extra = sorted([k for k in totals.keys() if k not in preferred_order])
    for stage in known + extra:
        total = totals.get(stage, 0.0)
        count = max(1, counts.get(stage, 0))
        avg_ms = 1000.0 * total / count
        share = 100.0 * total / wall_seconds
        lines.append(
            f"  {stage:>7}: total={total:.3f}s avg={avg_ms:.2f}ms count={count} share={share:.1f}%"
        )

    return "\n".join(lines)
