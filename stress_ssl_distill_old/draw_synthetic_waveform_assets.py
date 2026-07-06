from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


BLUE = "#2563EB"
NAVY = "#071A4D"
ORANGE = "#F97316"
BLACK = "#4B5563"
GREEN = "#2F855A"
LIGHT_BLUE = "#3B82F6"
LIGHT_GREEN = "#58A575"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize(values: list[float], amp: float) -> list[float]:
    mean = sum(values) / max(len(values), 1)
    centered = [value - mean for value in values]
    scale = max(max(abs(value) for value in centered), 1e-6)
    return [value / scale * amp for value in centered]


def gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def ecg_wave(n: int, beats: int = 4, noise: float = 0.012, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    values = [0.0 for _ in range(n)]
    for beat in range(beats):
        center = (beat + 0.62) / beats
        for idx in range(n):
            t = idx / max(n - 1, 1)
            values[idx] += 0.10 * gaussian(t, center - 0.060, 0.018)
            values[idx] -= 0.15 * gaussian(t, center - 0.018, 0.006)
            values[idx] += 1.10 * gaussian(t, center, 0.0045)
            values[idx] -= 0.34 * gaussian(t, center + 0.018, 0.008)
            values[idx] += 0.18 * gaussian(t, center + 0.075, 0.025)
    for idx in range(n):
        t = idx / max(n - 1, 1)
        values[idx] += 0.035 * math.sin(2 * math.pi * 2.1 * t) + rng.uniform(-noise, noise)
    return values


def ppg_wave(n: int, beats: int = 5, noise: float = 0.035, seed: int = 2) -> list[float]:
    rng = random.Random(seed)
    values = [0.0 for _ in range(n)]
    for beat in range(beats):
        center = (beat + 0.50) / beats
        for idx in range(n):
            t = idx / max(n - 1, 1)
            values[idx] += 0.70 * gaussian(t, center, 0.010)
            values[idx] -= 0.22 * gaussian(t, center + 0.025, 0.012)
            values[idx] += 0.16 * gaussian(t, center + 0.055, 0.020)
    for idx in range(n):
        t = idx / max(n - 1, 1)
        values[idx] += 0.06 * math.sin(2 * math.pi * 7.5 * t) + rng.uniform(-noise, noise)
    return values


def acc_wave(n: int, color_lane: int = 0, seed: int = 3) -> list[float]:
    rng = random.Random(seed + color_lane * 19)
    values = []
    for idx in range(n):
        t = idx / max(n - 1, 1)
        base = (
            0.12 * math.sin(2 * math.pi * (2.1 + 0.35 * color_lane) * t + color_lane)
            + 0.05 * math.sin(2 * math.pi * 12.0 * t)
            + rng.uniform(-0.07, 0.07)
        )
        values.append(base)
    for event in (0.22, 0.52, 0.78):
        for idx in range(n):
            t = idx / max(n - 1, 1)
            values[idx] += (0.55 - 0.10 * color_lane) * gaussian(t, event + 0.015 * color_lane, 0.006)
            values[idx] -= 0.20 * gaussian(t, event + 0.020 + 0.010 * color_lane, 0.010)
    return values


def eda_wave(n: int) -> list[float]:
    values = []
    for idx in range(n):
        t = idx / max(n - 1, 1)
        value = (
            0.48 * math.sin(2 * math.pi * 2.0 * t - 0.45)
            + 0.16 * math.sin(2 * math.pi * 4.0 * t + 0.20)
            + 0.70 * gaussian(t, 0.18, 0.11)
            - 0.55 * gaussian(t, 0.45, 0.10)
            + 0.40 * gaussian(t, 0.73, 0.15)
        )
        values.append(value)
    return values


def resp_wave(n: int) -> list[float]:
    values = []
    for idx in range(n):
        t = idx / max(n - 1, 1)
        slow = math.sin(2 * math.pi * 2.65 * t - 0.6)
        shaped = math.tanh(2.0 * slow)
        values.append(shaped + 0.08 * math.sin(2 * math.pi * 8 * t))
    return values


def polyline(values: list[float], x: float, y: float, width: float, amp: float) -> str:
    scaled = normalize(values, amp)
    n = len(scaled)
    points = []
    for idx, value in enumerate(scaled):
        px = x + idx / max(n - 1, 1) * width
        py = y - value
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def text(x: float, y: float, body: str, size: int = 18, color: str = "#111827", weight: int = 700, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escape(body)}</text>'
    )


def line_wave(
    values: list[float],
    x: float,
    y: float,
    width: float,
    amp: float,
    color: str,
    stroke: float = 2.0,
) -> str:
    return (
        f'<polyline points="{polyline(values, x, y, width, amp)}" fill="none" '
        f'stroke="{color}" stroke-width="{stroke:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def axis_icon(x: float, y: float, scale: float = 1.0) -> str:
    sw = 3.0 * scale
    return "\n".join(
        [
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y - 42 * scale:.1f}" stroke="{NAVY}" stroke-width="{sw}" stroke-linecap="round"/>',
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 50 * scale:.1f}" y2="{y:.1f}" stroke="{NAVY}" stroke-width="{sw}" stroke-linecap="round"/>',
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x - 35 * scale:.1f}" y2="{y + 26 * scale:.1f}" stroke="{NAVY}" stroke-width="{sw}" stroke-linecap="round"/>',
            f'<polygon points="{x - 4 * scale:.1f},{y - 42 * scale:.1f} {x + 4 * scale:.1f},{y - 42 * scale:.1f} {x:.1f},{y - 52 * scale:.1f}" fill="{NAVY}"/>',
            f'<polygon points="{x + 50 * scale:.1f},{y - 4 * scale:.1f} {x + 50 * scale:.1f},{y + 4 * scale:.1f} {x + 60 * scale:.1f},{y:.1f}" fill="{NAVY}"/>',
            f'<polygon points="{x - 35 * scale:.1f},{y + 20 * scale:.1f} {x - 30 * scale:.1f},{y + 28 * scale:.1f} {x - 43 * scale:.1f},{y + 31 * scale:.1f}" fill="{NAVY}"/>',
            text(x - 10 * scale, y - 56 * scale, "z", size=int(14 * scale), color=NAVY, weight=700, anchor="middle"),
            text(x + 70 * scale, y + 7 * scale, "z", size=int(14 * scale), color=NAVY, weight=700, anchor="middle"),
            text(x - 50 * scale, y + 42 * scale, "x", size=int(14 * scale), color=NAVY, weight=700, anchor="middle"),
        ]
    )


def rounded_panel(x: float, y: float, w: float, h: float, stroke: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="18" '
        f'fill="#FFFFFF" stroke="{stroke}" stroke-width="2.2"/>'
    )


def deployable_panel() -> str:
    w, h = 430, 420
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        rounded_panel(2, 2, w - 4, h - 4, LIGHT_BLUE),
        text(w / 2, 47, "Deployable watch", size=28, color="#0B3DAE", weight=800, anchor="middle"),
        text(w / 2, 83, "sensors", size=28, color="#0B3DAE", weight=800, anchor="middle"),
        text(w / 2, 124, "PPG/BVP + ACC", size=22, color="#111827", weight=700, anchor="middle"),
        text(23, 187, "PPG/BVP", size=18, color=NAVY, weight=700),
        line_wave(ppg_wave(230, beats=5, seed=4), 138, 180, 238, 33, BLUE, stroke=2.0),
        text(23, 270, "ACC (3-axis)", size=18, color=NAVY, weight=700),
        axis_icon(70, 348, scale=0.74),
        line_wave(acc_wave(230, 0, seed=8), 185, 254, 190, 18, BLACK, stroke=1.75),
        line_wave(acc_wave(230, 1, seed=8), 185, 300, 190, 18, ORANGE, stroke=1.75),
        line_wave(acc_wave(230, 2, seed=8), 185, 347, 190, 18, BLUE, stroke=1.75),
        "</svg>",
    ]
    return "\n".join(parts)


def privileged_panel() -> str:
    w, h = 430, 560
    x_label = 28
    x_wave = 125
    wave_w = 250
    y0 = 235
    step = 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        rounded_panel(2, 2, w - 4, h - 4, LIGHT_GREEN),
        text(w / 2, 70, "Privileged sensors", size=28, color="#0B6F2A", weight=800, anchor="middle"),
        text(w / 2, 108, "during training only", size=28, color="#0B6F2A", weight=800, anchor="middle"),
        text(w / 2, 162, "ECG / EDA / Resp /", size=24, color="#111827", weight=700, anchor="middle"),
        text(w / 2, 197, "E4 / Polar", size=24, color="#111827", weight=700, anchor="middle"),
        text(x_label, y0 + 5, "ECG", size=18, color="#111827", weight=700),
        line_wave(ecg_wave(230, beats=3, noise=0.006, seed=2), x_wave, y0, wave_w, 31, GREEN, stroke=2.0),
        text(x_label, y0 + step + 5, "EDA", size=18, color="#111827", weight=700),
        line_wave(eda_wave(230), x_wave, y0 + step, wave_w, 28, GREEN, stroke=2.0),
        text(x_label, y0 + 2 * step + 5, "Resp", size=18, color="#111827", weight=700),
        line_wave(resp_wave(230), x_wave, y0 + 2 * step, wave_w, 28, GREEN, stroke=2.0),
        text(x_label, y0 + 3 * step + 5, "E4", size=18, color="#111827", weight=700),
        line_wave(ecg_wave(240, beats=4, noise=0.045, seed=10), x_wave, y0 + 3 * step, wave_w, 29, GREEN, stroke=2.0),
        text(x_label, y0 + 4 * step + 5, "Polar", size=18, color="#111827", weight=700),
        line_wave(ecg_wave(240, beats=4, noise=0.032, seed=15), x_wave, y0 + 4 * step, wave_w, 32, GREEN, stroke=2.0),
        "</svg>",
    ]
    return "\n".join(parts)


def combined_panel() -> str:
    deploy = deployable_panel()
    priv = privileged_panel()
    deploy_inner = deploy.split(">", 1)[1].rsplit("</svg>", 1)[0]
    priv_inner = priv.split(">", 1)[1].rsplit("</svg>", 1)[0]
    w, h = 430, 995
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            f'<g transform="translate(0,0)">{deploy_inner}</g>',
            f'<g transform="translate(0,435)">{priv_inner}</g>',
            "</svg>",
        ]
    )


def individual_waveforms(output_dir: Path) -> None:
    specs = [
        ("ppg_bvp", ppg_wave(230, beats=5, seed=4), BLUE, 300, 64, 18),
        ("acc_x", acc_wave(230, 0, seed=8), BLACK, 300, 64, 16),
        ("acc_y", acc_wave(230, 1, seed=8), ORANGE, 300, 64, 16),
        ("acc_z", acc_wave(230, 2, seed=8), BLUE, 300, 64, 16),
        ("ecg", ecg_wave(230, beats=3, noise=0.006, seed=2), GREEN, 300, 64, 20),
        ("eda", eda_wave(230), GREEN, 300, 64, 19),
        ("resp", resp_wave(230), GREEN, 300, 64, 19),
        ("e4", ecg_wave(240, beats=4, noise=0.045, seed=10), GREEN, 300, 64, 19),
        ("polar", ecg_wave(240, beats=4, noise=0.032, seed=15), GREEN, 300, 64, 20),
    ]
    for name, values, color, width, height, amp in specs:
        svg = "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                line_wave(values, 4, height / 2, width - 8, amp, color, stroke=2.2),
                "</svg>",
            ]
        )
        (output_dir / f"{name}.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw synthetic editable waveform assets for method figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/waveform_assets/synthetic"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "deployable_watch_sensors.svg").write_text(deployable_panel(), encoding="utf-8")
    (args.output_dir / "privileged_training_sensors.svg").write_text(privileged_panel(), encoding="utf-8")
    (args.output_dir / "combined_sensor_panels.svg").write_text(combined_panel(), encoding="utf-8")
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    individual_waveforms(individual_dir)

    print(f"saved={args.output_dir / 'deployable_watch_sensors.svg'}")
    print(f"saved={args.output_dir / 'privileged_training_sensors.svg'}")
    print(f"saved={args.output_dir / 'combined_sensor_panels.svg'}")
    print(f"saved_individual_dir={individual_dir}")


if __name__ == "__main__":
    main()
