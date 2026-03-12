"""
generate_assets_dbz.py
----------------------
Generates ALL visual assets for the Gesture Vision Engine with
DBZ-accurate art style based on actual anime/game references:

  AURA SHEETS    — Upward-rising spiky flame wisps (SSJ style), not rings
  BEAM SHEETS    — Tapered energy beams with white core + coloured halo
  EXPLOSION      — Flash → expanding shockwave ring → debris scatter
  RING SHEETS    — Expanding power-up rings (like DBZ power level burst)
  SPARK SHEETS   — Jagged electric arcs (SSJ2 lightning / ki crackle)
  SMOKE SHEET    — Dark grey-brown smoke puffs
  SPIRIT BOMB    — Growing blue orb with orbiting wisps + equatorial ring
  TEXTURES       — Soft radial glow dots, beam gradient, shockwave ring
  HUD ASSETS     — Energy bar, cooldown ring, crosshair

All outputs: RGBA PNG, transparent background.
Run from project root:  python generate_assets_dbz.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math, os

# ── Output paths (relative to project root) ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITES_DIR = os.path.join(BASE_DIR, "assets", "effects", "sprites")
TEXTURES_DIR = os.path.join(BASE_DIR, "assets", "effects", "textures")
HUD_DIR = os.path.join(BASE_DIR, "assets", "ui", "hud")

for d in [SPRITES_DIR, TEXTURES_DIR, HUD_DIR]:
    os.makedirs(d, exist_ok=True)

# ── DBZ colour palette (anime-accurate) ──────────────────────────────────────
GOLD_CORE = (255, 255, 200)  # white-hot SSJ gold core
GOLD_MID = (255, 210, 30)  # bright SSJ gold
GOLD_OUTER = (255, 100, 0)  # deep orange-gold edge
BLUE_CORE = (220, 245, 255)  # white-blue ki core
BLUE_MID = (60, 160, 255)  # DBZ ki blue
BLUE_OUTER = (0, 50, 180)  # deep blue
CYAN_CORE = (210, 255, 255)  # kamehameha white-cyan
CYAN_MID = (80, 220, 255)  # kamehameha cyan
CYAN_OUTER = (0, 100, 200)  # kamehameha deep
RED_CORE = (255, 240, 200)  # red ki white core
RED_MID = (255, 60, 30)  # fiery red
RED_OUTER = (180, 10, 0)  # deep crimson
PURPLE_CORE = (255, 220, 255)  # purple ki core
PURPLE_MID = (200, 60, 255)  # spirit bomb purple
PURPLE_OUT = (100, 0, 180)  # deep purple
WHITE = (255, 255, 255)
GREY_SMOKE = (140, 130, 120)  # smoke colour

# ═════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def dist_field(w, h, cx, cy):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)


def new_frame(w, h):
    return np.zeros((h, w, 4), dtype=np.float32)


def to_img(data):
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), "RGBA")


def make_sheet(frames, path):
    if not frames:
        return
    w = sum(f.width for f in frames)
    h = max(f.height for f in frames)
    sheet = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for f in frames:
        sheet.paste(f, (x, 0))
        x += f.width
    sheet.save(path, "PNG")
    fname = os.path.basename(path)
    print(f"  {fname:<45}  {w}x{h}  {len(frames)} frames")


def radial_glow(data, cx, cy, r_inner, r_outer, color, alpha=255):
    """Paint a soft radial glow region."""
    h, w = data.shape[:2]
    d = dist_field(w, h, cx, cy)
    t = np.clip((d - r_inner) / max(r_outer - r_inner, 1), 0, 1)
    m = (1 - t) ** 2
    mask = d <= r_outer
    data[:, :, 0] = np.where(
        mask, np.minimum(255, data[:, :, 0] + color[0] * m * alpha / 255), data[:, :, 0]
    )
    data[:, :, 1] = np.where(
        mask, np.minimum(255, data[:, :, 1] + color[1] * m * alpha / 255), data[:, :, 1]
    )
    data[:, :, 2] = np.where(
        mask, np.minimum(255, data[:, :, 2] + color[2] * m * alpha / 255), data[:, :, 2]
    )
    data[:, :, 3] = np.where(
        mask, np.minimum(255, data[:, :, 3] + m * alpha), data[:, :, 3]
    )
    return data


def ring_glow(data, cx, cy, r, thick, color, alpha=220):
    h, w = data.shape[:2]
    d = dist_field(w, h, cx, cy)
    m = np.clip(1 - np.abs(d - r) / max(thick, 1), 0, 1) ** 2
    glow = np.clip(1 - np.abs(d - r) / max(thick * 2.8, 1), 0, 1) ** 3
    data[:, :, 0] = np.minimum(255, data[:, :, 0] + color[0] * (m + glow * 0.4))
    data[:, :, 1] = np.minimum(255, data[:, :, 1] + color[1] * (m + glow * 0.4))
    data[:, :, 2] = np.minimum(255, data[:, :, 2] + color[2] * (m + glow * 0.4))
    data[:, :, 3] = np.minimum(255, data[:, :, 3] + m * alpha + glow * alpha * 0.4)
    return data


def soft_blur(img, radius=2.0):
    return img.filter(ImageFilter.GaussianBlur(radius))


# ═════════════════════════════════════════════════════════════════════════════
# AURA SPRITESHEET
# DBZ reference: SSJ aura — vertical spiky flames rising from the ground up,
# semi-transparent, pulsing inward/outward, warm colour at base cooling to white tips
# ═════════════════════════════════════════════════════════════════════════════
def gen_aura_sheet(name, core_col, mid_col, outer_col, n_frames=8, fw=128, fh=192):
    frames = []
    for i in range(n_frames):
        phase = i / n_frames  # 0→1 animation phase
        pulse = 0.75 + 0.25 * math.sin(phase * 2 * math.pi)

        data = new_frame(fw, fh)
        cx = fw // 2

        # ── Base glow at bottom (ki pooling at feet) ──────────────────
        base_r = int(36 * pulse)
        data = radial_glow(
            data, cx, fh - 20, 0, base_r, outer_col, alpha=int(160 * pulse)
        )
        data = radial_glow(
            data, cx, fh - 20, 0, base_r // 2, mid_col, alpha=int(220 * pulse)
        )

        # ── Upward-rising flame spikes (signature DBZ aura look) ──────
        np.random.seed(i * 13 + 7)
        n_flames = 7
        for f_idx in range(n_flames):
            # Each flame: starts at bottom, tapers to tip at random height
            base_x = cx + np.random.uniform(-28, 28) * pulse
            flame_h = np.random.uniform(fh * 0.35, fh * 0.85) * pulse
            flame_bw = np.random.uniform(10, 22)  # base width
            lean = np.random.uniform(-8, 8)  # slight lean left/right

            steps = 40
            for s in range(steps):
                t = s / (steps - 1)  # 0=base, 1=tip
                fy = fh - 20 - t * flame_h
                fx = base_x + lean * t
                fw2 = flame_bw * (1 - t) ** 0.6  # taper

                # Colour: outer_col at base → core_col (white-ish) at tip
                cr = outer_col[0] + (core_col[0] - outer_col[0]) * t
                cg = outer_col[1] + (core_col[1] - outer_col[1]) * t
                cb = outer_col[2] + (core_col[2] - outer_col[2]) * t
                al = (1 - t * 0.6) * 200 * pulse

                d = dist_field(fw, fh, fx, fy)
                m = np.clip(1 - d / max(fw2, 1), 0, 1) ** 1.8

                data[:, :, 0] = np.minimum(255, data[:, :, 0] + cr * m * al / 255)
                data[:, :, 1] = np.minimum(255, data[:, :, 1] + cg * m * al / 255)
                data[:, :, 2] = np.minimum(255, data[:, :, 2] + cb * m * al / 255)
                data[:, :, 3] = np.minimum(255, data[:, :, 3] + m * al)

        # ── Inner bright core column (body glow) ──────────────────────
        for y in range(fh):
            t_y = (fh - y) / fh  # 0=bottom, 1=top
            glow = math.exp(-t_y * 3) * pulse
            col_w = int(12 * pulse * (1 - t_y * 0.5))
            d = dist_field(fw, fh, cx, y)
            cm = np.clip(1 - d / max(col_w, 1), 0, 1) ** 2
            data[:, :, 0] = np.minimum(
                255, data[:, :, 0] + core_col[0] * cm * glow * 180 / 255
            )
            data[:, :, 1] = np.minimum(
                255, data[:, :, 1] + core_col[1] * cm * glow * 180 / 255
            )
            data[:, :, 2] = np.minimum(
                255, data[:, :, 2] + core_col[2] * cm * glow * 180 / 255
            )
            data[:, :, 3] = np.minimum(255, data[:, :, 3] + cm * glow * 180)

        img = to_img(data)
        img = soft_blur(img, 1.5)
        frames.append(img)

    make_sheet(frames, os.path.join(SPRITES_DIR, f"aura_sheet_{name}.png"))


# ═════════════════════════════════════════════════════════════════════════════
# BEAM SPRITESHEET
# DBZ reference: Kamehameha/Galick Gun beams — tapered cylinder,
# bright white core surrounded by coloured halo, slight wave/turbulence
# ═════════════════════════════════════════════════════════════════════════════
def gen_beam_sheet(name, core_col, mid_col, outer_col, n_frames=8, fw=256, fh=80):
    frames = []
    cy = fh // 2

    for i in range(n_frames):
        phase = i / n_frames
        pulse = 0.85 + 0.15 * math.sin(phase * 2 * math.pi)
        data = new_frame(fw, fh)

        for x in range(fw):
            t_x = x / fw  # 0=origin, 1=tip
            taper = 1.0 - t_x * 0.35  # beam tapers toward tip
            wave_off = math.sin(t_x * 12 + phase * 2 * math.pi) * 2.5

            for y in range(fh):
                dy = abs(y - (cy + wave_off))
                hw = (fh * 0.42) * taper * pulse  # half-width at this x

                # Outer halo
                outer = math.exp(-(dy**2) / (hw**2 * 2.2))
                # Mid glow
                mid = math.exp(-(dy**2) / (hw**2 * 0.55))
                # White core
                core = math.exp(-(dy**2) / (hw**2 * 0.08))

                # x-fade: bright origin, slight fade at tip
                x_fade = 0.7 + 0.3 * (1 - t_x * 0.5)

                r = (
                    outer_col[0] * outer + mid_col[0] * mid + core_col[0] * core
                ) * x_fade
                g = (
                    outer_col[1] * outer + mid_col[1] * mid + core_col[1] * core
                ) * x_fade
                b = (
                    outer_col[2] * outer + mid_col[2] * mid + core_col[2] * core
                ) * x_fade
                a = (outer * 160 + mid * 220 + core * 255) * x_fade

                data[y, x, 0] = min(255, r)
                data[y, x, 1] = min(255, g)
                data[y, x, 2] = min(255, b)
                data[y, x, 3] = min(255, a)

        frames.append(to_img(data))

    make_sheet(frames, os.path.join(SPRITES_DIR, f"beam_sheet_{name}.png"))


# ═════════════════════════════════════════════════════════════════════════════
# EXPLOSION SPRITESHEET
# DBZ reference: Ki blast impact — white flash → expanding orange-gold ring
# → shockwave blast → debris sparks scattering outward → fade
# ═════════════════════════════════════════════════════════════════════════════
def gen_explosion_sheet(
    name, core_col, mid_col, outer_col, n_frames=10, fw=128, fh=128
):
    frames = []
    cx, cy = fw // 2, fh // 2

    for i in range(n_frames):
        t = i / (n_frames - 1)  # 0=impact, 1=fully expanded/faded
        data = new_frame(fw, fh)

        # Phase 1 (0–0.2): bright white impact flash
        if t < 0.25:
            flash_r = int(fw * 0.45 * (t / 0.25))
            flash_a = int(255 * (1 - t / 0.25) * 0.9) + int(255 * (t / 0.25))
            data = radial_glow(
                data, cx, cy, 0, flash_r, (255, 255, 255), alpha=min(255, flash_a)
            )
            data = radial_glow(data, cx, cy, 0, flash_r // 2, core_col, alpha=255)

        # Phase 2 (0.1–0.7): expanding shockwave ring
        if t > 0.08:
            ring_t = (t - 0.08) / 0.92
            ring_r = int(fw * 0.48 * ring_t)
            ring_th = max(2, int(14 * (1 - ring_t * 0.8)))
            ring_a = int(240 * math.sin(ring_t * math.pi))
            if ring_r > 0:
                data = ring_glow(data, cx, cy, ring_r, ring_th, mid_col, alpha=ring_a)
                # Outer glow ring
                data = ring_glow(
                    data, cx, cy, ring_r, ring_th * 2, outer_col, alpha=ring_a // 2
                )

        # Phase 3 (0.2–0.9): inner glow core shrinking
        core_r = max(1, int(fw * 0.22 * (1 - max(0, (t - 0.2) / 0.7))))
        if core_r > 2:
            core_a = int(200 * (1 - t))
            data = radial_glow(data, cx, cy, 0, core_r, mid_col, alpha=core_a)
            data = radial_glow(data, cx, cy, 0, core_r // 2, core_col, alpha=core_a)

        # Phase 4 (0.15–0.85): debris sparks flying outward
        if 0.15 < t < 0.85:
            np.random.seed(i * 7)
            debris_t = (t - 0.15) / 0.7
            for _ in range(14):
                angle = np.random.uniform(0, 2 * math.pi)
                r_dist = np.random.uniform(0.3, 1.0) * debris_t * fw * 0.46
                px = cx + math.cos(angle) * r_dist
                py = cy + math.sin(angle) * r_dist
                pr = max(1, int(5 * (1 - debris_t) * np.random.uniform(0.5, 1.5)))
                pa = int(220 * (1 - debris_t))
                data = radial_glow(data, px, py, 0, pr, mid_col, alpha=pa)

        img = to_img(data)
        img = soft_blur(img, 1.2)
        frames.append(img)

    make_sheet(frames, os.path.join(SPRITES_DIR, f"explosion_sheet_{name}.png"))


# ═════════════════════════════════════════════════════════════════════════════
# RING SPRITESHEET
# DBZ reference: Power-up energy ring — solid ring expanding outward from
# body, used during transformations (SSJ, power level burst)
# ═════════════════════════════════════════════════════════════════════════════
def gen_ring_sheet(name, core_col, mid_col, outer_col, n_frames=10, fw=128, fh=128):
    frames = []
    cx, cy = fw // 2, fh // 2

    for i in range(n_frames):
        t = i / (n_frames - 1)
        data = new_frame(fw, fh)

        r_max = fw * 0.48
        r = t * r_max
        thick = max(2, int(16 * (1 - t * 0.65)))
        alpha = int(255 * math.sin(t * math.pi))

        # Inner flash fill at start
        if t < 0.35:
            fill_a = int(200 * (1 - t / 0.35))
            data = radial_glow(data, cx, cy, 0, int(r) + 4, core_col, alpha=fill_a)

        # Main ring
        if r > 1:
            data = ring_glow(data, cx, cy, r, thick, mid_col, alpha=alpha)
            # Bright inner edge of ring
            data = ring_glow(
                data,
                cx,
                cy,
                r - thick * 0.3,
                thick * 0.4,
                core_col,
                alpha=min(255, alpha + 60),
            )
            # Soft outer halo
            data = ring_glow(data, cx, cy, r, thick * 3, outer_col, alpha=alpha // 3)

        # Small energy sparks on the ring perimeter
        np.random.seed(i * 3 + 1)
        if r > 5:
            for _ in range(8):
                angle = np.random.uniform(0, 2 * math.pi)
                sx = cx + math.cos(angle) * r
                sy = cy + math.sin(angle) * r
                data = radial_glow(
                    data, sx, sy, 0, 5, core_col, alpha=min(255, alpha + 80)
                )

        img = to_img(data)
        img = soft_blur(img, 1.2)
        frames.append(img)

    make_sheet(frames, os.path.join(SPRITES_DIR, f"ring_sheet_{name}.png"))


# ═════════════════════════════════════════════════════════════════════════════
# SPARK SPRITESHEET
# DBZ reference: SSJ2 electric arcs — jagged forking lightning bolts
# crackling outward from center, white-hot at base, coloured at tips
# ═════════════════════════════════════════════════════════════════════════════
def gen_spark_sheet(name, core_col, mid_col, n_frames=8, fw=96, fh=96):
    frames = []
    cx, cy = fw // 2, fh // 2

    for i in range(n_frames):
        data = new_frame(fw, fh)
        np.random.seed(i * 17 + 3)

        # Core glow at center
        data = radial_glow(data, cx, cy, 0, 12, (255, 255, 255), alpha=240)
        data = radial_glow(data, cx, cy, 0, 20, mid_col, alpha=140)

        # Jagged lightning bolts radiating outward
        n_bolts = 6
        for b in range(n_bolts):
            base_angle = (b / n_bolts) * 2 * math.pi + i * 0.52
            length = np.random.uniform(22, 42)
            branches = 2  # forks per bolt

            for br in range(branches):
                angle_off = np.random.uniform(-0.25, 0.25)
                angle = base_angle + angle_off
                n_segs = 10
                px, py = float(cx), float(cy)

                for s in range(n_segs):
                    t2 = (s + 1) / n_segs
                    ex = cx + math.cos(angle) * length * t2
                    ey = cy + math.sin(angle) * length * t2
                    # Jitter perpendicular to bolt direction
                    perp = angle + math.pi / 2
                    jit = np.random.uniform(-6, 6) * (1 - t2)
                    jx = ex + math.cos(perp) * jit
                    jy = ey + math.sin(perp) * jit

                    # Draw segment as a series of glowing dots
                    seg_steps = 8
                    for ss in range(seg_steps):
                        st = ss / seg_steps
                        lx = px + (jx - px) * st
                        ly = py + (jy - py) * st
                        # White core, colour at edges
                        fade = 1 - t2 * 0.5
                        data = radial_glow(
                            data, lx, ly, 0, 2.5, (255, 255, 255), alpha=int(220 * fade)
                        )
                        data = radial_glow(
                            data, lx, ly, 1.5, 5, mid_col, alpha=int(150 * fade)
                        )
                    px, py = jx, jy

                # Tip flash
                tip_x = cx + math.cos(base_angle) * length
                tip_y = cy + math.sin(base_angle) * length
                data = radial_glow(data, tip_x, tip_y, 0, 4, core_col, alpha=180)

        img = to_img(data)
        img = soft_blur(img, 1.0)
        frames.append(img)

    make_sheet(frames, os.path.join(SPRITES_DIR, f"spark_sheet_{name}.png"))


# ═════════════════════════════════════════════════════════════════════════════
# SMOKE SPRITESHEET
# Reference: Dust/debris cloud after impact — dark grey-brown expanding puffs
# ═════════════════════════════════════════════════════════════════════════════
def gen_smoke_sheet(n_frames=8, fw=128, fh=128):
    frames = []
    cx = fw // 2

    for i in range(n_frames):
        t = i / (n_frames - 1)
        data = new_frame(fw, fh)

        # Smoke rises upward and expands
        cy = int(fh * 0.75 - t * fh * 0.45)

        np.random.seed(i * 5 + 2)
        n_puffs = 6
        for p in range(n_puffs):
            px = cx + np.random.uniform(-30, 30) * (0.5 + t * 0.5)
            py = cy + np.random.uniform(-20, 20)
            pr = int(12 + t * 40 + p * 7)
            # Smoke gets lighter and more transparent as it rises
            grey = int(GREY_SMOKE[0] + p * 10 * (1 - t))
            pa = int(max(0, 130 * (1 - t * 0.85) * np.random.uniform(0.5, 1.0)))
            col = (grey, grey - 10, grey - 20)
            data = radial_glow(data, px, py, 0, pr, col, alpha=pa)

        img = to_img(data)
        img = soft_blur(img, 3.5)
        frames.append(img)

    make_sheet(frames, os.path.join(SPRITES_DIR, "smoke_sheet.png"))


# ═════════════════════════════════════════════════════════════════════════════
# SPIRIT BOMB SPRITESHEET
# DBZ reference: Goku's spirit bomb — blue-white orb growing from fist-sized
# to planet-sized, with orbiting energy wisps swirling around equator,
# and streams of energy flowing inward from all directions
# ═════════════════════════════════════════════════════════════════════════════
def gen_spirit_bomb_sheet(n_frames=10, fw=128, fh=128):
    frames = []
    cx, cy = fw // 2, fh // 2

    for i in range(n_frames):
        t = i / (n_frames - 1)
        data = new_frame(fw, fh)

        r_orb = max(3, int(6 + t * 52))

        # Outer atmosphere halo
        data = radial_glow(
            data, cx, cy, r_orb * 0.8, r_orb * 2.4, BLUE_OUTER, alpha=int(100 * t)
        )

        # Mid glow
        data = radial_glow(
            data,
            cx,
            cy,
            r_orb * 0.5,
            r_orb * 1.3,
            BLUE_MID,
            alpha=int(200 * min(t + 0.2, 1)),
        )

        # Orb body — blue at edges, white at center
        d = dist_field(fw, fh, cx, cy)
        orb_m = np.clip(1 - d / max(r_orb, 1), 0, 1)
        ct = np.clip(d / max(r_orb * 0.6, 1), 0, 1)
        data[:, :, 0] += (255 * (1 - ct) + BLUE_MID[0] * ct) * orb_m * 240 / 255
        data[:, :, 1] += (255 * (1 - ct) + BLUE_MID[1] * ct) * orb_m * 240 / 255
        data[:, :, 2] += (255 * (1 - ct) + 255 * ct) * orb_m * 240 / 255
        data[:, :, 3] += orb_m * 240
        data = np.clip(data, 0, 255)

        # Equatorial ring (signature spirit bomb band)
        data = ring_glow(
            data,
            cx,
            cy,
            r_orb,
            max(1, int(r_orb * 0.15)),
            BLUE_CORE,
            alpha=int(180 * min(t + 0.1, 1)),
        )

        # Orbiting wisps (energy gathered from around)
        n_wisps = int(6 + t * 8)
        for w_idx in range(n_wisps):
            angle = (w_idx / n_wisps) * 2 * math.pi + t * math.pi * 2
            wr = r_orb * (1.1 + 0.2 * math.sin(w_idx * 1.7))
            wx = cx + math.cos(angle) * wr
            wy = cy + math.sin(angle) * wr * 0.55
            data = radial_glow(
                data, wx, wy, 0, max(2, int(5 * t)), (180, 220, 255), alpha=int(200 * t)
            )

        # Inward energy streams (ki being gathered)
        if t > 0.3:
            np.random.seed(i * 9)
            stream_t = (t - 0.3) / 0.7
            for _ in range(5):
                angle = np.random.uniform(0, 2 * math.pi)
                start_r = r_orb * (1.8 + np.random.uniform(0, 0.8))
                for step in range(12):
                    st = step / 11
                    sr = start_r * (1 - st * 0.7)
                    sx = cx + math.cos(angle) * sr
                    sy = cy + math.sin(angle) * sr
                    data = radial_glow(
                        data,
                        sx,
                        sy,
                        0,
                        3,
                        (180, 220, 255),
                        alpha=int(160 * stream_t * (1 - st)),
                    )

        img = to_img(data)
        img = soft_blur(img, 1.5)
        frames.append(img)

    make_sheet(frames, os.path.join(SPRITES_DIR, "spirit_bomb_sheet.png"))


# ═════════════════════════════════════════════════════════════════════════════
# TEXTURES (single PNG files)
# ═════════════════════════════════════════════════════════════════════════════


def gen_particle_glow():
    """White soft radial glow dot — used by particle system for all ki sparks."""
    SIZE = 64
    data = new_frame(SIZE, SIZE)
    cx = cy = SIZE // 2
    data = radial_glow(data, cx, cy, 0, SIZE // 2, (255, 255, 255), alpha=255)
    data = radial_glow(data, cx, cy, 0, SIZE // 4, (255, 255, 255), alpha=255)
    data = radial_glow(data, cx, cy, 0, SIZE // 8, (255, 255, 255), alpha=255)
    img = to_img(data)
    img = soft_blur(img, 1.5)
    img.save(os.path.join(TEXTURES_DIR, "particle_glow.png"))
    print(f"   particle_glow.png                          64x64")


def gen_colored_particles():
    """Coloured ki glow dots — white-hot core → colour → transparent."""
    for pname, c_inner, c_mid, c_out in [
        ("gold", GOLD_CORE, GOLD_MID, GOLD_OUTER),
        ("blue", BLUE_CORE, BLUE_MID, BLUE_OUTER),
        ("red", RED_CORE, RED_MID, RED_OUTER),
        ("purple", PURPLE_CORE, PURPLE_MID, PURPLE_OUT),
    ]:
        SIZE = 64
        data = new_frame(SIZE, SIZE)
        cx = cy = SIZE // 2
        data = radial_glow(data, cx, cy, SIZE // 3, SIZE // 2, c_out, alpha=120)
        data = radial_glow(data, cx, cy, SIZE // 6, SIZE // 3, c_mid, alpha=210)
        data = radial_glow(data, cx, cy, 0, SIZE // 6, c_inner, alpha=255)
        # Bright white hot core
        data = radial_glow(data, cx, cy, 0, SIZE // 12, (255, 255, 255), alpha=255)
        img = to_img(data)
        img = soft_blur(img, 1.5)
        img.save(os.path.join(TEXTURES_DIR, f"particle_{pname}.png"))
        print(f"  particle_{pname}.png                        64x64")


def gen_beam_core():
    """Horizontal beam texture — white core → cyan halo. Tileable horizontally."""
    W, H = 256, 64
    data = new_frame(W, H)
    cy = H // 2

    # Outer halo
    for y in range(H):
        dy = abs(y - cy)
        outer = math.exp(-(dy**2) / ((H * 0.38) ** 2))
        mid = math.exp(-(dy**2) / ((H * 0.18) ** 2))
        core = math.exp(-(dy**2) / ((H * 0.07) ** 2))
        for x in range(W):
            data[y, x, 0] = min(
                255, CYAN_OUTER[0] * outer + CYAN_MID[0] * mid + 255 * core
            )
            data[y, x, 1] = min(
                255, CYAN_OUTER[1] * outer + CYAN_MID[1] * mid + 255 * core
            )
            data[y, x, 2] = min(
                255, CYAN_OUTER[2] * outer + CYAN_MID[2] * mid + 255 * core
            )
            data[y, x, 3] = min(255, outer * 160 + mid * 220 + core * 255)

    to_img(data).save(os.path.join(TEXTURES_DIR, "beam_core.png"))
    print(f"  beam_core.png                             256x64")


def gen_beam_core_gold():
    """Gold beam texture — white core → gold → orange halo."""
    W, H = 256, 64
    data = new_frame(W, H)
    cy = H // 2
    for y in range(H):
        dy = abs(y - cy)
        outer = math.exp(-(dy**2) / ((H * 0.38) ** 2))
        mid = math.exp(-(dy**2) / ((H * 0.18) ** 2))
        core = math.exp(-(dy**2) / ((H * 0.07) ** 2))
        for x in range(W):
            data[y, x, 0] = min(
                255, GOLD_OUTER[0] * outer + GOLD_MID[0] * mid + 255 * core
            )
            data[y, x, 1] = min(
                255, GOLD_OUTER[1] * outer + GOLD_MID[1] * mid + 255 * core
            )
            data[y, x, 2] = min(
                255, GOLD_OUTER[2] * outer + GOLD_MID[2] * mid + 255 * core
            )
            data[y, x, 3] = min(255, outer * 160 + mid * 220 + core * 255)
    to_img(data).save(os.path.join(TEXTURES_DIR, "beam_core_gold.png"))
    print(f"  beam_core_gold.png                        256x64")


def gen_shockwave():
    """Single shockwave ring texture — expanding ring on transparent background."""
    SIZE = 256
    data = new_frame(SIZE, SIZE)
    cx = cy = SIZE // 2
    r_mid = SIZE * 0.42
    data = ring_glow(data, cx, cy, r_mid, SIZE * 0.06, CYAN_MID, alpha=220)
    data = ring_glow(data, cx, cy, r_mid, SIZE * 0.12, CYAN_OUTER, alpha=100)
    # Inner flash
    data = radial_glow(data, cx, cy, 0, SIZE * 0.35, CYAN_CORE, alpha=60)
    img = to_img(data)
    img = soft_blur(img, 2.0)
    img.save(os.path.join(TEXTURES_DIR, "shockwave.png"))
    print(f"   shockwave.png                             256x256")


# ═════════════════════════════════════════════════════════════════════════════
# HUD ASSETS
# ═════════════════════════════════════════════════════════════════════════════


def gen_hud_assets():
    # Energy bar background
    W, H = 240, 22
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=5, fill=(15, 15, 25, 210))
    draw.rounded_rectangle(
        [0, 0, W - 1, H - 1], radius=5, outline=(60, 60, 100, 255), width=1
    )
    img.save(os.path.join(HUD_DIR, "energy_bar_bg.png"))
    print(f"  energy_bar_bg.png                          {W}x{H}")

    # Energy bar fill — green(full) → gold(mid) → red(low), left to right
    iw, ih = W - 4, H - 4
    data = np.zeros((ih, iw, 4), dtype=np.uint8)
    for x in range(iw):
        t = x / iw
        if t > 0.6:
            r, g, b = (
                int(60 + (t - 0.6) / 0.4 * (255 - 60)),
                220,
                int(80 * (1 - (t - 0.6) / 0.4)),
            )
        elif t > 0.3:
            tt = (t - 0.3) / 0.3
            r, g, b = (
                int(255 * tt + GOLD_MID[0] * (1 - tt)),
                int(GOLD_MID[1]),
                int(GOLD_MID[2] * (1 - tt)),
            )
        else:
            tt = t / 0.3
            r, g, b = 255, int(60 * tt), 30
        data[:, x] = [r, g, b, 228]
    Image.fromarray(data, "RGBA").save(os.path.join(HUD_DIR, "energy_bar_fill.png"))
    print(f"   energy_bar_fill.png                        {iw}x{ih}")

    # Energy bar frame — glowing gold border
    img3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(img3).rounded_rectangle(
        [0, 0, W - 1, H - 1], radius=5, outline=(*GOLD_MID, 200), width=2
    )
    img3_b = img3.filter(ImageFilter.GaussianBlur(1.5))
    Image.alpha_composite(img3_b, img3).save(
        os.path.join(HUD_DIR, "energy_bar_frame.png")
    )
    print(f"   energy_bar_frame.png                       {W}x{H}")

    # Cooldown ring
    SIZE = 64
    data = new_frame(SIZE, SIZE)
    cx = cy = SIZE // 2
    data = ring_glow(data, cx, cy, SIZE * 0.41, SIZE * 0.07, GOLD_MID, alpha=200)
    to_img(data).save(os.path.join(HUD_DIR, "cooldown_ring.png"))
    print(f"   cooldown_ring.png                          64x64")

    # Crosshair
    SIZE = 32
    img4 = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d4 = ImageDraw.Draw(img4)
    cx4 = cy4 = SIZE // 2
    gap = 4
    d4.line([(cx4 - 12, cy4), (cx4 - gap, cy4)], fill=(*CYAN_MID, 200), width=2)
    d4.line([(cx4 + gap, cy4), (cx4 + 12, cy4)], fill=(*CYAN_MID, 200), width=2)
    d4.line([(cx4, cy4 - 12), (cx4, cy4 - gap)], fill=(*CYAN_MID, 200), width=2)
    d4.line([(cx4, cy4 + gap), (cx4, cy4 + 12)], fill=(*CYAN_MID, 200), width=2)
    d4.ellipse([cx4 - 3, cy4 - 3, cx4 + 3, cy4 + 3], outline=(*CYAN_MID, 200), width=1)
    img4.save(os.path.join(HUD_DIR, "crosshair.png"))
    print(f"  crosshair.png                              32x32")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  DBZ Gesture Vision Engine — Asset Generator")
    print("=" * 62)

    print("\n  Aura Spritesheets (SSJ upward flame spikes):")
    gen_aura_sheet("gold", GOLD_CORE, GOLD_MID, GOLD_OUTER)
    gen_aura_sheet("blue", BLUE_CORE, BLUE_MID, BLUE_OUTER)
    gen_aura_sheet("red", RED_CORE, RED_MID, RED_OUTER)
    gen_aura_sheet("purple", PURPLE_CORE, PURPLE_MID, PURPLE_OUT)
    gen_aura_sheet("white", (255, 255, 255), (240, 240, 255), (180, 180, 220))

    print("\n  Beam Spritesheets (tapered ki beams):")
    gen_beam_sheet("blue", BLUE_CORE, BLUE_MID, BLUE_OUTER)
    gen_beam_sheet("gold", GOLD_CORE, GOLD_MID, GOLD_OUTER)
    gen_beam_sheet("white", (255, 255, 255), (230, 240, 255), (160, 180, 220))
    gen_beam_sheet("red", RED_CORE, RED_MID, RED_OUTER)

    print("\n Explosion Spritesheets (flash → ring → debris):")
    gen_explosion_sheet("gold", GOLD_CORE, GOLD_MID, GOLD_OUTER)
    gen_explosion_sheet("blue", BLUE_CORE, BLUE_MID, BLUE_OUTER)
    gen_explosion_sheet("white", (255, 255, 255), (220, 230, 255), (150, 160, 200))

    print("\n Ring Spritesheets (power-up expanding rings):")
    gen_ring_sheet("gold", GOLD_CORE, GOLD_MID, GOLD_OUTER)
    gen_ring_sheet("blue", BLUE_CORE, BLUE_MID, BLUE_OUTER)
    gen_ring_sheet("purple", PURPLE_CORE, PURPLE_MID, PURPLE_OUT)

    print("\n  Spark Spritesheets (SSJ2 lightning arcs):")
    gen_spark_sheet("gold", GOLD_CORE, GOLD_MID)
    gen_spark_sheet("blue", BLUE_CORE, BLUE_MID)
    gen_spark_sheet("white", (255, 255, 255), (220, 230, 255))

    print("\n🌫️   Other Spritesheets:")
    gen_smoke_sheet()
    gen_spirit_bomb_sheet()

    print("\n  Textures (single PNGs):")
    gen_particle_glow()
    gen_colored_particles()
    gen_beam_core()
    gen_beam_core_gold()
    gen_shockwave()

    print("\n HUD Assets:")
    gen_hud_assets()

    sprites = len(os.listdir(SPRITES_DIR))
    textures = len(os.listdir(TEXTURES_DIR))
    hud = len(os.listdir(HUD_DIR))
    total = sprites + textures + hud
    print(f"\n{'='*62}")
    print(f"  {sprites} spritesheets   {textures} textures   {hud} HUD assets")
    print(f"  {total} total files generated")
    print(f"{'='*62}\n")
