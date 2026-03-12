"""
generate_icons_dbz.py
---------------------
Generates DBZ-accurate gesture icons based on actual DBZ anime/game references:

  CHARGING    — Two fists at sides, gold ki orb forming between palms (classic charge pose)
  FIRING      — Arm extended forward, open palm, blue ki blast at fingertips
  KAMEHAMEHA  — Both hands cupped side-by-side pushed forward, cyan beam erupting
  SPIRIT BOMB — One arm raised, massive blue-white orb above with orbiting wisps
  POWER UP    — Super Saiyan golden spiky aura explosion around figure
  TELEPORT    — Goku Instant Transmission: two fingers on forehead, green flash
  BLOCK       — Arms crossed in X, purple ki barrier dome with impact sparks

All icons: 128x128 RGBA PNG, transparent background.
Run from your project root:  python generate_icons_dbz.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math, os

# ── Output path ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICONS_DIR = os.path.join(BASE_DIR, "assets", "ui", "icons")
os.makedirs(ICONS_DIR, exist_ok=True)

SIZE = 128
CX = SIZE // 2
CY = SIZE // 2

# ── DBZ anime-accurate colour palette ───────────────────────────────────────
GOLD_CORE = (255, 255, 210)
GOLD_MID = (255, 210, 30)
GOLD_OUTER = (255, 130, 0)
BLUE_CORE = (220, 245, 255)
BLUE_MID = (60, 160, 255)
BLUE_OUTER = (0, 60, 180)
CYAN_CORE = (200, 255, 255)
CYAN_MID = (80, 220, 255)
CYAN_OUTER = (0, 120, 200)
GREEN_MID = (80, 255, 160)
PURPLE_MID = (180, 60, 255)
WHITE = (255, 255, 255)
SKIN = (220, 175, 120)
HAIR = (20, 20, 20)
ORANGE_GI = (255, 120, 20)
BLUE_GI = (30, 60, 160)
DARK = (15, 15, 30)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def blank():
    return Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))


def arr_blank():
    return np.zeros((SIZE, SIZE, 4), dtype=np.float32)


def dist_field(cx=CX, cy=CY):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)


def paint_orb(data, cx, cy, r, c_inner, c_mid, c_outer):
    d = dist_field(cx, cy)
    halo = np.clip(1 - d / (r * 2.4), 0, 1) ** 2.5
    mid = np.clip(1 - d / (r * 1.15), 0, 1) ** 2.0
    core = np.clip(1 - d / max(r * 0.45, 1), 0, 1) ** 1.4
    for col, mask, alpha in [
        (c_outer, halo, 120),
        (c_mid, mid, 210),
        (c_inner, core, 255),
    ]:
        data[:, :, 0] = np.minimum(255, data[:, :, 0] + col[0] * mask * alpha / 255)
        data[:, :, 1] = np.minimum(255, data[:, :, 1] + col[1] * mask * alpha / 255)
        data[:, :, 2] = np.minimum(255, data[:, :, 2] + col[2] * mask * alpha / 255)
        data[:, :, 3] = np.minimum(255, data[:, :, 3] + mask * alpha)
    data[:, :, 0] = np.minimum(255, data[:, :, 0] + c_inner[0] * core)
    data[:, :, 1] = np.minimum(255, data[:, :, 1] + c_inner[1] * core)
    data[:, :, 2] = np.minimum(255, data[:, :, 2] + c_inner[2] * core)
    data[:, :, 3] = np.minimum(255, data[:, :, 3] + core * 255)
    return data


def paint_ring(data, cx, cy, r, thick, color, alpha=200):
    d = dist_field(cx, cy)
    ring = np.clip(1 - np.abs(d - r) / max(thick, 1), 0, 1) ** 2
    glow = np.clip(1 - np.abs(d - r) / max(thick * 2.5, 1), 0, 1) ** 3
    data[:, :, 0] = np.minimum(255, data[:, :, 0] + color[0] * (ring + glow * 0.35))
    data[:, :, 1] = np.minimum(255, data[:, :, 1] + color[1] * (ring + glow * 0.35))
    data[:, :, 2] = np.minimum(255, data[:, :, 2] + color[2] * (ring + glow * 0.35))
    data[:, :, 3] = np.minimum(255, data[:, :, 3] + ring * alpha + glow * alpha * 0.35)
    return data


def point_glow(data, cx, cy, r, color, alpha=255):
    d = dist_field(cx, cy)
    m = np.clip(1 - d / max(r, 1), 0, 1) ** 2
    data[:, :, 0] = np.minimum(255, data[:, :, 0] + color[0] * m * alpha / 255)
    data[:, :, 1] = np.minimum(255, data[:, :, 1] + color[1] * m * alpha / 255)
    data[:, :, 2] = np.minimum(255, data[:, :, 2] + color[2] * m * alpha / 255)
    data[:, :, 3] = np.minimum(255, data[:, :, 3] + m * alpha)
    return data


def jagged_line(draw, x1, y1, x2, y2, color, width=2, jitter=5, seed=0, segs=10):
    np.random.seed(seed)
    pts = [(x1, y1)]
    for s in range(1, segs):
        t = s / segs
        mx = x1 + (x2 - x1) * t + np.random.uniform(-jitter, jitter) * (1 - t)
        my = y1 + (y2 - y1) * t + np.random.uniform(-jitter, jitter) * (1 - t)
        pts.append((int(mx), int(my)))
    pts.append((x2, y2))
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(*color, 200), width=width + 1)
        draw.line(
            [pts[i], pts[i + 1]], fill=(255, 255, 255, 160), width=max(1, width - 1)
        )


def apply_glow(img, color, radius=13, strength=2.0):
    r, g, b = color[:3]
    alpha = img.split()[3]
    blurred = alpha.filter(ImageFilter.GaussianBlur(radius))
    arr = np.array(blurred).astype(np.float32) / 255.0
    glow = np.zeros((SIZE, SIZE, 4), dtype=np.float32)
    glow[:, :, 0] = r * arr
    glow[:, :, 1] = g * arr
    glow[:, :, 2] = b * arr
    glow[:, :, 3] = np.clip(arr * 255 * strength, 0, 255)
    return Image.alpha_composite(Image.fromarray(glow.astype(np.uint8), "RGBA"), img)


def to_img(data):
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), "RGBA")


def save(img, name, glow_col):
    img = apply_glow(img, glow_col, radius=11, strength=1.9)
    path = os.path.join(ICONS_DIR, f"{name}.png")
    img.save(path, "PNG")
    print(f"    {name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CHARGING
# ═══════════════════════════════════════════════════════════════════════════
def gen_charging():
    data = arr_blank()
    data = paint_orb(
        data, CX, CY, r=26, c_inner=GOLD_CORE, c_mid=GOLD_MID, c_outer=GOLD_OUTER
    )
    data = paint_ring(data, CX, CY, r=34, thick=4, color=GOLD_MID, alpha=160)
    data = paint_ring(data, CX, CY, r=46, thick=3, color=GOLD_OUTER, alpha=80)
    img = to_img(data)
    draw = ImageDraw.Draw(img)
    # Left fist
    draw.rounded_rectangle(
        [8, 52, 36, 78], radius=8, fill=(*SKIN, 240), outline=(*DARK, 255), width=2
    )
    draw.rounded_rectangle(
        [8, 58, 34, 74], radius=7, fill=(*ORANGE_GI, 220), outline=(*DARK, 255), width=2
    )
    draw.ellipse([8, 52, 22, 66], fill=(*SKIN, 230), outline=(*DARK, 200), width=1)
    for ky in [59, 64, 69, 74]:
        draw.line([(11, ky), (33, ky)], fill=(*DARK, 70), width=1)
    # Right fist
    draw.rounded_rectangle(
        [92, 52, 120, 78], radius=8, fill=(*SKIN, 240), outline=(*DARK, 255), width=2
    )
    draw.rounded_rectangle(
        [94, 58, 120, 74],
        radius=7,
        fill=(*ORANGE_GI, 220),
        outline=(*DARK, 255),
        width=2,
    )
    draw.ellipse([106, 52, 120, 66], fill=(*SKIN, 230), outline=(*DARK, 200), width=1)
    for ky in [59, 64, 69, 74]:
        draw.line([(95, ky), (117, ky)], fill=(*DARK, 70), width=1)
    # Lightning arcs
    jagged_line(draw, 36, 64, CX - 10, CY, GOLD_CORE, width=2, jitter=5, seed=1)
    jagged_line(draw, 92, 64, CX + 10, CY, GOLD_CORE, width=2, jitter=5, seed=2)
    jagged_line(draw, 36, 68, CX - 8, CY + 4, GOLD_MID, width=1, jitter=4, seed=3)
    jagged_line(draw, 92, 68, CX + 8, CY + 4, GOLD_MID, width=1, jitter=4, seed=4)
    # Ki sparks around orb
    for i, angle in enumerate(range(0, 360, 40)):
        ax = int(CX + math.cos(math.radians(angle)) * 24)
        ay = int(CY + math.sin(math.radians(angle)) * 24)
        r = 2 + (i % 2)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(*GOLD_CORE, 230))
    save(img, "gesture_charging", GOLD_MID)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FIRING
# ═══════════════════════════════════════════════════════════════════════════
def gen_firing():
    data = arr_blank()
    data = paint_orb(
        data, 96, CY, r=20, c_inner=BLUE_CORE, c_mid=BLUE_MID, c_outer=BLUE_OUTER
    )
    for x in range(10, 78):
        t = (x - 10) / 68
        for dy in range(-5, 6):
            y = CY + dy
            if 0 <= y < SIZE:
                fade = t * max(0, 1 - abs(dy) / 6) * 0.55
                data[y, x, 0] = min(255, data[y, x, 0] + BLUE_MID[0] * fade)
                data[y, x, 1] = min(255, data[y, x, 1] + BLUE_MID[1] * fade)
                data[y, x, 2] = min(255, data[y, x, 2] + BLUE_MID[2] * fade)
                data[y, x, 3] = min(255, data[y, x, 3] + 180 * fade)
    img = to_img(data)
    draw = ImageDraw.Draw(img)
    # Arm
    draw.rounded_rectangle(
        [10, 57, 80, 71],
        radius=9,
        fill=(*ORANGE_GI, 230),
        outline=(*DARK, 255),
        width=2,
    )
    # Open palm
    draw.rounded_rectangle(
        [72, 51, 96, 77], radius=7, fill=(*SKIN, 240), outline=(*DARK, 255), width=2
    )
    for i, fy in enumerate([53, 58, 63, 68]):
        draw.rounded_rectangle(
            [88, fy, 104, fy + 7],
            radius=3,
            fill=(*SKIN, 230),
            outline=(*DARK, 200),
            width=1,
        )
    draw.rounded_rectangle(
        [74, 73, 90, 82], radius=4, fill=(*SKIN, 230), outline=(*DARK, 200), width=1
    )
    # Palm glow
    data2 = np.array(img).astype(np.float32)
    data2 = point_glow(data2, 88, CY, 14, BLUE_CORE, alpha=200)
    img = to_img(data2)
    draw = ImageDraw.Draw(img)
    # Motion lines
    for i, yl in enumerate([54, 58, 62, 66, 70, 74]):
        draw.line(
            [(8, yl), (8 + 22 - i * 2, yl)], fill=(*BLUE_CORE, 140 - i * 18), width=1
        )
    save(img, "gesture_firing", BLUE_MID)


# ═══════════════════════════════════════════════════════════════════════════
# 3. KAMEHAMEHA
# ═══════════════════════════════════════════════════════════════════════════
def gen_kamehameha():
    data = arr_blank()
    BEAM_CY = CY + 8

    # Outer beam glow
    for y in range(SIZE):
        dy = abs(y - BEAM_CY)
        outer = math.exp(-(dy**2) / (20**2)) * 0.65
        if outer > 0.01:
            for x in range(50, SIZE):
                ramp = min(1.0, (x - 50) / 18)
                b = outer * ramp
                data[y, x, 0] = min(255, data[y, x, 0] + CYAN_MID[0] * b)
                data[y, x, 1] = min(255, data[y, x, 1] + CYAN_MID[1] * b)
                data[y, x, 2] = min(255, data[y, x, 2] + CYAN_MID[2] * b)
                data[y, x, 3] = min(255, data[y, x, 3] + 210 * b)

    # Bright white-cyan beam core
    for y in range(SIZE):
        dy = abs(y - BEAM_CY)
        core = math.exp(-(dy**2) / (6**2))
        if core > 0.01:
            for x in range(46, SIZE):
                data[y, x, 0] = min(255, data[y, x, 0] + 255 * core)
                data[y, x, 1] = min(255, data[y, x, 1] + 255 * core)
                data[y, x, 2] = min(255, data[y, x, 2] + 255 * core)
                data[y, x, 3] = min(255, data[y, x, 3] + 255 * core)

    # Orb at palm origin
    data = paint_orb(
        data,
        44,
        BEAM_CY,
        r=18,
        c_inner=(255, 255, 255),
        c_mid=CYAN_MID,
        c_outer=CYAN_OUTER,
    )

    img = to_img(data)
    draw = ImageDraw.Draw(img)

    # Upper cupped hand
    draw.rounded_rectangle(
        [10, 28, 50, 52], radius=9, fill=(*SKIN, 240), outline=(*DARK, 255), width=2
    )
    draw.ellipse([10, 24, 26, 40], fill=(*SKIN, 220), outline=(*DARK, 220), width=2)
    for fx in [18, 26, 34, 42]:
        draw.line([(fx, 30), (fx, 50)], fill=(*DARK, 90), width=1)
    draw.rounded_rectangle(
        [10, 48, 50, 60], radius=4, fill=(*BLUE_GI, 220), outline=(*DARK, 255), width=2
    )

    # Lower cupped hand
    draw.rounded_rectangle(
        [10, 62, 50, 86], radius=9, fill=(*SKIN, 240), outline=(*DARK, 255), width=2
    )
    draw.ellipse([10, 72, 26, 88], fill=(*SKIN, 220), outline=(*DARK, 220), width=2)
    for fx in [18, 26, 34, 42]:
        draw.line([(fx, 64), (fx, 84)], fill=(*DARK, 90), width=1)
    draw.rounded_rectangle(
        [10, 56, 50, 66], radius=4, fill=(*BLUE_GI, 220), outline=(*DARK, 255), width=2
    )

    # Energy crackling between palms
    jagged_line(draw, 46, 50, 46, 62, (255, 255, 255), width=1, jitter=3, seed=7)
    jagged_line(draw, 40, 50, 42, 62, CYAN_MID, width=1, jitter=4, seed=8)

    # Beam wave lines
    for yl in [BEAM_CY - 18, BEAM_CY - 12, BEAM_CY + 12, BEAM_CY + 18]:
        draw.line([(55, yl), (SIZE - 4, yl)], fill=(*CYAN_MID, 60), width=1)

    save(img, "gesture_kamehameha", CYAN_MID)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SPIRIT BOMB
# ═══════════════════════════════════════════════════════════════════════════
def gen_spirit_bomb():
    data = arr_blank()
    ORB_CX, ORB_CY, ORB_R = CX, 32, 28

    d = dist_field(ORB_CX, ORB_CY)
    halo = np.clip(1 - d / (ORB_R * 2.8), 0, 1) ** 2.5
    mid = np.clip(1 - d / (ORB_R * 1.3), 0, 1) ** 2
    core = np.clip(1 - d / (ORB_R * 0.6), 0, 1) ** 1.5
    inner = np.clip(1 - d / (ORB_R * 0.25), 0, 1) ** 1.2

    data[:, :, 0] += BLUE_OUTER[0] * halo * 0.5
    data[:, :, 1] += BLUE_OUTER[1] * halo * 0.5
    data[:, :, 2] += BLUE_OUTER[2] * halo * 0.5
    data[:, :, 3] += halo * 100

    data[:, :, 0] += BLUE_MID[0] * mid * 0.9
    data[:, :, 1] += BLUE_MID[1] * mid * 0.9
    data[:, :, 2] += BLUE_MID[2] * mid * 0.9
    data[:, :, 3] += mid * 210

    ct = np.clip(d / (ORB_R * 0.5), 0, 1)
    data[:, :, 0] += (255 * (1 - ct) + BLUE_CORE[0] * ct) * core
    data[:, :, 1] += (255 * (1 - ct) + BLUE_CORE[1] * ct) * core
    data[:, :, 2] += (255 * (1 - ct) + 255 * ct) * core
    data[:, :, 3] += core * 245

    data[:, :, 0] += 255 * inner
    data[:, :, 1] += 255 * inner
    data[:, :, 2] += 255 * inner
    data[:, :, 3] += inner * 255

    # Orbiting wisps
    for i in range(14):
        angle = (i / 14) * 2 * math.pi
        wr = ORB_R * (1.05 + 0.18 * math.sin(i * 1.9))
        wx = ORB_CX + math.cos(angle) * wr
        wy = ORB_CY + math.sin(angle) * wr * 0.55
        data = point_glow(data, wx, wy, 5, (180, 220, 255), alpha=180)

    data = paint_ring(
        data, ORB_CX, ORB_CY, r=ORB_R + 1, thick=3, color=BLUE_CORE, alpha=140
    )
    img = to_img(data)
    draw = ImageDraw.Draw(img)

    # Body
    draw.rounded_rectangle(
        [CX - 10, 70, CX + 10, 100],
        radius=5,
        fill=(*ORANGE_GI, 230),
        outline=(*DARK, 255),
        width=2,
    )
    draw.rectangle([CX - 10, 86, CX + 10, 92], fill=(*DARK, 200))

    # Right arm raised straight up
    draw.rounded_rectangle(
        [CX + 8, 52, CX + 18, 78],
        radius=5,
        fill=(*ORANGE_GI, 220),
        outline=(*DARK, 255),
        width=2,
    )
    draw.rounded_rectangle(
        [CX + 10, 38, CX + 18, 60],
        radius=4,
        fill=(*SKIN, 230),
        outline=(*DARK, 255),
        width=2,
    )
    draw.ellipse(
        [CX + 6, 52, CX + 22, 66], fill=(*SKIN, 230), outline=(*DARK, 255), width=2
    )

    # Left arm relaxed
    draw.rounded_rectangle(
        [CX - 18, 70, CX - 8, 96],
        radius=5,
        fill=(*ORANGE_GI, 220),
        outline=(*DARK, 255),
        width=2,
    )

    # Energy stream to orb
    jagged_line(
        draw,
        CX + 14,
        52,
        ORB_CX + 4,
        ORB_CY + ORB_R,
        BLUE_CORE,
        width=2,
        jitter=3,
        seed=10,
    )
    save(img, "gesture_spirit_bomb", BLUE_MID)


# ═══════════════════════════════════════════════════════════════════════════
# 5. POWER UP
# ═══════════════════════════════════════════════════════════════════════════
def gen_power_up():
    data = arr_blank()
    np.random.seed(42)

    # Spiky SSJ aura
    for s in range(18):
        angle = (s / 18) * 2 * math.pi + np.random.uniform(-0.1, 0.1)
        length = 44 + np.random.uniform(-8, 14)
        base_w = np.random.uniform(7, 14)
        for step in range(32):
            tt = step / 31
            r = tt * length
            sx = CX + math.cos(angle) * r
            sy = CY + math.sin(angle) * r
            sw = base_w * (1 - tt) ** 0.7
            d = dist_field(sx, sy)
            m = np.clip(1 - d / max(sw, 1), 0, 1) ** 2
            t_c = tt**0.5
            cr = GOLD_MID[0] + (GOLD_CORE[0] - GOLD_MID[0]) * t_c
            cg = GOLD_MID[1] + (GOLD_CORE[1] - GOLD_MID[1]) * t_c
            cb = GOLD_MID[2] + (GOLD_CORE[2] - GOLD_MID[2]) * t_c
            al = (1 - tt * 0.75) * 240
            data[:, :, 0] = np.minimum(255, data[:, :, 0] + cr * m * al / 255)
            data[:, :, 1] = np.minimum(255, data[:, :, 1] + cg * m * al / 255)
            data[:, :, 2] = np.minimum(255, data[:, :, 2] + cb * m * al / 255)
            data[:, :, 3] = np.minimum(255, data[:, :, 3] + m * al)

    data = paint_orb(
        data, CX, CY, r=20, c_inner=GOLD_CORE, c_mid=GOLD_MID, c_outer=GOLD_OUTER
    )
    data = paint_ring(data, CX, CY, r=24, thick=4, color=GOLD_MID, alpha=160)
    img = to_img(data)
    draw = ImageDraw.Draw(img)

    # Spiky golden SSJ hair
    spike_col = (*GOLD_MID, 240)
    draw.polygon([(CX - 5, CY - 20), (CX + 5, CY - 20), (CX, CY - 40)], fill=spike_col)
    draw.polygon(
        [(CX - 14, CY - 18), (CX - 6, CY - 18), (CX - 12, CY - 34)], fill=spike_col
    )
    draw.polygon(
        [(CX + 6, CY - 18), (CX + 14, CY - 18), (CX + 12, CY - 34)], fill=spike_col
    )
    draw.polygon(
        [(CX - 18, CY - 14), (CX - 10, CY - 14), (CX - 16, CY - 28)], fill=spike_col
    )
    draw.polygon(
        [(CX + 10, CY - 14), (CX + 18, CY - 14), (CX + 16, CY - 28)], fill=spike_col
    )

    # Head
    draw.ellipse(
        [CX - 10, CY - 20, CX + 10, CY - 4],
        fill=(*SKIN, 240),
        outline=(*DARK, 255),
        width=2,
    )
    # Intense glowing eyes
    draw.ellipse([CX - 7, CY - 15, CX - 2, CY - 11], fill=(255, 255, 200, 255))
    draw.ellipse([CX + 2, CY - 15, CX + 7, CY - 11], fill=(255, 255, 200, 255))
    # Body
    draw.rounded_rectangle(
        [CX - 8, CY - 4, CX + 8, CY + 14],
        radius=3,
        fill=(*ORANGE_GI, 230),
        outline=(*DARK, 255),
        width=2,
    )
    # Arms spread back
    draw.line([(CX - 8, CY + 2), (CX - 24, CY + 10)], fill=(*ORANGE_GI, 230), width=5)
    draw.line([(CX + 8, CY + 2), (CX + 24, CY + 10)], fill=(*ORANGE_GI, 230), width=5)
    save(img, "gesture_power_up", GOLD_MID)


# ═══════════════════════════════════════════════════════════════════════════
# 6. TELEPORT (Instant Transmission)
# ═══════════════════════════════════════════════════════════════════════════
def gen_teleport():
    data = arr_blank()
    FOCUS_X, FOCUS_Y = CX + 14, 42

    # IT flash rings
    for r_ring, al in [(50, 60), (38, 90), (24, 120), (14, 150)]:
        data = paint_ring(
            data, FOCUS_X, FOCUS_Y, r=r_ring, thick=4, color=GREEN_MID, alpha=al
        )

    data = paint_orb(
        data,
        FOCUS_X,
        FOCUS_Y,
        r=12,
        c_inner=(255, 255, 255),
        c_mid=GREEN_MID,
        c_outer=(0, 180, 80),
    )

    # Speed warp streaks
    for angle_deg in range(0, 360, 22):
        angle = math.radians(angle_deg)
        for step in range(18):
            t = step / 17
            r = 14 + t * 46
            sx = int(FOCUS_X + math.cos(angle) * r)
            sy = int(FOCUS_Y + math.sin(angle) * r)
            if 0 <= sx < SIZE and 0 <= sy < SIZE:
                fade = (1 - t) * 0.6
                data[sy, sx, 0] = min(255, data[sy, sx, 0] + 255 * fade)
                data[sy, sx, 1] = min(255, data[sy, sx, 1] + 255 * fade)
                data[sy, sx, 2] = min(255, data[sy, sx, 2] + 255 * fade)
                data[sy, sx, 3] = min(255, data[sy, sx, 3] + 200 * fade)

    img = to_img(data)
    draw = ImageDraw.Draw(img)

    # Spiky dark hair
    for hx, hy, hw in [
        (CX - 8, 14, 5),
        (CX - 2, 10, 5),
        (CX + 4, 12, 5),
        (CX + 10, 16, 4),
    ]:
        draw.polygon([(hx - hw, 28), (hx + hw, 28), (hx, hy)], fill=(*HAIR, 240))

    # Head
    draw.ellipse(
        [CX - 14, 22, CX + 14, 52], fill=(*SKIN, 240), outline=(*DARK, 255), width=2
    )
    # Closed eyes (concentration)
    draw.line([(CX - 9, 36), (CX - 3, 36)], fill=(*DARK, 220), width=2)
    draw.line([(CX + 3, 36), (CX + 9, 36)], fill=(*DARK, 220), width=2)
    # Calm mouth
    draw.arc([CX - 5, 40, CX + 5, 46], start=0, end=180, fill=(*DARK, 180), width=1)

    # Body
    draw.rounded_rectangle(
        [CX - 12, 52, CX + 12, 90],
        radius=5,
        fill=(*ORANGE_GI, 225),
        outline=(*DARK, 255),
        width=2,
    )
    # Right arm raising to head
    draw.line([(CX + 10, 62), (CX + 20, 44)], fill=(*ORANGE_GI, 200), width=7)
    draw.line([(CX + 10, 62), (CX + 20, 44)], fill=(*SKIN, 160), width=5)
    # Wrist / hand
    draw.ellipse(
        [CX + 14, 36, CX + 26, 50], fill=(*SKIN, 235), outline=(*DARK, 220), width=2
    )
    # Two fingers on forehead (index + middle)
    draw.rounded_rectangle(
        [CX + 14, 28, CX + 20, 42],
        radius=3,
        fill=(*SKIN, 235),
        outline=(*DARK, 200),
        width=1,
    )
    draw.rounded_rectangle(
        [CX + 21, 28, CX + 27, 42],
        radius=3,
        fill=(*SKIN, 235),
        outline=(*DARK, 200),
        width=1,
    )
    # Left arm relaxed
    draw.line([(CX - 10, 60), (CX - 22, 82)], fill=(*ORANGE_GI, 220), width=6)
    draw.ellipse(
        [CX - 28, 78, CX - 16, 90], fill=(*SKIN, 220), outline=(*DARK, 200), width=2
    )

    # Fingertip glow
    data2 = np.array(img).astype(np.float32)
    data2 = point_glow(data2, CX + 19, 34, 10, GREEN_MID, alpha=200)
    img = to_img(data2)
    save(img, "gesture_teleport", GREEN_MID)


# ═══════════════════════════════════════════════════════════════════════════
# 7. BLOCK
# ═══════════════════════════════════════════════════════════════════════════
def gen_block():
    data = arr_blank()
    d = dist_field(CX, CY + 6)
    shield = np.clip(1 - np.abs(d - 50) / 8, 0, 1) ** 2
    inner = np.clip(1 - np.abs(d - 46) / 5, 0, 1) ** 2
    fill = np.clip(1 - d / 52, 0, 1) ** 3 * 0.18

    data[:, :, 0] += PURPLE_MID[0] * shield
    data[:, :, 1] += PURPLE_MID[1] * shield
    data[:, :, 2] += PURPLE_MID[2] * shield
    data[:, :, 3] += shield * 210
    data[:, :, 0] += 255 * inner * 0.6
    data[:, :, 1] += 180 * inner * 0.6
    data[:, :, 2] += 255 * inner * 0.6
    data[:, :, 3] += inner * 160
    data[:, :, 0] += PURPLE_MID[0] * fill
    data[:, :, 1] += PURPLE_MID[1] * fill
    data[:, :, 2] += PURPLE_MID[2] * fill
    data[:, :, 3] += fill * 200

    # Impact sparks on shield
    for angle_deg in [20, 55, 95, 135, 170, 215, 270, 320]:
        angle = math.radians(angle_deg)
        ix = CX + math.cos(angle) * 48
        iy = CY + 6 + math.sin(angle) * 48
        data = point_glow(data, ix, iy, 7, (255, 255, 255), alpha=230)
        for step in range(8):
            t = step / 7
            sx = int(ix + math.cos(angle) * t * 12)
            sy = int(iy + math.sin(angle) * t * 12)
            if 0 <= sx < SIZE and 0 <= sy < SIZE:
                data[sy, sx, 0] = min(255, data[sy, sx, 0] + 255 * (1 - t))
                data[sy, sx, 1] = min(255, data[sy, sx, 1] + 220 * (1 - t))
                data[sy, sx, 2] = min(255, data[sy, sx, 2] + 255 * (1 - t))
                data[sy, sx, 3] = min(255, data[sy, sx, 3] + 220 * (1 - t))

    img = to_img(data)
    draw = ImageDraw.Draw(img)

    # Crossed arms X
    draw.line([(CX - 30, 28), (CX + 30, 90)], fill=(*ORANGE_GI, 240), width=18)
    draw.line([(CX - 30, 28), (CX + 30, 90)], fill=(*DARK, 180), width=1)
    draw.line([(CX + 30, 28), (CX - 30, 90)], fill=(*ORANGE_GI, 240), width=18)
    draw.line([(CX + 30, 28), (CX - 30, 90)], fill=(*DARK, 180), width=1)
    # Cel-shade highlight
    draw.line([(CX - 28, 30), (CX + 28, 88)], fill=(255, 200, 100, 100), width=5)
    draw.line([(CX + 28, 30), (CX - 28, 88)], fill=(255, 200, 100, 100), width=5)
    # Fists at top corners
    for fx, fy in [(CX - 30, 26), (CX + 30, 26)]:
        draw.ellipse(
            [fx - 10, fy - 10, fx + 10, fy + 10],
            fill=(*SKIN, 240),
            outline=(*DARK, 255),
            width=2,
        )
        draw.arc(
            [fx - 8, fy - 6, fx + 8, fy + 2],
            start=200,
            end=340,
            fill=(*DARK, 120),
            width=1,
        )

    # Center ki burst at X intersection
    data2 = np.array(img).astype(np.float32)
    data2 = paint_orb(
        data2,
        CX,
        CY + 10,
        r=12,
        c_inner=(255, 255, 255),
        c_mid=(220, 150, 255),
        c_outer=PURPLE_MID,
    )
    img = to_img(data2)
    save(img, "gesture_block", PURPLE_MID)


# ═══════════════════════════════════════════════════════════════════════════
# PREVIEW
# ═══════════════════════════════════════════════════════════════════════════
def gen_preview():
    entries = [
        ("gesture_charging", "CHARGING", GOLD_MID),
        ("gesture_firing", "FIRING", BLUE_MID),
        ("gesture_kamehameha", "KAMEHAMEHA", CYAN_MID),
        ("gesture_spirit_bomb", "SPIRIT BOMB", BLUE_MID),
        ("gesture_power_up", "POWER UP", GOLD_MID),
        ("gesture_teleport", "TELEPORT", GREEN_MID),
        ("gesture_block", "BLOCK", PURPLE_MID),
    ]
    COLS = 4
    ROWS = math.ceil(len(entries) / COLS)
    PAD = 16
    LBL_H = 20
    CW = SIZE + PAD * 2
    CH = SIZE + LBL_H + PAD * 2
    canvas = Image.new("RGB", (COLS * CW, ROWS * CH), (10, 10, 20))
    draw = ImageDraw.Draw(canvas)
    for idx, (fname, label, color) in enumerate(entries):
        col = idx % COLS
        row = idx // COLS
        x = col * CW + PAD
        y = row * CH + PAD
        draw.rounded_rectangle(
            [col * CW + 4, row * CH + 4, col * CW + CW - 4, row * CH + CH - 4],
            radius=10,
            fill=(22, 22, 36),
        )
        path = os.path.join(ICONS_DIR, f"{fname}.png")
        if os.path.exists(path):
            icon = Image.open(path).convert("RGBA")
            canvas.paste(icon, (x, y), icon)
        draw.text(
            (x + SIZE // 2, y + SIZE + 4), label, fill=tuple(color[:3]), anchor="mt"
        )
    canvas.save(os.path.join(ICONS_DIR, "PREVIEW_ICONS.png"))
    print(f"\n  PREVIEW_ICONS.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  DBZ Gesture Vision Engine — Icon Generator v2")
    print("=" * 56 + "\n")
    gen_charging()
    gen_firing()
    gen_kamehameha()
    gen_spirit_bomb()
    gen_power_up()
    gen_teleport()
    gen_block()
    gen_preview()
    n = len([f for f in os.listdir(ICONS_DIR) if f.endswith(".png")])
    print(f"\n{'='*56}")
    print(f"    {n} files saved to  assets/ui/icons/")
    print(f"{'='*56}\n")
