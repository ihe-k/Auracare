"""
Generate the Task 4 picture-description stimulus.

Why this is drawn rather than downloaded
    The draft app pointed at an Unsplash photograph of a dog, which is a weak
    stimulus (little to describe, no action, no people) and an external
    dependency that breaks offline. The established clinical pictures - the
    Cookie Theft card from the Boston Diagnostic Aphasia Examination, the
    Picnic scene from the WAB - are under copyright and cannot be shipped.

    So this draws an original scene to the same design brief: high contrast,
    several simultaneous actions, people of different ages, objects in clear
    spatial relationships, and at least one element that invites inference
    rather than naming. That gives a connected-speech sample with enough
    content to measure information density against.

Describable content, for scoring reference
    actions   : boy flying a kite, girl throwing a ball, dog chasing the ball,
                man reading on a bench, woman pushing a pram, ducks on the pond
    objects   : kite, ball, bench, pram, book, pond, ducks, tree, sun, clouds,
                ice cream van, bin, path, fence
    inference : the dog has slipped its lead (the lead is on the ground)

Usage
    python scripts/make_stimulus.py [--out assets/stimulus_park.png]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 1200, 800

SKY = (176, 216, 240)
GRASS = (124, 179, 96)
GRASS_DARK = (104, 159, 76)
PATH = (222, 206, 174)
POND = (86, 156, 198)
POND_EDGE = (62, 126, 166)
TREE_TRUNK = (102, 72, 48)
TREE_LEAF = (58, 126, 62)
TREE_LEAF_2 = (72, 148, 74)
SUN = (252, 214, 92)
WHITE = (250, 250, 250)
OUTLINE = (34, 34, 40)
RED = (214, 68, 60)
BLUE = (54, 104, 184)
YELLOW = (238, 190, 60)
PURPLE = (138, 86, 170)
ORANGE = (232, 136, 54)
BROWN = (142, 96, 58)
PINK = (226, 138, 158)
SKIN = (238, 198, 164)
SKIN_2 = (176, 124, 88)


def ellipse(d, cx, cy, rx, ry, fill, outline=OUTLINE, width=3):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill, outline=outline,
              width=width)


def person(
    d,
    x,
    y,
    scale=1.0,
    shirt=BLUE,
    trousers=(64, 72, 92),
    skin=SKIN,
    hair=(58, 44, 36),
    arm_angle=-40.0,
    facing=1,
):
    """A simple standing figure, anchored at the feet."""
    s = scale
    head_r = int(16 * s)
    head_cy = int(y - 118 * s)

    # legs
    d.line([(x - 10 * s, y), (x - 4 * s, y - 52 * s)], fill=trousers,
           width=max(3, int(11 * s)))
    d.line([(x + 10 * s, y), (x + 4 * s, y - 52 * s)], fill=trousers,
           width=max(3, int(11 * s)))
    # torso
    d.line([(x, y - 50 * s), (x, y - 100 * s)], fill=shirt,
           width=max(4, int(22 * s)))
    # far arm
    d.line([(x, y - 92 * s), (x - facing * 26 * s, y - 64 * s)], fill=shirt,
           width=max(3, int(9 * s)))
    # near arm, raised by arm_angle degrees
    rad = math.radians(arm_angle)
    ax = x + facing * math.cos(rad) * 34 * s
    ay = (y - 92 * s) + math.sin(rad) * 34 * s
    d.line([(x, y - 92 * s), (ax, ay)], fill=shirt, width=max(3, int(9 * s)))
    d.ellipse([ax - 5 * s, ay - 5 * s, ax + 5 * s, ay + 5 * s], fill=skin)
    # head
    ellipse(d, x, head_cy, head_r, head_r, skin, width=max(2, int(3 * s)))
    d.chord([x - head_r, head_cy - head_r, x + head_r, head_cy + head_r],
            180, 360, fill=hair)
    return (ax, ay)


def draw_scene() -> Image.Image:
    img = Image.new("RGB", (W, H), SKY)
    d = ImageDraw.Draw(img)

    # --- sky ---------------------------------------------------------------
    ellipse(d, 1075, 110, 58, 58, SUN, outline=(232, 184, 60), width=4)
    for cx, cy, r in ((210, 120, 40), (265, 134, 52), (320, 120, 36),
                      (700, 90, 34), (748, 102, 44), (796, 92, 30)):
        ellipse(d, cx, cy, r, r * 0.72, WHITE, outline=(210, 220, 232), width=2)

    # --- ground ------------------------------------------------------------
    d.rectangle([0, 470, W, H], fill=GRASS)
    d.line([(0, 470), (W, 470)], fill=GRASS_DARK, width=4)

    # path
    d.polygon([(0, 690), (W, 610), (W, 668), (0, 760)], fill=PATH)
    d.line([(0, 690), (W, 610)], fill=(198, 180, 150), width=3)
    d.line([(0, 760), (W, 668)], fill=(198, 180, 150), width=3)

    # fence along the back
    for fx in range(40, W - 20, 54):
        d.rectangle([fx, 440, fx + 9, 492], fill=(198, 186, 160), outline=OUTLINE,
                    width=2)
    d.rectangle([30, 452, W - 30, 461], fill=(210, 198, 172), outline=OUTLINE,
                width=2)

    # --- pond --------------------------------------------------------------
    d.ellipse([60, 560, 400, 700], fill=POND, outline=POND_EDGE, width=5)
    for i, (dx, dy, sc) in enumerate(((120, 600, 1.0), (190, 626, 1.1),
                                      (255, 596, 0.9))):
        # duck: body, head, beak
        ellipse(d, dx, dy, 26 * sc, 16 * sc, (248, 244, 236), width=3)
        ellipse(d, dx + 20 * sc, dy - 16 * sc, 10 * sc, 10 * sc, (248, 244, 236),
                width=3)
        d.polygon([(dx + 29 * sc, dy - 17 * sc), (dx + 42 * sc, dy - 13 * sc),
                   (dx + 29 * sc, dy - 9 * sc)], fill=ORANGE, outline=OUTLINE)
    for ry in (648, 664, 680):
        d.arc([150, ry, 330, ry + 22], 200, 340, fill=POND_EDGE, width=3)

    # --- big tree ----------------------------------------------------------
    d.rectangle([980, 430, 1026, 620], fill=TREE_TRUNK, outline=OUTLINE, width=3)
    for cx, cy, r in ((1003, 360, 96), (930, 400, 70), (1076, 398, 72),
                      (960, 318, 58), (1050, 322, 54)):
        ellipse(d, cx, cy, r, r * 0.9, TREE_LEAF, width=3)
    for cx, cy, r in ((985, 350, 44), (1035, 372, 36)):
        ellipse(d, cx, cy, r, r * 0.86, TREE_LEAF_2, outline=None, width=0)

    # --- ice cream van -----------------------------------------------------
    vx, vy = 760, 560
    d.rectangle([vx, vy - 86, vx + 210, vy], fill=WHITE, outline=OUTLINE, width=4)
    d.rectangle([vx + 150, vy - 126, vx + 210, vy - 86], fill=WHITE,
                outline=OUTLINE, width=4)
    d.rectangle([vx + 162, vy - 118, vx + 200, vy - 94], fill=(188, 220, 240),
                outline=OUTLINE, width=3)
    d.rectangle([vx + 18, vy - 72, vx + 118, vy - 30], fill=(190, 226, 238),
                outline=OUTLINE, width=3)
    d.rectangle([vx, vy - 96, vx + 210, vy - 86], fill=PINK, outline=OUTLINE,
                width=3)
    ellipse(d, vx + 44, vy + 4, 22, 22, (54, 54, 60), outline=OUTLINE, width=3)
    ellipse(d, vx + 170, vy + 4, 22, 22, (54, 54, 60), outline=OUTLINE, width=3)
    # cone on the roof
    d.polygon([(vx + 60, vy - 96), (vx + 78, vy - 140), (vx + 96, vy - 96)],
              fill=(226, 190, 128), outline=OUTLINE)
    ellipse(d, vx + 78, vy - 146, 15, 13, (248, 232, 226), width=3)

    # --- bench and man reading --------------------------------------------
    # Kept in the mid-ground so the foreground band below y=690 stays clear
    # for the girl, the ball and the dog.
    bx, by = 505, 598
    d.rectangle([bx, by, bx + 190, by + 14], fill=BROWN, outline=OUTLINE, width=3)
    d.rectangle([bx, by - 46, bx + 190, by - 32], fill=BROWN, outline=OUTLINE,
                width=3)
    d.rectangle([bx + 8, by + 14, bx + 20, by + 56], fill=(112, 76, 46),
                outline=OUTLINE, width=3)
    d.rectangle([bx + 170, by + 14, bx + 182, by + 56], fill=(112, 76, 46),
                outline=OUTLINE, width=3)
    # seated man: thigh along the seat, shin down to the ground
    d.line([(bx + 60, by - 2), (bx + 112, by - 2)], fill=(64, 72, 92), width=16)
    d.line([(bx + 112, by - 2), (bx + 118, by + 58)], fill=(64, 72, 92), width=14)
    d.polygon([(bx + 112, by + 54), (bx + 136, by + 58), (bx + 134, by + 66),
               (bx + 110, by + 64)], fill=(44, 40, 38), outline=OUTLINE)
    d.line([(bx + 60, by - 6), (bx + 60, by - 60)], fill=PURPLE, width=24)
    ellipse(d, bx + 60, by - 82, 17, 17, SKIN_2, width=3)
    d.chord([bx + 43, by - 99, bx + 77, by - 65], 180, 360, fill=(40, 34, 30))
    # the book, held out in front
    d.polygon([(bx + 78, by - 40), (bx + 120, by - 52), (bx + 120, by - 20),
               (bx + 78, by - 10)], fill=WHITE, outline=OUTLINE)
    d.line([(bx + 99, by - 46), (bx + 99, by - 15)], fill=OUTLINE, width=2)
    d.line([(bx + 60, by - 48), (bx + 82, by - 30)], fill=PURPLE, width=9)

    # --- boy flying a kite -------------------------------------------------
    hand = person(d, 300, 556, scale=0.82, shirt=RED, trousers=(50, 62, 96),
                  arm_angle=-72.0, facing=1)
    kx, ky = 470, 196
    d.line([hand, (kx, ky + 46)], fill=(70, 70, 76), width=2)
    d.polygon([(kx, ky - 50), (kx + 52, ky), (kx, ky + 50), (kx - 52, ky)],
              fill=YELLOW, outline=OUTLINE)
    d.line([(kx, ky - 50), (kx, ky + 50)], fill=OUTLINE, width=2)
    d.line([(kx - 52, ky), (kx + 52, ky)], fill=OUTLINE, width=2)
    for i, (tx, ty) in enumerate(((kx + 6, ky + 74), (kx - 4, ky + 100),
                                  (kx + 8, ky + 126))):
        d.polygon([(tx - 10, ty), (tx + 10, ty - 6), (tx + 6, ty + 10)],
                  fill=RED, outline=OUTLINE)

    # --- girl throwing a ball, dog chasing --------------------------------
    # Foreground band, left to right: girl -> ball in flight -> dog -> pram.
    # Each element gets its own horizontal lane so nothing overlaps.
    gh = person(d, 330, 786, scale=0.9, shirt=YELLOW, trousers=(142, 76, 150),
                skin=SKIN, hair=(96, 60, 36), arm_angle=-62.0, facing=1)
    ball_x, ball_y = 520, 714
    for i, f in enumerate((0.3, 0.55, 0.8)):
        tx = gh[0] + (ball_x - gh[0]) * f
        ty = gh[1] + (ball_y - gh[1]) * f - 14 * math.sin(math.pi * f)
        d.ellipse([tx - 3, ty - 3, tx + 3, ty + 3], fill=(236, 180, 176))
    ellipse(d, ball_x, ball_y, 17, 17, RED, width=3)
    d.arc([ball_x - 17, ball_y - 17, ball_x + 17, ball_y + 17], 200, 340,
          fill=WHITE, width=3)

    # dog: body, head, legs, tail, ears
    dx, dy = 742, 730
    ellipse(d, dx, dy, 46, 25, (198, 150, 92), width=3)
    ellipse(d, dx + 44, dy - 22, 21, 19, (198, 150, 92), width=3)
    d.polygon([(dx + 32, dy - 36), (dx + 40, dy - 58), (dx + 50, dy - 38)],
              fill=(168, 122, 70), outline=OUTLINE)
    d.polygon([(dx + 58, dy - 16), (dx + 78, dy - 12), (dx + 58, dy - 6)],
              fill=(60, 52, 46), outline=OUTLINE)
    for lx, ly in ((dx - 28, dy + 20), (dx - 12, dy + 22), (dx + 20, dy + 22),
                   (dx + 34, dy + 20)):
        d.line([(lx, ly), (lx - 4, ly + 28)], fill=(178, 132, 80), width=9)
        d.line([(lx - 4, ly + 28), (lx + 6, ly + 30)], fill=(178, 132, 80),
               width=7)
    d.line([(dx - 44, dy - 8), (dx - 76, dy - 40)], fill=(198, 150, 92), width=10)
    # the trailing lead - the inference cue
    d.line([(dx + 36, dy - 30), (dx + 4, dy + 34), (dx - 56, dy + 48),
            (dx - 120, dy + 36)], fill=(190, 60, 70), width=5)
    ellipse(d, dx - 126, dy + 35, 13, 9, None, outline=(190, 60, 70), width=5)

    # --- woman with a pram -------------------------------------------------
    px, py = 1096, 772
    person(d, px, py, scale=0.9, shirt=(72, 160, 148), trousers=(52, 60, 80),
           skin=SKIN_2, hair=(30, 26, 24), arm_angle=14.0, facing=-1)
    d.polygon([(px - 112, py - 58), (px - 32, py - 72), (px - 38, py - 20),
               (px - 106, py - 14)], fill=(96, 112, 180), outline=OUTLINE,
              width=3)
    d.arc([px - 120, py - 106, px - 46, py - 38], 150, 350,
          fill=(72, 88, 156), width=14)
    ellipse(d, px - 102, py, 16, 16, (54, 54, 60), outline=OUTLINE, width=3)
    ellipse(d, px - 46, py, 16, 16, (54, 54, 60), outline=OUTLINE, width=3)
    d.line([(px - 38, py - 66), (px - 16, py - 80)], fill=(72, 88, 156),
           width=7)

    # --- litter bin, kept clear of the figures -----------------------------
    d.polygon([(84, 710), (130, 710), (122, 778), (92, 778)],
              fill=(96, 104, 112), outline=OUTLINE, width=3)
    d.rectangle([78, 698, 136, 712], fill=(118, 126, 134), outline=OUTLINE,
                width=3)

    # --- foreground grass tufts -------------------------------------------
    for gx in range(20, W, 73):
        gy = 782
        for off in (-7, 0, 7):
            d.line([(gx + off, gy), (gx + off * 2, gy - 18)], fill=GRASS_DARK,
                   width=3)

    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "stimulus_park.png",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    img = draw_scene()
    img.save(args.out, format="PNG", optimize=True)
    print(f"wrote {args.out}  ({img.size[0]}x{img.size[1]}, "
          f"{args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
