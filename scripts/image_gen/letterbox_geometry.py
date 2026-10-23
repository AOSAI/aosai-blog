"""Draw the scale-and-pad example in docs/ComputerVision/YOLOSeries/07_LetterBox.md."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aosai-blog-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle


OUT = Path("docs/.vuepress/public/ComputerVision/YOLOSeries/07_letterbox_geometry.png")
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
INK = "#30343F"
MUTED = "#667085"
TEAL = "#0F8E8A"
PURPLE = "#6E5AAB"
CORAL = "#D65E5A"


def draw_scene(ax, scale, top_padding=0):
    """Draw identical image content at each stage, in scaled pixel coordinates."""
    width, height = 1280 * scale, 720 * scale
    ax.add_patch(Rectangle((0, top_padding), width, height, color="#EDF5F8"))
    ax.add_patch(Rectangle((0, top_padding + 445 * scale), width, 275 * scale,
                           color="#DDE6DE"))
    ax.add_patch(Rectangle((0, top_padding + 510 * scale), width, 210 * scale,
                           color="#CBD4D9"))
    ax.add_patch(Polygon(
        [(440 * scale, top_padding + 350 * scale),
         (545 * scale, top_padding + 270 * scale),
         (745 * scale, top_padding + 270 * scale),
         (865 * scale, top_padding + 350 * scale)],
        closed=True, facecolor="#71A4B4", edgecolor="#477589", linewidth=1.4,
    ))
    ax.add_patch(Rectangle((370 * scale, top_padding + 350 * scale),
                           550 * scale, 140 * scale,
                           facecolor="#71A4B4", edgecolor="#477589", linewidth=1.4))
    for x in (495, 805):
        ax.add_patch(Circle((x * scale, top_padding + 495 * scale),
                            36 * scale, facecolor="#354854", edgecolor="white", linewidth=1.2))
    ax.add_patch(Rectangle((320 * scale, top_padding + 180 * scale),
                           640 * scale, 360 * scale,
                           facecolor="none", edgecolor=CORAL, linewidth=2.7))


def stage(fig, bounds, shape, scale, top_padding=0):
    ax = fig.add_axes(bounds)
    width, height = shape
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if top_padding:
        ax.add_patch(Rectangle((0, 0), width, height, color="#E7E9EC"))
    draw_scene(ax, scale, top_padding)
    ax.add_patch(Rectangle((0, 0), width, height, facecolor="none",
                           edgecolor=TEAL if top_padding else PURPLE,
                           linewidth=2.0))
    if top_padding:
        ax.text(width / 2, top_padding / 2, "上方填充 140 px", ha="center", va="center",
                fontsize=12, color=INK)
        ax.text(width / 2, height - top_padding / 2, "下方填充 140 px",
                ha="center", va="center", fontsize=12, color=INK)


def main():
    font_manager.fontManager.addfont(str(FONT))
    normal = font_manager.FontProperties(fname=str(FONT))
    bold = font_manager.FontProperties(fname=str(BOLD))
    plt.rcParams["font.family"] = normal.get_name()

    fig = plt.figure(figsize=(14.4, 6.4), facecolor="white")
    fig.text(0.5, 0.955, "同一张图：先等比缩小，再填补空白", ha="center", va="center",
             fontproperties=font_manager.FontProperties(fname=str(BOLD), size=23), color=INK)
    fig.text(0.5, 0.895, "三个画布和框按同一比例绘制；珊瑚色框始终指向同一物体",
             ha="center", va="center", fontsize=13.5, color=MUTED)

    headings = [
        (3.6 / 14.4, "① 原图", "1280 × 720"),
        (9.0 / 14.4, "② 等比缩放", "640 × 360 · r = 0.5"),
        (12.6 / 14.4, "③ 上下补边", "640 × 640"),
    ]
    for center, title, detail in headings:
        fig.text(center, 0.825, title, ha="center", va="center",
                 fontproperties=font_manager.FontProperties(fname=str(BOLD), size=17), color=INK)
        fig.text(center, 0.785, detail, ha="center", va="center", fontsize=12.5, color=MUTED)

    # Physical axes widths remain proportional to pixel widths: 0.005 in / px.
    stage(fig, [0.4 / 14.4, 1.2 / 6.4, 6.4 / 14.4, 3.6 / 6.4],
          (1280, 720), scale=1)
    stage(fig, [7.4 / 14.4, 2.1 / 6.4, 3.2 / 14.4, 1.8 / 6.4],
          (640, 360), scale=0.5)
    stage(fig, [11.0 / 14.4, 1.4 / 6.4, 3.2 / 14.4, 3.2 / 6.4],
          (640, 640), scale=0.5, top_padding=140)

    for start, end in ((6.85, 7.30), (10.65, 10.90)):
        plt.annotate("", xy=(end / 14.4, 3.0 / 6.4), xytext=(start / 14.4, 3.0 / 6.4),
                     xycoords="figure fraction", textcoords="figure fraction",
                     arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": TEAL})

    fig.text(0.5, 0.095,
             "框坐标   (320, 180, 960, 540)   →   (160, 90, 480, 270)   →   (160, 230, 480, 410)",
             ha="center", va="center", fontsize=14.5, color=INK)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
