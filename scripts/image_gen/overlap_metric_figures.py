"""Generate the CIoU containment diagram as standalone HTML."""

from pathlib import Path


OUT = Path(__file__).resolve().parent
STYLE = """
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #fbfaf7; color: #25334a; font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
  main { width: min(100%, 1020px); margin: 0 auto; padding: 20px; }
  .diagram-container { width: 100%; overflow-x: auto; }
  svg { display: block; width: 100%; min-width: 960px; height: auto; }
  .title { font: 600 27px "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; fill: #25334a; }
  .panel-title { font: 600 18px "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; fill: #25334a; }
  .label { font: 500 15px "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; fill: #53627a; }
  .score { font: 600 20px "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; fill: #26334b; }
  .note { font: 500 15px "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; fill: #53627a; }
  .base { fill: #38aaa0; fill-opacity: .10; stroke: #228f86; stroke-width: 3; }
  .small { fill: #ed736b; fill-opacity: .24; stroke: #df6158; stroke-width: 3; }
  .base-center { fill: #228f86; }
  .small-center { fill: #df6158; }
  .guide { fill: none; stroke: #9ba6b6; stroke-width: 1.6; stroke-dasharray: 5 5; }
  @media print { .diagram-container { overflow-x: visible; } svg { min-width: 0; } }
</style>
"""


def frame(slug, title, description, panels):
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{STYLE}</head>
<body><main><div class="diagram-container">
<svg viewBox="0 0 960 440" role="img" aria-labelledby="{slug}-title {slug}-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="{slug}-title">{title}</title>
  <desc id="{slug}-desc">{description}</desc>
  <rect x="0" y="0" width="960" height="440" fill="#fbfaf7" />
  <text x="480" y="44" text-anchor="middle" class="title">{title}</text>
  <line x1="480" y1="74" x2="480" y2="370" stroke="#dbe1e6" stroke-width="1.5" />
  {panels}
</svg></div></main></body></html>
"""


def panel(cx, title, inner_x, inner_y, score):
    bx, by, size, small = cx - 105, 118, 210, 126
    icx, icy = inner_x + small / 2, inner_y + small / 2
    reference_mark = (
        f'<circle cx="{cx}" cy="223" r="11" fill="none" stroke="#228f86" stroke-width="3" />'
        if icx == cx and icy == 223
        else f'<circle cx="{cx}" cy="223" r="5.5" class="base-center" />'
    )
    center_marks = f"""
  <path d="M {icx:g} {icy:g} H {cx} V 223" class="guide" />
  {reference_mark}
  <circle cx="{icx:g}" cy="{icy:g}" r="5.5" class="small-center" />
  <text x="{cx}" y="345" text-anchor="middle" class="label">青点：参考中心　珊瑚点：候选中心</text>"""
    return f"""
  <text x="{cx}" y="91" text-anchor="middle" class="panel-title">{title}</text>
  <rect x="{bx}" y="{by}" width="{size}" height="{size}" class="base" />
  <rect x="{inner_x}" y="{inner_y}" width="{small}" height="{small}" class="small" />
  {center_marks}
  <text x="{cx}" y="385" text-anchor="middle" class="score">{score}</text>"""


def main():
    ciou = panel(255, "A · 两框中心重合", 192, 160, "CIoU = 0.36")
    ciou += panel(705, "B · 候选中心偏移 20×20", 600, 118, "CIoU = 0.32")
    ciou += '<text x="480" y="425" text-anchor="middle" class="note">两例宽高比相同，宽高比项为 0；B 的中心距离项为 0.04</text>'
    (OUT / "09_ciou_containment.html").write_text(frame(
        "ciou-containment", "IoU 同为 0.36，CIoU 区分中心偏移",
        "两个同样面积的候选框位于参考框内部。居中时 CIoU 为 0.36，靠角时中心距离罚项为 0.04，CIoU 为 0.32。", ciou
    ))


if __name__ == "__main__":
    main()
