"""Generate a single consistent per-class mIoU example."""

from pathlib import Path


OUT = Path(__file__).resolve().parent / "08_miou_grid.html"
COLORS = ("#dce2e7", "#69a9d6", "#ed8070")
TARGET = ((0, 0, 1), (0, 2, 1))
PRED = ((0, 1, 1), (0, 2, 0))


def grid(values, left):
    cells = []
    for row, labels in enumerate(values):
        for col, label in enumerate(labels):
            x, y = left + col * 65, 152 + row * 65
            cells.append(
                f'<rect x="{x}" y="{y}" width="57" height="57" rx="8" '
                f'fill="{COLORS[label]}" stroke="#fff" stroke-width="2" />'
                f'<text x="{x + 28.5}" y="{y + 37}" text-anchor="middle" '
                f'class="cell">{label}</text>'
            )
    return "".join(cells)


def main():
    svg = f"""<svg viewBox="0 0 960 430" role="img" aria-labelledby="miou-title miou-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="miou-title">mIoU：每类分别算像素 IoU，再求平均</title>
  <desc id="miou-desc">同一组两行三列的标注图与预测图。背景 IoU 为二除以四，道路为一除以三，车为一，三类平均约零点六一一。</desc>
  <rect width="960" height="430" fill="#fbfaf7" />
  <text x="480" y="47" text-anchor="middle" class="title">mIoU：先算每个类别的 IoU，再求平均</text>
  <rect x="45" y="88" width="260" height="270" rx="18" class="panel" />
  <rect x="350" y="88" width="260" height="270" rx="18" class="panel" />
  <rect x="655" y="88" width="260" height="270" rx="18" class="panel" />
  <text x="175" y="127" text-anchor="middle" class="section">标注类别图</text>
  <text x="480" y="127" text-anchor="middle" class="section">预测类别图</text>
  <text x="785" y="127" text-anchor="middle" class="section">逐类 IoU</text>
  {grid(TARGET, 81)}
  {grid(PRED, 386)}
  <text x="175" y="325" text-anchor="middle" class="muted">每格代表一个像素</text>
  <text x="480" y="325" text-anchor="middle" class="muted">与左图逐格比较</text>
  <text x="785" y="168" text-anchor="middle" class="score">背景：2 / 4 = 0.500</text>
  <text x="785" y="208" text-anchor="middle" class="score">道路：1 / 3 ≈ 0.333</text>
  <text x="785" y="248" text-anchor="middle" class="score">车：1 / 1 = 1.000</text>
  <line x1="695" x2="875" y1="273" y2="273" stroke="#d9e0e4" stroke-width="2" />
  <text x="785" y="316" text-anchor="middle" class="result">mIoU ≈ 0.611</text>
  <rect x="259" y="384" width="16" height="16" rx="3" fill="{COLORS[0]}" />
  <text x="283" y="398" class="legend">0 背景</text>
  <rect x="410" y="384" width="16" height="16" rx="3" fill="{COLORS[1]}" />
  <text x="434" y="398" class="legend">1 道路</text>
  <rect x="561" y="384" width="16" height="16" rx="3" fill="{COLORS[2]}" />
  <text x="585" y="398" class="legend">2 车</text>
</svg>"""
    OUT.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<style>body{margin:0;background:#fbfaf7;font-family:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif}'
        'svg{display:block;width:100%;max-width:960px;margin:auto}'
        '.title{font:600 27px "Noto Sans SC","PingFang SC",sans-serif;fill:#25334a}'
        '.panel{fill:#fff;stroke:#dce3e6;stroke-width:2}'
        '.section{font:600 20px "Noto Sans SC","PingFang SC",sans-serif;fill:#25334a}'
        '.cell{font:600 19px "Noto Sans SC","PingFang SC",sans-serif;fill:#26334b}'
        '.muted{font:500 15px "Noto Sans SC","PingFang SC",sans-serif;fill:#53627a}'
        '.score{font:500 18px "Noto Sans SC","PingFang SC",sans-serif;fill:#3d4c63}'
        '.result{font:600 25px "Noto Sans SC","PingFang SC",sans-serif;fill:#25334a}'
        '.legend{font:500 15px "Noto Sans SC","PingFang SC",sans-serif;fill:#53627a}'
        '</style></head><body>' + svg + '</body></html>'
    )


if __name__ == "__main__":
    main()
