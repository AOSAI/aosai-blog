"""Export the inline SVG from a self-contained diagram HTML file."""

import argparse
from pathlib import Path
from xml.etree import ElementTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    html = args.source.read_text(encoding="utf-8")
    start = html.index("<svg ")
    end = html.index("</svg>", start) + len("</svg>")
    svg = html[start:end]
    ElementTree.fromstring(svg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
