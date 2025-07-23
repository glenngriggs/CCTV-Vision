#!/usr/bin/env python3
"""
CLI helper for the person-search system.

Usage
-----
python -m person_search.run --video path/to/input.mp4 \
                            --output outputs/annotated.mp4 \
                            --preset blue
"""
import argparse, sys
from pathlib import Path

from .system import CompleteVideoProcessor

_PRESET_QUERIES = {
    "all":   {},                         # no filtering
    "blue":  {"shirt_color": "blue"},    # tests light-blue fix
    "black": {"shirt_color": "black"}    # tests dark-gray-as-black fix
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the video person-search pipeline.")
    p.add_argument("--video", required=True, help="Input video file")
    p.add_argument("--output", default=None, help="Optional output *.mp4 path")
    p.add_argument("--preset",
                   choices=list(_PRESET_QUERIES),
                   default="all",
                   help="Quick query presets")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    if not Path(args.video).exists():
        sys.exit(f"❌ Video not found: {args.video}")

    processor = CompleteVideoProcessor()      # model loads here
    query     = _PRESET_QUERIES[args.preset]

    # Human-friendly default output name
    output_path = args.output or (
        Path(args.video).with_stem(f"{Path(args.video).stem}_annotated")
        .with_suffix(".mp4")
    )

    results = processor.process_video_complete(
        video_path=args.video,
        query=query,
        output_path=str(output_path),
        show_progress=True
    )

    if results["success"]:
        print(f"\n✅ Done!  Matches found: {len(results['matches'])}")
        print(f"📂 Output written to: {output_path}")
    else:
        sys.exit(f"❌ Processing failed: {results.get('error')}")

if __name__ == "__main__":
    main()
