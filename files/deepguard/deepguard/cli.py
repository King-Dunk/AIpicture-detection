#!/usr/bin/env python3
"""
DeepGuard CLI - Detect AI-generated images from the terminal.

Usage:
    deepguard scan image.jpg
    deepguard scan image.jpg --heatmap
    deepguard scan folder/ --batch
    deepguard scan image.jpg --threshold 0.6 --json
"""

import argparse
import json
import sys
from pathlib import Path

from deepguard.detector import DeepfakeDetector

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# ANSI colors (no external deps needed)
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}
  ██████╗ ███████╗███████╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
  ██║  ██║█████╗  █████╗  ██████╔╝██║  ███╗██║   ██║███████║██████╔╝██║  ██║
  ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
  ██████╔╝███████╗███████╗██║     ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
  ╚═════╝ ╚══════╝╚══════╝╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
{RESET}{DIM}  AI-Generated Image Detector | github.com/yourhandle/deepguard{RESET}
"""
    print(banner)


def render_bar(value: float, width: int = 30) -> str:
    """Render a simple ASCII progress bar."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:.1%}"


def color_for_risk(risk: str) -> str:
    return {
        "LOW":    GREEN,
        "MEDIUM": YELLOW,
        "HIGH":   RED,
    }.get(risk, RESET)


def print_result(result, verbose: bool = False):
    risk_color = color_for_risk(result.risk_level)
    verdict_str = f"{RED}🤖 AI-GENERATED{RESET}" if result.is_ai_generated else f"{GREEN}✅ LIKELY REAL{RESET}"

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"  {BOLD}File:{RESET}     {result.filepath}")
    print(f"  {BOLD}Verdict:{RESET}  {verdict_str}")
    print(f"  {BOLD}Score:{RESET}    {render_bar(result.confidence)}")
    print(f"  {BOLD}Risk:{RESET}     {risk_color}{BOLD}{result.risk_level}{RESET}")

    if verbose:
        print(f"\n  {DIM}Signal breakdown:{RESET}")
        for signal_name, score in result.signals.items():
            label = signal_name.replace("_", " ").title()
            bar = render_bar(score, width=20)
            print(f"    {label:<22} {bar}")

    if result.heatmap_path:
        print(f"\n  {DIM}Heatmap saved: {result.heatmap_path}{RESET}")

    print(f"{BOLD}{'─' * 60}{RESET}\n")


def scan_single(args):
    detector = DeepfakeDetector(threshold=args.threshold)
    try:
        result = detector.detect(args.target, save_heatmap=args.heatmap)
    except FileNotFoundError as e:
        print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}Detection failed: {e}{RESET}")
        sys.exit(1)

    if args.json:
        output = {
            "filepath": result.filepath,
            "is_ai_generated": result.is_ai_generated,
            "confidence": round(result.confidence, 4),
            "risk_level": result.risk_level,
            "signals": {k: round(v, 4) for k, v in result.signals.items()},
        }
        print(json.dumps(output, indent=2))
    else:
        print_result(result, verbose=args.verbose)


def scan_batch(args):
    target = Path(args.target)
    if target.is_file():
        # Single file passed with --batch flag — just scan it
        images = [target]
    elif target.is_dir():
        images = [p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        print(f"{RED}Error: '{target}' is not a valid file or directory.{RESET}")
        sys.exit(1)

    if not images:
        print(f"{YELLOW}No images found in {target}{RESET}")
        sys.exit(0)

    detector = DeepfakeDetector(threshold=args.threshold)
    results = []
    ai_count = 0

    print(f"\n{CYAN}Scanning {len(images)} image(s)...{RESET}\n")

    for img_path in images:
        try:
            result = detector.detect(str(img_path), save_heatmap=args.heatmap)
            results.append(result)
            if result.is_ai_generated:
                ai_count += 1
            if not args.json:
                print_result(result, verbose=args.verbose)
        except Exception as e:
            print(f"{YELLOW}  Skipping {img_path.name}: {e}{RESET}")

    # Summary
    total = len(results)
    if args.json:
        summary = {
            "total_scanned": total,
            "ai_detected": ai_count,
            "real_detected": total - ai_count,
            "results": [
                {
                    "filepath": r.filepath,
                    "is_ai_generated": r.is_ai_generated,
                    "confidence": round(r.confidence, 4),
                    "risk_level": r.risk_level,
                }
                for r in results
            ],
        }
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"  {BOLD}Scan Summary{RESET}")
        print(f"  Total scanned : {total}")
        print(f"  AI-generated  : {RED}{ai_count}{RESET}")
        print(f"  Likely real   : {GREEN}{total - ai_count}{RESET}")
        print(f"{BOLD}{'═' * 60}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="deepguard",
        description="DeepGuard — AI-Generated Image Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepguard scan photo.jpg
  deepguard scan photo.jpg --heatmap --verbose
  deepguard scan images/ --batch
  deepguard scan photo.jpg --threshold 0.6 --json
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan image(s) for AI generation")
    scan_parser.add_argument("target", help="Image file or directory to scan")
    scan_parser.add_argument("--batch",     action="store_true", help="Scan all images in a directory")
    scan_parser.add_argument("--heatmap",   action="store_true", help="Save anomaly heatmap")
    scan_parser.add_argument("--verbose",   action="store_true", help="Show per-signal breakdown")
    scan_parser.add_argument("--json",      action="store_true", help="Output results as JSON")
    scan_parser.add_argument("--threshold", type=float, default=0.5,
                             help="Detection threshold 0.0-1.0 (default: 0.5)")

    args = parser.parse_args()

    if args.command is None:
        print_banner()
        parser.print_help()
        sys.exit(0)

    if args.command == "scan":
        if not args.json:
            print_banner()
        target = Path(args.target)
        if args.batch or target.is_dir():
            scan_batch(args)
        else:
            scan_single(args)


if __name__ == "__main__":
    main()
