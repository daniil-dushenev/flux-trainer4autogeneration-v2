#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def find_pngs(root: Path) -> list[Path]:
    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.png']


def main() -> int:
    parser = argparse.ArgumentParser(description='Delete all .png files recursively in a folder.')
    parser.add_argument('folder', type=Path, help='Target folder')
    parser.add_argument('--yes', action='store_true', help='Actually delete files (without this flag, dry-run only)')
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        print(f'Error: folder not found: {folder}')
        return 1

    png_files = find_pngs(folder)
    print(f'Found {len(png_files)} .png files in {folder}')

    if not args.yes:
        print('Dry-run mode. Use --yes to delete.')
        for p in png_files[:20]:
            print(p)
        if len(png_files) > 20:
            print(f'... and {len(png_files) - 20} more')
        return 0

    deleted = 0
    for p in png_files:
        p.unlink()
        deleted += 1

    print(f'Deleted {deleted} .png files from {folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
