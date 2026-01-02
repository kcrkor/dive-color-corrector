#!/usr/bin/env python3
"""Bump the version number in __version__."""

import re
from pathlib import Path

VERSION_FILE = Path("src/dive_color_corrector/__init__.py")


def bump(part: str) -> None:
    if VERSION_FILE.exists():
        content = VERSION_FILE.read_text()
    else:
        print(f"Error: {VERSION_FILE} not found.")
        return

    match = re.search(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"', content, re.MULTILINE)
    if not match:
        print("Error: Version string not found in expected format.")
        return

    major, minor, patch = map(int, match.groups())

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        print(f"Error: Invalid part '{part}'. Use 'major', 'minor', or 'patch'.")
        return

    new_version = f"{major}.{minor}.{patch}"
    new_content = re.sub(
        r'^(__version__ = ")(\d+\.\d+\.\d+)(")',
        rf"\g<1>{new_version}\g<3>",
        content,
        flags=re.MULTILINE,
    )

    VERSION_FILE.write_text(new_content)
    print(f"Bumped version to {new_version}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python bump_version.py <major|minor|patch>")
        sys.exit(1)
    bump(sys.argv[1])
