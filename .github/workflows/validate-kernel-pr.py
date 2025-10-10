#!/usr/bin/env python3
"""
Script to validate kernel PR title and directory structure
Usage: validate-kernel-pr.py <pr_title>
Returns exit code 0 if valid, 1 if invalid, and outputs kernel name to stdout
"""

import sys
import re
from pathlib import Path


def validate_kernel_name(kernel_name: str) -> bool:
    """
    Validate that the kernel name is safe and follows expected patterns.
    Only allow alphanumeric characters, hyphens, and underscores.
    """
    if not kernel_name:
        return False

    # Only allow alphanumeric characters, hyphens, and underscores
    if not re.match(r"^[a-zA-Z0-9_-]+$", kernel_name):
        return False

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: validate-kernel-pr.py <pr_title>", file=sys.stderr)
        sys.exit(1)

    pr_title = sys.argv[1]

    if ":" not in pr_title:
        print("No colon found in PR title, skipping build", file=sys.stderr)
        sys.exit(1)

    kernel_name = pr_title.split(":", 1)[0].strip()

    if not validate_kernel_name(kernel_name):
        print(f"Invalid kernel name '{kernel_name}', skipping build", file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(kernel_name)
    if not kernel_path.exists() or not kernel_path.is_dir():
        print(f"Kernel '{kernel_name}' does not exist, skipping build", file=sys.stderr)
        sys.exit(1)

    flake_nix = kernel_path / "flake.nix"
    build_toml = kernel_path / "build.toml"

    if not flake_nix.exists() or not build_toml.exists():
        print(
            f"Kernel '{kernel_name}' missing flake.nix or build.toml, skipping build",
            file=sys.stderr,
        )
        sys.exit(1)

    print(kernel_name)
    sys.exit(0)


if __name__ == "__main__":
    main()
