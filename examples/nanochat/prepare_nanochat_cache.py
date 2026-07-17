#!/usr/bin/env python3
"""Create or verify the trusted canonical autoresearch cache manifest."""

import argparse

from nanochat_cache import create_manifest, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    identity = create_manifest() if args.action == "create" else verify_manifest(rehash_data=True)["identity_sha256"]
    print(identity)


if __name__ == "__main__":
    main()
