#!/usr/bin/env python3
"""Create or verify the trusted canonical autoresearch cache manifest."""

import argparse

from nanochat_cache import create_manifest, prepare_data_snapshot, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    if args.action == "create":
        identity = create_manifest()
    else:
        dataset = verify_manifest(rehash_data=True)
        prepare_data_snapshot(dataset)
        identity = dataset["identity_sha256"]
    print(identity)


if __name__ == "__main__":
    main()
