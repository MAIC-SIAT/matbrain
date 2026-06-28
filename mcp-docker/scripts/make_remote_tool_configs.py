#!/usr/bin/env python3
"""Generate Mat-MCP tool configs that point to a remote MCP host.

The checked-in configs use localhost for single-machine development.  For
Docker/cluster training, the rollout container may run on a different host from
the MCP stack.  This script rewrites only the hostname part of every server_url
and keeps ports and paths unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml


DEFAULT_INPUTS = [
    Path("mcp-docker/configs/structure_property_tools.yaml"),
    Path("mcp-docker/configs/evidence_tools.yaml"),
]


def rewrite_url(url: str, host: str, scheme: str | None = None) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"server_url is not absolute: {url}")
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{host}{port}"
    return urlunparse(
        (
            scheme or parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def convert_file(input_path: Path, output_path: Path, host: str, scheme: str | None) -> int:
    data = yaml.safe_load(input_path.read_text(encoding="utf-8")) or {}
    count = 0
    for item in data.get("tools", []):
        if "server_url" in item:
            item["server_url"] = rewrite_url(str(item["server_url"]), host=host, scheme=scheme)
            count += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="Remote MCP host or IP, e.g. 127.0.0.1")
    parser.add_argument("--output-dir", type=Path, default=Path("mcp-docker/configs/remote"))
    parser.add_argument("--scheme", default=None, help="Optional URL scheme override, normally omitted.")
    parser.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    args = parser.parse_args()

    for input_path in args.inputs:
        suffix = f".{args.host}.yaml"
        output_path = args.output_dir / f"{input_path.stem}{suffix}"
        count = convert_file(input_path, output_path, host=args.host, scheme=args.scheme)
        print(f"{input_path} -> {output_path} ({count} URLs)")


if __name__ == "__main__":
    main()
