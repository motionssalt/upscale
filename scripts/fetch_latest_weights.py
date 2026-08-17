#!/usr/bin/env python3
"""
MOTIONSALT Upscaler — weights auto-mirror helper.

Purpose:
    Check upstream HuggingFace model repos for newer versions of the weight
    files this project uses (AnimeJaNai V3 SuperUltraCompact 2x,
    AnimeJaNai V3 UltraCompact 2x, Real-ESRGAN AnimeVideo v3 2x). If any
    upstream file has a newer commit than what is currently mirrored on this
    repo's latest GitHub Release, download the new files and publish a new
    Release with them attached.

    This script is what the Colab notebook indirectly relies on: the notebook
    never talks to HuggingFace itself — it only pulls from this repo's
    Releases. This script is the ONLY place HuggingFace is contacted.

Usage:
    # Automated (from GitHub Actions):
    python scripts/fetch_latest_weights.py --repo owner/motionsalt-upscaler

    # Manual seed (first-ever release, before the Action has anything to
    # compare against — run this once locally with a PAT):
    GITHUB_TOKEN=ghp_xxx python scripts/fetch_latest_weights.py \\
        --repo owner/motionsalt-upscaler --force

Behavior:
    - If HuggingFace is unreachable      -> exit 0, do nothing (safe fallback).
    - If nothing changed upstream        -> exit 0, do nothing.
    - If something changed               -> download, create new Release, exit 0.
    - If GitHub upload fails             -> exit non-zero (Action will show red).

The Colab notebook is unaffected in every one of these cases as long as at
least one Release already exists in this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Config — the exact upstream sources this project mirrors.
# ---------------------------------------------------------------------------
# Each entry says: "on HuggingFace, the file at <hf_repo>/<hf_path> is what
# we save into the GitHub Release as <mirror_name>".
#
# NOTE: HuggingFace repo paths below are the widely-published community
# mirrors of these models at time of writing. If an upstream author moves
# them, only this table needs to be updated — the Colab notebook itself
# doesn't care, because it downloads from GitHub, not from here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpstreamFile:
    tier: str                 # "LOW" / "MEDIUM" / "HIGH"
    mirror_name: str          # filename inside the GitHub Release
    hf_repo: str              # "namespace/repo" on HuggingFace
    hf_path: str              # path to the file inside that repo
    hf_revision: str = "main" # branch/tag/commit on HuggingFace

SOURCES: list[UpstreamFile] = [
    UpstreamFile(
        tier="LOW",
        mirror_name="2x_AnimeJaNaiV3_SuperUltraCompact.pth",
        hf_repo="the-database/mpv-upscale-2x_animejanai",
        hf_path="2x_AnimeJaNaiV3_SuperUltraCompact.pth",
    ),
    UpstreamFile(
        tier="MEDIUM",
        mirror_name="2x_AnimeJaNaiV3_UltraCompact.pth",
        hf_repo="the-database/mpv-upscale-2x_animejanai",
        hf_path="2x_AnimeJaNaiV3_UltraCompact.pth",
    ),
    UpstreamFile(
        tier="HIGH",
        mirror_name="realesr-animevideov3.pth",
        hf_repo="ai-forever/Real-ESRGAN",
        hf_path="realesr-animevideov3.pth",
    ),
]

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co"  # /<repo>/resolve/<rev>/<path>

GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Small HTTP helper (stdlib only, so this script has no external dependencies
# — the GitHub Action can run it without pip install).
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: Optional[dict] = None, timeout: int = 30) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()

def _http_json(url: str, headers: Optional[dict] = None, timeout: int = 30) -> dict:
    return json.loads(_http_get(url, headers=headers, timeout=timeout).decode("utf-8"))

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# HuggingFace side: figure out the "current" version of each upstream file.
# We treat the file's last-commit SHA on its branch as its version identity.
# ---------------------------------------------------------------------------

def hf_latest_commit(repo: str, path: str, revision: str = "main") -> Optional[str]:
    """
    Ask HuggingFace what commit last modified this file. Returns the commit SHA
    string, or None if HuggingFace is unreachable / the file no longer exists.
    """
    url = f"{HF_API}/models/{repo}/commits/{revision}?path={path}"
    try:
        data = _http_json(url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[hf] unreachable for {repo}:{path} ({e}) — treating as no-change.")
        return None
    if isinstance(data, list) and data:
        return data[0].get("id") or data[0].get("commit", {}).get("id")
    if isinstance(data, dict):
        return data.get("id") or data.get("sha")
    return None

def hf_download(repo: str, path: str, revision: str, dest: Path) -> None:
    url = f"{HF_RESOLVE}/{repo}/resolve/{revision}/{path}"
    print(f"[hf] downloading {url}")
    req = Request(url, headers={"User-Agent": "motionsalt-upscaler-mirror"})
    with urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


# ---------------------------------------------------------------------------
# GitHub side: read the latest Release's manifest, publish a new one, upload
# assets. Uses the `gh` CLI if available (it is, in GitHub Actions runners),
# otherwise falls back to the REST API.
# ---------------------------------------------------------------------------

MANIFEST_NAME = "manifest.json"

def gh_latest_release_manifest(repo: str, token: str) -> Optional[dict]:
    """
    Fetch manifest.json from the latest Release's assets, if any.
    manifest.json is a small file we upload alongside the weights so we can
    tell later, without downloading multi-GB .pth files, what version each
    mirrored file corresponds to.
    """
    url = f"{GITHUB_API}/repos/{repo}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "motionsalt-upscaler-mirror",
    }
    try:
        rel = _http_json(url, headers=headers)
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    for asset in rel.get("assets", []):
        if asset.get("name") == MANIFEST_NAME:
            asset_url = asset["url"]
            manifest_bytes = _http_get(
                asset_url,
                headers={**headers, "Accept": "application/octet-stream"},
            )
            return json.loads(manifest_bytes.decode("utf-8"))
    return {"files": {}}  # release exists but has no manifest — treat as empty

def gh_next_tag(repo: str, token: str) -> str:
    """Compute the next weights-vX.Y.Z tag (minor bump) from the latest one."""
    url = f"{GITHUB_API}/repos/{repo}/releases"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "motionsalt-upscaler-mirror",
    }
    try:
        releases = _http_json(url, headers=headers)
    except HTTPError:
        releases = []
    latest = 0, 0, 0
    for rel in releases or []:
        m = re.match(r"weights-v(\d+)\.(\d+)\.(\d+)$", rel.get("tag_name", ""))
        if m:
            v = tuple(int(x) for x in m.groups())
            if v > latest:
                latest = v
    if latest == (0, 0, 0):
        return "weights-v1.0.0"
    maj, minr, patch = latest
    return f"weights-v{maj}.{minr + 1}.0"

def gh_publish_release(repo: str, tag: str, notes: str, files: list[Path]) -> None:
    """Publish a Release with the given assets attached, via the `gh` CLI."""
    if not shutil.which("gh"):
        raise RuntimeError(
            "The `gh` CLI is required to publish releases. It is preinstalled "
            "on GitHub Actions runners. For local seeding, install it from "
            "https://cli.github.com/."
        )
    cmd = [
        "gh", "release", "create", tag,
        "--repo", repo,
        "--title", f"Model weights {tag}",
        "--notes", notes,
        *[str(f) for f in files],
    ]
    print(f"[gh] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Mirror upstream weight files into this repo's Releases.")
    ap.add_argument("--repo", required=True, help="owner/name of THIS repo (e.g. someone/motionsalt-upscaler)")
    ap.add_argument("--force", action="store_true", help="Publish even if nothing changed (seed initial release).")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable is required.", file=sys.stderr)
        return 2

    # 1) What do we currently mirror?
    print(f"[mirror] reading current manifest from {args.repo} …")
    current = gh_latest_release_manifest(args.repo, token)
    current_files = (current or {}).get("files", {}) if current is not None else {}

    # 2) Check each upstream file for a newer commit.
    changed: list[tuple[UpstreamFile, str]] = []
    upstream_status: dict[str, dict] = {}
    for src in SOURCES:
        latest = hf_latest_commit(src.hf_repo, src.hf_path, src.hf_revision)
        upstream_status[src.mirror_name] = {
            "hf_repo": src.hf_repo,
            "hf_path": src.hf_path,
            "hf_revision": src.hf_revision,
            "hf_commit": latest,
            "tier": src.tier,
        }
        if latest is None:
            print(f"[mirror] {src.mirror_name}: upstream unreachable — leaving mirror as-is.")
            continue
        current_commit = (current_files.get(src.mirror_name) or {}).get("hf_commit")
        if current_commit == latest:
            print(f"[mirror] {src.mirror_name}: up to date ({latest[:8]}).")
        else:
            print(f"[mirror] {src.mirror_name}: OUTDATED ({current_commit} -> {latest[:8]}).")
            changed.append((src, latest))

    if not changed and not args.force:
        print("[mirror] nothing to do. Existing GitHub Release is still current.")
        return 0

    # 3) Download the changed files (plus any unchanged files we still need to
    #    re-attach so the new release is self-contained).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        download_specs: list[UpstreamFile] = []
        for src in SOURCES:
            latest = upstream_status[src.mirror_name]["hf_commit"]
            if latest is None:
                # Upstream unreachable for this file — reuse whatever we already
                # had (skip; a partial release is worse than no release).
                print(f"[mirror] skipping {src.mirror_name}: no upstream info this run.")
                continue
            download_specs.append(src)

        if not download_specs:
            print("[mirror] no downloadable files this run. Aborting.")
            return 0

        assets: list[Path] = []
        manifest: dict = {"generated_at": int(time.time()), "files": {}}
        for src in download_specs:
            dest = tmp / src.mirror_name
            hf_download(src.hf_repo, src.hf_path, src.hf_revision, dest)
            digest = _sha256(dest)
            manifest["files"][src.mirror_name] = {
                "tier": src.tier,
                "hf_repo": src.hf_repo,
                "hf_path": src.hf_path,
                "hf_revision": src.hf_revision,
                "hf_commit": upstream_status[src.mirror_name]["hf_commit"],
                "sha256": digest,
                "size_bytes": dest.stat().st_size,
            }
            assets.append(dest)

        # Write the manifest and attach it too.
        manifest_path = tmp / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2))
        assets.append(manifest_path)

        # 4) Publish new Release.
        tag = gh_next_tag(args.repo, token)
        notes_lines = [
            "Automated mirror of upstream anime upscaling model weights.",
            "",
            "The Colab notebook downloads these files directly and does not "
            "contact HuggingFace at runtime.",
            "",
            "**Files:**",
        ]
        for name, meta in manifest["files"].items():
            short = (meta["hf_commit"] or "")[:8]
            notes_lines.append(
                f"- `{name}` ({meta['tier']}) — from `{meta['hf_repo']}` @ `{short}`"
            )
        gh_publish_release(args.repo, tag, "\n".join(notes_lines), assets)
        print(f"[mirror] published {tag} with {len(assets)} assets.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
