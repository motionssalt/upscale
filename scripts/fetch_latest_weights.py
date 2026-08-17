#!/usr/bin/env python3
"""
MOTIONSALT Upscaler — weights auto-mirror helper.

Purpose:
    Check upstream sources for newer versions of the weight files this
    project uses (AnimeJaNai V3 SuperUltraCompact 2x, AnimeJaNai V3
    UltraCompact 2x, Real-ESRGAN AnimeVideo v3 2x). If any upstream file
    is newer than what is currently mirrored on this repo's latest GitHub
    Release, download the new files and publish a new Release with them
    attached.

    The Colab notebook never talks to upstream hosts itself — it only pulls
    from this repo's Releases. This script is the ONLY place upstream is
    contacted.

Usage:
    # Automated (from GitHub Actions):
    python scripts/fetch_latest_weights.py --repo owner/motionsalt-upscaler

    # Manual seed (first-ever release, before the Action has anything to
    # compare against — run this once locally with a PAT):
    GITHUB_TOKEN=ghp_xxx python scripts/fetch_latest_weights.py \
        --repo owner/motionsalt-upscaler --force

    # End-to-end test without publishing (probes + downloads + manifest):
    python scripts/fetch_latest_weights.py --repo owner/repo --force --dry-run

Behavior (consistent for every source):
    - Each source is probed and downloaded INDEPENDENTLY — one bad source
      can never abort the others.
    - Source failed, previous mirror exists   -> carry the previous file
      forward into the new Release, exit 0, but emit a ::warning::
      annotation and a job-summary section so it is visible on the
      Actions tab without opening the log.
    - Source failed, NO previous mirror       -> exit 1 with an ::error::
      annotation. A missing mirror means the notebook would 404, so this
      must be loud.
    - Every source failed                     -> exit 1, nothing published.
    - Sources OK, nothing changed             -> exit 0, do nothing.
    - Sources OK, something changed           -> new Release, exit 0.

Upstream layout notes (why the table below looks the way it does):
    - Real-ESRGAN AnimeVideo v3: `realesr-animevideov3.pth` is published by
      the original author (xinntao) as a GitHub Release asset on tag
      v0.2.5.0. The HuggingFace repo `ai-forever/Real-ESRGAN` does NOT host
      this file (only RealESRGAN_x2/x4/x8.pth) — pointing there 404s.
    - AnimeJaNai V3: the HuggingFace repo
      `the-database/mpv-upscale-2x_animejanai` does not exist (HF returns
      401 for nonexistent repos as an anti-enumeration measure, which made
      this look like an auth problem). The models are actually distributed
      on GitHub Releases, bundled inside
      `2x_AnimeJaNai_HD_V3_ModelsOnly.zip` on tag 3.0.0 — so those two
      entries below name the zip asset plus the member to extract.
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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Config — the exact upstream sources this project mirrors.
# ---------------------------------------------------------------------------
# Each entry says: "the upstream asset identified by (kind, repo, tag,
# asset_name) is what we save into the GitHub Release as <mirror_name>".
# `mirror_name` must stay byte-identical to the filenames hard-coded in the
# Colab notebook — the notebook downloads
# https://github.com/<repo>/releases/download/<tag>/<mirror_name>.
#
# kind="gh-release": file lives as (or inside) a GitHub Release asset.
#     member=None            -> the asset IS the file.
#     member="dir/file.pth"  -> the asset is a zip; extract this member.
# kind="hf": file lives in a HuggingFace repo at hf_repo/hf_path@hf_revision.
#     If the HF repo is ever gated, set an HF_TOKEN repo secret and it will
#     be sent as a Bearer token automatically.
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
HF_RESOLVE = "https://huggingface.co"  # /<repo>/resolve/<rev>/<path>

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Source:
    tier: str                        # "LOW" / "MEDIUM" / "HIGH"
    mirror_name: str                 # filename inside our GitHub Release
    kind: str = "gh-release"         # "gh-release" | "hf"
    # gh-release fields:
    repo: str = ""                   # "owner/name" on GitHub
    tag: str = ""                    # release tag
    asset_name: str = ""             # release asset filename
    member: Optional[str] = None     # path inside the asset, if it is a zip
    # hf fields:
    hf_repo: str = ""                # "namespace/repo" on HuggingFace
    hf_path: str = ""                # file path inside that repo
    hf_revision: str = "main"

    def describe(self) -> str:
        if self.kind == "gh-release":
            loc = f"github:{self.repo}@{self.tag}/{self.asset_name}"
            if self.member:
                loc += f"::{self.member}"
            return loc
        return f"hf:{self.hf_repo}/{self.hf_path}@{self.hf_revision}"


SOURCES: list[Source] = [
    Source(
        tier="LOW",
        mirror_name="2x_AnimeJaNaiV3_SuperUltraCompact.pth",
        kind="gh-release",
        repo="the-database/mpv-upscale-2x_animejanai",
        tag="3.0.0",
        asset_name="2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
        member="2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth",
    ),
    Source(
        tier="MEDIUM",
        mirror_name="2x_AnimeJaNaiV3_UltraCompact.pth",
        kind="gh-release",
        repo="the-database/mpv-upscale-2x_animejanai",
        tag="3.0.0",
        asset_name="2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
        member="2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_UltraCompact.pth",
    ),
    Source(
        tier="HIGH",
        mirror_name="realesr-animevideov3.pth",
        kind="gh-release",
        repo="xinntao/Real-ESRGAN",
        tag="v0.2.5.0",
        asset_name="realesr-animevideov3.pth",
        member=None,
    ),
]


# ---------------------------------------------------------------------------
# Small HTTP helpers (stdlib only — the Action needs no pip install step).
# ---------------------------------------------------------------------------

USER_AGENT = "motionsalt-upscaler-mirror"


def _http_get(url: str, headers: Optional[dict] = None, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_json(url: str, headers: Optional[dict] = None, timeout: int = 60) -> dict:
    return json.loads(_http_get(url, headers=headers, timeout=timeout).decode("utf-8"))


def _download(url: str, dest: Path, headers: Optional[dict] = None, timeout: int = 600) -> None:
    req = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# GitHub Actions visibility helpers — every failure mode is surfaced either
# as an annotation (shows on the run's Summary page) or in the step summary
# markdown, never only as a raw log line.
# ---------------------------------------------------------------------------


def _summary(markdown: str) -> None:
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
    print(markdown)


def _annotate(level: str, message: str) -> None:
    # level: "warning" | "error" — rendered on the Actions run summary page.
    safe = message.replace("\n", "%0A").replace("\r", "")
    print(f"::{level}::{safe}")


def _summary_table(results: list["FetchResult"], current_files: dict) -> None:
    _summary("| Mirror file | Tier | Upstream status | Version |")
    _summary("| --- | --- | --- | --- |")
    for r in results:
        if r.error:
            status = f":x: failed — {r.error[:80]}"
            version = (current_files.get(r.src.mirror_name) or {}).get("version") \
                or (current_files.get(r.src.mirror_name) or {}).get("hf_commit") \
                or "—"
        else:
            status = ":white_check_mark: ok"
            version = r.version or "—"
        _summary(f"| `{r.src.mirror_name}` | {r.src.tier} | {status} | {version} |")


# ---------------------------------------------------------------------------
# Upstream probing: figure out the "current" version identity of each source.
# For GitHub Release assets we use "<asset_id>@<updated_at>" — it changes
# whenever the author re-uploads the asset. For HF we use the last commit.
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    src: Source
    version: Optional[str] = None
    error: Optional[str] = None
    downloaded_path: Optional[Path] = None


def gh_release_asset_meta(repo: str, tag: str, asset_name: str,
                          token: Optional[str] = None,
                          _release_cache: dict = {}) -> dict:
    """Return the asset metadata dict for a release asset. Raises on failure."""
    # Fall back to GITHUB_TOKEN if no token was passed explicitly. Even for
    # PUBLIC upstream repos this matters: unauthenticated GitHub API allows
    # only 60 requests/hour per IP, which shared Actions runners can burn
    # through quickly. Authenticated requests get 5000/hour.
    if not token:
        token = os.environ.get("GITHUB_TOKEN")
    key = (repo, tag)
    if key not in _release_cache:
        url = f"{GITHUB_API}/repos/{repo}/releases/tags/{tag}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        _release_cache[key] = _http_json(url, headers=headers)
    rel = _release_cache[key]
    for asset in rel.get("assets", []) or []:
        if asset.get("name") == asset_name:
            return asset
    raise FileNotFoundError(
        f"asset {asset_name!r} not found on release {repo}@{tag} "
        f"(has: {[a.get('name') for a in rel.get('assets', [])]})"
    )


def hf_latest_commit(repo: str, path: str, revision: str,
                     hf_token: Optional[str]) -> str:
    """Last commit SHA touching this file on HF. Raises on failure."""
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    data = _http_json(
        f"https://huggingface.co/api/models/{repo}/commits/{revision}?path={path}",
        headers=headers,
    )
    if isinstance(data, list) and data:
        return data[0].get("id") or data[0].get("commit", {}).get("id")
    if isinstance(data, dict) and (data.get("id") or data.get("sha")):
        return data.get("id") or data.get("sha")
    raise RuntimeError(f"no commit info returned for {repo}/{path}@{revision}")


def probe_source(src: Source, hf_token: Optional[str]) -> FetchResult:
    """Probe one source. NEVER raises — errors come back on the result."""
    try:
        if src.kind == "gh-release":
            asset = gh_release_asset_meta(src.repo, src.tag, src.asset_name)
            version = f"{asset['id']}@{asset.get('updated_at', '?')}"
        else:
            version = hf_latest_commit(src.hf_repo, src.hf_path,
                                       src.hf_revision, hf_token)
        return FetchResult(src=src, version=version)
    except HTTPError as e:
        return FetchResult(src=src, error=f"HTTP {e.code} {e.reason} ({src.describe()})")
    except (URLError, TimeoutError, OSError, json.JSONDecodeError,
            FileNotFoundError, RuntimeError, KeyError) as e:
        return FetchResult(src=src, error=f"{e} ({src.describe()})")


# ---------------------------------------------------------------------------
# Downloading: each source independently; zip members extracted in place.
# ---------------------------------------------------------------------------


def download_source(src: Source, dest_dir: Path, hf_token: Optional[str],
                    _asset_cache: dict = {}) -> Path:
    """Download one source into dest_dir/mirror_name. Raises on failure."""
    dest = dest_dir / src.mirror_name
    if src.kind == "gh-release":
        asset = gh_release_asset_meta(src.repo, src.tag, src.asset_name)
        url = asset["browser_download_url"]
        if src.member:
            # Two AnimeJaNai entries share one zip — download it once.
            key = (src.repo, src.tag, src.asset_name)
            if key not in _asset_cache:
                zpath = dest_dir / f".cache-{src.asset_name}"
                print(f"[gh-release] downloading {url}")
                _download(url, zpath)
                _asset_cache[key] = zpath
            with zipfile.ZipFile(_asset_cache[key]) as zf:
                with zf.open(src.member) as zin, dest.open("wb") as out:
                    shutil.copyfileobj(zin, out, length=1 << 20)
        else:
            print(f"[gh-release] downloading {url}")
            _download(url, dest)
    else:
        url = f"{HF_RESOLVE}/{src.hf_repo}/resolve/{src.hf_revision}/{src.hf_path}"
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        print(f"[hf] downloading {url}")
        _download(url, dest, headers=headers)
    size = dest.stat().st_size
    if size == 0:
        raise RuntimeError(f"downloaded 0 bytes for {src.mirror_name}")
    print(f"  [OK] {src.mirror_name}: {size:,} bytes")
    return dest


# ---------------------------------------------------------------------------
# Our own GitHub Release side: read manifest, carry forward, publish.
# ---------------------------------------------------------------------------


def gh_latest_release(repo: str, token: str) -> Optional[dict]:
    url = f"{GITHUB_API}/repos/{repo}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        return _http_json(url, headers=headers)
    except HTTPError as e:
        if e.code == 404:
            return None
        raise


def gh_latest_release_manifest(repo: str, token: str) -> tuple[Optional[dict], dict]:
    """
    Returns (manifest_dict_or_None, {asset_name: browser_download_url}).
    The asset map lets us carry forward previous weight files when an
    upstream source is broken but we already mirrored it before.
    """
    rel = gh_latest_release(repo, token)
    if rel is None:
        return None, {}
    asset_urls = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    if MANIFEST_NAME not in asset_urls:
        return {"files": {}}, asset_urls
    manifest = json.loads(_http_get(asset_urls[MANIFEST_NAME]).decode("utf-8"))
    return manifest, asset_urls


def gh_next_tag(repo: str, token: str) -> str:
    url = f"{GITHUB_API}/repos/{repo}/releases"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        releases = _http_json(url, headers=headers)
    except HTTPError:
        releases = []
    latest = (0, 0, 0)
    for rel in releases or []:
        m = re.match(r"weights-v(\d+)\.(\d+)\.(\d+)$", rel.get("tag_name", ""))
        if m:
            v = tuple(int(x) for x in m.groups())
            if v > latest:
                latest = v
    if latest == (0, 0, 0):
        return "weights-v1.0.0"
    maj, minr, _ = latest
    return f"weights-v{maj}.{minr + 1}.0"


def gh_publish_release(repo: str, tag: str, notes: str, files: list[Path]) -> None:
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
    ap = argparse.ArgumentParser(
        description="Mirror upstream weight files into this repo's Releases.")
    ap.add_argument("--repo", required=True,
                    help="owner/name of THIS repo (e.g. someone/motionsalt-upscaler)")
    ap.add_argument("--force", action="store_true",
                    help="Publish even if nothing changed (seed initial release).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Probe + download + build manifest, but publish nothing.")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: GITHUB_TOKEN environment variable is required.", file=sys.stderr)
        return 2
    hf_token = os.environ.get("HF_TOKEN")  # only needed if an HF source is gated

    _summary("## Weights mirror — run report")
    _summary("")

    # 1) What do we currently mirror?
    print(f"[mirror] reading current manifest from {args.repo} …")
    manifest, prev_asset_urls = ({}, {}) if args.dry_run and not token else \
        gh_latest_release_manifest(args.repo, token or "")
    current_files = (manifest or {}).get("files", {}) if manifest is not None else {}

    # 2) Probe every source, each one isolated from the others.
    results: list[FetchResult] = []
    for src in SOURCES:
        r = probe_source(src, hf_token)
        results.append(r)
        if r.error:
            print(f"[mirror] {src.mirror_name}: PROBE FAILED — {r.error}")
        else:
            print(f"[mirror] {src.mirror_name}: upstream version={r.version}")

    def old_version(name: str) -> Optional[str]:
        entry = current_files.get(name) or {}
        return entry.get("version") or entry.get("hf_commit")

    changed: list[FetchResult] = []
    probe_failures: list[FetchResult] = []
    for r in results:
        if r.error:
            probe_failures.append(r)
        elif old_version(r.src.mirror_name) != r.version:
            changed.append(r)

    # 3) A probe failure with NO previous mirror is a hard error: that file
    #    has never been mirrored, so the notebook would 404 on it. This is
    #    the "silent stale-forever" bug — it must fail loudly, and it must
    #    do so whether or not anything else changed.
    unresolved = [r for r in probe_failures if r.src.mirror_name not in prev_asset_urls]
    if unresolved:
        _annotate("error",
                  "Upstream sources failed and no previous mirror exists to fall "
                  "back on: " + "; ".join(f"{r.src.mirror_name} ({r.error})"
                                          for r in unresolved))
        _summary("### :x: Unresolvable source failures (no prior mirror)")
        for r in unresolved:
            _summary(f"- **{r.src.mirror_name}** — {r.error}")
        _summary("")
        _summary_table(results, current_files)
        return 1

    # 4) Nothing changed and not forcing: warn about carried-forward files,
    #    then exit cleanly.
    if not changed and not args.force:
        _summary("_All reachable sources are up to date._")
        _summary("")
        _summary_table(results, current_files)
        for r in probe_failures:
            _annotate("warning",
                      f"{r.src.mirror_name}: upstream probe failed "
                      f"({r.error}); existing mirror left in place.")
        return 0

    # 5) Download every source we can; carry forward previous files for
    #    sources that are broken this run.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assets: list[Path] = []
        new_manifest: dict = {"generated_at": int(time.time()), "files": {}}
        carried_forward: list[str] = []
        download_failures: list[FetchResult] = []

        for r in results:
            if r.error:
                continue  # handled below via carry-forward (all have priors here)
            try:
                path = download_source(r.src, tmp, hf_token)
                r.downloaded_path = path
            except Exception as e:  # noqa: BLE001 — isolation is the point
                r.error = f"download failed: {e}"
                r.downloaded_path = None
                download_failures.append(r)
                continue
            new_manifest["files"][r.src.mirror_name] = {
                "tier": r.src.tier,
                "source": r.src.describe(),
                "version": r.version,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            assets.append(path)

        # Any source that failed (probe or download) carries its previous
        # mirror forward, so the new Release stays self-contained.
        for r in probe_failures + download_failures:
            prev_url = prev_asset_urls.get(r.src.mirror_name)
            prev_meta = current_files.get(r.src.mirror_name) or {}
            dest = tmp / r.src.mirror_name
            print(f"[mirror] carrying forward previous {r.src.mirror_name}")
            _download(prev_url, dest)
            new_manifest["files"][r.src.mirror_name] = {
                **prev_meta,
                "tier": r.src.tier,
                "carried_forward": True,
                "upstream_error": r.error,
            }
            assets.append(dest)
            carried_forward.append(r.src.mirror_name)

        succeeded = [r for r in results
                     if r.error is None and r.downloaded_path is not None]

        # A run where nothing was fetched and nothing could be carried is
        # unpublishable — we'd be cutting an empty Release.
        if not assets:
            _annotate("error",
                      "Every upstream source failed this run. No new Release "
                      "published.")
            _summary("### :x: Total outage — nothing could be mirrored")
            _summary_table(results, current_files)
            return 1

        for r in probe_failures + download_failures:
            _annotate("warning",
                      f"{r.src.mirror_name}: {r.error} — carried previous "
                      f"mirror forward into the new Release.")

        manifest_path = tmp / MANIFEST_NAME
        manifest_path.write_text(json.dumps(new_manifest, indent=2))
        assets.append(manifest_path)

        # 6) Publish (skipped under --dry-run).
        tag = gh_next_tag(args.repo, token) if token else "weights-vDRYRUN"
        notes = [
            "Automated mirror of upstream anime upscaling model weights.",
            "",
            "The Colab notebook downloads these files directly and never "
            "contacts upstream hosts at runtime.",
            "",
            "**Files:**",
        ]
        for name, meta in new_manifest["files"].items():
            line = f"- `{name}` ({meta['tier']}) — {meta.get('source', 'prior mirror')}"
            if meta.get("carried_forward"):
                line += " — ⚠️ carried forward (upstream fetch failed this run)"
            notes.append(line)

        _summary_table(results, new_manifest["files"])
        if carried_forward:
            _summary("")
            _summary("### :warning: Carried forward (upstream failed this run)")
            for name in carried_forward:
                _summary(f"- `{name}`")

        if args.dry_run:
            print(f"[dry-run] would publish {tag} with {len(assets)} assets:")
            for a in assets:
                print(f"[dry-run]   {a.name} ({a.stat().st_size:,} bytes)")
            return 0

        gh_publish_release(args.repo, tag, "\n".join(notes), assets)
        print(f"[mirror] published {tag} with {len(assets)} assets.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
