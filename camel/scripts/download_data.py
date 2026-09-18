#!/usr/bin/env python3
"""Fetch the three benchmarks CAMEL is evaluated on.

    python3 -m camel.scripts.download_data              # all three
    python3 -m camel.scripts.download_data socialmem    # just one

Everything lands under `paths.root` from config.yaml (./data by default), in
the layout the loaders in camel/data.py expect:

    data/
      EverMemBench/dataset_download/<topic>/{dialogue.json,qa_<topic>.json}
      GroupMemBench/data/final/<domain>/synthetic_domain_channels_*.json
      SocialMemBench/{conversations,qa,networks,personas}.parquet

SocialMemBench is CC BY 4.0 and downloads directly. EverMemBench and
GroupMemBench are released by their own authors under their own terms, so this
script prints where to request them rather than mirroring copies we have no
right to redistribute.
"""
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camel import config  # noqa: E402

SOCIAL_BASE = ("https://huggingface.co/datasets/anon4data/socialmembench"
               "/resolve/main")
SOCIAL_FILES = ("conversations", "qa", "networks", "personas")

# We deliberately do not hardcode download URLs for these two: they are
# released by their own authors under their own terms, and a stale or guessed
# mirror URL is worse than none. The arXiv ID is the stable pointer.
MANUAL = {
    "evermem": ("EverMemBench", "arXiv:2602.01313", "dataset_download/"),
    "groupmem": ("GroupMemBench", "arXiv:2605.14498", "data/final/"),
}


def fetch_socialmem(root):
    dest = root / "SocialMemBench"
    dest.mkdir(parents=True, exist_ok=True)
    for name in SOCIAL_FILES:
        out = dest / f"{name}.parquet"
        if out.exists() and out.stat().st_size > 0:
            print(f"  {name}.parquet present ({out.stat().st_size // 1024} KB)")
            continue
        print(f"  downloading {name}.parquet ...", flush=True)
        try:
            urllib.request.urlretrieve(f"{SOCIAL_BASE}/{name}.parquet", out)
        except urllib.error.URLError as e:
            out.unlink(missing_ok=True)
            print(f"  FAILED: {e}\n  Fetch it by hand from {SOCIAL_BASE}")
            return False
        print(f"    {out.stat().st_size // 1024} KB")
    print(f"SocialMemBench ready at {dest}")
    return True


def explain_manual(key, root):
    name, url, subdir = MANUAL[key]
    target = root / name / subdir
    have = target.exists() and any(target.iterdir())
    status = "present" if have else "MISSING"
    print(f"\n{name}: {status}")
    if have:
        print(f"  {target}")
        return True
    print(f"  Distributed by its own authors; we do not redistribute it.")
    print(f"  See {url} for the release the authors point to, then place it")
    print(f"  so that this path exists:")
    print(f"    {target}")
    return False


def main():
    want = sys.argv[1:] or ["socialmem", "evermem", "groupmem"]
    root = Path(config.ROOT)
    root.mkdir(parents=True, exist_ok=True)
    print(f"data root: {root.resolve()}")

    ok = True
    if "socialmem" in want:
        print("\nSocialMemBench (CC BY 4.0):")
        ok &= fetch_socialmem(root)
    for key in ("evermem", "groupmem"):
        if key in want:
            ok &= explain_manual(key, root)

    print()
    if ok:
        print("All requested datasets are in place.")
    else:
        print("Some datasets still need to be fetched by hand (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
