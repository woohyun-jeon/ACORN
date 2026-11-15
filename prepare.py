import argparse
import os

from acorn.data import build_baer_context, build_baer_patches, build_hls_context, build_hls_patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hls", "baer"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    patches, context = os.path.join(a.out, "patches"), os.path.join(a.out, "context")
    if a.dataset == "hls":
        n = build_hls_patches(a.root, patches)
        m = build_hls_context(a.root, context)
    else:
        n = build_baer_patches(a.root, patches, a.workers)
        m = build_baer_context(a.root, context)
    print(f"{a.dataset}: {n} labelled tiles in {patches}, {m} context tiles in {context}")


if __name__ == "__main__":
    main()
