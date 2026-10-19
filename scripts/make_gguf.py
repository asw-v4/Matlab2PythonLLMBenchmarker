"""Convert a pinned Hugging Face model to a Q8_0 GGUF and write its manifest.

    uv run --group convert python scripts/make_gguf.py olmo3-7b-instruct

Needs network (Hugging Face, GitHub). Run once per model, not during benchmarking.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bench import gguf_build as g  # noqa: E402

ALLOW = ["*.json", "*.safetensors", "*.txt", "*.jinja"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="key in configs/models.json")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "models.json")
    ap.add_argument("--llama-root", type=Path, default=Path(r"Z:\Coding\_tools\llama.cpp"))
    ap.add_argument("--models-root", type=Path, default=Path(r"Z:\Coding\_tools\models"))
    ap.add_argument("--keep-bf16", action="store_true", help="keep the ~15 GB intermediate file")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    cfg = g.load_config(args.config)
    m = cfg["models"][args.model]
    lc = cfg["llama_cpp"]
    out_dir = args.models_root / args.model
    src_dir = out_dir / "source"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] downloading {m['hf_repo']} @ {m['revision'][:10]}")
    snapshot_download(repo_id=m["hf_repo"], revision=m["revision"], local_dir=src_dir, allow_patterns=ALLOW)

    print(f"[2/5] llama.cpp source @ {lc['commit'][:10]}")
    llama_src = g.fetch_llama_source(args.llama_root, lc["commit"])

    bf16 = out_dir / f"{args.model}-bf16.gguf"
    final = out_dir / f"{args.model}-{m['quant']}.gguf"
    print("[3/5] converting to bf16 GGUF")
    subprocess.run(g.convert_command(sys.executable, llama_src, src_dir, bf16, "bf16"),
                   env=g.converter_env(os.environ, llama_src), check=True)

    print(f"[4/5] quantising to {m['quant']}")
    quantize = args.llama_root / lc["build"] / "vulkan" / "llama-quantize.exe"
    subprocess.run(g.quantize_command(quantize, bf16, final, m["quant"]), check=True)

    print("[5/5] hashing and writing manifest")
    src_hashes = {p.name: g.sha256_file(p) for p in sorted(src_dir.iterdir()) if p.is_file()}
    manifest = g.build_manifest(
        name=args.model, hf_repo=m["hf_repo"], revision=m["revision"], quant=m["quant"],
        llama_build=lc["build"], llama_commit=lc["commit"],
        backend_build_note=f"quantised with prebuilt {lc['build']} win-vulkan llama-quantize",
        source_files=src_hashes, gguf_path=final, gguf_sha256=g.sha256_file(final),
    )
    text = json.dumps(manifest, indent=2) + "\n"
    (out_dir / "manifest.json").write_text(text, encoding="utf-8", newline="\n")
    repo_manifest = REPO / "models" / "manifests" / f"{args.model}.json"
    repo_manifest.parent.mkdir(parents=True, exist_ok=True)
    repo_manifest.write_text(text, encoding="utf-8", newline="\n")
    if not args.keep_bf16:
        bf16.unlink(missing_ok=True)
    print(f"done: {final}\nmanifest: {repo_manifest}")


if __name__ == "__main__":
    main()
