# Third-party code and data

`tt_qwen36/` is this port (Apache-2.0) unless listed here. `tools/`, `examples/`, `tt-model.yaml` and
`pyproject.toml` are packaging. Everything below came from somewhere else and keeps its own licence.

| path | upstream | licence | licence text |
| --- | --- | --- | --- |
| `tt_qwen36/ref/third_party/strata_tokenizer.py` | Strata 0.2.12 (build 0.1.58) `tools/strata_tokenizer.py`, copied unchanged apart from the header line | MIT | `third_party_licenses/Strata-MIT.txt` |
| `tt_qwen36/q36/kernels/route.cpp`, `route_rows.cpp` (exp table and polynomial only) | Lottolabs Qwen3.6-35B-A3B runtime `routing_math.hpp` / `topk_routing.cpp` | Apache-2.0 | `LICENSE` |
| `code/models/common` (model repository / container image only) | [tt-metal](https://github.com/tenstorrent/tt-metal) `58648d4` | Apache-2.0 | tt-metal's `LICENSE` in the image |
| `bfp8/` (model repository only) | BF16 weights of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) `995ad96`, converted by `tools/convert_bfp8.py` | Apache-2.0 | `LICENSE`, `NOTICE` |

Design and numerics credits without copied code (see `NOTICE`): sjettTT/tt-qwen-3.8-flash-next (Apache-2.0) for the
ttnn op recipes the GDN / attention / MoE / MTP device code grew from.

The base weights (Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150, Apache-2.0) are downloaded from the Hugging Face
Hub at a pinned revision and are not redistributed here.
