# PUBLISH-READY: release checklist for tt-qwen3.6-35b-a3b-p100a

Status: **ready to publish** (2026-11-14). Image built, tested end to end on a p100a from fresh caches, leak gate
clean. Nothing has been pushed, uploaded or created on GitHub or the Hugging Face Hub; the steps under
*To publish* are the owner's.

## What is in this repository

| | |
|---|---|
| code | `tt_qwen36/` (server, HTTP layer, model and kernels), `tools/convert_bfp8.py`, `examples/` |
| package | `tt-model.yaml` (tt-model 5.1 container, kind `http-server`, tt-metal `58648d4`) |
| model card draft | `hf/README.md` (goes to the HF repo as `README.md`, merged with the generated Provenance section) |
| licences | `LICENSE` (Apache-2.0), `NOTICE`, `THIRD-PARTY.md`, `third_party_licenses/Strata-MIT.txt` |
| notes | `PORTING_NOTES.md` |

The would-be HF repo (`sunandbean/qwen3.6-35b-a3b-p100a`) is the directory `tt-model package --container` stages,
plus four additions made by the private staging step:

| path | size | from |
|---|---:|---|
| `image/` | 935.5 MB (24 blobs) | OCI layout of image `tt-model/qwen3.6-35b-a3b-p100a:306f2a55e613` (4.18 GB unpacked) |
| `code/` | 6 MB | `tt_qwen36/`, `tools/` (this repo) + tt-metal `models/common`, + `THIRD-PARTY.md` and licence texts |
| `tt_kernel_manifest.json`, `requirements.lock` | small | tt-model (build host name scrubbed) |
| `README.md` | small | `hf/README.md` + the generated Provenance section |
| `LICENSE`, `NOTICE`, `PORTING_NOTES.md` | small | this repo |
| `bfp8/` | **6.01 GB** (272 tensorbins + `SHA256SUMS`) | the step-7 conversion, byte-identical to `tools/convert_bfp8.py` output |

## Decision: BFP8 upgrade layers (owner, 2026-11-14): option (b)

Ship the converted BFP8 tensors in our own HF repo (`bfp8/`, Apache-2.0, `NOTICE` names the source: Qwen BF16 at
revision `995ad96eacd98c81ed38be0c5b274b04031597b0`; the conversion script is included), **on by default**. The
Lottolabs checkpoint stays the base weight source, pinned to `c355c41d4fd079c6df3c4ba9bab1c40f37c17f09`.
Off switch: `Q36_BFP8_SOURCE=off` (also the `long-262k` profile).

**Size correction:** the "+2.84 GB" is the extra *card DRAM* over the BFP4 copies the upgrade replaces. The files
are the full BFP8 matrices: **272 files, 6.01 GB**. Option (b) therefore adds 6.0 GB to the HF repo and to every
user's first download, not 2.8 GB.

Each option is one setting (`serve.env` `Q36_BFP8_SOURCE` in `tt-model.yaml`):

| | (a) user converts locally `bf16` | **(b) ship `bfp8/` (chosen)** `hf` | (c) default off `off` |
|---|---|---|---|
| extra download | ~67 GB (Qwen BF16 safetensors) | **6.0 GB** (`bfp8/` of our repo) | 0 |
| total download | ~88 GB | **~27 GB** | ~21 GB |
| extra disk | 67 GB BF16 + 6.0 GB converted cache | **6.0 GB** | 0 |
| first start, extra work | the 67 GB download + 37 s converting 272 files on the CPU | **the 6.0 GB download + one sha256 pass** | none |
| later starts | ~35 s | **~35 s** | ~35 s |
| KL vs BF16 / top-1 (1,532 positions) | 0.0130 / 0.971 | **0.0130 / 0.971** | 0.0173 / 0.963 |
| weights on the card | 23.75 GB | **23.75 GB** | 20.91 GB |
| 262k context | no (off needed) | no (off needed) | yes |
| what we host | nothing derived | **derived weights (Apache-2.0, NOTICE)** | nothing derived |

Conversion time for (a): **37 s** for all 272 files on this box's CPU, from a local BF16 copy (2026-11-14,
`tools/convert_bfp8.py --check` in the 58648d4 release image, no card). The 67 GB download dominates (a) completely.
**Reproducible:** that run produced **272/272 files byte-identical** to `tt_qwen36/bfp8_sha256.txt`, i.e. to the files
(b) ships, which are the files step 7's gate measured.

## Leak gate (mandatory)

The gate is a private script (`q36pkg_leakgate.py`, kept outside this repository because its pattern list itself
names the private identifiers). It scans:

- **files:** every tracked and untracked (not ignored) file of this repo, and the staged HF repo dir (merged card,
  `tt_kernel_manifest.json`, `code/`, licences; `bfp8/` tensorbins by name only).
- **history:** `git log --all -p --format=raw` (authors, committers, messages, every diff), every ref, `.git/config`.
- **image:** `docker image inspect` (labels, env, cmd) and `docker history`, then every file in every layer
  (`docker save`). The image in the staged OCI layout `image/` has the same manifest digest
  (`sha256:306f2a55e613…`, 23 layers).

Two tiers of patterns:
- **Owner-specific (block anywhere, image included):** owner name and handle (the package's own two repo ids are
  allowlisted), build and workstation host names, this box's IPs, private service and lease names, private tree
  names, tokens and keys.
- **Broad (block in the package, history, image config and the package code in the image; counted elsewhere):**
  home-directory paths, e-mail addresses, private IPv4 ranges, the carrier-grade NAT range (Tailscale),
  university-account words, private-service words, workflow words.

Allowed exactly: the package's two repo ids, the public identity (`Sunbin Kim`,
`40445991+SunandBean@users.noreply.github.com`), the Co-Authored-By noreply address, and `/home/tt` (the image
user's home).

Command:

```sh
python3 q36pkg_leakgate.py --repo tt-qwen3.6-35b-a3b-p100a \
  --image tt-model/qwen3.6-35b-a3b-p100a:306f2a55e613 --extra build/qwen3.6-35b-a3b-p100a
```

Found and fixed on the way:
- tt-model writes the build host name into `tt_kernel_manifest.json` (`producer.hostname`). The staging step sets
  it to null; re-check after every `tt-model package`.
- The owner's name appeared in this checklist and in its history before the identity was chosen. The text is fixed
  and the history was squashed to one root commit.

Upstream text in the image (counted, not blocking): home-directory paths 563, e-mail addresses 6,786,
university-account words 285, private IPv4 99, service words 241, CGNAT addresses 18. These are Debian copyright
files, Python's `ipaddress`, sympy and similar; the owner-specific tier has 0 hits in all 58,701 image files.

Final result, on this repository as committed (one root commit, public identity), image `306f2a55e613` and the
staged HF repo dir (2026-11-14 20:36):

```
leak gate: {"files scanned": 440, "data files (names only)": 272, "commits scanned": 1, "image layers": 24,
            "image files": 58701, "image text files scanned": 41386}
CLEAN: 0 blocking hits
```

## Local end-to-end test (as an outsider)

Fresh caches only: a new empty Hugging Face cache filled from the Hub without a token (Lottolabs `checkpoint/` at
`c355c41`: 2,033 files, 21 GB, all 2,033 identical to Lottolabs' SHA256SUMS), a new empty kernel cache. `bfp8/` cannot
come from the Hub before the upload, so its 272 files were placed in the model's tt-model weight cache, which the
default source (`hf`) uses before downloading. The download path itself is untested until the repo exists: test it
first after the private push (`tt-model serve` on a machine without the files).

**Part 1, the package code on the card** (tt-metal's `58648d4` release image, read-only package mount,
2026-11-14 18:07-18:09):

| | |
|---|---|
| start, cold kernel cache | weights 23.75 GB in 33 s, model ready in 73 s, first request accepted 80 s after `docker run` |
| `GET /v1/models` | `qwen3.6-35b-a3b-p100a` |
| OpenAI `/v1/chat/completions` (examples/openai_chat.py, greedy, 31-token prompt) | correct 2-sentence answer, 57 tokens, **83.0 tok/s** decode, 33/52 drafts accepted |
| Anthropic `/v1/messages` (examples/anthropic_messages.py) | correct 3-item list, `end_turn`, 80 tokens, 1.24 s |
| OpenAI again (warm, 28 of 31 prompt tokens from the prefix cache) | **90.0 tok/s**, 41/62 drafts accepted; same answer up to a near-tie verb in the second sentence |
| exit | server stopped by itself after 1 idle minute |

The first attempt (17:53) exited after 3 s: tt-metal writes `generated/` under the working directory, which was the
read-only package mount. Fixed in the server (it moves to a writable directory first).

**Part 2, the tt-model image** `tt-model/qwen3.6-35b-a3b-p100a:306f2a55e613` (2026-11-14 20:05-20:08, between two of
another job's batches on the shared card):

The `docker run` is exactly what `tt-model serve --print` composes for this package, with chip 0:
`--device /dev/tenstorrent/0`, hugepages, the HF cache on `/hf`, tt-model's per-model kernel, weight and tensor
caches, `TT_MODEL_WEIGHTS_REVISION`, `Q36_BFP8_SOURCE=hf`, and the command `python -m tt_qwen36.server --host 0.0.0.0
--port 8098`. Four changes were needed for the shared test box's rules: `-d`, `--rm`, a different container name, and
`--idle-exit 1` appended. tt-model's own `serve` refuses on that box (see Known issues).

| | |
|---|---|
| start, fresh caches, cold kernel cache | weights 23.75 GB in 33 s, model ready in 73 s, ready line 80 s after `docker run` |
| `GET /v1/models` | `qwen3.6-35b-a3b-p100a` |
| OpenAI `/v1/chat/completions` (greedy, 31-token prompt) | correct answer, 57 tokens, **83.1 tok/s**, 33/52 drafts accepted; word for word the answer of part 1 |
| Anthropic `/v1/messages` | correct 3-item list, `end_turn`, 80 tokens, 1.24 s; same as part 1 |
| OpenAI again (warm, 28/31 prompt tokens cached) | **91.2 tok/s**, 41/62 drafts accepted |
| exit | stopped by itself after 1 idle minute |

The first image (`fd23726bd43a`, 19:25) exited after 10 s: `No module named 'regex'` (the tokenizer's
pre-tokenizer; the release image has it, the tt-model image's venv did not). Fixed: `regex` is in
`runtime.packages`, and the image's build-time `verify:` now imports every module of the package, so a missing
dependency fails the build instead of a user's first start.

## To publish (owner)

Done (2026-11-14):
- Copyright holder in `NOTICE`: Sunbin Kim.
- The single root commit is authored and committed as `Sunbin Kim <40445991+SunandBean@users.noreply.github.com>`
  (the approved public identity; allowlisted in the leak gate, which is clean again with it).
- A real `tt-model serve` of the local package on a p100a (below).

Remaining, in this order:
1. Create the GitHub repo `SunandBean/tt-qwen3.6-35b-a3b-p100a` (empty), push `main`, tag `v0.1.0`
   (`tt-model.yaml` points `extra_code` at that tag).
2. Rebuild from the GitHub ref: `tt-model package --container tt-model.yaml`. The local build used a bare clone at
   a neutral local path; the code is the same commit, only the recorded source differs. The tt-metal build is
   cached, so this takes minutes.
3. Stage with the private `q36pkg_stage.py`: it merges the card, copies the licences, adds `bfp8/` and **nulls
   `producer.hostname`** in `tt_kernel_manifest.json`. Run the leak gate on the repo, the rebuilt image and the
   staged dir; every hit blocks.
4. `tt-model push <dir> --private`, check the repo page, then test the download path on a machine **without**
   `bfp8/` in its caches: `tt-model serve sunandbean/qwen3.6-35b-a3b-p100a` must fetch `bfp8/` (6.0 GB) from the
   Hub. Optionally pin `Q36_BFP8_REVISION` to the commit that added `bfp8/`.
5. Make the repo public (`--publish` to list it in the tt-model catalog).
6. Remove this file, or keep it as release notes.

## Real `tt-model serve` test (2026-11-14 21:29-21:31)

The way a user serves a locally built package: `tt-model serve build/qwen3.6-35b-a3b-p100a/tt_kernel_manifest.json`
(tt-model's documented local path; the Hub path needs the published repo). Default profile, default port, tt-model's
own boot watch and `tt-model stop`. On the shared test box the always-on photo service, which holds a card grant,
was stopped for the test (about 1.5 min) and restarted. Weights: `HF_HOME` = the fresh cache from the outsider
download; tt-model's per-model caches in the default `~/.cache/tt-model/qwen3.6-35b-a3b-p100a`, with `bfp8/` placed
in its `weights/` because the HF repo does not exist yet.

| | |
|---|---|
| boot checklist | weights `Lottolabs/...@c355c41d` on host, host ready (docker, tt devices, hugepages), image `306f2a55e613`, container started |
| ready | **1 min 20 s**, endpoint `http://127.0.0.1:20000` |
| `GET /v1/models` | `qwen3.6-35b-a3b-p100a` |
| OpenAI `/v1/chat/completions` | correct answer, 57 tokens, **84.0 tok/s** |
| Anthropic `/v1/messages` | correct answer, `end_turn`, 80 tokens, 1.23 s |
| `tt-model stop` | clean shutdown, 829 ms |

The photo service was back right after its restart (`status: "ok"`, `error: null`, holding the card) and reported
`idle` again once it yielded the card to the next job (owner-verified 21:39). The owner accepted this as healthy.

## Known issues (not blocking)

- `tt-model serve` refuses on a machine where any other container already holds `/dev/tenstorrent`, even an idle
  one; this is tt-model's busy-chip check, not the package. Users who share the card with another service have to
  stop that service first (as in the test above).
- `P100A_Q36_FUSED=0` (the unfused GDN path, a debug switch) references an undefined `qkv`. It was already in the
  gated step-7 code; the default fused path is unaffected.
- Greedy output can differ at a near-tie token between a cold request and one resumed from the prefix cache (on the
  model card).
