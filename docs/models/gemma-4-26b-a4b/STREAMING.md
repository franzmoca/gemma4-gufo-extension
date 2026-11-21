# Streaming the 26B-A4B's experts (experimental)

This branch (`feat/expert-stream`) can leave the 26B-A4B's routed experts in
the GGUF and read them on demand into a cache in RAM, using Gufo's experimental
expert store (`<gufo/ext/experimental/expert_stream.hpp>`, extension API
0.3.28). The store is not in a Gufo release; build against the Gufo branch
`feat/expert-stream` of franzmoca/gufo, which `flake.nix` pins on this branch.
The branch stays experimental: the 26B-A4B fits in memory on Strix Halo, so
resident weights remain the default and the right choice there. It is the
second user of the store after the DeepSeek V4.1 Flash extension, and shows
the API carries a model with a different layout.

## Use

```sh
gufo serve llm -m gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf --stream-experts
gufo serve llm -m gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf --stream-experts \
  --expert-cache-bytes 8589934592
```

- `--stream-experts` streams every layer's routed `[gate | up]` and down
  tensors; BF16 expert tensors (layer 29 of UD-Q8_K_XL) stay resident.
- `--expert-cache-bytes N` bounds the cache (it requires `--stream-experts`).
  Without it, the cache takes the memory left: serving sizes it after the
  sessions and the prompt cache (the runner declares 2 GiB of headroom and
  1 GiB of prompt cache per session), and `gufo prompt` and `gufo bench`
  right after loading, leaving 2 GiB free.
- Once sized, the cache is filled at startup with experts interleaved across
  layers, so the first requests wait less on reads. A cache that holds every
  streamed expert starts with all of them (UD-Q4_K_XL 13.4 GiB in about 3 s,
  UD-Q8_K_XL 21.9 GiB in about 6 s); a 4 GiB cache fills in about 1 s.
- `GEMMA4_EXPERT_TRACE=PATH` writes the store's routing trace, which Gufo's
  `tools/expert_stream/` replays offline.

## How it runs

Each layer routes as before; the executor then hands the routes to the store
on its own stream (`Prepare`), which reads the missing experts and returns one
device table per piece with each expert's address, in a cache slot or the read
stage. The routed GEMV, prefill GEMM and fused GeGLU kernels take that table
in place of a base pointer and stride; with resident weights they run
unchanged. The kernels read whole k-iterations, up to 4 KiB past an expert,
so the store keeps that margin readable after every slot and stage row.
At decode widths the experts are queued after the dense MLP on the expert
stream, since the host waits for the routes inside `Prepare`.

## Results

Strix Halo `gfx1151`, accelerator-performance profile, `gufo serve` with the
Q8_0 MTP drafter, 12 varied prompts × 256 tokens greedy, a fresh server per
arm, each arm run twice in reversed order. Every reply and draft count is
identical to the resident build in every arm, also on UD-Q6_K_XL and with a
2,855-token prompt (spot checks), and served through the declared cache.

| Quant | Arm | Decode (tok/s) | Prefill (tok/s) |
| --- | --- | ---: | ---: |
| UD-Q4_K_XL | resident | 106.3 | 555 |
| | streamed, every expert cached | 103.9 | 541 |
| | streamed, 8 GiB cache | 43.0 / 48.3 | |
| | streamed, 4 GiB cache | 16.7 / 16.9 | |
| | streamed, 2 GiB cache | 9.9 / 10.0 | |
| UD-Q8_K_XL | resident | 87.9 | 406 |
| | streamed, every expert cached | 86.2 | 399 |
| | streamed, 16 GiB cache | 53.2 / 55.3 | |
| | streamed, 8 GiB cache | 16.6 / 16.8 | |

With every expert cached, decode costs 2–2.5% (the per-layer wait for the
routes) and prefill 2–3%. Before the startup read, the same arm decoded at
88–89 (UD-Q4_K_XL) and 73–74 tok/s (UD-Q8_K_XL), reading 13 and 21 GiB on
first use. Smaller caches fall steeply: misses are read on demand at
2–3.7 GiB/s with nothing read ahead.

## Not done

- Prefetch for partial caches (`Store::Predict` from the router's next-layer
  scores, as DeepSeek V4.1 Flash does) and a ranking from routing
  histograms (`Policy::ranking`); both would raise the partial-cache rows above.
- An expert-tails file on a second drive (`Store::AttachTails`).
- The functional suites and the Pi agent replay were not run with streaming.
