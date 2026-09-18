# ipt_tilde — guide for LLM agents and assistants

This file is written for a language model (Claude Code, Codex, Cursor, Copilot, a
chat assistant fed with repository context) helping someone build, debug or
extend the `ipt~` Max/MSP external and the `pipo.ipt` MuBu module. It states how
the pieces fit together, the rules that are not obvious from the code, and the
traps that cost the most time. Everything here was checked against the `dev`
branch at v1.2.0. When this file and the code disagree, the code wins.

Humans: the README is the friendlier entry point. This file is denser on purpose.

---

## 1. What this repository is

Two Max externals that run a TorchScript playing-technique classifier on live
audio, plus a CLI example, all consuming the same C library:

| Component | Path | Built with | Role |
|---|---|---|---|
| `ipt~` | `ipt_tilde/ipt_tilde.cpp` | min-api (C++ Max SDK) | signal-rate object: audio in, class index / name / distribution out |
| `pipo.ipt` | `pipo.ipt/PiPoIPT.h`, `pipo.ipt/pipo.ipt.cpp` | pipo-sdk | PiPo module for MuBu hosts: `pipo~` (real time) and `mubu.process` (offline) |
| `ipt_example` | `app/ipt_example/main.cpp` | plain C | smallest possible consumer of the C ABI; used by CI as a runtime check |
| libipt | `libipt/` (git submodule) | CMake, libtorch 2.4.1 | the inference core behind a C ABI: `libipt/include/ipt.h` |

Models are trained and exported by
[ipt_recognition](https://github.com/nbrochec/ipt_recognition), which has its
own `AGENTS.md`. This repo never touches torch directly: everything goes through
`ipt.h`. Keep it that way.

---

## 2. Repository map

```
CMakeLists.txt        root: PACKAGE_VERSION (single source of the version), adds libipt, app, ipt_tilde, pipo.ipt
ipt_tilde/            ipt~ source, its CMake, a min-api unit test (ipt_tilde_test.cpp)
pipo.ipt/             PiPoIPT.h (the module), pipo.ipt.cpp (Max wrapper), demo patch, PiPoIPT_schema.md
app/ipt_example/      CLI consumer of ipt.h
libipt/               submodule, tracked on its `dev` branch (.gitmodules: branch = dev)
min-api/              submodule, Cycling '74 min-api
pipo.ipt/pipo-sdk/    submodule, IRCAM pipo-sdk (branch develop)
externals/            build output (.mxo / .mxe64); gitignored
support/              Windows build output (ipt.dll + torch DLLs); gitignored
build/, build-xcode/, build_libipt/   CMake trees; gitignored
media/                logo
.github/workflows/ci.yml   macOS arm64 + Windows x64 build, runtime check, unit test
```

Clone with `--recurse-submodules`. A tree without `libipt/include/ipt.h` is a
tree without submodules.

---

## 3. Building

macOS, Apple Silicon only (libtorch is arm64-only here, no universal binary):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target ipt_tilde pipo.ipt -j 8
```

Windows x64 (Visual Studio 2022):

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config Release -j 8
```

Facts that matter when a build fails:

- `CMAKE_POLICY_VERSION_MINIMUM=3.5` is needed with CMake ≥ 4 because
  min-api's test mock declares an ancient `cmake_minimum_required`.
- On first configure, libipt's CMake **downloads libtorch** into
  `libipt/libs/libtorch` (macOS: the Anaconda `pytorch-2.4.1-py3.12_0` build,
  about 20× faster on CPU than the pytorch.org macOS binary; Windows: the
  official MSVC CPU zip). The download is retried three times and checked
  against a pinned SHA-256 in `libipt/CMakeLists.txt`. Same torch version on
  both platforms, so `.ts` models are interchangeable.
- On macOS, libipt pins the sysroot to Xcode's bundled SDK: the Command Line
  Tools SDK ships a libc++ that calls `__builtin_clzg`, which the libtorch 2.4
  headers cannot compile with Apple clang 16. If configure cannot find Xcode's
  SDK, `xcode-select -p` must point at `/Applications/Xcode.app/...`.
- `static library kineto_LIBRARY-NOTFOUND not found` is a harmless warning
  from the Anaconda torch build.
- macOS output is a self-contained bundle: `externals/<name>.mxo/Contents/MacOS/`
  holds the external, `libipt.dylib`, `libtorch.dylib`, `libtorch_cpu.dylib`
  and `libc10.dylib`, with `@loader_path` rpaths and an ad-hoc `codesign`.
  Windows output is a flat `.mxe64` in `externals/` and the DLL closure in
  `support/`; Max finds DLLs through a package's `support/` folder.
- Install by copying `externals/` (and `support/` on Windows) into
  `~/Documents/Max 9/Packages/ipt_tilde/`.
- Release signing is done internally at IRCAM, not in this repo. Do not add
  signing scripts.

### Version

`set(PACKAGE_VERSION "x.y.z")` in the root `CMakeLists.txt` is the only place
to bump. It is passed to `ipt~` as the `IPT_PACKAGE_VERSION` define and printed
in the console banner; the README "NEW vX.Y.Z" heading is updated by hand.

---

## 4. `ipt~`

**Arguments:** `ipt~ <model> [device]`. `<model>` is a path to a `.ts` file
(absolute, or a name found through Max's search path). `[device]` is `CPU`
(default), `CUDA` or `MPS`.

**Attributes** (all settable at runtime):

| Attribute | Default | Meaning |
|---|---|---|
| `@enabled` | 1 | 0 stops feeding audio to the model |
| `@sensitivity` | 1.0 | 0..1; smoothing time constant is `(1 − sensitivity) × sensitivityrange` ms, so 1.0 = no smoothing |
| `@sensitivityrange` | 2000 | ms; upper bound of the smoothing window |
| `@threshold` | −80 | dB RMS gate; blocks below it are not classified (≤ −80 disables the gate) |
| `@window` | 20 | ms; length of the RMS gate window |
| `@confidence` | 0.0 | if the top probability is below it, outputs `-1` and `no_confidence` instead of the class |
| `@period` | 0 | ms; 0 = emit every inference, >0 = emit at most once per period, results in between are integrated |
| `@verbose` | 0 | print libipt error strings instead of a generic message |

**Messages:** `classnames` sends `classnames <c1> <c2> …` out of the dumpout.

**Outlets, left to right:** class index (int, `-1` on low confidence); class
name (symbol, `no_confidence` on low confidence); probability distribution
(list of floats, always sent); dumpout (`latency <ms>` after every output,
`classnames …` on request).

**Threading, the rule that must not be broken:** the perform routine only
pushes samples into a lock-free FIFO (`m_audio_fifo`). A dedicated
`std::thread` (`main_loop`) loads the model, drains the FIFO, calls
`ipt_process`, and pushes raw distributions into a second FIFO. A min-api
`timer` (`deliverer`) on the main thread drains that FIFO, smooths with
`ipt_smooth`, and sends to the outlets. **No torch call ever runs on the audio
thread.** Model loading happens on the worker, so instantiation returns before
the model is ready; `dspsetup` waits for it.

---

## 5. `pipo.ipt`

A PiPo module hosted by MuBu objects; it has no inlets of its own. Typical
patch: `pipo~ ipt @ipt.model /path/to/model.ts`. Attributes are prefixed by the
host as `ipt.<name>`:

| Attribute | Default | Meaning |
|---|---|---|
| `model` | "" | path to the `.ts`; relative names are resolved against Max's search path; spaces and quotes are handled |
| `device` | CPU | CPU / CUDA / MPS |
| `sensitivity` | 0.5 | as in `ipt~` |
| `sensitivityrange` | 500 | ms |
| `threshold` | −80 | dB |
| `window` | 20 | ms |
| `confidence` | 0.2 | **not applied**: the module has one output stream and always emits the distribution |
| `period` | 0 | ms, paced on the frame time |
| `offline` | 0 | 1 forces batch mode |
| `threads` | 0 | torch intra-op threads; 0 = torch default; process-global and only effective before the first model load in the Max process |

Output: one frame per inference with `num_classes` columns, labelled with the
class names, at rate `sr / maxFrames`.

**Two modes, chosen in `streamAttributes`:**

- **Real time** (`pipo~`, no time tags, `@offline 0`): `frames()` is called
  from the MSP perform routine. It only enqueues samples into a lock-free FIFO
  and dequeues finished distributions; a worker thread runs `ipt_process`.
  Smoothing and `propagateFrames` stay on the host thread. This mirrors `ipt~`.
  Before v1.2.0 the forward ran synchronously in `frames()`, which saturated the
  CPU and killed the audio: do not reintroduce that.
- **Offline** (`mubu.process`, time-tagged input, or `@offline 1`):
  `frames()` collects windows with `ipt_acquire_window` and runs them in one
  forward pass with `ipt_classify_batch` every 128 windows and in `finalize()`.
  Results are identical to per-block classification; smoothing uses the frame
  time, not the wall clock.

The model is loaded in `streamAttributes` on the host's thread, under a mutex
that the worker also takes around `ipt_process`, so a reload never races an
inference. `ipt_smooth` and the attribute setters are called from the host
thread while the worker may be inside `ipt_process`; libipt tolerates that
because they touch disjoint state, the same pattern `ipt~` uses.

`pipo.ipt/PiPoIPT_schema.md` documents the module in more detail.

---

## 6. The libipt contract

Everything needed is in `libipt/include/ipt.h`; the essentials:

- Lifecycle: `ipt_create(path, device, threshold_db, window_ms)` →
  `ipt_initialize_model` → `ipt_init_buffers(sr, block)` → repeated
  `ipt_process(samples, n, out_dist, cap, &latency)` → `ipt_destroy`.
- `ipt_process` returns `> 0` (number of classes, a distribution was
  produced), `0` (still buffering or below the gate) or `< 0` (an `IPT_ERR_*`
  code; read `ipt_last_error()`, thread-local).
- The distribution is **raw softmax** from libipt; smoothing is a separate call,
  `ipt_smooth(dist, n, frame_time_ms, out, cap)`, with `frame_time_ms < 0`
  meaning "use the wall clock".
- `ipt_set_num_threads(n)` is process-global and only effective before the
  first `ipt_initialize_model` in the process.
- The header says all calls belong on one thread. In practice both externals
  run `ipt_process` on a worker and setters/`ipt_smooth` on the host thread;
  what must never happen is two threads inside `ipt_process`, or destroying the
  handle while another thread uses it.
- Model requirements: a TorchScript module whose `forward` takes
  `(batch, 1, segment_length)` float32 raw audio and returns logits
  `(batch, num_classes)`, with exported `get_sr()`, `get_seglen()`,
  `get_classnames()`. libipt resamples the host audio to `get_sr()` (r8brain)
  and accumulates `get_seglen()` samples per inference, so the segment length is
  the latency floor. `libipt/tests/dummy.ts` (sr 24000, seglen 12000, 4 classes)
  is the model CI uses.

libipt is a separate repository with its own CI, `ARCHITECTURE.md` and
`libipt_schema.md`. Change it there, then bump the submodule here:

```bash
git -C libipt fetch origin dev && git -C libipt checkout <sha>
git add libipt && git commit -m "bump libipt to <sha> (<why>)"
```

CI checks out submodules recursively, so the gitlink must point at a commit
that exists on the libipt remote.

---

## 7. CI

`.github/workflows/ci.yml` runs on pushes to `dev` and `main`, on pull
requests, by hand, and **twice a week on a schedule** (Mon and Thu 05:00 UTC).
Two jobs:

- macOS 14 (Apple Silicon): configure, build, check that both `.mxo` bundles
  and the CLI exist, run the CLI against `libipt/tests/dummy.ts`, `ctest`.
- Windows: same, with `.mxe64` files and the DLL closure in `support/`.

libtorch is cached with a fixed key per platform. GitHub evicts caches unread
for 7 days; the schedule exists to keep them warm because a cold cache means a
CDN download during configure. Scheduled runs use the default branch (`main`),
so it is the `main`-scoped cache that stays warm; `dev` runs fall back to it.
The remaining Node 20 deprecation warnings come from `actions/checkout@v4` and
`actions/cache@v4`.

---

## 8. How to work in this repo as an agent

- **Commits and pushes are made by the maintainer.** Prepare the change,
  build it, and hand back the exact `git` commands. Commit only when asked in
  so many words, and never add a Co-Authored-By or similar trailer.
- Rebuild the target you touched and check the produced bundle
  (`strings externals/<x>.mxo/Contents/MacOS/<x> | grep …` is often enough)
  before reporting done. Real-time behaviour is verified by the maintainer in
  Max; say clearly what was and was not tested.
- Anything that runs in the perform routine (`operator()` in `ipt~`,
  `frames()` in real-time `pipo.ipt`) must be allocation-free and must not
  call into torch. Push to a FIFO, return.
- Do not edit files under `min-api/`, `pipo.ipt/pipo-sdk/` or `libipt/` from
  this repo; they are submodules. `PIPO_MAX_CLASS` and `ext_main` live in the
  pipo-sdk macro, which is why `pipo.ipt` has no per-class hook of its own.
- Keep the two externals' attribute names and semantics aligned; users move
  between them.
- Add a README bullet under the current "NEW vX.Y.Z" heading for anything
  user-visible, and bump `PACKAGE_VERSION` only when the maintainer says so.
- License is CC-BY-NC-4.0 (non-commercial); credits go in the `ipt~` console
  banner (`maxclass_setup`) and the README, not in per-object stamps that no
  one sees.

---

## 9. Related

- [ipt_recognition](https://github.com/nbrochec/ipt_recognition) — training
  and export; read its `AGENTS.md` for the data pipeline.
- [libipt](https://github.com/nbrochec/libipt) — the C ABI and its internals.
- Tutorials and videos are linked from the README; papers: Brochec et al.
  2024/2025/2026, Fiorini et al. 2025.
