# PiPoIPT - overview

`PiPoIPT.h` is a **PiPo module**: it plugs into a processing chain (pipo~ in real time, or mubu.process offline), receives audio blocks, and outputs a **probability distribution over the model's classes**.

Since **v1.2.0** it no longer embeds the inference core — it drives [**libipt**](https://github.com/nbrochec/libipt) through its **C ABI** (`ipt.h`). The classifier is an opaque `ipt_classifier*` handle; no torch or C++ core types appear here.

```
              +---------------------------------------------------------+
              |                      PiPoIPT                             |
              |        (PiPo wrapper around libipt's C ABI, ipt.h)       |
              +---------------------------------------------------------+

   AUDIO ------------------+                          +----------> OUTPUT
   (blocks)                |                          |      (prob per class)

  +----------------------------------------------------------------------+
  | 1. streamAttributes()   <- called ONCE at startup / model change     |
  |    - picks the mode: offline (batch) vs real-time                    |
  |        offline_ = @offline 1  OR  host sends time-tags               |
  |    - resolves the model path (ipt_resolve_model_path)                |
  |    - ipt_create(path, IPT_DEVICE_CPU/CUDA/MPS, threshold, window)    |
  |      ipt_initialize_model()   load TorchScript + parse metadata     |
  |      ipt_init_buffers(sr, maxFrames)                                 |
  |    - ipt_num_classes / ipt_get_class_name -> output stream format    |
  |    - resets state (batch, period); sync_attributes()                |
  +----------------------------------------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  | 2. frames()   <- called for EVERY audio block                        |
  |                                                                      |
  |    multi-channel audio --> summed to MONO (inputbuf_)                |
  |                                                                      |
  |    sync_attributes()  (applies @threshold @window @sensitivity       |
  |                        only when a value actually changed, via       |
  |                        ipt_set_energy_threshold / _threshold_window / |
  |                        _smoothing_tau)                               |
  |                                                                      |
  |    ipt_acquire_window()  --> window ready? (energy gating here)      |
  |        - returns a malloc'd float* the CALLER OWNS                    |
  |        - if energy too low -> len 0, skip this block                 |
  |        - else -> window ptr appended to batch_windows_ (+ its time)  |
  |                                                                      |
  |    +- REAL-TIME (offline_=false) --> flush_batch() right away        |
  |    |      (no added latency)                                         |
  |    +- OFFLINE --> accumulate up to MAX_BATCH (128), then flush       |
  +----------------------------------------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  | 3. flush_batch()   <- the heart of the inference                     |
  |                                                                      |
  |    ipt_classify_batch(batch_windows_, window_length_, ...)           |
  |         > a SINGLE forward pass for all windows in the batch         |
  |           (= far faster offline, identical results)                  |
  |         > writes N x numclasses raw distributions into batch_dist_   |
  |                                                                      |
  |    for each result, in order:                                        |
  |      - ipt_smooth(dist, numclasses, frame_time, ...)                 |
  |        temporal smoothing of the distribution in libipt              |
  |        (tau set by @sensitivity/@sensitivityrange; fed the FRAME     |
  |         time, not the wall clock, so bursts keep their spacing)      |
  |      - @period: 0 = emit on every inference,                         |
  |                 >0 = at most one emit every N ms                     |
  |      - propagateFrames() --> distribution sent to next module        |
  |                                                                      |
  |    ipt_free_window() for every window in the batch                   |
  +----------------------------------------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  | 4. finalize()   <- end of the input stream                           |
  |    - flush_batch() to drain the windows still pending                |
  |    - propagateFinalize()                                             |
  +----------------------------------------------------------------------+

  ~PiPoIPT()  free_batch_windows()  (ipt_free_window on any pending batch)
              ipt_destroy(classifier_)
```

## The 4 key ideas

1. **Same result, two speeds.** Every block always produces exactly one window. The only difference between real-time and offline is *when* the inference runs: immediately (no latency) vs in batches of 128 (a single `ipt_classify_batch` forward pass = far faster). The results are identical.

2. **Two filtering stages.**
   - *Before* the model: `ipt_acquire_window()` does **energy gating** (`@threshold`/`@window`) - if it's too quiet, no window is returned and we skip.
   - *After* the model: `ipt_smooth()` (libipt's leaky integrator) smooths the distribution over time (`@sensitivity`/`@sensitivityrange` → `tau`), and `@period` limits the output rate.

3. **Windows are caller-owned.** `ipt_acquire_window()` hands back a `float*` that libipt allocated; PiPoIPT keeps the raw pointers in `batch_windows_` and **must free each one** with `ipt_free_window()` — done after every flush, and in the destructor for any un-flushed batch. This is the main thing to get right versus the old value-returning C++ API.

4. **`PiPoPathAttr`** (top of the file): a small utility class that repairs file paths containing spaces, because Max splits the message into several atoms at every space. Without it, `/Users/me/Max 9/model.ts` would be truncated to `/Users/me/Max`.
