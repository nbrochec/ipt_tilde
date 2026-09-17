// -*- mode: c++; c-basic-offset:2 -*-

#include "PiPo.h"
#include "ipt.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include "readerwriterqueue/readerwriterqueue.h"

// Portable "is this path readable?" — MSVC has no unistd.h/R_OK.
#if defined(_WIN32)
#include <io.h>
#define IPT_PATH_READABLE(p) (_access((p), 4) == 0)
#else
#include <unistd.h>
#define IPT_PATH_READABLE(p) (access((p), R_OK) == 0)
#endif

// String attribute for file paths. The host splits a message like
//   ipt.model /Users/me/Max 9/model.ts
// into separate atoms at every space and feeds them one by one, so a plain
// string scalar would only keep "/Users/me/Max". This re-joins all atoms with
// spaces and normalizes the result so spaced paths load correctly.
class PiPoPathAttr : public PiPoScalarAttr<const char *>
{
public:
  PiPoPathAttr (PiPo *pipo, const char *name, const char *descr, bool changesStream,
                const char *initVal = "")
  : PiPoScalarAttr<const char *>(pipo, name, descr, changesStream, initVal)
  {
    if (initVal) path_ = initVal;
  }

  void set (unsigned int i, const char *val, bool silently = false) override
  {
    if (val == NULL) val = "";
    if (i == 0)
      path_ = val;
    else
    {
      path_ += ' ';
      path_ += val;
    }
    normalize_path();
    this->changed(silently);
  }

  void clone (Attr *other) override
  {
    if (auto *o = dynamic_cast<PiPoPathAttr *>(other))
      path_ = o->path_;
  }

  const char *get (void)                      { return path_.c_str(); }
  const char *getStr (unsigned int i = 0) override { return path_.c_str(); }
  PiPo::Atom  getAtom (unsigned int i) override    { return PiPo::Atom(path_.c_str()); }

private:
  // Make the path robust to how it arrives from the patch:
  //  - non-breaking spaces (U+00A0 == bytes C2 A0) -> regular space
  //  - trim leading/trailing whitespace and quote characters (Max treats '
  //    as a literal, so paths can arrive wrapped in ' or " — any number of them)
  void normalize_path ()
  {
    std::string s;
    s.reserve(path_.size());
    for (std::size_t i = 0; i < path_.size(); ++i)
    {
      if (i + 1 < path_.size()
          && static_cast<unsigned char>(path_[i])     == 0xC2
          && static_cast<unsigned char>(path_[i + 1]) == 0xA0)
      {
        s += ' ';
        ++i;
      }
      else
        s += path_[i];
    }
    path_.swap(s);

    const char *trimset = " \t\r\n\"'";
    auto a = path_.find_first_not_of(trimset);
    if (a == std::string::npos) { path_.clear(); return; }
    auto z = path_.find_last_not_of(trimset);
    path_ = path_.substr(a, z - a + 1);
  }

  std::string path_;
};

// Implemented by the host wrapper (pipo.ipt.cpp), where the Max SDK is available.
// Resolves a possibly-relative model filename against Max's search path:
//   - returns an absolute path if the file is found in the search path,
//   - returns the input unchanged if it is already readable (absolute/cwd-relative),
//   - returns empty for empty input.
std::string ipt_resolve_model_path (const char *name);

// Implemented by the host wrapper (pipo.ipt.cpp). Prints the version/credits
// banner to the Max console once per Max session, like ipt~ does.
void ipt_post_banner_once ();

class PiPoIPT : public PiPo
{
private:
  ipt_classifier*     classifier_ = nullptr;
  int                 numclasses_ = 0;
  int                 window_length_ = 0;   // segment length reported by ipt_acquire_window
  std::vector<double> inputbuf_;

  // Batched inference: every block still produces one window (so results are
  // identical to per-block classification), but windows are collected and run
  // through the model in a single forward pass, which is far faster offline.
  // Windows are flushed when MAX_BATCH accumulate, and the remainder in finalize().
  // Windows are owned by libipt (ipt_acquire_window) and freed with ipt_free_window.
  static constexpr std::size_t MAX_BATCH = 128;
  std::vector<float*>             batch_windows_;
  std::vector<double>             batch_times_;   // frame time (ms) of each window
  bool                            offline_ = false;  // batch only offline (mubu.process)

  // buffers for one flush: raw distributions from ipt_classify_batch, and the
  // smoothed distribution awaiting output. Smoothing lives in libipt now.
  std::vector<float> batch_dist_;         // numclasses_ * batch size, filled by classify_batch
  std::vector<float> pending_;            // latest smoothed distribution awaiting output
  double             last_output_time_ = -1e18;  // frame time (ms) of last @period emit

  // Real-time mode (pipo~): the host calls frames() from the MSP perform routine,
  // i.e. the audio thread, so the torch forward must NOT run there. Mirroring
  // ipt~: frames() only pushes samples into a lock-free FIFO and pops finished
  // raw distributions; a dedicated worker thread runs ipt_process() and pushes
  // the results back. Smoothing and output stay on the host thread.
  struct RawResult { std::vector<float> dist; };
  moodycamel::ReaderWriterQueue<double>    audio_fifo_{16384};
  moodycamel::ReaderWriterQueue<RawResult> result_fifo_{64};
  std::thread        worker_;
  std::atomic<bool>  running_{false};    // worker thread lifetime
  std::atomic<bool>  rt_active_{false};  // worker should drain audio_fifo_ and classify
  std::mutex         classifier_mutex_;  // held by the worker around ipt_process, and by
                                         // streamAttributes while (re)creating classifier_
  std::vector<float> worker_dist_;       // worker-side scratch buffer for ipt_process
  bool               have_result_ = false; // at least one distribution since model load

  // last values pushed to the classifier, so we only re-apply on change
  float applied_threshold_   = NAN;
  float applied_window_      = NAN;
  float applied_sensitivity_ = NAN;
  float applied_sensrange_   = NAN;

  static double clampd (double x, double lo, double hi)
  { return std::max(lo, std::min(x, hi)); }

  // Push current attribute values into the classifier.
  // Cheap: only calls a setter when the value actually changed since the last sync.
  void sync_attributes ()
  {
    if (!classifier_) return;

    float threshold = threshold_attr_.get();
    if (threshold != applied_threshold_)
    {
      ipt_set_energy_threshold(classifier_, threshold);
      applied_threshold_ = threshold;
    }

    float window = window_attr_.get();
    if (window != applied_window_)
    {
      ipt_set_threshold_window(classifier_, static_cast<int>(window));
      applied_window_ = window;
    }

    float sensitivity = sensitivity_attr_.get();
    float sensrange   = sensitivityrange_attr_.get();
    if (sensitivity != applied_sensitivity_  ||  sensrange != applied_sensrange_)
    {
      // sensitivity 1 -> tau 0 (no smoothing, most reactive); lower sensitivity -> more smoothing
      double tau = (1.0 - clampd(static_cast<double>(sensitivity), 0.0, 1.0))
                 * static_cast<double>(sensrange);
      ipt_set_smoothing_tau(classifier_, tau);
      applied_sensitivity_ = sensitivity;
      applied_sensrange_   = sensrange;
    }
  }

  // Worker thread: real-time inference off the audio thread.
  void worker_loop ()
  {
    std::vector<double> buffered;
    buffered.reserve(16384);

    while (running_.load(std::memory_order_relaxed))
    {
      bool did_work = false;

      if (rt_active_.load(std::memory_order_acquire))
      {
        buffered.clear();
        double sample;
        while (audio_fifo_.try_dequeue(sample))
          buffered.push_back(sample);

        if (!buffered.empty())
        {
          std::lock_guard<std::mutex> lock(classifier_mutex_);
          if (classifier_ && rt_active_.load(std::memory_order_acquire))
          {
            did_work = true;
            int n = ipt_process(classifier_, buffered.data(), static_cast<int>(buffered.size()),
                                worker_dist_.data(), static_cast<int>(worker_dist_.size()), nullptr);
            if (n > 0)
            {
              RawResult r;
              r.dist.assign(worker_dist_.begin(),
                            worker_dist_.begin() + std::min<int>(n, static_cast<int>(worker_dist_.size())));
              result_fifo_.try_enqueue(std::move(r));   // dropped if the host is not consuming
            }
            else if (n < 0)
              printf("pipo.ipt ERROR %s\n", ipt_last_error());
          }
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(did_work ? 1 : (rt_active_ ? 1 : 10)));
    }
  }

  // Drop anything queued between host and worker (on model reload / mode change).
  void drain_fifos ()
  {
    double d;
    while (audio_fifo_.try_dequeue(d)) {}
    RawResult r;
    while (result_fifo_.try_dequeue(r)) {}
    have_result_ = false;
  }

  // Free any windows still held by a pending (unflushed) batch.
  void free_batch_windows ()
  {
    for (float* w : batch_windows_)
      ipt_free_window(w);
    batch_windows_.clear();
    batch_times_.clear();
  }

public:
  PiPoPathAttr                    modelname_attr_;
  PiPoScalarAttr<PiPo::Enumerate> device_attr_;
  PiPoScalarAttr<float>           sensitivity_attr_;
  PiPoScalarAttr<float>           sensitivityrange_attr_;
  PiPoScalarAttr<float>           threshold_attr_;
  PiPoScalarAttr<float>           window_attr_;
  PiPoScalarAttr<float>           confidence_attr_;
  PiPoScalarAttr<float>           period_attr_;
  PiPoScalarAttr<bool>            offline_attr_;
  PiPoScalarAttr<int>             threads_attr_;

  PiPoIPT (Parent *parent, PiPo *receiver = NULL)
  : PiPo(parent, receiver),
    modelname_attr_            (this, "model", "Absolute path to model.ts", true, ""),
    device_attr_               (this, "device", "Device type", true, 0),
    sensitivity_attr_          (this, "sensitivity", "Adjust the sensitivity of classification output", false, 0.5),
    sensitivityrange_attr_     (this, "sensitivityrange", "Set the time window for sensitivity scaling", false, 500),
    threshold_attr_            (this, "threshold", "Set the energy threshold for classification", false, -80),
    window_attr_               (this, "window", "Set the sliding window size for energy thresholding", false, 20),
    confidence_attr_           (this, "confidence", "Set the minimum confidence threshold for classification output", false, 0.2),
    period_attr_               (this, "period", "Set the output period in ms (0 = output every inference)", false, 0),
    offline_attr_              (this, "offline", "Batch inference for offline use like mubu.process (1), or classify per block in real-time (0)", true, false),
    threads_attr_              (this, "threads", "Number of torch intra-op threads for inference (0 = torch default; process-global, effective before the first model load)", true, 0)
  {
    device_attr_.addEnumItem("CPU",  "Use CPU");
    device_attr_.addEnumItem("CUDA", "NVIDIA GPU");
    device_attr_.addEnumItem("MPS",  "AppleSilicon GPU");

    ipt_post_banner_once();

    worker_dist_.resize(256);
    running_ = true;
    worker_  = std::thread(&PiPoIPT::worker_loop, this);
  }

  ~PiPoIPT (void)
  {
    rt_active_ = false;
    running_   = false;
    if (worker_.joinable()) worker_.join();

    free_batch_windows();
    if (classifier_) ipt_destroy(classifier_);
  }

  int streamAttributes (bool hasTimeTags, double rate, double offset,
                        unsigned int width, unsigned int height,
                        const char **labels, bool hasVarSize,
                        double domain, unsigned int maxFrames)
  {
    int ret = PIPO_ERROR;
    double sr = rate;

    // Batch inference for offline use. Enable explicitly with @offline 1 (e.g. in
    // mubu.process); otherwise auto-enable when the host sends time-tagged frames.
    // In real-time (pipo~, @offline 0, no time tags) we classify per block to
    // avoid the latency of waiting for a batch to fill.
    offline_ = offline_attr_.get() || hasTimeTags;

    // load model: resolve relative names against the host's search path
    std::string resolved_path = ipt_resolve_model_path(modelname_attr_.getStr());
    const char *model_path = resolved_path.c_str();

    ipt_device device = IPT_DEVICE_CPU;
    switch (device_attr_.get())
    {
      case 1: device = IPT_DEVICE_CUDA; break;
      case 2: device = IPT_DEVICE_MPS;  break;
      default: device = IPT_DEVICE_CPU; break;
    }

    if (model_path  &&  model_path[0] != 0)
    {
      // pre-flight: report a clear error if the path is not readable, instead
      // of letting torch throw a cryptic fopen errno 2
      if (!IPT_PATH_READABLE(model_path))
      {
        std::string msg = "cannot read model file: ";
        msg += model_path;
        signalError(msg.c_str());
        return PIPO_ERROR;
      }

      // stop the worker from touching the classifier while we swap it
      rt_active_ = false;
      std::lock_guard<std::mutex> lock(classifier_mutex_);
      drain_fifos();

      // discard any previously loaded model / pending batch
      free_batch_windows();
      if (classifier_) { ipt_destroy(classifier_); classifier_ = nullptr; }

      // torch intra-op thread count (process-global, only effective before the
      // first model load in this process)
      ipt_set_num_threads(threads_attr_.get());

      classifier_ = ipt_create(model_path, device,
                               threshold_attr_.get(), static_cast<int>(window_attr_.get()));
      if (!classifier_  ||  ipt_initialize_model(classifier_) != IPT_OK)
      {
        std::string msg = "cannot load model: ";
        msg += ipt_last_error();
        if (classifier_) { ipt_destroy(classifier_); classifier_ = nullptr; }
        signalError(msg.c_str());
        return PIPO_ERROR;
      }
      // real-time: the worker drains whatever accumulated in the FIFO since its
      // last pass, which can be larger than one host block
      int block = offline_ ? static_cast<int>(maxFrames)
                           : std::max<int>(static_cast<int>(maxFrames), 4096);
      ipt_init_buffers(classifier_, static_cast<int>(sr), block);
      inputbuf_.reserve(maxFrames);

      // reset batching and output-period state for the freshly loaded model,
      // then push the current attribute values into the classifier
      last_output_time_ = -1e18;
      batch_windows_.reserve(MAX_BATCH);
      batch_times_.reserve(MAX_BATCH);
      applied_threshold_ = applied_window_ = applied_sensitivity_ = applied_sensrange_ = NAN;
      sync_attributes();

      // query model output parameters
      numclasses_ = ipt_num_classes(classifier_);
      pending_.assign(numclasses_, 0.f);
      if (static_cast<int>(worker_dist_.size()) < numclasses_)
        worker_dist_.resize(numclasses_);

      std::vector<std::string> classnames(numclasses_);
      const char **out_labels = new const char *[numclasses_];
      for (int i = 0; i < numclasses_; i++)
      {
        char name[256];
        ipt_get_class_name(classifier_, i, name, sizeof(name));
        classnames[i] = name;
        out_labels[i] = classnames[i].c_str();
      }

      // determine output stream parameters
      double out_framerate = sr / maxFrames;
      ret = propagateStreamAttributes(true, out_framerate, 0, numclasses_, 1,
                                          out_labels, false, 0.001 / out_framerate, 1);
      delete [] out_labels;

      // hand the classifier to the worker in real-time mode
      rt_active_ = !offline_;
    }

    return ret;
  }

  int frames (double time, double weight, PiPoValue *values, unsigned int size, unsigned int num)
  {
    if (!classifier_)
      return PIPO_ERROR;

    // reduce this block to mono
    if (size == 1)
      inputbuf_.assign(values, values + num);
    else
    {
      inputbuf_.assign(num, 0);
      for (unsigned int i = 0; i < num; i++)
      {
        for (unsigned int j = 0; j < size; j++)
          inputbuf_[i] += values[j];

        values += size;
      }
    }

    sync_attributes();   // apply any @threshold / @window / @sensitivity changes

    // real-time (pipo~): we are on the audio thread. Never run the model here:
    // hand the samples to the worker and emit whatever it has finished so far.
    if (!offline_)
      return frames_realtime(time);

    // offline (mubu.process): acquire the window this block would classify
    // (energy gating happens here), but defer the model forward: collect
    // windows and run them as one batch. The window is malloc'd by libipt; we
    // own it until ipt_free_window.
    float* window = nullptr;
    int len = ipt_acquire_window(classifier_, inputbuf_.data(),
                                 static_cast<int>(inputbuf_.size()), &window);
    if (len > 0)
    {
      window_length_ = len;
      batch_windows_.push_back(window);
      batch_times_.push_back(time);
    }

    if (batch_windows_.size() >= MAX_BATCH)
      return flush_batch();

    return PIPO_OK;
  }

  // Real-time block: enqueue audio for the worker, dequeue finished raw
  // distributions, smooth them in order, and emit the latest one (paced by
  // @period on frame time). No allocation on the steady-state path.
  int frames_realtime (double time)
  {
    for (double s : inputbuf_)
      audio_fifo_.try_enqueue(s);   // dropped if the worker falls behind

    bool got_new = false;
    RawResult r;
    while (result_fifo_.try_dequeue(r))
    {
      ipt_smooth(classifier_, r.dist.data(), static_cast<int>(r.dist.size()),
                 time, pending_.data(), static_cast<int>(pending_.size()));
      got_new = true;
    }
    if (!got_new)
      return PIPO_OK;
    have_result_ = true;

    int period_ms = static_cast<int>(period_attr_.get());
    if (period_ms <= 0  ||  time - last_output_time_ >= period_ms)
    {
      last_output_time_ = time;
      return propagateFrames(time, 0, pending_.data(), pending_.size(), 1);
    }
    return PIPO_OK;
  }

  // Classify all pending windows in a single forward pass, then smooth and emit
  // each result in order (so output is identical to per-block classification).
  int flush_batch ()
  {
    if (batch_windows_.empty())
      return PIPO_OK;

    int n = static_cast<int>(batch_windows_.size());
    batch_dist_.resize(static_cast<std::size_t>(n) * numclasses_);

    int got = ipt_classify_batch(classifier_,
                                 reinterpret_cast<const float* const*>(batch_windows_.data()),
                                 n, window_length_,
                                 batch_dist_.data(), static_cast<int>(batch_dist_.size()),
                                 nullptr);
    if (got < 0)
    {
      printf("pipo.ipt ERROR %s\n", ipt_last_error());
      signalError(ipt_last_error());
      free_batch_windows();
      return PIPO_ERROR;
    }

    int ret = PIPO_OK;
    int period_ms = static_cast<int>(period_attr_.get());

    for (int i = 0; i < got; i++)
    {
      double t = batch_times_[i];

      // smooth the raw distribution with libipt's leaky integrator. Pass the
      // frame time (not wall-clock) so time-aware smoothing keeps the real audio
      // spacing even though we process in a burst.
      ipt_smooth(classifier_, &batch_dist_[static_cast<std::size_t>(i) * numclasses_],
                 numclasses_, t, pending_.data(), static_cast<int>(pending_.size()));

      // @period: 0 -> emit every inference; >0 -> at most one emit per period (ms),
      // paced on frame time. @confidence does not gate (single outlet).
      if (period_ms <= 0  ||  t - last_output_time_ >= period_ms)
      {
        last_output_time_ = t;
        int r = propagateFrames(t, 0, pending_.data(), pending_.size(), 1);
        if (r != PIPO_OK)
          ret = r;
      }
    }

    free_batch_windows();
    return ret;
  }

  // End of input: flush the windows still pending in the batch.
  int finalize (double inputEnd)
  {
    int ret = flush_batch();
    if (ret != PIPO_OK)
      return ret;
    return propagateFinalize(inputEnd);
  }
};
