// -*- mode: c++; c-basic-offset:2 -*-

#include "PiPo.h"
#include "ipt_classifier.h"
#include "leaky_integrator.h"
#include "utility.h"
#include <chrono>
#include <cmath>
#include <string>
#include <cstring>
#include <unistd.h>

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

class PiPoIPT : public PiPo
{
private:
  std::unique_ptr<IptClassifier> classifier_;
  std::vector<double> inputbuf_;

  // Batched inference: every block still produces one window (so results are
  // identical to per-block classification), but windows are collected and run
  // through the model in a single forward pass, which is far faster offline.
  // Windows are flushed when MAX_BATCH accumulate, and the remainder in finalize().
  static constexpr std::size_t MAX_BATCH = 128;
  std::vector<std::vector<float>> batch_windows_;
  std::vector<double>             batch_times_;   // frame time (ms) of each window
  bool                            offline_ = false;  // batch only offline (mubu.process)

  // temporal smoothing of the class distribution (driven by sensitivity / sensitivityrange)
  LeakyIntegrator    integrator_;
  std::vector<float> pending_;            // latest smoothed distribution awaiting output
  bool               have_pending_ = false;
  double             last_output_time_ = -1e18;  // frame time (ms) of last @period emit

  // last values pushed to the classifier/integrator, so we only re-apply on change
  float applied_threshold_   = NAN;
  float applied_window_      = NAN;
  float applied_sensitivity_ = NAN;
  float applied_sensrange_   = NAN;

  // Push current attribute values into the classifier / integrator.
  // Cheap: only calls a setter when the value actually changed since the last sync.
  void sync_attributes ()
  {
    float threshold = threshold_attr_.get();
    if (threshold != applied_threshold_)
    {
      classifier_->set_energy_threshold(threshold);
      applied_threshold_ = threshold;
    }

    float window = window_attr_.get();
    if (window != applied_window_)
    {
      classifier_->set_threshold_window(static_cast<int>(window));
      applied_window_ = window;
    }

    float sensitivity = sensitivity_attr_.get();
    float sensrange   = sensitivityrange_attr_.get();
    if (sensitivity != applied_sensitivity_  ||  sensrange != applied_sensrange_)
    {
      // sensitivity 1 -> tau 0 (no smoothing, most reactive); lower sensitivity -> more smoothing
      double tau = (1.0 - util::clamp(static_cast<double>(sensitivity), 0.0, 1.0))
                 * static_cast<double>(sensrange);
      integrator_.set_tau(tau);
      applied_sensitivity_ = sensitivity;
      applied_sensrange_   = sensrange;
    }
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
    offline_attr_              (this, "offline", "Batch inference for offline use like mubu.process (1), or classify per block in real-time (0)", true, false)
  {
    device_attr_.addEnumItem("CPU",  "Use CPU");
    device_attr_.addEnumItem("CUDA", "NVIDIA GPU");
    device_attr_.addEnumItem("MPS",  "AppleSilicon GPU");
  }

  ~PiPoIPT (void)
  { }

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

    auto device = torch::kCPU;
    switch (device_attr_.get())
    {
      case 1: device = torch::kCUDA;    break;
      case 2: device = torch::kMPS;         break;
      default: device = torch::kCPU;    break;
    }

    if (model_path  &&  model_path[0] != 0)
    {
      // pre-flight: report a clear error if the path is not readable, instead
      // of letting torch throw a cryptic fopen errno 2
      if (access(model_path, R_OK) != 0)
      {
        std::string msg = "cannot read model file: ";
        msg += model_path;
        signalError(msg.c_str());
        return PIPO_ERROR;
      }

      try {
        classifier_ = std::make_unique<IptClassifier>(std::string(model_path), device);
        classifier_->initialize_model();
        classifier_->initialize_buffers(sr, maxFrames);
        inputbuf_.reserve(maxFrames);

        // reset smoothing, batching and output-period state for the freshly loaded
        // model, then push the current attribute values into classifier / integrator
        integrator_   = LeakyIntegrator{};
        have_pending_ = false;
        last_output_time_ = -1e18;
        batch_windows_.clear();
        batch_times_.clear();
        batch_windows_.reserve(MAX_BATCH);
        batch_times_.reserve(MAX_BATCH);
        applied_threshold_ = applied_window_ = applied_sensitivity_ = applied_sensrange_ = NAN;
        sync_attributes();
      }
      catch (const std::exception& e)  // catches c10::Error from torch as well as std::runtime_error
      {
        printf("ERROR %s\n", e.what());
        signalError(e.what());
        return PIPO_ERROR;
      }

      // query model output parameters
      std::vector<std::string> classnames = *classifier_->get_class_names();
      int numclasses = classnames.size();

      const char **out_labels = new const char *[numclasses];
      for (int i = 0; i < numclasses; i++)
            out_labels[i] = classnames[i].c_str();

      // determine output stream parameters
      double out_framerate = sr / maxFrames;
      ret = propagateStreamAttributes(true, out_framerate, 0, numclasses, 1,
                                          out_labels, false, 0.001 / out_framerate, 1);
      delete [] out_labels;
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

    // acquire the window this block would classify (energy gating happens here),
    // but defer the model forward: collect windows and run them as one batch.
    std::optional<std::vector<float>> window = classifier_->acquire_window(std::move(inputbuf_));
    if (window)
    {
      batch_windows_.push_back(std::move(*window));
      batch_times_.push_back(time);
    }

    // real-time: classify this block immediately (no batching, no added latency).
    // offline: keep collecting until the batch is full, then flush one big batch.
    if (!offline_  ||  batch_windows_.size() >= MAX_BATCH)
      return flush_batch();

    return PIPO_OK;
  }

  // Classify all pending windows in a single forward pass, then smooth and emit
  // each result in order (so output is identical to per-block classification).
  int flush_batch ()
  {
    if (batch_windows_.empty())
      return PIPO_OK;

    std::vector<ClassificationResult> results;
    try {
      results = classifier_->classify(batch_windows_);
    }
    catch (const std::exception& e)  // catches c10::Error from torch as well as std::runtime_error
    {
      printf("pipo.ipt ERROR %s\n", e.what());
      signalError(e.what());
      batch_windows_.clear();
      batch_times_.clear();
      return PIPO_ERROR;
    }

    int ret = PIPO_OK;
    int period_ms = static_cast<int>(period_attr_.get());

    for (std::size_t i = 0; i < results.size(); i++)
    {
      double t = batch_times_[i];

      // feed the integrator the frame time (not wall-clock) so its time-aware
      // smoothing keeps the real audio spacing even though we process in a burst
      auto ts = std::chrono::steady_clock::time_point{}
              + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double, std::milli>(t));
      pending_ = integrator_.process(results[i].distribution, ts);

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

    batch_windows_.clear();
    batch_times_.clear();
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
