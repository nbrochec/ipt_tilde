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

  // temporal smoothing of the class distribution (driven by sensitivity / sensitivityrange)
  LeakyIntegrator    integrator_;
  std::vector<float> pending_;            // latest smoothed distribution awaiting output
  bool               have_pending_ = false;
  std::chrono::steady_clock::time_point last_output_;  // for @period output pacing

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

  PiPoIPT (Parent *parent, PiPo *receiver = NULL)
  : PiPo(parent, receiver),
    modelname_attr_            (this, "model", "Absolute path to model.ts", true, ""),
    device_attr_               (this, "device", "Device type", true, 0),
    sensitivity_attr_          (this, "sensitivity", "Adjust the sensitivity of classification output", false, 0.5),
    sensitivityrange_attr_     (this, "sensitivityrange", "Set the time window for sensitivity scaling", false, 500),
    threshold_attr_            (this, "threshold", "Set the energy threshold for classification", false, -80),
    window_attr_               (this, "window", "Set the sliding window size for energy thresholding", false, 20),
    confidence_attr_           (this, "confidence", "Set the minimum confidence threshold for classification output", false, 0.2),
    period_attr_               (this, "period", "Set the processing period in ms", false, 0)
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

        // reset smoothing + output-period state for the freshly loaded model,
        // then push the current attribute values into the classifier / integrator
        integrator_   = LeakyIntegrator{};
        have_pending_ = false;
        last_output_  = std::chrono::steady_clock::now();
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
    int ret = PIPO_ERROR;

    if (classifier_)
    {
      if (size == 1)
        inputbuf_.assign(values, values + num); // copy to required double vector
      else // copy to required double vector and reduce to mono
      {
        inputbuf_.assign(num, 0); // set to num zeros
        for (int i = 0; i < num; i++)
        {
          for (int j = 0; j < size; j++)
            inputbuf_[i] += values[j];

          values += size;
        }
      }

      std::optional<ClassificationResult> result;
      try {
        sync_attributes();   // apply any @threshold / @window / @sensitivity changes
        result = classifier_->process(std::move(inputbuf_));
      }
      catch (const std::exception& e)  // catches c10::Error from torch as well as std::runtime_error
      {
        printf("pipo.ipt ERROR %s\n", e.what());
        signalError(e.what());
        return PIPO_ERROR;
      }

      // Always process through integrator if we have a result
      // This ensures temporal smoothing (LeakyIntegrator) is always applied
      if (result)
      {
        pending_ = integrator_.process(result->distribution);
        have_pending_ = true;
      }

      // @period: controls how often we emit the output distribution
      // When period > 0, emission is gated - only emit when period has elapsed
      // When period == 0, emit every frame we have pending data
      bool should_emit = false;
      int period_ms = static_cast<int>(period_attr_.get());
      
      if (period_ms <= 0)
      {
        // Period disabled: emit every frame we have data
        should_emit = have_pending_;
      }
      else
      {
        // Period enabled: only emit when period has elapsed AND we have pending data
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_output_).count();
        should_emit = (elapsed >= period_ms && have_pending_);
      }

      // @confidence: in the original ipt_tilde, confidence does NOT gate emission
      // It always emits the distribution, but sends -1 and "no_confidence" on other outlets
      // Since pipo.ipt only has one output (the distribution), we always emit it
      // regardless of confidence level, matching the original behavior
      if (should_emit && have_pending_)
      {
        last_output_ = std::chrono::steady_clock::now();
        have_pending_ = false;
        
        // Always emit the smoothed distribution from LeakyIntegrator
        // (matching original ipt_tilde behavior where distribution is always sent)
        ret = propagateFrames(time, 0, pending_.data(), pending_.size(), 1);
      }
      else
      {
        // Should not emit: either period not elapsed or no pending data
        ret = PIPO_OK;
      }
    }

    return ret;
  }
};
