// -*- mode: c++; c-basic-offset:2 -*-

#include "PiPo.h"
#include "ipt_classifier.h"
#include "leaky_integrator.h"
#include "utility.h"
#include <chrono>
#include <cmath>

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
  PiPoScalarAttr<const char *>    modelname_attr_;
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

    // load model
    const char *model_path = modelname_attr_.getStr();
    printf("modelname %s, sr %f\n", model_path, sr);

    auto device = torch::kCPU;
    switch (device_attr_.get())
    {
      case 1: device = torch::kCUDA;    break;
      case 2: device = torch::kMPS;         break;
      default: device = torch::kCPU;    break;
    }

    if (model_path  &&  model_path[0] != 0)
    {
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

      if (result)
      {
        // temporal smoothing: sensitivity / sensitivityrange set the integrator's tau
        pending_      = integrator_.process(result->distribution);
        have_pending_ = true;
      }

      // @period: 0 -> output every inference; >0 -> one smoothed output per period (ms)
      bool emit = false;
      int  period_ms = static_cast<int>(period_attr_.get());
      if (period_ms <= 0)
        emit = have_pending_;
      else
      {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_output_).count();
        if (elapsed >= period_ms)
        {
          emit = have_pending_;
          last_output_ = now;
        }
      }

      if (emit)
      {
        have_pending_ = false;
        // @confidence: only output when the top class is confident enough,
        // otherwise suppress this block's output (single-outlet analogue of "no_confidence")
        std::size_t idx = util::argmax(pending_);
        if (pending_[idx] >= confidence_attr_.get())
          ret = propagateFrames(time, 0, pending_.data(), pending_.size(), 1);
        else
          ret = PIPO_OK;
      }
      else
        ret = PIPO_OK;
    }

    return ret;
  }
};
