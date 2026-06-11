// -*- mode: c++; c-basic-offset:2 -*-

#include "PiPo.h"
#include "ipt_classifier.h"

class PiPoIPT : public PiPo
{
private:
  std::unique_ptr<IptClassifier> classifier_;
  std::vector<double> inputbuf_;

public:
  PiPoScalarAttr<const char *>    modelname_attr_;
  PiPoScalarAttr<PiPo::Enumerate> device_attr_;
  PiPoScalarAttr<float>           sensitivity_attr_;
  PiPoScalarAttr<float>           sensitivityrange_attr_;
  PiPoScalarAttr<float>           threshold_attr_;
  PiPoScalarAttr<float>           window_attr_;
  PiPoScalarAttr<float>           confidence_attr_;
  PiPoScalarAttr<int>             period_attr_;

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
      }
      catch (std::runtime_error& e)
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
        result = classifier_->process(std::move(inputbuf_));
      }
      catch (std::runtime_error& e)
      {
        printf("pipo.ipt ERROR %s\n", e.what());
        signalError(e.what());
        return PIPO_ERROR;
      }

      if (result)
      {
        ClassificationResult res = *result;
        auto classprob = res.distribution;
        ret = propagateFrames(time, 0, classprob.data(), classprob.size(), 1);
      }
      else
        ret = PIPO_OK;
    }

    return ret;
  }
};
