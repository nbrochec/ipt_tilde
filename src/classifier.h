
#ifndef IPT_MAX_CLASSIFIER_H
#define IPT_MAX_CLASSIFIER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>


struct ClassificationResult {
    std::vector<float> distribution;
    double inference_latency_ms;
};


// ==============================================================================================

class IptModel {
public:
    static const inline std::string SAMPLE_RATE_METHOD = "get_sr";
    static const inline std::string SEGMENT_LENGTH_METHOD = "get_seglen";
    static const inline std::string CLASS_NAMES_METHOD = "get_classnames";


    /** @throws c10::Error if model cannot be loaded */
    explicit IptModel(const std::string& path) {
        m_model = torch::jit::load(path);
        m_sample_rate = parse_sample_rate(m_model);
        m_segment_length = parse_segment_length(m_model);
        m_class_names = parse_class_names(m_model);
    }

    const std::vector<std::string>& get_class_names() const {
        return m_class_names;
    }

    const std::string& get_class_name(std::size_t index) const {
        return m_class_names[index];
    }

    int get_segment_length() const {
        return m_segment_length;
    }

    int get_sample_rate() const {
        return m_sample_rate;
    }

    torch::jit::Module& model() {
        return m_model;
    }


private:
    /** @throws c10::Error if model cannot parse sample rate */
    static int parse_sample_rate(torch::jit::Module& model) {
        return model.get_method(SAMPLE_RATE_METHOD)(std::vector<c10::IValue>()).to<int>();
    }

    /** @throws c10::Error if model cannot parse segment length */
    static int parse_segment_length(torch::jit::Module& model) {
        return model.get_method(SEGMENT_LENGTH_METHOD)(std::vector<c10::IValue>()).to<int>();
    }

    /** @throws c10::Error if model cannot parse segment length */
    static std::vector<std::string> parse_class_names(torch::jit::Module& model) {
        throw c10::Error("Not implemented"); // TODO
    }


    torch::jit::Module m_model;
    int m_sample_rate;
    int m_segment_length;
    std::vector<std::string> m_class_names;

};


// ==============================================================================================

class Classifier {
public:
    static const inline std::string CLASSIFY_METHOD = "forward";

    /**
     * @note Make sure to initialize the object on the same thread that will call `classify()`
     * @throws c10::Error if model cannot be loaded
     * */
    explicit Classifier(const std::string& model_path, torch::DeviceType device)
            : m_model(model_path), m_device(device) {
        at::init_num_threads();
        auto& model = m_model.model();
        model.eval();
        model.to(m_device);
    }

    /** @throws c10::Error if classification fails */
    ClassificationResult classify(std::vector<float>&& windowed_buffer) {
        auto tensor_in = vector2tensor(std::move(windowed_buffer));
        tensor_in = tensor_in.to(m_device);
        std::vector<torch::jit::IValue> inputs = {tensor_in};
        auto t1 = std::chrono::high_resolution_clock::now();
        auto tensor_out = m_model.model().get_method(CLASSIFY_METHOD)(inputs).toTensor();
        auto t2 = std::chrono::high_resolution_clock::now();
        tensor_out = tensor_out.to(torch::kCPU);

        auto v = tensor2vector(tensor_out);
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        return ClassificationResult{v, static_cast<double>(latency_ns) / 1e6};
    }


    const IptModel get_model() const {
        return m_model;
    }


    static std::vector<float> tensor2vector(const at::Tensor& tensor) {
        std::vector<float> v;
        v.reserve(tensor.numel());
        auto out_ptr = tensor.contiguous().data_ptr<float>();

        std::copy(out_ptr, out_ptr + tensor.numel(), std::back_inserter(v));

        return v;
    }


    static at::Tensor vector2tensor(std::vector<float>&& v) {
        return torch::from_blob(v.data(), {1, 1, static_cast<long long>(v.size())}, torch::kFloat32);
    }


private:
    torch::DeviceType m_device;
    IptModel m_model;
};


#endif //IPT_MAX_CLASSIFIER_H
