
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

class Model {
public:
    static const inline std::string CLASSIFY_METHOD = "forward";

    static const inline std::string SAMPLE_RATE_METHOD = "get_sr";
    static const inline std::string SEGMENT_LENGTH_METHOD = "get_seglen";
    static const inline std::string CLASS_NAMES_METHOD = "get_classnames";

    /**
     * @note Make sure to initialize the object on the same thread that will call `classify()`
     * @throws c10::Error if model cannot be loaded
     * */
    explicit Model(const std::string& model_path, torch::DeviceType device) : m_device(device) {
        at::init_num_threads();

        m_model = torch::jit::load(model_path);
        m_model.eval();
        m_model.to(m_device);

        m_sample_rate = parse_sample_rate(m_model);
        m_segment_length = parse_segment_length(m_model);
        m_class_names = parse_class_names(m_model);
    }

    /** @throws c10::Error if classification fails */
    ClassificationResult classify(std::vector<float> windowed_buffer) {
        auto tensor_in = vector2tensor(windowed_buffer);

        tensor_in = tensor_in.to(m_device);
        std::vector<torch::jit::IValue> inputs = {tensor_in};
        auto t1 = std::chrono::high_resolution_clock::now();
        auto tensor_out = m_model.get_method("forward")(inputs).toTensor();
        auto t2 = std::chrono::high_resolution_clock::now();
        
        tensor_out = torch::softmax(tensor_out, /*dim=*/-1);
        
        tensor_out = tensor_out.to(torch::kCPU);

        auto v = tensor2vector(tensor_out);
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        return ClassificationResult{v, static_cast<double>(latency_ns) / 1e6};
    }


    /** Classify several windows in a single forward pass.
     *  @param windows each must be exactly get_segment_length() samples long
     *  @returns one ClassificationResult per window, in the same order
     *  @throws c10::Error if classification fails */
    std::vector<ClassificationResult> classify(const std::vector<std::vector<float>>& windows) {
        if (windows.empty()) {
            return {};
        }

        const long batch = static_cast<long>(windows.size());
        const long length = static_cast<long>(windows.front().size());

        // pack the windows contiguously into a [batch, 1, length] tensor
        std::vector<float> flat;
        flat.reserve(static_cast<std::size_t>(batch) * static_cast<std::size_t>(length));
        for (const auto& w: windows) {
            flat.insert(flat.end(), w.begin(), w.end());
        }

        auto tensor_in = torch::from_blob(flat.data(), {batch, 1, length}, torch::kFloat32).to(m_device);
        std::vector<torch::jit::IValue> inputs = {tensor_in};

        auto t1 = std::chrono::high_resolution_clock::now();
        auto tensor_out = m_model.get_method("forward")(inputs).toTensor();
        auto t2 = std::chrono::high_resolution_clock::now();

        tensor_out = torch::softmax(tensor_out, /*dim=*/-1).to(torch::kCPU).contiguous();

        const long num_classes = tensor_out.size(-1);
        const float* out_ptr = tensor_out.data_ptr<float>();

        // share the (single) forward-pass cost evenly across the batch
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double latency_per_window = static_cast<double>(latency_ns) / 1e6 / static_cast<double>(batch);

        std::vector<ClassificationResult> results;
        results.reserve(static_cast<std::size_t>(batch));
        for (long b = 0; b < batch; ++b) {
            const float* row = out_ptr + b * num_classes;
            results.push_back(ClassificationResult{std::vector<float>(row, row + num_classes),
                                                   latency_per_window});
        }

        return results;
    }


    const std::vector<std::string>& get_class_names() const {
        return m_class_names;
    }


    int get_segment_length() const {
        return m_segment_length;
    }


    int get_sample_rate() const {
        return m_sample_rate;
    }


    static std::vector<float> tensor2vector(const at::Tensor& tensor) {
        std::vector<float> v;
        v.reserve(tensor.numel());
        auto out_ptr = tensor.contiguous().data_ptr<float>();

        std::copy(out_ptr, out_ptr + tensor.numel(), std::back_inserter(v));

        return v;
    }


    static at::Tensor vector2tensor(std::vector<float>& v) {
        return torch::from_blob(v.data(), {1, 1, static_cast<long long>(v.size())}, torch::kFloat32);
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
        auto classnames = model.get_method(CLASS_NAMES_METHOD)(std::vector<c10::IValue>()).toList();

        std::vector<std::string> v;

        for (const auto& classname: classnames) {
            v.emplace_back(classname.get().to<std::string>());
        }

        return v;
    }


    torch::DeviceType m_device;

    torch::jit::Module m_model;

    int m_sample_rate;
    int m_segment_length;
    std::vector<std::string> m_class_names;
};


#endif //IPT_MAX_CLASSIFIER_H
