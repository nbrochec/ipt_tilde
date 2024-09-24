#include "c74_min.h"
#include <torch/script.h>
#include <torch/torch.h>

using namespace c74::min;

struct ClassificationResult {
    std::vector<float> distribution;
    std::size_t argmax;
};

// ==============================================================================================

class IptClassifier {
public:
    explicit IptClassifier(const std::string& path) {
        try {
            m_buffer = std::vector<float>(m_size, 0.0f);
            at::init_num_threads();
            m_model = torch::jit::load(path);
            m_model.eval();
            m_model.to(torch::kCPU);
        } catch (c10::Error& e) {
            throw std::runtime_error(e.what());
        }
    }

    std::optional<ClassificationResult> process(std::vector<double>&& input) {
        std::optional<ClassificationResult> result = std::nullopt;
        for (auto& sample: input) {
            m_buffer[m_write_index] = static_cast<float>(sample);
            m_write_index = (m_write_index + 1) % m_size;

            if (m_write_index == 0) {
                result = analyze_buffer();
            }
        }
        return result;
    }


    ClassificationResult analyze_buffer() {
        auto tensor_in = vector2tensor(m_buffer);
        std::vector<torch::jit::IValue> inputs = {tensor_in};
        auto tensor_out = m_model.get_method("forward")(inputs).toTensor();

        auto v = tensor2vector<float>(tensor_out);
        auto amax = argmax(v);

        return ClassificationResult{v, amax};
    }

private:
    static std::size_t argmax(const std::vector<float>& v) {
        float max = v[0];
        std::size_t index = 0;
        for (std::size_t i = 1; i < v.size(); i++) {
            if (v[i] > max) {
                max = v[i];
                index = i;
            }
        }
        return index;
    }

    template<typename T>
    static std::vector<T> tensor2vector(const at::Tensor& tensor) {
        std::vector<float> v;
        v.reserve(tensor.numel());
        auto out_ptr = tensor.contiguous().data_ptr<float>();

        std::copy(out_ptr, out_ptr + tensor.numel(), std::back_inserter(v));

        return v;
    }

    static at::Tensor vector2tensor(std::vector<float>& v) {
        return torch::from_blob(v.data(), {1, 1, static_cast<long long>(v.size())}, torch::kFloat32);
    }

    std::size_t m_size = 7168;

    std::vector<float> m_buffer;
    std::size_t m_write_index = 0;

    torch::jit::Module m_model;

};


// ==============================================================================================

class ipt_tilde : public object<ipt_tilde>, public vector_operator<> {
private:
    std::thread m_processing_thread;
    c74::min::fifo<double> m_audio_fifo{16384};
    c74::min::fifo<ClassificationResult> m_event_fifo{100};

    std::atomic<bool> m_running = false;
    std::string m_path;

public:
    MIN_DESCRIPTION{""};                   // TODO
    MIN_TAGS{""};                          // TODO
    MIN_AUTHOR{""};                        // TODO
    MIN_RELATED{""};                       // TODO

    inlet<> inlet_main{this, "(signal) audio input", ""}; // TODO

    outlet<> outlet_main{this, "(int) selected class index", ""};
    outlet<> outlet_distribution{this, "(list) class probability distribution", ""};
    outlet<> dumpout{this, "(any) dumpout"};

    explicit ipt_tilde(const atoms& args = {}) {
        if (!args.empty() && args[0].type() == c74::min::message_type::symbol_argument) {
            m_path = static_cast<std::string>(args[0]);
            m_running = true;
            m_processing_thread = std::thread([&]() { this->process(); });

        } else {
            error("Missing argument: filepath to model");
        }
    }

    ~ipt_tilde() override {
        if (m_running) {
            m_running = false;
            m_processing_thread.join();
        }
    }


    timer<> deliverer{
            this, MIN_FUNCTION {
                ClassificationResult result;
                while (m_event_fifo.try_dequeue(result)) {
                    atoms distribution;
                    for (std::size_t i = 0; i < result.distribution.size(); ++i) {
                        distribution.emplace_back(result.distribution[i]);
                    }

                    atoms argmax;
                    argmax.emplace_back(result.argmax);


                    outlet_main.send(argmax);
                    outlet_distribution.send(distribution);
                }
                return {};
            }
    };


    void operator()(audio_bundle in, audio_bundle) override {
        if (in.channel_count() > 0 && m_running) {
            for (auto i = 0; i < in.frame_count(); ++i) {
                m_audio_fifo.try_enqueue(in.samples(0)[i]);
            }
        }
    }


    attribute<bool> verbose{ this, "verbose", false};


    // TODO: Dummy maxclass_setup. Remove or replace
    message<> maxclass_setup{this, "maxclass_setup", MIN_FUNCTION {
        cout << "PyTorch version: "
             << TORCH_VERSION_MAJOR << "."
             << TORCH_VERSION_MINOR << "."
             << TORCH_VERSION_PATCH << endl;
        return {};
    }};

private:

    void process() {
        try {
            IptClassifier classifier{m_path};

            while (m_running) {
                std::vector<double> buffered_audio;
                double sample;
                while (m_audio_fifo.try_dequeue(sample)) {
                    buffered_audio.push_back(sample);
                }

                if (!buffered_audio.empty()) {
                    auto label = classifier.process(std::move(buffered_audio));
                    if (label) {
                        m_event_fifo.try_enqueue(*label);
                        deliverer.delay(0.0);
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } catch (std::runtime_error& e) {
            if (verbose.get()) {
                cerr << e.what() << endl;
            } else {
                cerr << "error during loading" << endl;
            }
        }

    }


};


MIN_EXTERNAL(ipt_tilde);