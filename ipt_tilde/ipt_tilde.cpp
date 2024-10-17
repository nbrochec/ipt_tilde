#include "c74_min.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>

using namespace c74::min;

struct ClassificationResult {
    std::vector<float> distribution;
    std::size_t argmax;
};

// ==============================================================================================

class IptClassifier {
public:
    explicit IptClassifier(const std::string& path) {
        m_buffer = std::vector<float>(m_size, 0.0f);
        at::init_num_threads();
        m_model = torch::jit::load(path);
        m_model.eval();
        m_model.to(torch::kCPU);
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


private:
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

class LeakyIntegrator {
public:
    std::vector<float> process(const std::vector<float>& input) {
        auto current_time = std::chrono::system_clock::now();

        if (!m_last_callback || m_tau < 1e-6 || m_previous_value.size() != input.size()) {
            m_last_callback = current_time;
            m_previous_value = input;
            return input;
        }

        auto elapsed_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - *m_last_callback).count());

        elapsed_time = std::min(m_tau, std::max(0.0, elapsed_time));

        auto output = integrate(input, elapsed_time);
        m_previous_value = output;
        m_last_callback = current_time;

        return output;
    }

    void set_tau(double tau) {
        m_tau = tau;
    }


private:
    std::vector<float> integrate(const std::vector<float>& current_value, double elapsed_time) {
        std::vector<float> result;
        result.reserve(current_value.size());

        auto dt = elapsed_time / m_tau;

        for (std::size_t i = 0; i < current_value.size(); i++) {
            result.emplace_back((1 - dt) * m_previous_value.at(i) + dt * current_value.at(i));
        }
        return result;
    }


    std::optional<std::chrono::time_point<std::chrono::system_clock>> m_last_callback;
    std::vector<float> m_previous_value;

    double m_tau = 0.0;


};


// ==============================================================================================

class ipt_tilde : public object<ipt_tilde>, public vector_operator<> {
private:
    std::thread m_processing_thread;
    c74::min::fifo<double> m_audio_fifo{16384};
    c74::min::fifo<ClassificationResult> m_event_fifo{100};

    std::atomic<bool> m_running = false;
    std::atomic<bool> m_enabled = true;
    std::string m_path;

    LeakyIntegrator m_integrator;

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
            auto path = std::string(args[0]);

            if (path.size() >= 3 && path.substr(path.length() - 3) != ".ts") {
                path = path + ".ts";
            }

            try {
                m_path = static_cast<std::string>(c74::min::path(path));
                m_running = true;
                m_processing_thread = std::thread([&]() { this->process(); });
            } catch (std::runtime_error& e) {
                // failed to locate file
                error(e.what());
            }

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
                    auto distribution = m_integrator.process(result.distribution);

                    atoms distribution_atms;

                    for (const auto& v : distribution) {
                        distribution_atms.emplace_back(v);
                    }

                    atoms argmax;
                    argmax.emplace_back(IptClassifier::argmax(distribution));

                    outlet_main.send(argmax);
                    outlet_distribution.send(distribution_atms);
                }
                return {};
            }
    };


    void operator()(audio_bundle in, audio_bundle) override {
        if (in.channel_count() > 0 && m_running && m_enabled) {
            for (auto i = 0; i < in.frame_count(); ++i) {
                m_audio_fifo.try_enqueue(in.samples(0)[i]);
            }
        }
    }


    attribute<bool> verbose{this, "verbose", false};

    attribute<bool> enabled{this, "enabled", true, setter{
            MIN_FUNCTION {
                if (args[0].type() == c74::min::message_type::int_argument) {
                    m_enabled = static_cast<bool>(args[0]);
                    return args;
                }

                cerr << "bad argument for message \"enabled\"" << endl;
                return enabled;
            }
    }
    };

    attribute<double> sensitivity{this, "sensitivity", 1.0, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && args[0].type() == c74::min::message_type::float_argument) {
                    auto tau = std::min(1.0, std::max(0.0, static_cast<double>(args[0])));
                    m_integrator.set_tau((1.0-tau) * static_cast<double>(sensitivityrange.get()));
                    return {tau};
                }

                cerr << "bad argument for message \"sensitivity\"" << endl;
                return sensitivity;
            }
    }
    };

    attribute<int> sensitivityrange{this, "sensitivityrange", 2000, setter{
            MIN_FUNCTION {
                if (args.size() == 1
                    && args[0].type() == c74::min::message_type::int_argument
                    && static_cast<int>(args[0]) > 0) {
                    m_integrator.set_tau(sensitivity.get() * (1.0 - static_cast<double>(args[0])));
                    return args;
                }

                cerr << "bad argument for message \"sensitivityrange\"" << endl;
                return sensitivityrange;
            }
    }
    };


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
        std::unique_ptr<IptClassifier> classifier;

        try {
            classifier = std::make_unique<IptClassifier>(m_path);
        } catch (c10::Error& e) {
            if (verbose.get()) {
                cerr << e.what() << endl;
            } else {
                cerr << "error during loading" << endl;
            }
            return;
        }

        try {
            while (m_running) {
                if (m_enabled) {
                    std::vector<double> buffered_audio;
                    double sample;
                    while (m_audio_fifo.try_dequeue(sample)) {
                        buffered_audio.push_back(sample);
                    }

                    if (!buffered_audio.empty()) {
                        auto label = classifier->process(std::move(buffered_audio));
                        if (label) {
                            m_event_fifo.try_enqueue(*label);
                            deliverer.delay(0.0);
                        }
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

        } catch (c10::Error& e) {
            if (verbose.get()) {
                cerr << e.what() << endl;
            } else {
                cerr << "model architecture is not compatible" << endl;
            }
        }

    }

};


MIN_EXTERNAL(ipt_tilde);