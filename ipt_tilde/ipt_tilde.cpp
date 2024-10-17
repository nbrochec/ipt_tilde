#include "c74_min.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>

using namespace c74::min;

struct ClassificationResult {
    std::vector<float> distribution;
    std::size_t argmax;
    double inference_latency_ms;
};

// ==============================================================================================

class IptClassifier {
public:
    explicit IptClassifier(const std::string& path, torch::DeviceType device)
    : m_device(device) {
        m_buffer = std::vector<float>(m_size, 0.0f);
        at::init_num_threads();
        m_model = torch::jit::load(path);
        m_model.eval();
        m_model.to(m_device);
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
        tensor_in  = tensor_in.to(m_device); // TODO: Might need a critical section here
        std::vector<torch::jit::IValue> inputs = {tensor_in};
        auto t1 = std::chrono::high_resolution_clock::now();
        auto tensor_out = m_model.get_method("forward")(inputs).toTensor();
        auto t2 = std::chrono::high_resolution_clock::now();

        auto v = tensor2vector<float>(tensor_out);
        auto amax = argmax(v);
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        return ClassificationResult{v, amax, static_cast<double>(latency_ns) / 1e6};
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

    torch::DeviceType m_device;

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
        try {
            auto model_path = parse_model_path(args);
            auto device_type = parse_device_type(args);

            m_processing_thread = std::thread(&ipt_tilde::main_loop, this, std::move(model_path), device_type);
        } catch (std::runtime_error& e) {
            error(e.what());
        }
    }


    ~ipt_tilde() override {
        if (m_processing_thread.joinable()) {
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

                    for (const auto& v: distribution) {
                        distribution_atms.emplace_back(v);
                    }

                    outlet_main.send(static_cast<long>(IptClassifier::argmax(distribution)));
                    outlet_distribution.send(distribution_atms);

                    atoms latency{"latency"};
                    latency.emplace_back(result.inference_latency_ms);
                    dumpout.send(latency);
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
                    m_integrator.set_tau((1.0 - tau) * static_cast<double>(sensitivityrange.get()));
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


private:
    void main_loop(std::string&& path, torch::DeviceType device) {
        std::unique_ptr<IptClassifier> classifier;

        try {
            classifier = std::make_unique<IptClassifier>(path, device);
            m_running = true;
        } catch (const std::exception& e) {
            if (verbose.get()) {
                cerr << e.what() << endl;
            } else {
                cerr << "error during loading" << endl;
            }
            return;
        } catch (...) {
            cerr << "unknown error during loading" << endl;
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
                        auto result = classifier->process(std::move(buffered_audio));
                        if (result) {
                            m_event_fifo.try_enqueue(*result);
                            deliverer.delay(0.0);
                        }
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

        } catch (const std::exception& e) {
            if (verbose.get()) {
                cerr << e.what() << endl;
            } else {
                cerr << "model architecture is not compatible" << endl;
            }
        } catch (...) {
            cerr << "unknown error" << endl;
        }

    }

    static std::string parse_model_path(const atoms& args) {
        if (args.empty()) {
            throw std::runtime_error("Missing argument: filepath to model");
        }

        if (args[0].type() != c74::min::message_type::symbol_argument) {
            throw std::runtime_error("first argument must be a filepath");
        }

        auto path = std::string(args[0]);

        if (path.size() >= 3 && path.substr(path.length() - 3) != ".ts") {
            path = path + ".ts";
        }

        // If relative path, look for file in max filepath and throws std::runtime_error if fails it to locate it
        path = static_cast<std::string>(c74::min::path(path));

        return path;
    }


    torch::DeviceType parse_device_type(const atoms& args) {
        if (args.size() < 2) {
            return torch::kCPU;
        }

        if (args[1].type() == c74::min::message_type::symbol_argument) {
            auto device_str = std::string(args[1]);
            std::transform(device_str.begin(), device_str.end(), device_str.begin(), [](unsigned char c) {
                return std::toupper(c);
            });

            if (device_str == "CPU") {
                return torch::kCPU;
            } else if (device_str == "CUDA") {
                return torch::kCUDA;
            } else if (device_str == "MPS") {
                return torch::kMPS;
            } else {
                cwarn << "unknown device type \"" << device_str << "\", defaulting to CPU" << endl;
                return torch::kCPU;
            }
        } else if (args[1].type() == c74::min::message_type::int_argument) {
            auto device_idx = static_cast<int>(args[1]);

            if (device_idx < 0 || device_idx >= static_cast<int>(torch::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)) {
                cwarn << "unknown device type \"" << device_idx << "\", defaulting to CPU" << endl;
                return torch::kCPU;
            }

            return static_cast<torch::DeviceType>(device_idx);
        }

        cwarn << "bad argument for message \"model\", defaulting to CPU << endl";
        return torch::kCPU;
    }
};


MIN_EXTERNAL(ipt_tilde);