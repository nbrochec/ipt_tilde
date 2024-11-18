#include "c74_min.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <CDSPResampler.h>
#include <chrono>

#include "ipt_classifier.h"
#include "leaky_integrator.h"

using namespace c74::min;

// ==============================================================================================


// ==============================================================================================


// ==============================================================================================

class ipt_tilde : public object<ipt_tilde>, public vector_operator<> {
private:
    std::unique_ptr<IptClassifier> m_classifier;

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
    
    
    attribute<double> threshold{ this, "threshold", -120.0, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && args[0].type() == c74::min::message_type::float_argument) {
                    // Note: ignored on first call, as m_classifier is not yet initialized
                    if (m_classifier) {
                        m_classifier->set_energy_threshold(static_cast<float>(args[0]));
                    }

					return args;
				}
                
                cerr << "bad argument for message \"threshold\"" << endl;
                return threshold;
            }
        }
    };


private:
    void main_loop(std::string&& path, torch::DeviceType device) {
        try {
            m_classifier = std::make_unique<IptClassifier>(path, device);
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
                        auto result = m_classifier->process(std::move(buffered_audio));
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