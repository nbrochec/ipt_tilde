#include "c74_min.h"
#include <torch/script.h>
#include <chrono>

#include "ipt_classifier.h"
#include "leaky_integrator.h"
#include "utility.h"

using namespace c74::min;

struct Docs {
    static const inline title VERBOSE_TITLE = "Verbose";
    static const inline title ENABLED_TITLE = "Enabled";
    static const inline title SENSITIVITY_TITLE = "Sensitivity";
    static const inline title SENSITIVITY_RANGE_TITLE = "Sensitivity Range";
    static const inline title THRESHOLD_TITLE = "Threshold";
    static const inline title WINDOW_TITLE = "Window";

    static const inline description VERBOSE_DESCRIPTION = "Toggle verbose output in case of errors."
                                                          " When this attribute is enabled, the full pytorch error will"
                                                          " be printed to the console yada yada blabla more text etc.";

    static const inline description ENABLED_DESCRIPTION = "TODO";
    static const inline description SENSITIVITY_DESCRIPTION = "TODO";
    static const inline description SENSITIVITY_RANGE_DESCRIPTION = "TODO";
    static const inline description THRESHOLD_DESCRIPTION = "TODO";
    static const inline description WINDOW_DESCRIPTION = "TODO";


    static const inline description CLASS_NAMES_DESCRIPTION = "TODO";
};


class ipt_tilde : public object<ipt_tilde>, public vector_operator<> {
private:
    std::unique_ptr<IptClassifier> m_classifier;

    std::thread m_processing_thread;
    c74::min::fifo<double> m_audio_fifo{16384};
    c74::min::fifo<ClassificationResult> m_event_fifo{100};


    std::atomic<bool> m_running = false; // lifetime control of internal classification thread
    std::atomic<bool> m_enabled = true;  // user-controlled flag to manually disable output

    // flag indicating whether m_classifier's `initialize_model()` has been called (independently of success)
    std::atomic<bool> m_model_initialized = false;

    LeakyIntegrator m_integrator;

    std::optional<std::vector<std::string>> m_class_names;


public:
    MIN_DESCRIPTION{""};                   // TODO
    MIN_TAGS{""};                          // TODO
    MIN_AUTHOR{"Nicolas Brochec, Joakim Borg, Marco Fiorini"};                        // TODO
    MIN_RELATED{""};                       // TODO

    inlet<> inlet_main{this, "(signal) audio input", ""}; // TODO

    outlet<> outlet_main{this, "(int) selected class index", ""};
    outlet<> outlet_classname{this, "(symbol) selected class name", ""};
    outlet<> outlet_distribution{this, "(list) class probability distribution", ""};
    outlet<> dumpout{this, "(any) dumpout"};

    explicit ipt_tilde(const atoms& args = {}) {
        try {
            auto model_path = parse_model_path(args);
            auto device_type = parse_device_type(args);
            m_classifier = std::make_unique<IptClassifier>(model_path, device_type);
        } catch (std::runtime_error& e) {
            error(e.what());
        }

        // Note: Object construction is finalized in `setup` message
    }


    ~ipt_tilde() override {
        if (m_processing_thread.joinable()) {
            m_running = false;
            m_processing_thread.join();
        }
    }
    
    // BOOT STAMP
    message<> maxclass_setup{
        this, "maxclass_setup",
        [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
            cout << " ipt~ v1.0 (2025) "
                << " by Nicolas Brochec, Joakim Borg, Marco Fiorini" << endl;
            cout << "IRCAM, ERC REACH" << endl;
            return {};
        }
    };


    timer<> deliverer{
            this, MIN_FUNCTION {
                ClassificationResult result;
                while (m_event_fifo.try_dequeue(result)) {
                    // If there are events in the event fifo, it means that the main loop is running and
                    // that the model therefore is initialized, so these assertions should never be hit
                    assert(m_model_initialized);
                    assert(m_classifier);

                    if (!m_class_names) {
                        m_class_names = *m_classifier->get_class_names();
                    }


                    auto distribution = m_integrator.process(result.distribution);

                    atoms distribution_atms;

                    for (const auto& v: distribution) {
                        distribution_atms.emplace_back(v);
                    }

                    auto index = static_cast<long>(util::argmax(distribution));

                    outlet_main.send(index);
                    outlet_classname.send(m_class_names->at(static_cast<std::size_t>(index)));
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


    attribute<bool> verbose{this, "verbose", false, Docs::VERBOSE_TITLE, Docs::VERBOSE_DESCRIPTION};


    attribute<bool> enabled{this, "enabled", true, Docs::ENABLED_TITLE, Docs::ENABLED_DESCRIPTION, setter{
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


    attribute<double> sensitivity{this, "sensitivity", 1.0, Docs::SENSITIVITY_TITLE, Docs::SENSITIVITY_DESCRIPTION, setter{
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


    attribute<int> sensitivityrange{this, "sensitivityrange", 2000, Docs::SENSITIVITY_RANGE_TITLE, Docs::SENSITIVITY_RANGE_DESCRIPTION, setter{
        MIN_FUNCTION {
            if (args.size() == 1
                && args[0].type() == c74::min::message_type::int_argument
                && static_cast<int>(args[0]) > 0) {
                
                // Retrieve the current sensitivity value
                double current_sensitivity = sensitivity.get();

                // Ensure sensitivity is within bounds
                current_sensitivity = std::clamp(current_sensitivity, 0.0, 1.0);

                // Update the tau value
                double new_tau = (1.0 - current_sensitivity) * static_cast<double>(args[0]);
                new_tau = std::max(new_tau, 1e-6);  // Avoid zero or negative tau
                m_integrator.set_tau(new_tau);

                // Return the updated sensitivityrange value
                return args;
            }

            cerr << "bad argument for message \"sensitivityrange\"" << endl;
            return sensitivityrange;
        }
    }};


    attribute<double> threshold{this, "threshold", EnergyThreshold::MINIMUM_THRESHOLD, Docs::THRESHOLD_TITLE, Docs::THRESHOLD_DESCRIPTION, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && (args[0].type() == c74::min::message_type::float_argument
                                         || args[0].type() == c74::min::message_type::int_argument)) {
                    // Note: ignored on first call, as m_classifier is not yet initialized.
                    //       In this case, it will be passed through the `setup` message instead
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


    attribute<int> window{this, "window", IptClassifier::DEFAULT_THRESHOLD_WINDOW_MS, Docs::WINDOW_TITLE, Docs::WINDOW_DESCRIPTION, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && (args[0].type() == c74::min::message_type::int_argument
                                         || args[0].type() == c74::min::message_type::float_argument)) {
                    // Note: ignored on first call, as m_classifier is not yet initialized.
                    //       In this case, it will be passed through the `setup` message instead
                    if (m_classifier) {
                        m_classifier->set_threshold_window(static_cast<int>(args[0]));
                    }

                    return args;
                }

                cerr << "bad argument for message \"threshold\"" << endl;
                return threshold;
            }
    }
    };

    message<> classnames{this, "classnames", Docs::CLASS_NAMES_DESCRIPTION, setter{MIN_FUNCTION {
        if (inlet != 0) {
            cerr << "invalid message \"classnames\" for inlet " << inlet << endl;
            return {};
        }

        if (!args.empty()) {
            cwarn << "extra argument for message \"classnames\"" << endl;
        }

        if (!m_running) {
            cerr << "cannot get classnames: no model has been loaded" << endl;
            return {};
        }

        // If model has been successfully initialize, we can be sure that the model has valid class names
        if (!m_class_names) {
            m_class_names = *m_classifier->get_class_names();
        }

        atoms names{"classnames"};
        for (const auto& n: *m_class_names) {
            names.emplace_back(n);
        }

        dumpout.send(names);

        return {};
    }}};


    // Note: Special function called internally by the min-api after the constructor and all attributes
    // have been initialized. This function cannot be called directly by a user
    message<> setup{this, "setup", MIN_FUNCTION {
        m_classifier->set_energy_threshold(threshold.get());
        m_classifier->set_threshold_window(window.get());

        // since m_classifier is initialized in ctor, we can be sure that it's fully initialized when thread is launched
        m_processing_thread = std::thread(&ipt_tilde::main_loop, this);
        return {};
    }};


    // Note: Special function called internally when audio is enabled. This function cannot be called directly by user.
    //       Also note that this function is called on the main thread, not the audio thread.
    message<> dspsetup{this, "dspsetup", MIN_FUNCTION {
        int sample_rate = args[0];
        int vector_length = args[1];

        // In the rare case of thread initialization not being done by the time dsp is enabled, wait until init finishes
        while (!m_model_initialized) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // If model initialization was successful: initialize buffers
        if (m_running) {
            m_classifier->initialize_buffers(sample_rate, vector_length);
        }

        return {};
    }};


private:
    void main_loop() {
        try {
            m_classifier->initialize_model();
            m_class_names = m_classifier->get_class_names();
            m_running = true;
        } catch (const std::exception& e) {
            if (verbose.get()) {
                cerr << e.what() << endl;
            } else {
                cerr << "error during loading" << endl;
            }
        } catch (...) {
            cerr << "unknown error during loading" << endl;
        }

        m_model_initialized = true; // true independently of success

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
