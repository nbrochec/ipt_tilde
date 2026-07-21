#include "c74_min.h"
#include <chrono>
#include <algorithm>
#include <cctype>

#include "ipt.h"

using namespace c74::min;

// Local replacements for the former util::/IptClassifier constants and helpers,
// which lived in the embedded core now owned by libipt (behind the C ABI).
namespace {
    constexpr double DEFAULT_MINIMUM_THRESHOLD    = -80.0;  // was EnergyThreshold::MINIMUM_THRESHOLD
    constexpr int    DEFAULT_THRESHOLD_WINDOW_MS  = 20;     // was IptClassifier::DEFAULT_THRESHOLD_WINDOW_MS

    std::size_t argmax(const std::vector<float>& v) {
        std::size_t index = 0;
        for (std::size_t i = 1; i < v.size(); ++i) {
            if (v[i] > v[index]) index = i;
        }
        return index;
    }
} // namespace

struct Docs {
    static const inline title VERBOSE_TITLE = "Verbose";
    static const inline title ENABLED_TITLE = "Enabled";
    static const inline title SENSITIVITY_TITLE = "Sensitivity";
    static const inline title SENSITIVITY_RANGE_TITLE = "Sensitivity Range";
    static const inline title THRESHOLD_TITLE = "Threshold";
    static const inline title WINDOW_TITLE = "Window";
    static const inline title CONFIDENCE_TITLE = "Confidence";
    static const inline title PERIOD_TITLE = "Period";

    static const inline description VERBOSE_DESCRIPTION = "Enable or disable verbose logging."
                                                          " When set to @verbose @1, the object provides detailed"
                                                          " logging to the Max console, useful for debugging.";

    static const inline description ENABLED_DESCRIPTION = "Enable or disable classification model operation."
                                                            " When set to @enabled @1, the model is operating."
                                                            " When set to @enabled @0, the model is disabled.";
    static const inline description SENSITIVITY_DESCRIPTION = "Adjust the sensitivity of classification output."
                                                            " Use a @float between @0. and @1. A higher sensitivity allows"
                                                            " quicker reactions to changes in audio input, while lower"
                                                            " sensitivity smooths the output.";   
    static const inline description SENSITIVITY_RANGE_DESCRIPTION = "Set the time window for sensitivity scaling."
                                                            " Use a @float between @0. and @2000. Control the duration"
                                                            " in milliseconds of the temporal window over which"
                                                            " the model's confidence is smoothed.";
    static const inline description THRESHOLD_DESCRIPTION = "Set the energy threshold for classification."
                                                            " Use a @float between @-70 and @0. Controls the minimum energy"
                                                            " level in dB required for a signal to be considered for"
                                                            " classification. Adjust to ignore background noise or quiet input.";
    static const inline description WINDOW_DESCRIPTION = "Set the sliding window size for energy thresholding."
                                                            " Specifies the time window with a @float, in milliseconds,"
                                                            " over which the energy threshold is applied. Larger" 
                                                            " windows smooth the thresholding response.";
    static const inline description CLASS_NAMES_DESCRIPTION = "Message to retrieve the list of class names from the model."
                                                            " Outputs the class names associated with the loaded model"
                                                            " via the dumpout outlet.";
    static const inline description CONFIDENCE_DESCRIPTION = "Set the minimum confidence threshold for classification output."
                                                            " Use a @float between @0. and @1. When the highest probability"
                                                            " is below this threshold, outputs 'no_confidence' instead of"
                                                            " the predicted class name.";
    static const inline description PERIOD_DESCRIPTION = "Set the output period in milliseconds."
                                                            " Use an @int of @0 or greater. When set to @0, each inference"
                                                            " result is output immediately (default). When set to a value"
                                                            " greater than @0, inference runs continuously and all results"
                                                            " within each period are accumulated by the leaky integrator"
                                                            " before a single smoothed output is sent. Longer periods"
                                                            " produce more smoothing.";

};


class ipt_tilde : public object<ipt_tilde>, public vector_operator<> {
private:
    // raw classification result stamped with its production time (steady-clock
    // ms), so that results drained together keep their real temporal spacing when
    // smoothed by ipt_smooth() on the main thread.
    struct TimedResult {
        std::vector<float> distribution;   // raw (unsmoothed) distribution
        double             latency_ms = 0.0;
        double             time_ms = 0.0;  // production time, steady clock, in ms
    };

    ipt_classifier* m_classifier = nullptr;

    std::thread m_processing_thread;
    c74::min::fifo<double> m_audio_fifo{16384};
    c74::min::fifo<TimedResult> m_event_fifo{100};


    std::atomic<bool> m_running = false; // lifetime control of internal classification thread
    std::atomic<bool> m_enabled = true;  // user-controlled flag to manually disable output

    // flag indicating whether m_classifier's model has been initialized (independently of success)
    std::atomic<bool> m_model_initialized = false;

    // Smoothing (ipt_smooth / ipt_set_smoothing_tau) lives entirely on the main
    // thread (deliverer + attribute setters), touching state disjoint from the
    // worker's ipt_process — mirroring how the old LeakyIntegrator was used.

    std::optional<std::vector<std::string>> m_class_names;

    // steady-clock now(), in milliseconds, for stamping results
    static double now_ms() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Pull the class names from the loaded model into m_class_names (once).
    void fetch_class_names() {
        int n = ipt_num_classes(m_classifier);
        if (n <= 0) return;
        std::vector<std::string> names(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            char buf[256];
            ipt_get_class_name(m_classifier, i, buf, sizeof(buf));
            names[static_cast<std::size_t>(i)] = buf;
        }
        m_class_names = std::move(names);
    }

public:
    MIN_DESCRIPTION{"Real-time Instrumental Playing Technique (IPT) recognition using a pre-trained classification model."};
    MIN_TAGS{""}; // TODO
    MIN_AUTHOR{"Nicolas Brochec, Joakim Borg, Marco Fiorini"};
    MIN_RELATED{""};  // TODO

    inlet<> inlet_main{this, "(signal) audio input", ""};

    outlet<> outlet_main{this, "(int) recognized class index", "Outputs the index of the class with higher detection probability."};
    outlet<> outlet_classname{this, "(symbol) recognized class name", "Outputs the name of selected class with higher detection probability."};
    outlet<> outlet_distribution{this, "(list) class probability distribution", "Outputs the class probability distribution as a list."};
    outlet<> dumpout{this, "(any) dumpout", "Outputs miscellaneous data like latency and class names."};

    argument<symbol> model_path_arg {this, "model", "Filepath to the TorchScript model to load. This argument is required. Use absolute path for your model or add your model to the Max file preferences list." };
    argument<symbol> device_arg {this, "device", "Device to use for inference: 'CPU', 'CUDA', or 'MPS'. Optional, defaults to 'CPU'." };

    explicit ipt_tilde(const atoms& args = {}) {
        try {
            auto model_path = parse_model_path(args);
            auto device = parse_device_type(args);
            // Defaults here; threshold/window attribute values are pushed in `setup`.
            m_classifier = ipt_create(model_path.c_str(), device,
                                      DEFAULT_MINIMUM_THRESHOLD, DEFAULT_THRESHOLD_WINDOW_MS);
            if (!m_classifier) {
                error(ipt_last_error());
            }
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
        if (m_classifier) {
            ipt_destroy(m_classifier);
            m_classifier = nullptr;
        }
    }
    
    // BOOT STAMP
    message<> maxclass_setup{
        this, "maxclass_setup",
        [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
            cout << " ipt~ v1.1.0 (2026) "
            << "by Nicolas Brochec" << endl;
            cout << " based on original work by Nicolas Brochec, Joakim Borg, and Marco Fiorini" << endl;
            cout << " IRCAM, RepMus REACH team" << endl;
            return {};
        }
    };


    timer<> deliverer{
            this, MIN_FUNCTION {
                assert(m_model_initialized);
                assert(m_classifier);

                if (!m_class_names) {
                    fetch_class_names();
                }

                TimedResult timed;
                std::vector<float> distribution;
                double latency_ms = 0.0;
                bool has_result = false;

                // Drain all pending results, smoothing each with libipt's leaky
                // integrator on this (single) thread. Pass the production frame
                // time so bursts keep their real spacing. The last smoothed
                // distribution is the one we emit.
                while (m_event_fifo.try_dequeue(timed)) {
                    distribution.resize(timed.distribution.size());
                    ipt_smooth(m_classifier,
                               timed.distribution.data(), static_cast<int>(timed.distribution.size()),
                               timed.time_ms,
                               distribution.data(), static_cast<int>(distribution.size()));
                    latency_ms = timed.latency_ms;
                    has_result = true;
                }

                if (!has_result) {
                    return {};
                }

                atoms distribution_atms;
                for (const auto& v: distribution) {
                    distribution_atms.emplace_back(v);
                }

                auto index = static_cast<long>(argmax(distribution));
                auto max_confidence = distribution[static_cast<std::size_t>(index)];

                if (max_confidence >= confidence.get()) {
                    outlet_main.send(index);
                    outlet_classname.send(m_class_names->at(static_cast<std::size_t>(index)));
                    outlet_distribution.send(distribution_atms);
                } else {
                    outlet_main.send(-1);
                    outlet_classname.send("no_confidence");
                    outlet_distribution.send(distribution_atms);
                }

                atoms latency{"latency"};
                latency.emplace_back(latency_ms);
                dumpout.send(latency);

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
                    ipt_set_smoothing_tau(m_classifier, (1.0 - tau) * static_cast<double>(sensitivityrange.get()));
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
                ipt_set_smoothing_tau(m_classifier, new_tau);

                // Return the updated sensitivityrange value
                return args;
            }

            cerr << "bad argument for message \"sensitivityrange\"" << endl;
            return sensitivityrange;
        }
    }};


    attribute<double> threshold{this, "threshold", DEFAULT_MINIMUM_THRESHOLD, Docs::THRESHOLD_TITLE, Docs::THRESHOLD_DESCRIPTION, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && (args[0].type() == c74::min::message_type::float_argument
                                         || args[0].type() == c74::min::message_type::int_argument)) {
                    // Note: ignored on first call, as m_classifier is not yet initialized.
                    //       In this case, it will be passed through the `setup` message instead
                    if (m_classifier) {
                        ipt_set_energy_threshold(m_classifier, static_cast<double>(args[0]));
                    }

                    return args;
                }

                cerr << "bad argument for message \"threshold\"" << endl;
                return threshold;
            }
    }
    };


    attribute<int> window{this, "window", DEFAULT_THRESHOLD_WINDOW_MS, Docs::WINDOW_TITLE, Docs::WINDOW_DESCRIPTION, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && (args[0].type() == c74::min::message_type::int_argument
                                         || args[0].type() == c74::min::message_type::float_argument)) {
                    // Note: ignored on first call, as m_classifier is not yet initialized.
                    //       In this case, it will be passed through the `setup` message instead
                    if (m_classifier) {
                        ipt_set_threshold_window(m_classifier, static_cast<int>(args[0]));
                    }

                    return args;
                }

                cerr << "bad argument for message \"threshold\"" << endl;
                return threshold;
            }
    }
    };
    
    attribute<double> confidence{this, "confidence", 0.0, Docs::CONFIDENCE_TITLE, Docs::CONFIDENCE_DESCRIPTION, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && args[0].type() == c74::min::message_type::float_argument) {
                    auto conf = std::min(1.0, std::max(0.0, static_cast<double>(args[0])));
                    return {conf};
                }

                cerr << "bad argument for message \"confidence\"" << endl;
                return confidence;
            }
    }
    };


    attribute<int> period{this, "period", 0, Docs::PERIOD_TITLE, Docs::PERIOD_DESCRIPTION, setter{
            MIN_FUNCTION {
                if (args.size() == 1 && (args[0].type() == c74::min::message_type::int_argument
                                          || args[0].type() == c74::min::message_type::float_argument)) {
                    auto ms = std::max(0, static_cast<int>(args[0]));
                    return {ms};
                }

                cerr << "bad argument for message \"period\"" << endl;
                return period;
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

        // If model has been successfully initialized, we can be sure that the model has valid class names
        if (!m_class_names) {
            fetch_class_names();
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
        if (m_classifier) {
            ipt_set_energy_threshold(m_classifier, threshold.get());
            ipt_set_threshold_window(m_classifier, window.get());

            // since m_classifier is created in the ctor, we can be sure it's valid when the thread is launched
            m_processing_thread = std::thread(&ipt_tilde::main_loop, this);
        }
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
            ipt_init_buffers(m_classifier, sample_rate, vector_length);
        }

        return {};
    }};


private:
    void main_loop() {
        // Model load: libipt reports failure via status code + ipt_last_error()
        // rather than exceptions.
        if (ipt_initialize_model(m_classifier) == IPT_OK) {
            fetch_class_names();
            m_running = true;
        } else {
            if (verbose.get()) {
                cerr << ipt_last_error() << endl;
            } else {
                cerr << "error during loading" << endl;
            }
        }

        m_model_initialized = true; // true independently of success

        std::vector<float> dist(256);
        double latency_ms = 0.0;
        auto last_output = std::chrono::steady_clock::now();

        while (m_running) {
            if (m_enabled) {
                std::vector<double> buffered_audio;
                double sample;
                while (m_audio_fifo.try_dequeue(sample)) {
                    buffered_audio.push_back(sample);
                }

                if (!buffered_audio.empty()) {
                    int n = ipt_process(m_classifier,
                                        buffered_audio.data(), static_cast<int>(buffered_audio.size()),
                                        dist.data(), static_cast<int>(dist.size()), &latency_ms);
                    if (n < 0) {
                        if (verbose.get()) cerr << ipt_last_error() << endl;
                        else               cerr << "model architecture is not compatible" << endl;
                    } else if (n > 0) {
                        // enqueue the RAW distribution; smoothing happens on the
                        // main thread (deliverer) to keep integrator use single-threaded
                        TimedResult timed;
                        timed.distribution.assign(dist.begin(),
                                                  dist.begin() + std::min<int>(n, static_cast<int>(dist.size())));
                        timed.latency_ms = latency_ms;
                        timed.time_ms    = now_ms();
                        m_event_fifo.try_enqueue(std::move(timed));

                        if (period.get() == 0) {
                            deliverer.delay(0.0);
                        }
                    }
                }
            }

            if (period.get() > 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_output).count();
                if (elapsed >= period.get()) {
                    deliverer.delay(0.0);
                    last_output = now;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
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


    ipt_device parse_device_type(const atoms& args) {
        if (args.size() < 2) {
            return IPT_DEVICE_CPU;
        }

        if (args[1].type() == c74::min::message_type::symbol_argument) {
            auto device_str = std::string(args[1]);
            std::transform(device_str.begin(), device_str.end(), device_str.begin(), [](unsigned char c) {
                return std::toupper(c);
            });

            if (device_str == "CPU") {
                return IPT_DEVICE_CPU;
            } else if (device_str == "CUDA") {
                return IPT_DEVICE_CUDA;
            } else if (device_str == "MPS") {
                return IPT_DEVICE_MPS;
            } else {
                cwarn << "unknown device type \"" << device_str << "\", defaulting to CPU" << endl;
                return IPT_DEVICE_CPU;
            }
        } else if (args[1].type() == c74::min::message_type::int_argument) {
            switch (static_cast<int>(args[1])) {
                case 0:  return IPT_DEVICE_CPU;
                case 1:  return IPT_DEVICE_CUDA;
                case 2:  return IPT_DEVICE_MPS;
                default:
                    cwarn << "unknown device type \"" << static_cast<int>(args[1]) << "\", defaulting to CPU" << endl;
                    return IPT_DEVICE_CPU;
            }
        }

        cwarn << "bad argument for message \"model\", defaulting to CPU" << endl;
        return IPT_DEVICE_CPU;
    }
};


MIN_EXTERNAL(ipt_tilde);
