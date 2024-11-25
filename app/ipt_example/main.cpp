#include "ipt_classifier.h"
#include <random>


static inline std::vector<double> random_vector(std::size_t n
                                                , std::mt19937& rng
                                                , std::uniform_real_distribution<>& dist) {
    std::vector<double> vs(n);
    for (double & v : vs) {
        v = dist(rng);
    }

    return vs;
}



int main() {
    std::string model_path = "/Users/joakimborg/Downloads/test_20241104_222325.ts";
    auto device = torch::kCPU;
    double energy_threshold_db = EnergyThreshold::MINIMUM_THRESHOLD;
    int energy_threshold_ms = 20;

    int sr = 44100;
    int input_vector_length = 64;

    std::size_t num_inferences = 10;

    auto classifier = IptClassifier{model_path, device, energy_threshold_db, energy_threshold_ms};

    classifier.initialize_model();
    classifier.initialize_buffers(sr, input_vector_length);

    std::vector<ClassificationResult> output_classes;

    // Here, we're just using random values, which obviously won't yield good results in terms of classification
    std::random_device rd;
    auto rng = std::mt19937(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    while (num_inferences > 0) {
        auto audio_input_vector = random_vector(input_vector_length, rng, dist);

        auto result = classifier.process(std::move(audio_input_vector));

        if (result) {
            output_classes.push_back(*result);
            --num_inferences;
        }
    }

    auto class_names = *classifier.get_class_names();

    for (const auto& output : output_classes) {
        auto class_index = util::argmax(output.distribution);
        std::cout << class_names[class_index] << std::endl;
    }

    return 0;
}