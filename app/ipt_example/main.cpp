#include "ipt_classifier.h"



int main() {
    std::string model_path = "/Users/joakimborg/Downloads/test_20241104_222325.ts";
    auto device = torch::kCPU;
    double energy_threshold_db = EnergyThreshold::MINIMUM_THRESHOLD;
    int energy_threshold_ms = 20;

    int sr = 44100;
    int input_vector_length = 512;

    IptClassifier classifier{model_path, device, energy_threshold_db, energy_threshold_ms};

    classifier.initialize_model();
    classifier.initialize_buffers(sr, input_vector_length);

    return 0;
}