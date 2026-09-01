/*
 * Standalone CLI consumer of libipt — no C++ inference core, no torch, just
 * the C ABI in ipt.h. Feeds synthetic audio blocks and prints the top class
 * whenever the model produces a classification.
 *
 * Run: ./ipt_example /path/to/model.ts
 */
#include "ipt.h"

#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = argc > 1
        ? argv[1]
        : "/Users/joakimborg/Downloads/test_20241104_222325.ts";

    const int sr                  = 44100;
    const int input_vector_length = 64;

    ipt_classifier* clf = ipt_create(model_path, IPT_DEVICE_CPU,
                                     /*energy_threshold_db=*/-80.0,
                                     /*threshold_window_ms=*/20);
    if (!clf) {
        std::fprintf(stderr, "ipt_create failed: %s\n", ipt_last_error());
        return 1;
    }
    if (ipt_initialize_model(clf) != IPT_OK) {
        std::fprintf(stderr, "model load failed: %s\n", ipt_last_error());
        ipt_destroy(clf);
        return 1;
    }
    ipt_init_buffers(clf, sr, input_vector_length);

    const int nclasses = ipt_num_classes(clf);
    std::vector<float> dist(nclasses > 0 ? nclasses : 256);

    // Random audio, which obviously won't yield meaningful classifications —
    // this just exercises the full processing path end-to-end.
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    std::size_t num_inferences = 10;
    while (num_inferences > 0) {
        std::vector<double> block(input_vector_length);
        for (double& v : block) v = uni(rng);

        int n = ipt_process(clf, block.data(), (int) block.size(),
                            dist.data(), (int) dist.size(), nullptr);
        if (n < 0) {
            std::fprintf(stderr, "process error: %s\n", ipt_last_error());
            break;
        }
        if (n > 0) {
            int best = 0;
            for (int i = 1; i < n; ++i)
                if (dist[i] > dist[best]) best = i;

            char name[128];
            ipt_get_class_name(clf, best, name, sizeof(name));
            std::printf("%s\n", name);
            --num_inferences;
        }
    }

    ipt_destroy(clf);
    return 0;
}
