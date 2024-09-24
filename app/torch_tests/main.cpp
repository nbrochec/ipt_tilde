
#include <torch/script.h>
#include <torch/torch.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>

template<typename T>
inline std::vector<T> tensor2vector(const at::Tensor& tensor) {

    std::vector<T> v;
    v.reserve(static_cast<std::size_t>(tensor.numel()));
    for (int i = 0; i < tensor.numel(); i++) {
        v.push_back(tensor[i].item<T>());
    }

    return v;
}

template<typename T>
inline std::vector<T> ivalue2vector(const at::IValue& ivalue) {
    return tensor2vector<T>(ivalue.toTensor());
}


inline at::Tensor vector2tensor(std::vector<float>& v) {
    return torch::from_blob(v.data(), {1, 1, static_cast<long long>(v.size())}, torch::kFloat32);
}


inline void read_audio_file(const juce::File& file, juce::AudioBuffer<float>& buffer) {
    // 1. Initialize AudioFormatManager and register formats
    juce::AudioFormatManager afm;
    afm.registerBasicFormats();

    // 2. Create an AudioFormatReader for the file
    std::unique_ptr<juce::AudioFormatReader> reader(afm.createReaderFor(file));

    if (reader != nullptr) {
        // 3. Prepare the buffer to have the appropriate number of channels and samples
        buffer.setSize((int) reader->numChannels, (int) reader->lengthInSamples);

        // 4. Read the audio data into the buffer
        reader->read(&buffer, 0, (int) reader->lengthInSamples, 0, true, true);
    } else {
        juce::Logger::writeToLog("Failed to create AudioFormatReader for file.");
    }
}


inline bool write_audio_file(const juce::AudioBuffer<float>& buffer, const juce::File& file, double sr, int bit_depth) {
    juce::AudioFormatManager afm;
    afm.registerBasicFormats();

    auto* format = afm.findFormatForFileExtension(file.getFileExtension());

    if (format == nullptr) {
        juce::Logger::writeToLog("Unsupported file format.");
        return false;
    }

    std::unique_ptr<juce::FileOutputStream> output_stream(file.createOutputStream());

    if (!output_stream->openedOk()) {
        juce::Logger::writeToLog("Failed to open output file.");
        return false;
    }

    std::unique_ptr<juce::AudioFormatWriter> writer(
            format->createWriterFor(output_stream.get(), sr
                                    , static_cast<std::uint32_t>(buffer.getNumChannels()), bit_depth, {}, 0));

    if (writer == nullptr) {
        juce::Logger::writeToLog("Failed to create AudioFormatWriter.");
        return false;
    }

    output_stream.release(); // ownership has been passed to the writer on successful initialization
    writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());

    return true;
}


// ==============================================================================================

int main() {
    juce::File in_file("/Users/joakimborg/Music/Debug/trapezoid_C_maj_scale_A4_440_triang.aif");
    juce::File out_file("./test.wav");
//    std::string path{"/Users/joakimborg/Downloads/percussion.ts"};
    std::string path{"/Users/joakimborg/Downloads/test_20240924_103536.ts"};
    std::string method_name{"forward"};

    int duration_s = 10;
    int sr = 44100;
    int bit_rate = 16;
    int num_channels = 1;

    int hop_size = 2048;

    std::cout << "PyTorch Tests\n";

    juce::AudioBuffer<float> input_buffer;
    read_audio_file(in_file, input_buffer);

//    juce::AudioBuffer<float> output_buffer{num_channels, duration_s * sr};

//    std::cout << "Input buffer: " << input_buffer.getNumSamples() << "\n";


    at::init_num_threads();

    // Load model
    auto model = torch::jit::load(path);
    model.eval();
    model.to(torch::kCPU);

    auto methods = model.get_methods();


    auto method = std::find_if(
            methods.begin()
            , methods.end()
            , [&method_name](const torch::jit::Method& m) { return m.name() == method_name; });

    if (method == methods.end()) {
        throw std::runtime_error("Could not find method " + method_name);
    }

    std::cout << "method name: " << method->name() << "\n";
    std::cout << "method num inputs: " << method->num_inputs() << "\n";

    std::size_t size = 7168;

//    auto params = ivalue2vector<int>(model.attr(method_name + "_params"));



    for (std::size_t j = 0; j < 20; ++j) {
        std::vector<float> v_batch;
        v_batch.reserve(size);

        for (std::size_t i = 0; i < size; ++i) {
            v_batch.push_back(input_buffer.getSample(0, j * size + i));
        }


        auto tensor_in = vector2tensor(v_batch);

//        std::cout << tensor_in  << "\n";


        std::vector<torch::jit::IValue> inputs = {tensor_in};
        auto tensor_out = model.get_method("forward")(inputs).toTensor();

        std::vector<float> output_v;
        output_v.reserve(tensor_out.numel());
         auto out_ptr = tensor_out.contiguous().data_ptr<float>();

         std::copy(out_ptr, out_ptr + tensor_out.numel(), std::back_inserter(output_v));



//        std::cout << tensor_out << "\n";

        for (auto& e : output_v) {
            std::cout << e << "\n";
        }

    }





//    auto in_dim = params[0];
//    auto in_ratio = params[1];
//    auto out_dim = params[2];
//    auto out_ratio = params[3];
//
//    int higher_ratio = 1;
//    int max_ratio = std::max(params[1], params[3]);
//    higher_ratio = std::max(higher_ratio, max_ratio);
//
//    std::cout << "Higher ratio: " << higher_ratio << "\n";
//
//
//    // TODO: This looks extremely redundant, there has to be a better way to create a Tensor from a vector
//
//    int n_batches = 1;
//    int n_vec = hop_size;
////
//    std::vector<float> batch;
//    batch.reserve(static_cast<std::size_t>(hop_size));
//
//    for (int j = 0; j < hop_size; ++j) {
//        batch.push_back(input_buffer.getSample(0, j));
//    }
//
//    auto tensor_in = torch::from_blob(batch.data(), {1, 1, 2048}, torch::kFloat32);
////    auto tensor_in = vector2tensor(batch);
//
//    std::cout << tensor_in << "\n";
//
//    auto cat_tensor_in = torch::cat(tensor_in, 1);
//
//    std::cout << cat_tensor_in << "\n";
//
//    cat_tensor_in = cat_tensor_in.reshape({in_dim, n_batches, -1, in_ratio});
//    cat_tensor_in = cat_tensor_in.select(-1, -1);
//    cat_tensor_in = cat_tensor_in.permute({1, 0, 2});
//
//    cat_tensor_in = cat_tensor_in.to(torch::kCPU);
//    std::vector<torch::jit::IValue> inputs = {cat_tensor_in};
//
//    for (auto& v : inputs) {
//        std::cout << v << "\n";
//    }


//
//    at::Tensor tensor_out;
//    try {
//        tensor_out = (*method)(inputs).toTensor();
//        tensor_out = tensor_out.repeat_interleave(out_ratio).reshape(
//                {n_batches, out_dim, -1});
//    } catch (const std::exception& e) {
//        std::cerr << e.what() << '\n';
//    }



    return 0;
}