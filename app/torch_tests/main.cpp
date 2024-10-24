
#include <torch/script.h>
#include <torch/torch.h>
#include <random>
//#include <juce_audio_basics/juce_audio_basics.h>
//#include <juce_audio_formats/juce_audio_formats.h>

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


//inline void read_audio_file(const juce::File& file, juce::AudioBuffer<float>& buffer) {
//    // 1. Initialize AudioFormatManager and register formats
//    juce::AudioFormatManager afm;
//    afm.registerBasicFormats();
//
//    // 2. Create an AudioFormatReader for the file
//    std::unique_ptr<juce::AudioFormatReader> reader(afm.createReaderFor(file));
//
//    if (reader != nullptr) {
//        // 3. Prepare the buffer to have the appropriate number of channels and samples
//        buffer.setSize((int) reader->numChannels, (int) reader->lengthInSamples);
//
//        // 4. Read the audio data into the buffer
//        reader->read(&buffer, 0, (int) reader->lengthInSamples, 0, true, true);
//    } else {
//        juce::Logger::writeToLog("Failed to create AudioFormatReader for file.");
//    }
//}
//
//
//inline bool write_audio_file(const juce::AudioBuffer<float>& buffer, const juce::File& file, double sr, int bit_depth) {
//    juce::AudioFormatManager afm;
//    afm.registerBasicFormats();
//
//    auto* format = afm.findFormatForFileExtension(file.getFileExtension());
//
//    if (format == nullptr) {
//        juce::Logger::writeToLog("Unsupported file format.");
//        return false;
//    }
//
//    std::unique_ptr<juce::FileOutputStream> output_stream(file.createOutputStream());
//
//    if (!output_stream->openedOk()) {
//        juce::Logger::writeToLog("Failed to open output file.");
//        return false;
//    }
//
//    std::unique_ptr<juce::AudioFormatWriter> writer(
//            format->createWriterFor(output_stream.get(), sr
//                                    , static_cast<std::uint32_t>(buffer.getNumChannels()), bit_depth, {}, 0));
//
//    if (writer == nullptr) {
//        juce::Logger::writeToLog("Failed to create AudioFormatWriter.");
//        return false;
//    }
//
//    output_stream.release(); // ownership has been passed to the writer on successful initialization
//    writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
//
//    return true;
//}


// ==============================================================================================

int main() {
//    juce::File in_file("/Users/joakimborg/Music/Debug/trapezoid_C_maj_scale_A4_440_triang.aif");
//    juce::File out_file("./test.wav");
//    std::string path{"/Users/joakimborg/Downloads/percussion.ts"};
    std::string path{"/Users/joakimborg/Downloads/export_ts_with_attributes/test_20241011_095212.ts"};
    std::string method_name{"get_attributes"};

//    int duration_s = 10;
//    int sr = 44100;
//    int bit_rate = 16;
//    int num_channels = 1;
//
//    int hop_size = 2048;

    std::cout << "PyTorch Tests\n";

    auto device = torch::kMPS;

//    juce::AudioBuffer<float> input_buffer;
//    read_audio_file(in_file, input_buffer);

//    juce::AudioBuffer<float> output_buffer{num_channels, duration_s * sr};

//    std::cout << "Input buffer: " << input_buffer.getNumSamples() << "\n";


    at::init_num_threads();

    // Load model
    auto model = torch::jit::load(path);
    model.eval();
    model.to(device);

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

    auto args = std::vector<torch::jit::IValue>{};
    auto v = method->operator()(args);
    std::cout << "Sample rate: " << v.to<int>() << "\n";

    std::size_t size = 7168;

    std::size_t buffer_size = size * 4;

    std::vector<float> input_buffer;
    input_buffer.reserve(buffer_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(0.0, 1.0);




    for (std::size_t i = 0; i < buffer_size; ++i) {
        input_buffer.push_back(static_cast<float>(distr(gen)));
    }

    std::cout << input_buffer.size() << "\n";


//    auto params = ivalue2vector<int>(model.attr(method_name + "_params"));



    for (std::size_t j = 0; j < 20; ++j) {
        std::vector<float> v_batch;
        v_batch.reserve(size);

        for (std::size_t i = 0; i < size; ++i) {
            v_batch.push_back(input_buffer.at(static_cast<std::size_t>(j * size + i)));
        }


        auto tensor_in = vector2tensor(v_batch);
        tensor_in = tensor_in.to(device);

        std::vector<torch::jit::IValue> inputs = {tensor_in};
        auto tensor_out = model.get_method("forward")(inputs).toTensor();

        tensor_out = tensor_out.to(torch::kCPU);

        std::vector<float> output_v;
        output_v.reserve(static_cast<std::size_t>(tensor_out.numel()));
         auto out_ptr = tensor_out.contiguous().data_ptr<float>();

         std::copy(out_ptr, out_ptr + tensor_out.numel(), std::back_inserter(output_v));


        for (auto& e : output_v) {
            std::cout << e << "\n";
        }

    }

    return 0;
}