<!-- ![](/media/logo.png) -->

# ipt_tilde

ipt~ is a Max/MSP external object for real-time classification of instrumental playing techniques.

This object loads and runs TorchScript (`.ts`) classification models, enabling low latency inference on CPU and MPS devices.

This project is related to [nime2025](https://github.com/nbrochec/nime2025) repository, where you can find the code used in our paper *Introducing EG-IPT and ipt~: a novel electric guitar dataset and a new Max/MSP object for real-time classification of instrumental playing techniques* presented during [NIME 2025](http://nime2025.org/), and train a classification model for electric guitar playing techniques.

### 🎥 Demo Video
A demonstration video of ipt~ detecting in real-time Instrumental Playing Techniques from the EG-IPT dataset is available [here](https://youtu.be/PFiWNnOd-vg).

## 💻 Build Instructions
<!-- (TODO: Formalize later) -->

### Requirements

+ macOS 10.13 or later
+ Apple Silicon processor M1 or later (Note: this external doesn't work on Intel processors at the moment)
+ Max 8.6 or later / Max 9.0.3 or later

### Building

- In a terminal, run the following commands:

```bash
git clone git@github.com:nbrochec/ipt_tilde.git --recurse-submodules
cd ipt_tilde
cmake -S . -B build DCMAKE_BUILD_TYPE=Release
cmake --build build --target ipt_tilde -j 8 --verbose
```

**Note:** The instructions above may trigger a CMake warning:  `static library kineto_LIBRARY-NOTFOUND not found.`  However, this does not appear to affect compilation or functionality.  Using the pre-compiled binaries from [PyTorch](https://pytorch.org/) will avoid this warning, but as of version 2.4.1, their CPU performance is approximately 20x slower compared to the Anaconda-provided binaries.

- Copy the produced `.mxo` external inside `~/Documents/Max 9/Packages/ipt_tilde/externals/`

## 🧠 About

This project is part of an ongoing research effort into the real-time recognition of instrumental playing techniques for interactive music systems.
If you use this work in your paper, please consider citing the following:

```bibtex
@inproceedings{fiorini2025egipt,
  title={Introducing EG-IPT and ipt~: a novel electric guitar dataset and a new Max/MSP object for real-time classification of instrumental playing techniques},
  author={Fiorini, Marco and Brochec, Nicolas and Borg, Joakim and Pasini, Riccardo},
  booktitle={NIME 2025},
  year={2025},
  address={Canberra, Australia}
}
```

## 📚 Related Work

If you are interested in this topic, please check out our other papers:
- [Fiorini and Brochec (2024)](https://hal.science/hal-04635907) - "Guiding Co-Creative Musical Agents through Real-Time Flute Instrumental Playing Technique Recognition"
- [Brochec et al. (2024)](https://hal.science/hal-04642673) - "Microphone-based Data Augmentation for Automatic Recognition of Instrumental Playing Techniques"
- [Brochec and Tanaka (2023)](https://hal.science/hal-04263718) - "Toward Real-Time Recognition of Instrumental Playing Techniques for Mixed Music: A Preliminary Analysis"

## 📜 License and Fundings

This project is released under a CC-BY-4.0 license.

This research is supported by the European Research Council (ERC) as part of the [Raising Co-creativity in Cyber-Human Musicianship (REACH) Project](https://reach.ircam.fr) directed by Gérard Assayag, under the European Union's Horizon 2020 research and innovation program (GA \#883313). 
Funding support for this work was provided by a Japanese Ministry of Education, Culture, Sports, Science and Technology (MEXT) scholarship to Nicolas Brochec. 
