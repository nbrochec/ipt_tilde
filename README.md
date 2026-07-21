<div align="center">
  <img src="media/logo.png" alt="ipt~ logo" width="300"/>
</div>

# ipt_tilde

ipt~ is a Max/MSP external object for real-time classification of instrumental playing techniques.

ipt~ is a core component of **SPIRIT** (System for Real-Time Recognition of Instrumental Playing Techniques).

This object loads and runs TorchScript (`.ts`) classification models, enabling low latency inference on CPU and MPS devices.

👉 Train your own playing techniques recognition model in following instructions from the [ipt_recognition](http://github.com/nbrochec/ipt_recognition) repository.

👉 Use ipt~ in your project using the self-contained bundle from the C API: [libipt](https://github.com/nbrochec/libipt)

### 💡 NEW v1.2.0
+  Inference is now powered by [libipt](https://github.com/nbrochec/libipt), a standalone C library, the IPT core is no longer embedded in this repo, but included as a submodule.
+ New attributes: `@period` allows you to throttle the output of ipt~ while keeping leaky integrator smoothing
+ Addition of a trumpet model into the ipt~ Max Package
+ [PiPo](https://github.com/ircam-ismm/pipo) (Plugin Interface for Processing Objects) module for usage in [MuBu](https://ircam-ismm.github.io/max-msp/mubu.html) (Multimodal Analysis of Sound and Motion), developed with [diemoschwarz](https://github.com/diemoschwarz)

### ⬆ Upcoming updates
+ Development of a VAMP Plugin, with [pierreguillot](https://github.com/pierreguillot)

### ⚙️ Requirements

+ macOS 10.13 or later
+ Apple Silicon processor M1 or later (Note: this external doesn't work on Intel processors at the moment)
+ Max 8.6 or later / Max 9.0.3 or later

### 💾 Installation

+ Go to [Releases](https://github.com/DYCI2/ipt_tilde/releases) and download the latest version of ipt~
+ Run the installer depending on your version of Max and follow the instructions

### 🎥 Video & Tutorials
+ [ipt~ recognizing from various instrument](https://reachcloud.ircam.fr/index.php/s/TEsMcZccaYHBYTr)
+ [Getting started](https://reachcloud.ircam.fr/index.php/s/gSSGoLfQDYBEent)
+ [How to train your own models](https://reachcloud.ircam.fr/index.php/s/wBbKaSmLs74MAQc)

## 🧠 About

This project is part of an ongoing research effort into the real-time recognition of instrumental playing techniques for interactive music systems.
If you use this work in your paper, please consider citing the followings:

```bibtex
@phdthesis{brochec:tel-05503298,
  TITLE = {{Advancing Human-Computer Interaction in Mixed Music: A Deep Learning Approach to Real-Time Instrumental Playing Technique Recognition}},
  AUTHOR = {Brochec, Nicolas},
  SCHOOL = {{Tokyo University of the Arts}},
  YEAR = {2026},
}

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
- [Brochec et al. (2026)](https://hal.science/hal-05547946) - "Automatic Hybrid Following in Real-Time Mixed Music: A Case Study with Antescofo and ipt~ for Flute Playing Techniques"
- [Brochec et al. (2025)](https://hal.science/hal-05061669) - "Interactive Music Co-Creation with an Instrumental Technique-Aware System: A Case Study with Flute and Somax2"
- [Fiorini and Brochec (2024)](https://hal.science/hal-04635907) - "Guiding Co-Creative Musical Agents through Real-Time Flute Instrumental Playing Technique Recognition"
- [Brochec et al. (2024)](https://hal.science/hal-04642673) - "Microphone-based Data Augmentation for Automatic Recognition of Instrumental Playing Techniques"

## 💻 Build Instructions

- In a terminal, run the following commands:

```bash
git clone git@github.com:DYCI2/ipt_tilde.git --recurse-submodules
cd ipt_tilde
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ipt_tilde -j 8 --verbose
```

- To build the PiPo module, add `pipo.ipt` to `--target` of previous command or run the following command:

```bash
cmake --build build --target pipo.ipt -j 8 --verbose
```

**Note:** The instructions above may trigger a CMake warning:  `static library kineto_LIBRARY-NOTFOUND not found.`  However, this does not appear to affect compilation or functionality.  Using the pre-compiled binaries from [PyTorch](https://pytorch.org/) will avoid this warning, but as of version 2.4.1, their CPU performance is approximately 20x slower compared to the Anaconda-provided binaries.

- Copy the produced `.mxo` external inside `~/Documents/Max 9/Packages/ipt_tilde/externals/`


## 📜 License and Fundings

This project is released under a CC-BY-NC-4.0 license.

This research is supported by the European Research Council (ERC) as part of the [Raising Co-creativity in Cyber-Human Musicianship (REACH) Project](https://reach.ircam.fr) directed by Gérard Assayag, under the European Union's Horizon 2020 research and innovation program (GA \#883313). 
Funding support for this work was provided by a Japanese Ministry of Education, Culture, Sports, Science and Technology (MEXT) scholarship to Nicolas Brochec. 
