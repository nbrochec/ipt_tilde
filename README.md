<div align="center">
  <img src="media/logo.png" alt="ipt~ logo" width="300"/>
</div>

# ipt~

ipt~ is a Max/MSP external object for real-time classification of instrumental playing techniques.

This object loads and runs TorchScript (`.ts`) classification models, enabling low latency inference on CPU and MPS devices. The package also includes `pipo.ipt`, the official PiPo module, which brings the same recognition to MuBu, both in real time and offline.

## 💡 NEW v1.2.0

+ `pipo.ipt` is the offical PiPo module for MuBu processing; batch mode is used automatically for offline hosts such as `mubu.process`, developed with [diemoschwarz](https://github.com/diemoschwarz).
+ Inference is now powered by [libipt](https://github.com/nbrochec/libipt), a standalone C library.
+ New attributes: `@period` allows you to throttle the output of ipt~ while keeping leaky integrator smoothing.
+ Addition of two trumpet models (with and without harmon mute) to the ipt~ Max Package.

## ⚙️ Requirements

+ macOS 10.13 or later with an Apple Silicon processor (M1 or later — Intel Macs are not supported),
  **or** Windows 10/11 on an x64 CPU with AVX2 (any desktop CPU from roughly 2013 onward)
+ Max 8.6 or later / Max 9.0.3 or later

## 💾 Installation

+ Go to [Releases](https://github.com/DYCI2/ipt_tilde/releases) and download the latest version of ipt~
+ Run the installer depending on your version of Max and follow the instructions

## 🎥 Videos & Tutorials

+ [ipt~ recognizing from various instrument](https://reachcloud.ircam.fr/index.php/s/TEsMcZccaYHBYTr)
+ [Getting started](https://reachcloud.ircam.fr/index.php/s/gSSGoLfQDYBEent)
+ [How to train your own models](https://reachcloud.ircam.fr/index.php/s/wBbKaSmLs74MAQc)

## 🔗 Related Projects

+ [ipt_recognition](http://github.com/nbrochec/ipt_recognition): train your own playing techniques recognition model
+ [libipt](https://github.com/nbrochec/libipt): the standalone C library behind ipt~, to use it in your own project
+ [IPT VAMP Plugin](https://github.com/Ircam-Partiels/ipt-vamp-plugin) by [pierreguillot](https://github.com/pierreguillot)

## 🧠 About

This project is part of an ongoing research effort into the real-time recognition of instrumental playing techniques for interactive music systems.

If you use this work in your paper, please cite the references listed in [CITATION.md](./CITATION.md), where you will also find our related publications.

## 📜 License and Fundings

This project is released under a CC-BY-NC-4.0 license.

This research is supported by the European Research Council (ERC) as part of the [Raising Co-creativity in Cyber-Human Musicianship (REACH) Project](https://reach.ircam.fr) directed by Gérard Assayag, under the European Union's Horizon 2020 research and innovation program (GA \#883313). 
Funding support for this work was provided by a Japanese Ministry of Education, Culture, Sports, Science and Technology (MEXT) scholarship to Nicolas Brochec. 

## 📇 Contact

Please write to `nicolas.brochec[at]ircam.fr` for any questions, or to share with us project made with ipt~

---

Building from source: [BUILDING.md](./BUILDING.md). Repository internals, threading rules and conventions (for developers, and for LLM agents): [AGENTS.md](./AGENTS.md)
