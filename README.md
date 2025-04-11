<!-- ![](/media/logo.png) -->

# ipt_tilde

ipt~ is a Max/MSP external object for real-time classification of instrumental playing techniques.

This object loads and runs TorchScript (`.ts`) classification models, enabling low latency inference on CPU and MPS devices.

This project is related to [nime2025](https://github.com/nbrochec/nime2025) repository. Where you can find the code used in our paper *Introducing EG-IPT and ipt~: a novel electric guitar dataset and a new Max/MSP object for real-time classification of instrumental playing techniques* presented during [NIME 2025](http://nime2025.org/).

## 💻 Build Instructions
<!-- (TODO: Formalize later) -->

(MacOS / ARM only, for now)
```bash
git clone git@github.com:nbrochec/ipt_tilde.git --recurse-submodules
cd ipt_tilde
cmake -S . -B build DCMAKE_BUILD_TYPE=Release
cmake --build build --target ipt_tilde -j 8 --verbose
```

Note: above instructions will result in a CMake warning: `static library kineto_LIBRARY-NOTFOUND not found.`. AFAICT, this doesn't seem to be a problem, compilation works regardless. Using the pre-compiled binaries directly from [pytorch](https://pytorch.org/) will avoid said warnings, but as of version 2.4.1, the performance is 20x worse on CPU than the Anaconda ones. 

## 🧠 About

This project is part of an ongoing research effort into the real-time recognition of instrumental playing techniques for interactive music systems.
If you use this work in your paper, please consider citing the following:

```bibtex
@inproceedings{fiorini2025egipt,
  title={Introducing EG-IPT and ipt~: a novel electric guitar dataset and a new Max/MSP object for real-time classification of instrumental playing techniques},
  author={Fiorini, Marco, and Brochec, Nicolas and Borg, Joakim and Pasini, Riccardo},
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

## 📜 License

This project is released under a CC-BY-4.0 license.


<!-- ## Roadmap

- [ ] licence (CC BY 4.0 ?)
- [ ] windows compilation
- [ ] notarization
- [ ] release
- [ ] Maxhelp and maxref (Marco) -->