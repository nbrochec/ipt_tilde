## Roadmap

- [x] Add GPU support (MPS)
- [ ] New methods from .ts file to be used :
    - `get_sr`: get the sampling rate (Hz)
    - `get_classnames`: get the name of each class in alphabetic order
    - `get_seglen`: get the number of samples on to make the inference
- [ ] Add outlet that returns the detected classname as a string/symbol. Outlets are then: classname, classindex, distribution, latency, dumpout?
- [ ] Add condition to start inference, absolute sum of samples must be higher than 0
- [ ] Add dB threshold to start inference? The user could specify it with a message/slider 
- [ ] Perform resampling before stacking samples for inference (number of samples is unchanged)
- [ ] Set the number of samples to 7168 by default, update with the value found with method `get_seglen`
- [ ] Path for relative folder "models" or any folder in the Max preferences, to load model without absolute path
- [ ] windows compilation
- [ ] notarization
- [ ] release
- [ ] Maxhelp and maxref (Marco)
- [ ] ~~Implement an onset detector (start inferring when the onset is detected, and stop when it is off).~~


## Build Instructions
(TODO: Formalize later)

(MacOS / ARM only, for now)
```bash
# TODO: Repo should be renamed ipt_tilde 
git clone git@github.com:nbrochec/ipt-max.git --recurse-submodules
cd ipt-max
cmake -S . -B build DCMAKE_BUILD_TYPE=Release
cmake --build build --target ipt_tilde -j 8 --verbose
```

Note: above instructions will result in a CMake warning: `static library kineto_LIBRARY-NOTFOUND not found.`. AFAICT, this doesn't seem to be a problem, compilation works regardless. Using the pre-compiled binaries directly from [pytorch](https://pytorch.org/) will avoid said warnings, but as of version 2.4.1, the performance is 20x worse on CPU than the Anaconda ones.  
