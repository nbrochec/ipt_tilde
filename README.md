## Roadmap

- [x] Add GPU support (MPS)
- [ ] Perform resampling before inference
- [ ] Implement an onset detector (start inferring when the onset is detected, and stop when it is off).


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
