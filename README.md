## Roadmap

- [ ] licence (CC BY 4.0 ?)
- [ ] windows compilation
- [ ] notarization
- [ ] release
- [ ] Maxhelp and maxref (Marco)

## Build Instructions
(TODO: Formalize later)

(MacOS / ARM only, for now)
```bash
git clone git@github.com:nbrochec/ipt_tilde.git --recurse-submodules
cd ipt_tilde
cmake -S . -B build DCMAKE_BUILD_TYPE=Release
cmake --build build --target ipt_tilde -j 8 --verbose
```

Note: above instructions will result in a CMake warning: `static library kineto_LIBRARY-NOTFOUND not found.`. AFAICT, this doesn't seem to be a problem, compilation works regardless. Using the pre-compiled binaries directly from [pytorch](https://pytorch.org/) will avoid said warnings, but as of version 2.4.1, the performance is 20x worse on CPU than the Anaconda ones.  
