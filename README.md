# Roadmap

- [ ] Add GPU support (MPS)
- [ ] Perform resampling before inference
- [ ] Measure and output inference latency
- [ ] Implement an onset detector (start inferring when the onset is detected, and stop when it is off).


## Build Instructions
(TODO: Formalize later)

```bash
# TODO: Repo should be renamed ipt_tilde 
git clone git@github.com:nbrochec/ipt-max.git --recurse-submodules
cd ipt-max
cmake -S . -B build DCMAKE_BUILD_TYPE=Release
cmake --build build --target ipt_tilde -j 8 --verbose
```