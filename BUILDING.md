# Building ipt~ from source

Most users do not need this file: signed installers for `ipt~` and `pipo.ipt` are
on the [Releases](https://github.com/DYCI2/ipt_tilde/releases) page. Build from
source if you want to modify the externals or target a platform that is not
distributed yet.

Inference is provided by [libipt](https://github.com/nbrochec/libipt), included
here as a git submodule, so always clone recursively. On the first configure,
libipt's CMake downloads a pinned libtorch (same version on both platforms, so
`.ts` models are interchangeable) into `libipt/libs/libtorch`.

## macOS (Apple Silicon)

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

**Note:** The instructions above may trigger a CMake warning:  `static library kineto_LIBRARY-NOTFOUND not found.`  However, this does not appear to affect compilation or functionality.  Using the pre-compiled binaries from [PyTorch](https://pytorch.org/) will avoid this warning, but as of version 2.4.1, their CPU performance is approximately 20x slower compared to the Anaconda-provided binaries. If your CMake version is 4.0 or later, add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`

- Copy the produced `.mxo` external inside `~/Documents/Max 9/Packages/ipt_tilde/externals/`

## Windows (x64)

You can build the external on a Windows machine. The Windows external build is not officially distributed yet.

- Requires Visual Studio 2022 or later and CMake. In a terminal:

```bat
git clone git@github.com:DYCI2/ipt_tilde.git --recurse-submodules
cd ipt_tilde
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config Release -j 8
```

- On first configure, CMake downloads the official libtorch for Windows (~176 MB) into `libipt/libs/libtorch`.
- The build produces the `.mxe64` externals in `externals/` and their runtime DLLs (`ipt.dll` + the libtorch closure) in `support/`. Copy **both folders** into `Documents/Max 9/Packages/ipt_tilde/` — Max finds the DLLs through the package's `support/` folder.

## Going further

[AGENTS.md](./AGENTS.md) documents the repository layout, the `ipt~` / `pipo.ipt`
threading model, the libipt C ABI contract, the CI job, and the remaining build
traps (SDK sysroot on macOS, version bumping, submodule updates).
