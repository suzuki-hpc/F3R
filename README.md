## Requirements

Hardware 

## Setup

##### Installing C++ compilers

The artifact assumes `icpx`, Intel OneAPI C++/D++ compiler, for CPU experiments and `nvc++`, the C++ compiler in the NVIDIA HPC SDK, for GPU experiments.

One can install them following the installation guide in the official webpage:

- `icpx`: 
- `nvc++`: 

While we used `icpx` version ??? and `nvc++` version, one may use newer versions since backward compatibility should exist.

##### Setup a Python environment

This artifact requires the saborn library to generate the figures.

The `requirements.txt` file includes the information of all packages required in the aritifact. So, one can install them as

```
pip install -r requirements.txt
```

Also, just installing `seaborn` would be ok, since it will install all dependencies.

```
pip install seaborn
```



## Execution

#### Task 1: preparing test matrices.

Download 

```
zsh download.sh
```

The artifact uses GNU Make to generate matrices in the HPCG and HPGMP benchmarks:

```
make CXX=<any C++17 compiler>
```

#### Task 2: compiling C++ source files

To ease the compilation, the aritficat contains two Makefiles in GNU Make, `Makefile` and `MakefileGPU.mk`. `Makefile` is to compile the sources for a CPU system:

```
make CXX=icpx
```

`MakefileGPU.mk` is a file to compile for a NVIDIA GPU system. The following command will be used:

```
make -f MakefileGPU CXX=nvc++
```

After that, 18 executables will be generated in the `bin` folder.

#### Task 3: running the executables

##### Main test on the CPU system: `suite.py t3c`

##### Main test on the GPU system: `suite.py t3g`

##### Parameter examination on the CPU system : `suite.py t3p`

#### Task 4: visualizing the numerical results

If Task 3 has been finished properly, the following command will produce all figures similar to those in the paper in the `fig` directory:

```
python plot.py
```



