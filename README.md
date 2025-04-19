# F3R: A Mixed-Precision Linear Solver

This repository contains the software artifact accompanying a research paper under review.

## Paper 

This artifact is associated with a research submission currently under review. The paper will be linked here upon publication.

## Hardware Requirements

For CPU experimtnes, 

- **CPU**: Intel Xeon Max 9480 (or similar, with AVX512 and fp16 support)

- **GPU**: NVIDIA A100 (80 GB) or equivalent CUDA-enabled GPU
- **RAM**: 40 GB
- **Note**: System must suport fp16 computing

## Software Requirements

- **Compilers**:
  - CPU Execution: Intel oneAPI DPC++/C++ Compiler (`icpx`), v2023.2.4+ with `-mavx512fp16`
  - GPU Execution: NVIDIA HPC SDK `nvc++`, v23.9+ with `-cuda`
  - Note: Compilers must support C++17
- **Python**: 3.8+
  - `Pandas >= 2.0.3`
  - `seaborn==0.13.2`
- **Build Tool**:
  - GNU Make 4.2.1

One may Install `icpx` from [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) and `nvc++` from [NVIDIA HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-239-downloads).

Install Python dependencies with:

```bash
pip install seaborn
```

or

```bash
pip install -r requirements.txt
```

## Setup Instructions

##### 1. Clone the repository

```bash
git clone https://github.com/suzuki-hpc/F3R.git
cd F3R
```

##### 2. Prepare datasets

```zsh
cd matrix
zsh download.sh # SuiteSparse
make # HPCG and HPGMP
```

##### 3. Compile solvers

```bash
cd work
make CXX=<your C++ copmiler> # for CPU-only Execution
make -f MakefileGPU CXX=nvc++ # for CPU-GPU Execution
```

## Execution

The complete experiment workflow:

```
T1 → T2 → T3_CPU + T3_GPU → T4
```

**T1**: Download/generate matrix data (in Setup Instructions)

**T2**: Compile solver code (in Setup Instructions)

**T3C/T3G**: Execute tests on the CPU/GPU node

**T4**: Visualize and save results

To perform T3C:

```bash
# in the `work` directory
python suite-cpu.py
python suite-cpu2.py
```

To perform T3G:

```bash
python suite-gpu.py
```

After that, numerical results will be in the `work` directory in CSV format. To visualize the results, please run a python script:

```zsh
python plot <1--6 corresponding to the figure number in the paper>
```

