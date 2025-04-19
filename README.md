# F3R: A Mixed-Precision Linear Solver

This repository contains the software artifact accompanying a research paper currently under review.

## Paper 

This artifact is associated with a research submission currently under peer review. The paper will be linked here upon publication.

## Hardware Requirements

- **CPU**: Intel Xeon Max 9480 (or similar, with AVX512 and fp16 support)
- **GPU**: NVIDIA A100 (80 GB) or equivalent CUDA-enabled GPU
- **RAM**: ≥ 40 GB
- **Note**: The system must suport fp16 computing

## Software Requirements

- **Compilers**:
  - **CPU Execution**: Intel oneAPI DPC++/C++ Compiler (`icpx`), version 2023.2.4+ with `-mavx512fp16`
  - **GPU Execution**: NVIDIA HPC SDK `nvc++`, version 23.9+ with `-cuda`
  - **Note**: Compilers must support C++17
- **Python**: Version 3.8+
  - `Pandas >= 2.0.3`
  - `seaborn==0.13.2`
- **Build Tool**:
  - GNU Make 4.2.1

You can Install `icpx` from the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) and `nvc++` from the [NVIDIA HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-239-downloads).

Install the required Python packages manually or with:

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
zsh download.sh # Download SuiteSparse matrices
make            # Generate HPCG and HPGMP matrices
```

##### 3. Compile solvers

```bash
cd work
make CXX=<your C++ copmiler>   # For CPU-only execution
make -f MakefileGPU CXX=nvc++  # For GPU execution
```

## Execution

The complete experiment workflow:

```
T1 → T2 → T3_C + T3_G → T4
```

**T1**: Download/generate matrix data (done in Setup)

**T2**: Compile solver code (done in Setup)

**T3_C/T3_G**: Execute tests

**T4**: Visualize and save results

##### Run CPU tests (T3_C):

```bash
# in the `work` directory
python suite-cpu.py
python suite-cpu2.py
```

##### Run GPU tests (T3_G):

```bash
python suite-gpu.py
```

##### Result Visualization (T4):
After execution, numerical results will be stored as CSV in the `work` directory. To generate figures corresponding to the paper:

```zsh
python plot.py 1  # Generates Figure 1
python plot.py 2  # Generates Figure 2
...
python plot.py 6  # Generates Figure 6
```

Each number corresponds to the figure number in the paper.

