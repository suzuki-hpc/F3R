# F3R: A Mixed-Precision Linear Solver

This repository contains the software artifact accompanying the following research paper.

Suzuki, Kengo, and Takeshi Iwashita. "A Nested Krylov Method Using Half-Precision Arithmetic." *In The International Conference for High Performance Computing, Networking, Storage and Analysis (SC ’25)*, November 16–21, 2025, St Louis, MO, USA. DOI: 10.1145/3712285.3759807

**Note**: Version **v1.0.1** corresponds to the following preprint (some figure and table numbers were updated in the published version):

Suzuki, Kengo, and Takeshi Iwashita. "A Nested Krylov Method Using Half-Precision Arithmetic." *arXiv preprint arXiv:2505.20719*(2025). DOI: 10.48550/arXiv.2505.20719

## Hardware Requirements

- **CPU**: Intel Xeon Max 9480 (or similar, with AVX512 and fp16 support)
- **GPU**: NVIDIA A100 (80 GB) or equivalent CUDA-enabled GPU
- **RAM**: ≥ 128 GB for the largest test case
- **Note**: The system must support fp16 computation

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

You can install `icpx` from the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) and `nvc++` from the [NVIDIA HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-239-downloads).

Install the required Python packages manually or with:

```bash
pip install -r requirements.txt
```

## Setup Instructions

##### 1. Clone the repository

```bash
git clone https://github.com/suzuki-hpc/F3R.git -b v1.0.2
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
make CXX=icpx                  # For CPU-only execution
make -f MakefileGPU CXX=nvc++  # For GPU execution
```

## Execution

The complete experiment workflow:

```
T1 → T2 → T3_C + T3_G → T4
```

**T1**: Download/generate matrix data (done in Setup)

**T2**: Compile solver code (done in Setup)

**T3_C/T3_G**: Perform CPU and GPU tests

**T4**: Visualize and save results

### Run CPU tests (T3_C)

#### Reproducing the results in Section 5.1

Run the following two independent commands:

```bash
# in the `work` directory
python suite-cpu.py <average> figure1a
python suite-cpu.py <average> figure1b
```

`<average>` is an integer parameter to specify the number of repetitions to compute the average. `1` would be sufficient to reproduce the general trend of the results; that is,

```
python suite-cpu.py 1 figure1a
python suite-cpu.py 1 figure1b
```

The commands above use only one-third of the test matrices to save time. If you would like to test all matrices, execute the following commands instead:

```
python suite-cpu.py <average> figure1a full
python suite-cpu.py <average> figure1b full
```

#### Reproducing the results in Section 6

Execute `suite-cpu2.py` with four different arguments corresponding to Figures 3, 4, 5, and 6:

```bash
# in the `work` directory
python suite-cpu2.py <average> figure3
python suite-cpu2.py <average> figure4
python suite-cpu2.py <average> figure5
python suite-cpu2.py <average> figure6
python suite-cpu2.py <average> figure7
```

`<average>` is the same parameter as for `suite-cpu.py`; that is, it specifies the number of repetitions. `1` would be sufficient to reproduce the results quickly.

These commands also use only one-third of the matrices to save time; you may pass `full` at the end of the commands to test all matrices, like 

```bash
python suite-cpu2.py 1 figure3 full
```

### Run GPU tests (T3_G)

#### Reproducing the results in Section 5.2

Run `suite-gpu.py` on a CPU-GPU system with parameters `<average>` and `figure2a` / `figure2b`:

```bash
# in the `work` directory
python suite-gpu.py <average> figure2a
python suite-gpu.py <average> figure2b
```

These two commands are independent of each other; one can perform them in parallel if the system accepts multiple jobs at the same time.

Similar to the script for CPU tests, `suite-gpu.py` uses only half the test matrices by default to save time. If you need a full reproduction, set `full`:

```bash
python suite-gpu.py <average> figure2a full
python suite-gpu.py <average> figure2b full
```

### Visualize Results (T4)

After execution, numerical results will be stored as CSV or TXT in the `work` directory. To generate tables and figures corresponding to the paper, execute the following commands:

```zsh
python plot.py table # Generates Tables 3 and 4

python plot.py 1     # Generates Figure 1
python plot.py 2     # Generates Figure 2
...
python plot.py 7     # Generates Figure 7
```

