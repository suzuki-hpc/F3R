# Reproduce results in the paper

## Hardware Requirements

- **CPU**: Intel Xeon Max 9480 (or similar, with AVX512 and fp16 support)
- **GPU**: NVIDIA A100 (80 GB) or equivalent CUDA-enabled GPU
- **RAM**: ≥ 128 GB for the largest test case
- **Note**: The system must support fp16 computation

## Software Requirements

- **Compilers**:
  - **CPU Execution**: Intel oneAPI DPC++/C++ Compiler (`icpx`), version 2023.2.4+ with `-mavx512fp16`
  - **GPU Execution**: NVIDIA CUDA Compiler `nvcc` version 23.9+, together with a compatible host C++ compiler (e.g., `g++`)
  - **Note**: Compilers must support C++20
- **Python**: Version 3.8+
  - `Pandas >= 2.0.3`
  - `seaborn==0.13.2`
- **Build Tool**:
  - GNU Make 4.2.1

Install the required Python packages manually or with:

```bash
pip install -r requirements.txt
```

## Setup Instructions

##### 1. Clone the repository

```bash
git clone https://github.com/suzuki-hpc/F3R.git -b v2.0.0
cd F3R/sc25
```

##### 2. Prepare datasets

```zsh
cd matrix
zsh download.sh # Download SuiteSparse matrices
make            # Generate HPCG and HPGMP matrices
```

##### 3. Compile solvers

```bash
make -f Makefile CXX=icpx    # For CPU-only execution
make -f MakefileGPU CXX=g++  # For GPU execution
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
# in the `sc25` directory
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

Execute `suite-cpu2.py` with four different arguments corresponding to Figures 3–7:

```bash
# in the `sc25` directory
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
# in the `sc25` directory
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

Experimental results are written to CSV or TXT files in the `work` directory. To generate tables and figures reported in the paper, execute:

```zsh
python plot.py table3 # Generates Table 3
python plot.py table4 # Generates Table 4

python plot.py 1     # Generates Figure 1
python plot.py 2     # Generates Figure 2
...
python plot.py 7     # Generates Figure 7
```

