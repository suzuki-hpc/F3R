# F3R: A Mixed-Precision Linear Solver

This repository contains the software artifact accompanying the following research paper:

**Kengo Suzuki** and **Takeshi Iwashita**.
<br>"A Nested Krylov Method Using Half-Precision Arithmetic."
<br>In *The International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25)*, November 16–21, 2025, St. Louis, MO, USA.
<br>DOI: 10.1145/3712285.3759807

## About This Repository

Version **v1.0.2** was used in the reproducibility evaluation process.
However, compilation issues were reported on some GPU systems.
To improve portability and reproducibility, we updated the source code and released **v2.0.0**.

In addition, the files and scripts used to reproduce the results reported in the SC '25 paper have been moved to the `sc25` directory.
The repository has also been reorganized to provide a more general framework for using the developed solvers.

For detailed instructions on reproducing the numerical results presented in the paper, please refer to the documentation in the `sc25` directory.

## Compilation Tests

We provide a set of simple unit tests based on **doctest**. The tests can be compiled and executed separately for both CPU and GPU targets as shown below.
Before using the source code, we recommend running these tests to verify that the software compiles and executes correctly on your system.

```bash
git clone --recursive https://github.com/suzuki-hpc/F3R.git
cd F3R/test
```

### CPU Test
```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DSENK_TARGETS_TO_BUILD=cpu
cmake --build build
cd build
./cpu_test
```

### GPU Test
```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DSENK_TARGETS_TO_BUILD=cuda
cmake --build build
cd build
./cuda_test
```

## Examples

Simple usage examples and solver demonstrations are provided in the `examples` directory. These examples will be expanded and improved in future releases.