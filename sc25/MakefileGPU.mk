NVCC := nvcc
CXX := g++
INCLUDE := -I ..
OPTIONS := -std=c++20 -O3 -arch=sm_80 --extended-lambda
XOPTIONS := -Xcompiler="-std=c++20 -O3 -fopenmp"
EXECDIR := bin
CLEAN_EXES := $(wildcard bin/*.exe)

all: $(EXECDIR) bicg cg gmres f3r

bicg: $(EXECDIR)/bicg64-gpu.exe $(EXECDIR)/bicg32-gpu.exe $(EXECDIR)/bicg16-gpu.exe
cg: $(EXECDIR)/cg64-gpu.exe $(EXECDIR)/cg32-gpu.exe $(EXECDIR)/cg16-gpu.exe
gmres: $(EXECDIR)/gm64-gpu.exe $(EXECDIR)/gm32-gpu.exe $(EXECDIR)/gm16-gpu.exe
f3r: $(EXECDIR)/f3r64-gpu.exe $(EXECDIR)/f3r32-gpu.exe $(EXECDIR)/f3r16-gpu.exe

$(EXECDIR)/bicg64-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_BiCG -DTYPE=double $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/bicg32-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_BiCG -DTYPE=float $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/bicg16-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_BiCG -DTYPE=__half $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg64-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_CG -DTYPE=double $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg32-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_CG -DTYPE=float $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg16-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_CG -DTYPE=__half $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm64-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_GM -DTYPE=double $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm32-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_GM -DTYPE=float $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm16-gpu.exe: src/krylov-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DSOLV_GM -DTYPE=__half $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r64-gpu.exe: src/f3r-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DDOUBLE -DTYPE=double $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r32-gpu.exe: src/f3r-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DFLOAT -DTYPE=float $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r16-gpu.exe: src/f3r-gpu.cu
	$(NVCC) -ccbin=$(CXX) $(OPTIONS) -DHALF -DTYPE=__half $(XOPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR):
	mkdir -p $(EXECDIR)

.PHONY: clean
clean:
	rm $(CLEAN_EXES)
