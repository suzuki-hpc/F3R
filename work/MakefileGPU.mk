CXX ?= nvc++
INCLUDE := -I ../library
OPTIONS := -std=c++17 -O3 -fopenmp -cuda
EXECDIR := bin
CLEAN_EXES := $(wildcard bin/*.exe)

all: $(EXECDIR) bicg cg gmres f3r

bicg: $(EXECDIR)/bicg64-gpu.exe $(EXECDIR)/bicg32-gpu.exe $(EXECDIR)/bicg16-gpu.exe
cg: $(EXECDIR)/cg64-gpu.exe $(EXECDIR)/cg32-gpu.exe $(EXECDIR)/cg16-gpu.exe
gmres: $(EXECDIR)/gm64-gpu.exe $(EXECDIR)/gm32-gpu.exe $(EXECDIR)/gm16-gpu.exe
f3r: $(EXECDIR)/f3r64-gpu.exe $(EXECDIR)/f3r32-gpu.exe $(EXECDIR)/f3r16-gpu.exe

$(EXECDIR)/bicg64-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_BiCG -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/bicg32-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_BiCG -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/bicg16-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_BiCG -DTYPE=__half $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg64-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_CG -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg32-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_CG -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg16-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_CG -DTYPE=__half $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm64-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_GM -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm32-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_GM -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm16-gpu.exe: src/krylov-gpu.cpp
	$(CXX) -DSOLV_GM -DTYPE=__half $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r64-gpu.exe: src/f3r-gpu.cpp
	$(CXX) -DDOUBLE -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r32-gpu.exe: src/f3r-gpu.cpp
	$(CXX) -DFLOAT -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r16-gpu.exe: src/f3r-gpu.cpp
	$(CXX) -DHALF -DTYPE=__half $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR):
	mkdir -p $(EXECDIR)

.PHONY: clean
clean:
	rm $(CLEAN_EXES)
