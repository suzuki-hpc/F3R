CXX ?= icpx
INCLUDE := -I ../library
OPTIONS := -std=c++17 -fast -qopenmp -xCORE-AVX512 -mavx512fp16 -qnextgen
EXECDIR := bin
CLEAN_EXES := $(filter-out bin/*-gpu.exe,$(wildcard bin/*.exe))

all: $(EXECDIR) bicg cg gmres f3r depth depth_log $(EXECDIR)/static.exe

bicg: $(EXECDIR)/bicg64.exe $(EXECDIR)/bicg32.exe $(EXECDIR)/bicg16.exe
cg: $(EXECDIR)/cg64.exe $(EXECDIR)/cg32.exe $(EXECDIR)/cg16.exe
gmres: $(EXECDIR)/gm64.exe $(EXECDIR)/gm32.exe $(EXECDIR)/gm16.exe
f3r: $(EXECDIR)/f3r64.exe $(EXECDIR)/f3r32.exe $(EXECDIR)/f3r16.exe
depth: $(EXECDIR)/f4.exe $(EXECDIR)/f3.exe $(EXECDIR)/f3h.exe $(EXECDIR)/f2.exe $(EXECDIR)/f2h.exe
depth_log: $(EXECDIR)/f3r16_log.exe $(EXECDIR)/f4_log.exe $(EXECDIR)/f3_log.exe $(EXECDIR)/f3h_log.exe $(EXECDIR)/f2_log.exe $(EXECDIR)/f2h_log.exe

$(EXECDIR)/bicg64.exe: src/krylov.cpp
	$(CXX) -DSOLV_BiCG -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/bicg32.exe: src/krylov.cpp
	$(CXX) -DSOLV_BiCG -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/bicg16.exe: src/krylov.cpp
	$(CXX) -DSOLV_BiCG -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg64.exe: src/krylov.cpp
	$(CXX) -DSOLV_CG -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg32.exe: src/krylov.cpp
	$(CXX) -DSOLV_CG -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/cg16.exe: src/krylov.cpp
	$(CXX) -DSOLV_CG -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm64.exe: src/krylov.cpp
	$(CXX) -DSOLV_GM -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm32.exe: src/krylov.cpp
	$(CXX) -DSOLV_GM -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/gm16.exe: src/krylov.cpp
	$(CXX) -DSOLV_GM -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r64.exe: src/f3r.cpp
	$(CXX) -DDOUBLE -DTYPE=double $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r32.exe: src/f3r.cpp
	$(CXX) -DFLOAT -DTYPE=float $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r16.exe: src/f3r.cpp
	$(CXX) -DHALF -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f4.exe: src/depth.cpp
	$(CXX) -DF4 -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3.exe: src/depth.cpp
	$(CXX) -DF3 -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3h.exe: src/depth.cpp
	$(CXX) -DF3H -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f2.exe: src/depth.cpp
	$(CXX) -DF2 -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f2h.exe: src/depth.cpp
	$(CXX) -DF2H -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3r16_log.exe: src/f3r.cpp
	$(CXX) -DHALF -DTYPE=_Float16 -DLOGGING $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f4_log.exe: src/depth.cpp
	$(CXX) -DF4 -DTYPE=_Float16 -DLOGGING $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3_log.exe: src/depth.cpp
	$(CXX) -DF3 -DTYPE=_Float16 -DLOGGING $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f3h_log.exe: src/depth.cpp
	$(CXX) -DF3H -DTYPE=_Float16 -DLOGGING $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f2_log.exe: src/depth.cpp
	$(CXX) -DF2 -DTYPE=_Float16 -DLOGGING $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/f2h_log.exe: src/depth.cpp
	$(CXX) -DF2H -DTYPE=_Float16 -DLOGGING $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR)/static.exe: src/static.cpp
	$(CXX) -DTYPE=_Float16 $(OPTIONS) $(INCLUDE) $^ -o $@

$(EXECDIR):
	mkdir -p $(EXECDIR)

.PHONY: clean
clean:
	rm $(CLEAN_EXES)
