CXX ?= icpx
CXXFLAGS = -O3 -std=c++17

SIZES = 7_7_7 8_7_7 8_8_7 8_8_8

all: $(foreach s,$(SIZES),hpcg_$(s).mtx hpgmp_$(s).mtx)

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

hpcg_%.mtx: hpcg.exe
	./$< $(subst _, ,$*)

hpgmp_%.mtx: hpgmp.exe
	./$< $(subst _, ,$*)
