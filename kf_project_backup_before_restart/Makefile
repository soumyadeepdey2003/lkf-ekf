# ---------------------------------------------------------------------------------------------
#  make cpu      : the 4 CPU programs (g++ + OpenMP)                       -> build/<name>
#  make cuda     : the 12 GPU / adaptive / crossover programs (nvcc)       -> build/<name>
#  make jetson   : same 12, with unified memory, for Jetson Orin Nano (-arch=sm_87)
#  make shim     : the 12 GPU programs on the CPU with a CUDA imitation (LOGIC TEST ONLY, no GPU needed)
#  make test     : unit tests of all math (gain cases, CORDIC, RECIP, Jacobian, EKF/LKF steps)
# ---------------------------------------------------------------------------------------------
CXX=g++
NVCC=nvcc
CXXFLAGS=-O3 -std=c++14 -fopenmp -Wall -Wextra -Icommon
NVFLAGS=-O3 -std=c++14 -Icommon -Xcompiler -fopenmp
FILTERS=lkf_orig lkf_new ekf_orig ekf_new
CPU_PROGS=$(foreach f,$(FILTERS),build/$(f)_cpu)
CUDA_PROGS=$(foreach f,$(FILTERS),build/$(f)_gpu build/$(f)_hetero build/$(f)_crossover)

all: cpu
cpu: $(CPU_PROGS)
cuda: $(CUDA_PROGS)
HDRS=$(wildcard common/*.h common/*.cuh detection/*.h timing/*.h)

build:
	mkdir -p build
build/lkf_%_cpu: lkf/lkf_%_cpu.cpp $(HDRS) | build
	$(CXX) $(CXXFLAGS) $< -o $@ -lm
build/ekf_%_cpu: ekf/ekf_%_cpu.cpp $(HDRS) | build
	$(CXX) $(CXXFLAGS) $< -o $@ -lm
build/lkf_%: lkf/lkf_%.cu $(HDRS) | build
	$(NVCC) $(NVFLAGS) $< -o $@
build/ekf_%: ekf/ekf_%.cu $(HDRS) | build
	$(NVCC) $(NVFLAGS) $< -o $@
jetson: NVFLAGS += -arch=sm_87 -DKF_UNIFIED_MEMORY
jetson: cuda

# --- logic test of the GPU programs without a GPU (timings meaningless) ---
SHIM_PROGS=$(foreach f,$(FILTERS),build/shim_$(f)_gpu build/shim_$(f)_hetero build/shim_$(f)_crossover)
shim: $(SHIM_PROGS)
build/shim_lkf_%: lkf/lkf_%.cu $(HDRS) | build
	$(CXX) $(CXXFLAGS) -x c++ -DKF_CUDA_SHIM $< -o $@ -lm
build/shim_ekf_%: ekf/ekf_%.cu $(HDRS) | build
	$(CXX) $(CXXFLAGS) -x c++ -DKF_CUDA_SHIM $< -o $@ -lm

test: | build
	$(CXX) -O1 -g -std=c++14 -Wall -Wextra -Icommon tests/test_ekf.cpp -o build/test_ekf -lm
	$(CXX) -O1 -g -std=c++14 -Wall -Wextra -Icommon tests/test_warp.cpp -o build/test_warp -lm
	$(CC) -O1 -g -Wall -Wextra -Icommon tests/test_gain.c -o build/test_gain -lm
	./build/test_gain && ./build/test_ekf && ./build/test_warp
# --- the timing report tool (file 18 built as a program): original versus upgraded from timing_results.csv ---
report: | build
	$(CXX) -O2 -std=c++14 -x c++ -DKF_TIMING_REPORT_MAIN timing/kf_timing.h -o build/kf_timing_report
	./build/kf_timing_report timing_results.csv
clean:
	rm -rf build crossover_*.csv timing_results.csv
