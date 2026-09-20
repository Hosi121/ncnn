# x86 Deconvolution Col2im check

This branch contains the code and data for the performance check. The PR
contains only the operator change and its regression tests. These files were
prepared with help from an AI assistant.

The baseline is `1857775c98921d71e1ec2b34dc7d292c72529423`. The candidate is
`e1574f9f7c4a8611e3c758a66f6cee42ac20edf9`. The probe measures complete FP32
Deconvolution calls through the SGEMM path. It does not measure a full model.

## Build and run

Use Linux x86 with CMake, Ninja, GCC, OpenMP, Python 3, and taskset.

```bash
git clone --depth 1 --branch bench/deconvolution-col2im https://github.com/Hosi121/ncnn.git ncnn-col2im
cd ncnn-col2im
git fetch --depth 1 origin 1857775c98921d71e1ec2b34dc7d292c72529423
git worktree add --detach ../ncnn-col2im-base 1857775c98921d71e1ec2b34dc7d292c72529423
cmake -S ../ncnn-col2im-base -B ../build-base -G Ninja -C "$PWD/benchmarks/col2im/options.cmake"
cmake --build ../build-base --target ncnn test_deconvolution -j 4
cmake -S . -B ../build-candidate -G Ninja -C "$PWD/benchmarks/col2im/options.cmake"
cmake --build ../build-candidate --target ncnn test_deconvolution -j 4
ctest --test-dir ../build-candidate -R '^test_deconvolution$' --output-on-failure
g++ -std=c++11 -O3 benchmarks/col2im/probe.cpp -I src -I ../build-candidate/src -L ../build-candidate/src -lncnn -fopenmp -o ../col2im-probe
python3 benchmarks/col2im/compare.py --probe ../col2im-probe --base-lib ../build-base/src --candidate-lib ../build-candidate/src --mode check
python3 benchmarks/col2im/compare.py --probe ../col2im-probe --base-lib ../build-base/src --candidate-lib ../build-candidate/src --mode bench > ../col2im-results.jsonl
```

Set `--cpus-one` and `--cpus-four` to CPUs available to your process if CPUs 0
and 0-3 are not available. Do not run another build or benchmark at the same
time. The supplied options match the local build. AVX512 and INT8 are disabled.

The check compares every output byte for 236 configurations. These include
rectangular kernels, output padding, crop padding, bias, activation, packs
1/4/8, one or four threads, and overlap or dilation fallback cases.

The benchmark uses eight warmup calls and nine samples of 15 calls each.
Each variant starts in a new process. The order changes between rounds.
Take the median of the nine samples in each process, then take the median
of the three process medians. Setup, input packing, and output checks are
outside the timed region. The output tensor and pool allocators are reused.
The benchmark checks the final output hash. It does not check every timed
iteration. The byte comparison is a separate check.

## Saved data

`results.json` contains all 26 measured conditions. `raw.jsonl` contains the
samples for the baseline and this candidate. Both use the original local
measurement from 2026-09-20: Intel Core Ultra 7 255H, WSL2, GCC 13.3.
`up3_out4` uses input shape 128x128 with eight input and four output channels.
Other shapes and options are in `compare.py`.

The overlap controls were 1.7% and 2.2% slower for one and four threads.
That path retains the existing loop. This result does not prove zero cost
for every fallback case. A prior unpublished 2x2-only prototype was also
3.0% faster on the large one-thread 2x2 case in a separate five-process
check. The candidate remains faster than the upstream baseline there.

No full-model, ARM, Vulkan, or AVX512 speed result is claimed.
