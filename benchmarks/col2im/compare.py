"""Compare complete Deconvolution calls from two ncnn shared libraries."""

import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess


def run(args, variant, shape, kernel, stride, threads, *, dilation=(1, 1),
        padding=0, output_padding=(0, 0), bias=1, packed=1, activation=0):
    libraries = {"base": args.base_lib, "candidate": args.candidate_lib}
    env = dict(os.environ, LD_LIBRARY_PATH=str(libraries[variant].resolve()),
               NCNN_PROBE_KH=str(kernel[0]), NCNN_PROBE_SH=str(stride[0]),
               NCNN_PROBE_DH=str(dilation[0]),
               NCNN_PROBE_OUTPUT_PAD=str(output_padding[1]),
               NCNN_PROBE_OUTPUT_PAD_H=str(output_padding[0]),
               NCNN_PROBE_ACTIVATION=str(activation))
    env.pop("OMP_WAIT_POLICY", None)
    env.pop("GOMP_SPINCOUNT", None)
    env.pop("NCNN_PROBE_CHECK", None)
    if args.mode == "check":
        env.update(NCNN_PROBE_CHECK="1", OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
    cpus = args.cpus_one if threads == 1 else args.cpus_four
    command = ["taskset", "-c", cpus, str(args.probe.resolve()),
               *map(str, [*shape, kernel[1], stride[1], dilation[1], padding,
                          bias, threads, packed, 1])]
    result = subprocess.run(command, env=env, capture_output=True, check=True)
    header, _, output = result.stdout.partition(b"\n")
    return json.loads(header), output


def check(args):
    cases = 0
    values = 0
    shapes = [(1, 1, 1, 1), (7, 9, 8, 8), (13, 11, 8, 4), (17, 19, 13, 3)]
    kernels = [(1, 1), (2, 2), (3, 3), (4, 4), (2, 3), (1, 5), (5, 1)]
    configurations = []
    for shape, kernel, threads, packed, bias in itertools.product(
            shapes, kernels, [1, 4], [0, 1], [0, 1]):
        index = len(configurations)
        configurations.append(dict(
            shape=shape, kernel=kernel, stride=kernel, threads=threads,
            packed=packed, bias=bias,
            output_padding=[(0, 0), (1, 0), (0, 1), (1, 2)][index % 4],
            padding=1 if min(shape[:2]) > 1 and index % 3 == 0 else 0,
            activation=[0, 1, 4][index % 3]))
    for kernel, stride, dilation in [((3, 3), (2, 2), (1, 1)),
                                      ((2, 2), (2, 2), (2, 2)),
                                      ((2, 3), (1, 2), (1, 2))]:
        for threads, packed in itertools.product([1, 4], [0, 1]):
            configurations.append(dict(shape=(31, 33, 16, 16), kernel=kernel,
                                       stride=stride, dilation=dilation,
                                       threads=threads, packed=packed))
    for options in configurations:
        _, expected = run(args, "base", **options)
        _, actual = run(args, "candidate", **options)
        if not expected or expected != actual:
            raise AssertionError(options)
        cases += 1
        values += len(actual) // 4
    print(json.dumps(dict(cases=cases, output_values=values, bit_equal=True)))


def bench(args):
    cases = [
        ("up2_small", (16, 16, 8, 8), (2, 2), (2, 2), {}),
        ("up2", (184, 184, 16, 16), (2, 2), (2, 2), {}),
        ("up2_large", (368, 368, 16, 16), (2, 2), (2, 2), {}),
        ("up3", (128, 128, 16, 16), (3, 3), (3, 3), {}),
        ("up4", (96, 96, 16, 16), (4, 4), (4, 4), {}),
        ("rect_2x3", (128, 128, 16, 16), (2, 3), (2, 3), {}),
        ("up2_pad", (184, 184, 16, 16), (2, 2), (2, 2), dict(output_padding=(1, 1))),
        ("pointwise", (184, 184, 16, 16), (1, 1), (1, 1), {}),
        ("up3_out3", (128, 128, 8, 3), (3, 3), (3, 3), {}),
        ("up3_out4", (128, 128, 8, 4), (3, 3), (3, 3), {}),
        ("up3_unpacked", (96, 96, 16, 16), (3, 3), (3, 3), dict(packed=0)),
        ("overlap", (64, 64, 16, 16), (3, 3), (2, 2), dict(padding=1)),
        ("dilation", (64, 64, 16, 16), (2, 2), (2, 2), dict(dilation=(2, 2))),
    ]
    for round_index in range(args.rounds):
        for threads in [1, 4]:
            for name, shape, kernel, stride, options in cases:
                pair = {}
                order = ["base", "candidate"] if round_index % 2 == 0 else ["candidate", "base"]
                for variant in order:
                    item, _ = run(args, variant, shape, kernel, stride, threads, **options)
                    item.update(case=name, variant=variant, round=round_index,
                                kernel_hw=kernel, stride_hw=stride,
                                median_us=statistics.median(item["us"]))
                    pair[variant] = item
                if pair["base"]["hash"] != pair["candidate"]["hash"]:
                    raise AssertionError((name, threads, round_index))
                for item in pair.values():
                    print(json.dumps(item), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--base-lib", type=Path, required=True)
    parser.add_argument("--candidate-lib", type=Path, required=True)
    parser.add_argument("--mode", choices=["check", "bench"], default="check")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpus-one", default="0")
    parser.add_argument("--cpus-four", default="0-3")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    (check if args.mode == "check" else bench)(args)
