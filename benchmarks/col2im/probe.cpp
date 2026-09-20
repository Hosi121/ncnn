#include "allocator.h"
#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <vector>

static void fill(ncnn::Mat& m, unsigned seed)
{
    float* p = m;
    for (size_t i = 0; i < m.total(); i++)
    {
        seed = seed * 1664525u + 1013904223u;
        p[i] = ((int)((seed >> 16) & 255) - 128) / 512.f;
    }
}

int main(int argc, char** argv)
{
    if (argc != 13)
        return 2;
    const int w = atoi(argv[1]);
    const int h = atoi(argv[2]);
    const int inch = atoi(argv[3]);
    const int outch = atoi(argv[4]);
    const int kernel = atoi(argv[5]);
    const int stride = atoi(argv[6]);
    const int dilation = atoi(argv[7]);
    const int kh = getenv("NCNN_PROBE_KH") ? atoi(getenv("NCNN_PROBE_KH")) : kernel;
    const int sh = getenv("NCNN_PROBE_SH") ? atoi(getenv("NCNN_PROBE_SH")) : stride;
    const int dh = getenv("NCNN_PROBE_DH") ? atoi(getenv("NCNN_PROBE_DH")) : dilation;
    const int pad = atoi(argv[8]);
    const int bias = atoi(argv[9]);
    const int threads = atoi(argv[10]);
    const int packed = atoi(argv[11]);
    const int sgemm = atoi(argv[12]);

    ncnn::UnlockedPoolAllocator blobs;
    ncnn::PoolAllocator workspace;
    ncnn::Option opt;
    opt.num_threads = threads;
    opt.use_packing_layout = packed;
    opt.use_sgemm_convolution = sgemm;
    opt.use_fp16_storage = false;
    opt.use_fp16_packed = false;
    opt.use_bf16_storage = false;
    opt.blob_allocator = &blobs;
    opt.workspace_allocator = &workspace;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(11, kh);
    pd.set(2, dilation);
    pd.set(12, dh);
    pd.set(3, stride);
    pd.set(13, sh);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * inch * kernel * kh);
    const char* output_pad = getenv("NCNN_PROBE_OUTPUT_PAD");
    if (output_pad)
    {
        pd.set(18, atoi(output_pad));
        pd.set(19, getenv("NCNN_PROBE_OUTPUT_PAD_H") ? atoi(getenv("NCNN_PROBE_OUTPUT_PAD_H")) : atoi(output_pad));
    }
    if (getenv("NCNN_PROBE_ACTIVATION"))
        pd.set(9, atoi(getenv("NCNN_PROBE_ACTIVATION")));
    ncnn::Mat weights[2];
    weights[0].create(outch * inch * kernel * kh);
    weights[1].create(outch);
    fill(weights[0], 13);
    fill(weights[1], 29);
    ncnn::Layer* layer = ncnn::create_layer_cpu("Deconvolution");
    if (!layer || layer->load_param(pd) || layer->load_model(ncnn::ModelBinFromMatArray(weights)) || layer->create_pipeline(opt))
        return 3;
    ncnn::Mat input(w, h, inch);
    fill(input, 43);
    const int pack = packed ? (ncnn::cpu_support_x86_avx2() && inch % 8 == 0 ? 8 : inch % 4 == 0 ? 4 : 1) : 1;
    ncnn::Mat bottom;
    ncnn::convert_packing(input, bottom, pack, opt);
    ncnn::Mat output;
    for (int i = 0; i < 8; i++)
        if (layer->forward(bottom, output, opt))
            return 4;
    std::vector<double> samples;
    const bool check = getenv("NCNN_PROBE_CHECK") != 0;
    for (int round = 0; round < (check ? 0 : 9); round++)
    {
        const int repeats = 15;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; i++)
            if (layer->forward(bottom, output, opt))
                return 5;
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
        samples.push_back(us);
    }
    ncnn::Mat unpacked;
    ncnn::convert_packing(output, unpacked, 1, opt);
    uint64_t hash = 1469598103934665603ull;
    for (int c = 0; c < unpacked.c; c++)
    {
        const unsigned char* bytes = unpacked.channel(c);
        for (size_t i = 0; i < (size_t)unpacked.w * unpacked.h * 4; i++)
            hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
    printf("{\"w\":%d,\"h\":%d,\"inch\":%d,\"outch\":%d,\"kernel\":%d,\"stride\":%d,\"dilation\":%d,\"padding\":%d,\"bias\":%d,\"threads\":%d,\"packing\":%d,\"sgemm\":%d,\"out_pack\":%d,\"hash\":\"%016llx\",\"us\":[", w, h, inch, outch, kernel, stride, dilation, pad, bias, threads, packed, sgemm, output.elempack, (unsigned long long)hash);
    for (size_t i = 0; i < samples.size(); i++)
        printf("%s%.6f", i ? "," : "", samples[i]);
    printf("]}\n");
    if (check)
    {
        for (int c = 0; c < unpacked.c; c++)
            fwrite((const float*)unpacked.channel(c), sizeof(float), (size_t)unpacked.w * unpacked.h, stdout);
    }
    layer->destroy_pipeline(opt);
    delete layer;
    return 0;
}
