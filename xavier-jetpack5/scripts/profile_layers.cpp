// Per-layer latency and energy for the NeuOS hash table, at whatever DVFS configuration
// the board is in. Each layer's Forward is repeated for at least MIN_MS so that the
// INA3221 rails (updated about once per ms) see enough of it; the GPU is synchronized
// before the clock and the energy counters are read.
//
// Output, one line per layer (the input format of hashtables/generate_hashtable.py):
//   <layer> <latency_us per call> <SYS5V mJ> <GPU mJ> <CPU mJ> <VDDRQ mJ>   (energy per call)
//
// usage: profile_layers deploy.prototxt weights.caffemodel [MIN_MS=200] [CONFIGS OUT_DIR]
//   With CONFIGS (lines of all-AGX.config) and OUT_DIR, the model is loaded once and each
//   configuration is applied with NeuOS' setters (DVFS/setconfig.h) before profiling it,
//   writing OUT_DIR/<id>.log; without them, the current configuration goes to stdout.
#include <caffe/caffe.hpp>
#include "DVFS/setconfig.h"
#define CLUSTER_FREQ_CUR "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq"
#define GPU_CUR "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq"
#include <cuda_runtime.h>
#include <energymon/energymon-odroid.h>

#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <random>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;

static std::string slurp(const char* path) {
  std::ifstream f(path);
  std::string v;
  std::getline(f, v);
  return v;
}

static void profile(Net<float>& net, energymon& em, double min_us, FILE* out) {
  uint64_t a[4], b[4];
  const auto& layers = net.layers();
  for (size_t i = 0; i < layers.size(); ++i) {
    const auto& bottom = net.bottom_vecs()[i];
    const auto& top = net.top_vecs()[i];
    for (int w = 0; w < 3; ++w) layers[i]->Forward(bottom, top);  // warm up
    cudaDeviceSynchronize();
    long calls = 0;
    em.fread(&em, a);
    const auto t0 = std::chrono::steady_clock::now();
    double el = 0;
    do {
      for (int r = 0; r < 10; ++r) layers[i]->Forward(bottom, top);
      calls += 10;
      cudaDeviceSynchronize();
      el = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    } while (el < min_us);
    em.fread(&em, b);
    fprintf(out, "%s %.3f", net.layer_names()[i].c_str(), el / calls);
    for (int ch = 0; ch < 4; ++ch) fprintf(out, " %.6f", (double)(b[ch] - a[ch]) / calls / 1e6);
    fprintf(out, "\n");
    fflush(out);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s deploy.prototxt weights.caffemodel [MIN_MS] [CONFIGS OUT_DIR]\n", argv[0]);
    return 2;
  }
  const double min_us = 1000.0 * (argc > 3 ? atof(argv[3]) : 200.0);
  FLAGS_minloglevel = 2;
  Caffe::set_mode(Caffe::GPU);
  Net<float> net(argv[1], caffe::TEST);
  net.CopyTrainedLayersFrom(argv[2]);

  Blob<float>* in = net.input_blobs()[0];
  // classify.bin reshapes the deploy input (batch 10) to a single image
  in->Reshape(1, in->channels(), in->height(), in->width());
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> u(-100.f, 100.f);
  float* d = in->mutable_cpu_data();
  for (int k = 0; k < in->count(); ++k) d[k] = u(rng);
  net.Reshape();

  energymon em;
  energymon_get_odroid(&em);
  if (em.finit(&em)) { perror("energymon init"); return 1; }

  if (argc < 6) {
    profile(net, em, min_us, stdout);
  } else {
    std::ifstream cfg(argv[4]);
    long id, cpu, gpu, emc;
    int mask[8];
    while (cfg >> id >> cpu >> mask[0] >> mask[1] >> mask[2] >> mask[3] >> mask[4] >> mask[5]
               >> mask[6] >> mask[7] >> gpu >> emc) {
      setCoresOnline(mask);
      setClusterFreq(cpu);
      setGpuFreq(gpu);
      setEmcFreq(emc);
      usleep(500000);
      std::string path = std::string(argv[5]) + "/" + std::to_string(id) + ".log";
      FILE* out = fopen(path.c_str(), "w");
      fprintf(out, "# config %ld cpu %ld mask %d%d%d%d%d%d%d%d gpu %ld emc %ld\n", id, cpu, mask[0], mask[1],
              mask[2], mask[3], mask[4], mask[5], mask[6], mask[7], gpu, emc);
      fprintf(out, "# measured: %s %s %s %s\n", slurp(CLUSTER_FREQ_CUR).c_str(), slurp(GPU_CUR).c_str(),
              slurp(EMC_FREQ).c_str(), slurp("/sys/devices/system/cpu/online").c_str());
      fprintf(out, "# layer latency_us sys5v_mJ gpu_mJ cpu_mJ vddrq_mJ\n");
      profile(net, em, min_us, out);
      fclose(out);
      printf("%ld done\n", id);
      fflush(stdout);
    }
  }
  em.ffinish(&em);
  return 0;
}
