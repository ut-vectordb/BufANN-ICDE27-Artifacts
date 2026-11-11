// bufann_index_builder.cpp
//
// Driver 1: Build a BufANN disk-resident index from a raw data file.
//
// Usage:
//   bufann_index_builder --data_type float|int8|uint8
//                         --data_file <path>
//                         --index_prefix <path>
//                         --dim <N>
//                        [--R <N>]               default 64
//                        [--L <N>]               default 100
//                        [--C <N>]               default 750
//                        [--alpha <f>]            default 1.2
//                        [--build_threads <N>]   default 8
//                        [--beamwidth <N>]        default 4
//                        [--pq_chunks <N>]        default 0 (no PQ)
//                        [--memory_budget_gb <f>] default 0 (in-memory build)
//                        [--build_temp_dir <path>]
//                        [--metric L2|cosine]     default L2
//                        [--buffer_pool_frames <N>] default 16384
//                        [--saturate_graph 0|1]   default 0
//
// Produces files: <index_prefix>.heap, .meta, .medoids, .centroids
// (and PQ files when pq_chunks > 0).
//
// Immediate insert repair is always enabled.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "distance.h"
#include "bufann/bufann_api.h"

using namespace diskann::inplace;

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

static std::string get_arg(int argc, char **argv, const std::string &flag,
                            const std::string &def = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) return argv[i + 1];
    }
    return def;
}

static bool has_flag(int argc, char **argv, const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Typed build dispatch
// ---------------------------------------------------------------------------

template <typename T>
static int do_build(int argc, char **argv) {
    BufANNConfig cfg;

    const std::string data_file    = get_arg(argc, argv, "--data_file");
    const std::string index_prefix = get_arg(argc, argv, "--index_prefix");
    const std::string build_temp   = get_arg(argc, argv, "--build_temp_dir", "");

    if (data_file.empty() || index_prefix.empty()) {
        std::cerr << "ERROR: --data_file and --index_prefix are required" << std::endl;
        return 1;
    }

    cfg.dim = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--dim", "0")));
    if (cfg.dim == 0) {
        std::cerr << "ERROR: --dim must be non-zero" << std::endl;
        return 1;
    }

    cfg.R                 = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--R",               "64")));
    cfg.L                 = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--L",              "100")));
    cfg.C                 = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--C",              "750")));
    cfg.alpha             = std::stof(get_arg(argc, argv, "--alpha",             "1.2"));
    cfg.build_threads     = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--build_threads",    "8")));
    cfg.beamwidth         = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--beamwidth",         "4")));
    cfg.pq_chunks         = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--pq_chunks",         "0")));
    cfg.memory_budget_gb  = std::stof(get_arg(argc, argv, "--memory_budget_gb",  "0.0"));
    cfg.buffer_pool_frames = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--buffer_pool_frames", "16384")));
    cfg.saturate_graph    = (get_arg(argc, argv, "--saturate_graph", "0") != "0");

    const std::string metric_str = get_arg(argc, argv, "--metric", "L2");
    if (metric_str == "cosine" || metric_str == "COSINE") {
        cfg.metric = diskann::Metric::COSINE;
    } else {
        cfg.metric = diskann::Metric::L2;
    }

    std::cout << "Building index: data=" << data_file
              << "  prefix=" << index_prefix
              << "  dim=" << cfg.dim
              << "  R=" << cfg.R << "  L=" << cfg.L
              << "  threads=" << cfg.build_threads
              << "  metric=" << metric_str
              << (cfg.memory_budget_gb > 0 ? "  mode=disk_stream" : "  mode=in_memory")
              << std::endl;
    std::cout.flush();

    try {
        BufANNIndex<T> *idx = bufann_build<T>(data_file, index_prefix, cfg, build_temp);
        std::cout << "Index built successfully at prefix: " << index_prefix << std::endl;
        bufann_free(idx);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (has_flag(argc, argv, "--help") || argc < 2) {
        std::cout << "Usage: bufann_index_builder --data_type float|int8|uint8\n"
                     "         --data_file <path> --index_prefix <path> --dim <N>\n"
                     "        [--R N] [--L N] [--C N] [--alpha f]\n"
                     "        [--build_threads N] [--beamwidth N] [--pq_chunks N]\n"
                     "        [--memory_budget_gb f] [--build_temp_dir path]\n"
                     "        [--metric L2|cosine] [--buffer_pool_frames N]\n"
                     "        [--saturate_graph 0|1]" << std::endl;
        return 0;
    }

    const std::string dtype = get_arg(argc, argv, "--data_type", "float");
    if (dtype == "float") return do_build<float>(argc, argv);
    if (dtype == "int8")  return do_build<int8_t>(argc, argv);
    if (dtype == "uint8") return do_build<uint8_t>(argc, argv);
    std::cerr << "ERROR: unknown --data_type '" << dtype
              << "'; expected float|int8|uint8" << std::endl;
    return 1;
}
