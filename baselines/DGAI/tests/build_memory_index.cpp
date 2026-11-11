#include <index.h>
#include <omp.h>
#include <string.h>
#include <numeric>
#include "utils.h"
#include <sys/mman.h>
#include <unistd.h>

template<typename T>
int build_in_memory_index(const std::string &data_path, const std::string &tags_file, const unsigned R,
                          const unsigned L, const float alpha, const std::string &save_path, const unsigned num_threads,
                          bool dynamic_index, bool single_file_index, pipeann::Metric distMetric) {
  pipeann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  _u64 data_num, data_dim;
  pipeann::get_bin_metadata(data_path, data_num, data_dim);
  std::cout << "Building in-memory index with parameters: data_file: " << data_path << "tags file: " << tags_file
            << " R: " << R << " L: " << L << " alpha: " << alpha << " index_path: " << save_path
            << " #threads: " << num_threads
            << ", using distance metric: " << (distMetric == pipeann::Metric::COSINE ? "cosine " : "l2 ");

  typedef uint32_t TagT;

  pipeann::Index<T, TagT> index(distMetric, data_dim, data_num, dynamic_index, single_file_index,
                                true);  // enable_tags forced to true!
  if (dynamic_index) {
    std::vector<TagT> tags(data_num);
    std::iota(tags.begin(), tags.end(), 0);

    auto s = std::chrono::high_resolution_clock::now();
    index.build(data_path.c_str(), data_num, paras, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
  } else {
    std::ifstream reader;
    std::cout << "Opening bin file " << tags_file << "... " << std::endl;
    reader.open(tags_file, std::ios::binary);
    reader.seekg(2 * sizeof(_u32), std::ios::beg);
    _u32 tags_size = data_num * data_dim;
    std::vector<TagT> tags(data_num);
    reader.read((char *) tags.data(), tags_size * sizeof(_u32));
    reader.close();
    std::cout << "First tag is " << tags[0] << std::endl;

    auto s = std::chrono::high_resolution_clock::now();
    index.build(data_path.c_str(), data_num, paras, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
  }
  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char **argv) {
  if (argc != 12) {
    std::cout << "Usage: " << argv[0]
              << " <data_type(int8/uint8/float)>  <data_file.bin>"
                 " <tags_file> (use null if no tags file is used) "
                 "<output_index_file> <dynamic_index(0/1)> <single_file_index(0/1)>"
              << " <R> <L> <alpha> <num_threads_to_use>"
              << " <distance_metric(l2/cosine case-sensitive)>."
              << " See README for more information on parameters." << std::endl;
    exit(-1);
  }

  int arg_no = 2;

  const std::string data_path(argv[arg_no++]);
  const std::string tags_file(argv[arg_no++]);
  const std::string save_path(argv[arg_no++]);
  bool dynamic_index = (bool) atoi(argv[arg_no++]);
  bool single_file_index = (bool) atoi(argv[arg_no++]);
  const unsigned R = (unsigned) atoi(argv[arg_no++]);
  const unsigned L = (unsigned) atoi(argv[arg_no++]);
  const float alpha = (float) atof(argv[arg_no++]);
  const unsigned num_threads = (unsigned) atoi(argv[arg_no++]);
  const std::string dist_metric_str = argv[arg_no++];
  enum pipeann::Metric distMetric = dist_metric_str == "cosine"
                                        ? pipeann::Metric::COSINE
                                        : pipeann::Metric::L2;  // set to l2 even if something else is chosen

  if (dist_metric_str != "l2" && distMetric == pipeann::Metric::L2) {
    std::cerr << "Unknown distance metric " << argv[argc - 1] << ". Setting metric to L2" << std::endl;
  }

  if (std::string(argv[1]) == std::string("int8"))
    build_in_memory_index<int8_t>(data_path, tags_file, R, L, alpha, save_path, num_threads, dynamic_index,
                                  single_file_index, distMetric);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_in_memory_index<uint8_t>(data_path, tags_file, R, L, alpha, save_path, num_threads, dynamic_index,
                                   single_file_index, distMetric);
  else if (std::string(argv[1]) == std::string("float"))
    build_in_memory_index<float>(data_path, tags_file, R, L, alpha, save_path, num_threads, dynamic_index,
                                 single_file_index, distMetric);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
