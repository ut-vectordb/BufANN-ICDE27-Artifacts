#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "bufann/inplace_defs.h"

class AlignedFileReader;

namespace diskann {
namespace inplace {

class FileManager {
 public:
    FileManager() = default;
    explicit FileManager(uint32_t page_size);
    ~FileManager();

    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;

    void init(uint32_t page_size);
    FileNo register_file(const std::string& path, bool truncate_file);

    uint32_t page_size() const { return _page_size; }
    const std::string& path(FileNo file_no) const;
    uint32_t total_pages(FileNo file_no) const;
    uint32_t grow_file(FileNo file_no, uint32_t page_count);
    void set_total_pages(FileNo file_no, uint32_t used_pages);

    void read_page(PageID page_id, char* buf);
    void write_page(PageID page_id, const char* buf);

    AlignedFileReader& reader(FileNo file_no);
    int fd(FileNo file_no) const;

 private:
    struct FileState;
    FileState& file(FileNo file_no) const;

    uint32_t _page_size = 0;
    mutable std::mutex _files_mtx;
    std::vector<std::unique_ptr<FileState>> _files;
};

}  // namespace inplace
}  // namespace diskann
