#include "bufann/buffer_pool/file_manager.h"

#include "ann_exception.h"
#include "linux_aligned_file_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace diskann {
namespace inplace {

struct FileManager::FileState {
    std::string path;
    int fd = -1;
    std::unique_ptr<AlignedFileReader> reader;
    std::atomic<uint32_t> total_pages{0};
    std::mutex grow_mtx;

    ~FileState() {
        if (reader) {
            reader->deregister_all_threads();
            reader.reset();
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
};

FileManager::FileManager(uint32_t page_size) {
    init(page_size);
}

FileManager::~FileManager() = default;

void FileManager::init(uint32_t page_size) {
    std::lock_guard<std::mutex> lock(_files_mtx);
    _page_size = page_size;
    _files.clear();
}

FileNo FileManager::register_file(const std::string& path, bool truncate_file) {
    if (_page_size == 0) {
        throw ANNException("FileManager::register_file before init", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    {
        std::lock_guard<std::mutex> lock(_files_mtx);
        if (_files.size() >= INVALID_FILE) {
            throw ANNException("FileManager file id space exhausted", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    int open_flags = O_RDWR | O_CREAT | O_DIRECT | O_LARGEFILE;
    if (truncate_file) open_flags |= O_TRUNC;
    int fd = ::open(path.c_str(), open_flags, 0644);
    if (fd < 0) {
        throw ANNException("Failed to open heap file: " + path, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    uint32_t total_pages = 0;
    if (!truncate_file) {
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            const int error = errno;
            ::close(fd);
            throw ANNException("Failed to stat heap file: " + path + ": errno=" + std::to_string(error) +
                                   " (" + std::string(std::strerror(error)) + ")",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        if (st.st_size < 0) {
            ::close(fd);
            throw ANNException("Heap file size is negative: " + path, -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        const uint64_t file_bytes = static_cast<uint64_t>(st.st_size);
        const uint64_t file_pages_u64 =
            (file_bytes + static_cast<uint64_t>(_page_size) - 1) / static_cast<uint64_t>(_page_size);
        if (file_pages_u64 > std::numeric_limits<uint32_t>::max()) {
            ::close(fd);
            throw ANNException("Heap file page count exceeds uint32_t range: " + path, -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        total_pages = static_cast<uint32_t>(file_pages_u64);
    }

    auto state = std::make_unique<FileState>();
    state->path = path;
    state->fd = fd;
    auto linux_reader = std::make_unique<LinuxAlignedFileReader>();
    linux_reader->use_external_fd(fd);
    state->reader = std::move(linux_reader);
    state->total_pages.store(total_pages, std::memory_order_release);

    std::lock_guard<std::mutex> lock(_files_mtx);
    if (_files.size() >= INVALID_FILE) {
        throw ANNException("FileManager file id space exhausted", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    const FileNo file_no = static_cast<FileNo>(_files.size());
    _files.emplace_back(std::move(state));
    return file_no;
}

FileManager::FileState& FileManager::file(FileNo file_no) const {
    std::lock_guard<std::mutex> lock(_files_mtx);
    if (file_no >= _files.size() || !_files[file_no]) {
        throw ANNException("FileManager: invalid file id " + std::to_string(file_no), -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    return *_files[file_no];
}

const std::string& FileManager::path(FileNo file_no) const {
    return file(file_no).path;
}

uint32_t FileManager::total_pages(FileNo file_no) const {
    return file(file_no).total_pages.load(std::memory_order_acquire);
}

uint32_t FileManager::grow_file(FileNo file_no, uint32_t page_count) {
    FileState& f = file(file_no);
    std::lock_guard<std::mutex> lock(f.grow_mtx);
    const uint32_t base = f.total_pages.load(std::memory_order_relaxed);
    const uint64_t max_pages = static_cast<uint64_t>(std::numeric_limits<off_t>::max()) /
                               static_cast<uint64_t>(_page_size);
    if (static_cast<uint64_t>(base) + page_count > max_pages) {
        throw ANNException("FileManager::grow_file: heap size limit exceeded", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const uint32_t new_total = base + page_count;
    const off_t new_size = static_cast<off_t>(new_total) * static_cast<off_t>(_page_size);
    if (::ftruncate(f.fd, new_size) != 0) {
        throw ANNException("FileManager::grow_file ftruncate failed: errno=" + std::to_string(errno) +
                               " (" + std::string(std::strerror(errno)) + ")",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    f.total_pages.store(new_total, std::memory_order_release);
    return base;
}

void FileManager::set_total_pages(FileNo file_no, uint32_t used_pages) {
    FileState& f = file(file_no);
    const uint32_t current = f.total_pages.load(std::memory_order_acquire);
    if (used_pages > current) {
        throw ANNException("FileManager::set_total_pages: used_pages=" + std::to_string(used_pages) +
                               " exceeds heap capacity=" + std::to_string(current),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    f.total_pages.store(used_pages, std::memory_order_release);
}

void FileManager::read_page(PageID page_id, char* buf) {
    FileState& f = file(page_id.first);
    ssize_t n = ::pread(f.fd, buf, _page_size, static_cast<off_t>(page_id.second) * _page_size);
    if (n < 0) n = 0;
    if (static_cast<size_t>(n) < _page_size) {
        memset(buf + n, 0, _page_size - n);
    }
}

void FileManager::write_page(PageID page_id, const char* buf) {
    FileState& f = file(page_id.first);
    ssize_t n = ::pwrite(f.fd, buf, _page_size, static_cast<off_t>(page_id.second) * _page_size);
    (void)n;
}


AlignedFileReader& FileManager::reader(FileNo file_no) {
    return *file(file_no).reader;
}

int FileManager::fd(FileNo file_no) const {
    return file(file_no).fd;
}

}  // namespace inplace
}  // namespace diskann
