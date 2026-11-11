#pragma once

#include <cstdint>
#include <limits>
#include <utility>

namespace diskann {
namespace inplace {

static constexpr uint32_t INVALID_PAGE = 0xFFFFFFFFu;
static constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;
using FileNo = uint32_t;
using PageNo = uint32_t;
typedef std::pair<FileNo, PageNo> PageID;
static constexpr FileNo INVALID_FILE = 0xFFFFFFFFu;

// Whether a pin will read the page or modify it. It selects how the pin holds
// the frame's single latch: READ takes it shared, WRITE takes it exclusive.
enum AccessMode : uint8_t { READ = 0, WRITE = 1 };

using TagType = uint32_t;
static constexpr TagType INVALID_TAG = 0xFFFFFFFFu;
static constexpr uint64_t LIVE_DELETE_GENERATION =
    std::numeric_limits<uint64_t>::max();

}  // namespace inplace
}  // namespace diskann
