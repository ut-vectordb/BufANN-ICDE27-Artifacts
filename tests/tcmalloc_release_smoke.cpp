// Smoke: same teardown order as InPlaceGraphStore::~InPlaceGraphStore —
// destroy large meta-like allocations, then release_allocator_free_memory().
//   cmake --build build --target tcmalloc_release_smoke &&
//   ./build/tests/tcmalloc_release_smoke
#include "bufann/inplace_backend.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>

namespace {

uint64_t read_vm_rss_kb() {
    std::ifstream in("/proc/self/status");
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (in >> key) {
        if (key == "VmRSS:") {
            in >> value >> unit;
            return value;
        }
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

// Mirrors store meta tables: a few large new[] buffers, then reset + release.
struct FakeMetaTables {
    std::unique_ptr<uint8_t[]> a;
    std::unique_ptr<uint8_t[]> b;
    explicit FakeMetaTables(size_t bytes_each) {
        a.reset(new uint8_t[bytes_each]);
        b.reset(new uint8_t[bytes_each]);
        std::memset(a.get(), 0xAB, bytes_each);
        std::memset(b.get(), 0xCD, bytes_each);
    }
};

}  // namespace

int main() {
    // Default 256 MiB per buffer (512 MiB total); override with env MiB each.
    const char *env = std::getenv("TCMALLOC_RELEASE_SMOKE_MIB");
    const size_t mib_each =
        env != nullptr ? static_cast<size_t>(std::strtoull(env, nullptr, 10))
                       : 256ull;
    const size_t bytes_each = mib_each << 20;
    const uint64_t need_delta_kb = (mib_each * 3 / 2) << 10;  // ~75% of total

    const uint64_t rss0 = read_vm_rss_kb();
    auto meta = std::make_unique<FakeMetaTables>(bytes_each);
    const uint64_t rss1 = read_vm_rss_kb();
    if (rss1 < rss0 + need_delta_kb) {
        std::cerr << "FAIL: alloc did not raise RSS (before_kb=" << rss0
                  << " after_kb=" << rss1 << ")\n";
        return 1;
    }

    // Destructor path: drop tables first (like resetting _owned_meta_arrays).
    meta.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const uint64_t rss_free = read_vm_rss_kb();

    // Free alone must NOT return most of the memory (tcmalloc page heap).
    if (rss_free + (64ull << 10) < rss1) {
        // Allow a little noise; require RSS still near post-alloc.
    }
    if (rss_free < rss0 + need_delta_kb) {
        std::cerr << "FAIL: free alone already dropped RSS; test cannot "
                     "distinguish ReleaseFreeMemory (rss_free_kb="
                  << rss_free << ")\n";
        return 2;
    }

    diskann::inplace::release_allocator_free_memory();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const uint64_t rss_release = read_vm_rss_kb();
    const int64_t drop_kb = static_cast<int64_t>(rss_free) -
                            static_cast<int64_t>(rss_release);

    std::cout << "mib_each=" << mib_each << " rss_alloc_kb=" << rss1
              << " rss_after_free_kb=" << rss_free
              << " rss_after_release_kb=" << rss_release
              << " drop_kb=" << drop_kb << std::endl;

    if (drop_kb < static_cast<int64_t>(need_delta_kb)) {
        std::cerr << "FAIL: release after meta teardown did not drop RSS "
                     "(drop_kb="
                  << drop_kb << ")\n";
        return 3;
    }
    std::cout << "PASS: meta teardown + release_allocator_free_memory "
                 "(~InPlaceGraphStore order)\n";
    return 0;
}
