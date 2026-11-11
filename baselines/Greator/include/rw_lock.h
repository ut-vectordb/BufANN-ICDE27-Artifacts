#pragma once

#include <pthread.h>
#include <system_error>

namespace diskann {

class RWlock {
 public:
  RWlock() {
    pthread_rwlockattr_t attr;
    check(pthread_rwlockattr_init(&attr), "pthread_rwlockattr_init");
    try {
      check(pthread_rwlockattr_setkind_np(
                &attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP),
            "pthread_rwlockattr_setkind_np");
      check(pthread_rwlock_init(&_lock, &attr), "pthread_rwlock_init");
    } catch (...) {
      pthread_rwlockattr_destroy(&attr);
      throw;
    }
    check(pthread_rwlockattr_destroy(&attr), "pthread_rwlockattr_destroy");
  }

  ~RWlock() { pthread_rwlock_destroy(&_lock); }

  RWlock(const RWlock&) = delete;
  RWlock& operator=(const RWlock&) = delete;

  void lock() { check(pthread_rwlock_wrlock(&_lock), "pthread_rwlock_wrlock"); }

  void unlock() noexcept { pthread_rwlock_unlock(&_lock); }

  void lock_shared() {
    check(pthread_rwlock_rdlock(&_lock), "pthread_rwlock_rdlock");
  }

  void unlock_shared() noexcept { pthread_rwlock_unlock(&_lock); }

 private:
  static void check(int rc, const char* what) {
    if (rc != 0) {
      throw std::system_error(rc, std::generic_category(), what);
    }
  }

  pthread_rwlock_t _lock{};
};

}  // namespace diskann
