// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iostream>

#ifdef EXEC_ENV_OLS
#include "ANNLoggingImpl.hpp"
#endif

#include "logger_impl.h"
#include "windows_customizations.h"

namespace diskann {

  DISKANN_DLLEXPORT ANNStreamBuf coutBuff(stdout);
  DISKANN_DLLEXPORT ANNStreamBuf cerrBuff(stderr);

  DISKANN_DLLEXPORT std::basic_ostream<char> cout(&coutBuff);
  DISKANN_DLLEXPORT std::basic_ostream<char> cerr(&cerrBuff);

  ANNStreamBuf::ANNStreamBuf(FILE* fp) {
    if (fp == nullptr) {
      throw diskann::ANNException(
          "File pointer passed to ANNStreamBuf() cannot be null", -1);
    }
    if (fp != stdout && fp != stderr) {
      throw diskann::ANNException(
          "The custom logger only supports stdout and stderr.", -1);
    }
    _fp = fp;
    _logLevel = (_fp == stdout) ? ANNIndex::LogLevel::LL_Info
                                : ANNIndex::LogLevel::LL_Error;
#ifdef EXEC_ENV_OLS
    _buf = new char[BUFFER_SIZE + 1];  // See comment in the header
#else
    // In non-OLS mode we write directly to FILE* in overflow/xsputn and keep
    // the stream buffer empty. Allocate one byte so pbase()/pptr() are valid.
    _buf = new char[1];
#endif

#ifdef EXEC_ENV_OLS
    std::memset(_buf, 0, (BUFFER_SIZE) * sizeof(char));
    setp(_buf, _buf + BUFFER_SIZE);
#else
    _buf[0] = '\0';
    setp(_buf, _buf);
#endif
  }

  ANNStreamBuf::~ANNStreamBuf() {
    sync();
    _fp = nullptr;  // we'll not close because we can't.
    delete[] _buf;
  }

  int ANNStreamBuf::overflow(int c) {
    std::lock_guard<std::mutex> lock(_mutex);
#ifdef EXEC_ENV_OLS
    if (c != EOF) {
      *pptr() = (char) c;
      pbump(1);
    }
    flush();
#else
    if (c != EOF) {
      char ch = (char) c;
      logImpl(&ch, 1);
    }
#endif
    return c;
  }

  std::streamsize ANNStreamBuf::xsputn(const char* s, std::streamsize count) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (count <= 0) {
      return 0;
    }
#ifdef EXEC_ENV_OLS
    for (std::streamsize i = 0; i < count; ++i) {
      if (pptr() == epptr()) {
        flush();
      }
      *pptr() = s[i];
      pbump(1);
    }
#else
    logImpl(const_cast<char*>(s), static_cast<int>(count));
#endif
    return count;
  }

  int ANNStreamBuf::sync() {
    std::lock_guard<std::mutex> lock(_mutex);
#ifdef EXEC_ENV_OLS
    flush();
#endif
    return 0;
  }

  int ANNStreamBuf::underflow() {
    throw diskann::ANNException(
        "Attempt to read on streambuf meant only for writing.", -1);
  }

  int ANNStreamBuf::flush() {
#ifdef EXEC_ENV_OLS
    const int num = (int) (pptr() - pbase());
    if (num > 0) {
      logImpl(pbase(), num);
      pbump(-num);
    }
    return num;
#else
    return 0;
#endif
  }
  void ANNStreamBuf::logImpl(char* str, int num) {
#ifdef EXEC_ENV_OLS
    str[num] = '\0';  // Safe. See the c'tor.
    DiskANNLogging(_logLevel, str);
#else
    fwrite(str, sizeof(char), num, _fp);
    fflush(_fp);
#endif
  }

}  // namespace diskann
