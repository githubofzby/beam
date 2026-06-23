#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

class FastScanner {
 public:
  explicit FastScanner(const std::string& path)
      : file_(std::fopen(path.c_str(), "rb")),
        buffer_(kBufferSize),
        idx_(0),
        size_(0) {
    if (!file_) {
      throw std::runtime_error("Failed to open file: " + path);
    }
  }

  ~FastScanner() {
    if (file_) {
      std::fclose(file_);
    }
  }

  bool ReadInt64(int64_t& out) {
    int c = NextChar();
    while (c <= ' ') {
      if (c == EOF) {
        return false;
      }
      c = NextChar();
    }
    int sign = 1;
    if (c == '-') {
      sign = -1;
      c = NextChar();
    }
    int64_t value = 0;
    while (c > ' ') {
      value = value * 10 + (c - '0');
      c = NextChar();
    }
    out = value * sign;
    return true;
  }

 private:
  int NextChar() {
    if (idx_ >= size_) {
      size_ = std::fread(buffer_.data(), 1, buffer_.size(), file_);
      idx_ = 0;
      if (size_ == 0) {
        return EOF;
      }
    }
    return static_cast<unsigned char>(buffer_[idx_++]);
  }

  static constexpr size_t kBufferSize = 1 << 20;
  FILE* file_;
  std::vector<char> buffer_;
  size_t idx_;
  size_t size_;
};
