#include "platform_io.h"

#include "ggml.h"

#include <fstream>

namespace llama_moe {

bool read_file_at(
        const std::string & path,
        uint64_t offset,
        size_t size,
        std::vector<uint8_t> & dst,
        read_result & result,
        std::string & error) {
    const int64_t t0 = ggml_time_us();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "failed to open file: " + path;
        return false;
    }

    file.seekg((std::streamoff) offset, std::ios::beg);
    if (!file) {
        error = "failed to seek file: " + path;
        return false;
    }

    dst.resize(size);
    file.read(reinterpret_cast<char *>(dst.data()), (std::streamsize) size);
    result.bytes = (size_t) file.gcount();
    result.elapsed_us = ggml_time_us() - t0;

    if (result.bytes != size) {
        error = "short read from file: " + path;
        return false;
    }

    return true;
}

} // namespace llama_moe