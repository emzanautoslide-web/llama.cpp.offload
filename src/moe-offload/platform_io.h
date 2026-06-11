#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llama_moe {

struct read_result {
    size_t bytes = 0;
    int64_t elapsed_us = 0;
};

bool read_file_at(
        const std::string & path,
        uint64_t offset,
        size_t size,
        std::vector<uint8_t> & dst,
        read_result & result,
        std::string & error);

} // namespace llama_moe