#include <cstdio>
#include <string>
#include <vector>

int llama_bench(int argc, char ** argv);

int main(int argc, char ** argv) {
    std::vector<std::string> translated;
    translated.reserve((size_t) argc);
    translated.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pp") {
            translated.push_back("-p");
        } else if (arg == "--tg") {
            translated.push_back("-n");
        } else if (arg == "--repeat") {
            translated.push_back("-r");
        } else if (arg == "--ctx") {
            if (i + 1 < argc) {
                ++i;
            }
            continue;
        } else if (arg == "--warmup") {
            if (i + 1 < argc) {
                ++i;
            }
            continue;
        } else {
            translated.push_back(arg);
        }
    }

    std::vector<char *> cargv;
    cargv.reserve(translated.size());
    for (auto & arg : translated) {
        cargv.push_back(const_cast<char *>(arg.c_str()));
    }

    return llama_bench((int) cargv.size(), cargv.data());
}