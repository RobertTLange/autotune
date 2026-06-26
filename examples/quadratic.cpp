#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

double parse_double(const char* value, const std::string& flag) {
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0') {
    throw std::runtime_error("invalid value for " + flag + ": " + value);
  }
  return parsed;
}

void print_help(const char* program) {
  std::cout << "Usage: " << program << " [ignored-source-path] [--x VALUE]\n";
}

}  // namespace

int main(int argc, char** argv) {
  double x = 0.0;

  try {
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--help") {
        print_help(argv[0]);
        return 0;
      }
      if (arg == "--x") {
        if (index + 1 >= argc) {
          throw std::runtime_error("missing value for " + arg);
        }
        x = parse_double(argv[++index], arg);
        continue;
      }
      if (arg.rfind("--", 0) == 0) {
        throw std::runtime_error("unknown flag: " + arg);
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }

  const double score = 1.0 - std::pow(x - 0.7, 2.0);
  std::cout << "autotune_metric=" << score << "\n";
  return 0;
}
