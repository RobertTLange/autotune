#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Config {
  double kp = 5.0;
  double ki = 0.4;
  double kd = 1.0;
};

struct PlantState {
  double position = 0.0;
  double velocity = 0.0;
};

double parse_double(const char* value, const std::string& flag) {
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0') {
    throw std::runtime_error("invalid value for " + flag + ": " + value);
  }
  return parsed;
}

double target_at(double time) {
  if (time < 1.5) {
    return 1.0;
  }
  if (time < 3.0) {
    return -0.4;
  }
  return 0.7;
}

double disturbance_at(double time) {
  return time > 2.2 && time < 2.9 ? -0.8 : 0.0;
}

double simulate_controller(const Config& config) {
  constexpr double dt = 0.01;
  constexpr int steps = 500;
  constexpr double damping = 0.7;
  constexpr double spring = 0.35;
  constexpr double force_limit = 12.0;
  constexpr double integral_limit = 3.0;

  PlantState state;
  double integral = 0.0;
  double previous_error = target_at(0.0) - state.position;
  double tracking_loss = 0.0;
  double effort_loss = 0.0;
  double overshoot_loss = 0.0;

  for (int step = 0; step < steps; ++step) {
    const double time = step * dt;
    const double target = target_at(time);
    const double error = target - state.position;
    integral = std::clamp(integral + error * dt, -integral_limit, integral_limit);
    const double derivative = (error - previous_error) / dt;
    previous_error = error;

    const double raw_force = config.kp * error + config.ki * integral + config.kd * derivative;
    const double force = std::clamp(raw_force, -force_limit, force_limit);
    const double acceleration = force + disturbance_at(time) - damping * state.velocity - spring * state.position;
    state.velocity += acceleration * dt;
    state.position += state.velocity * dt;

    tracking_loss += error * error * dt;
    effort_loss += force * force * dt;
    if ((target > 0.0 && state.position > target) || (target < 0.0 && state.position < target)) {
      overshoot_loss += std::abs(state.position - target) * dt;
    }
  }

  return 10.0 - (6.0 * tracking_loss + 0.015 * effort_loss + 3.0 * overshoot_loss);
}

void print_help(const char* program) {
  std::cout << "Usage: " << program << " [ignored-source-path] [--kp VALUE] [--ki VALUE] [--kd VALUE]\n";
}

Config parse_args(int argc, char** argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--help") {
      print_help(argv[0]);
      std::exit(0);
    }
    if (arg == "--kp" || arg == "--ki" || arg == "--kd") {
      if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + arg);
      }
      const double value = parse_double(argv[++index], arg);
      if (arg == "--kp") {
        config.kp = value;
      } else if (arg == "--ki") {
        config.ki = value;
      } else {
        config.kd = value;
      }
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      throw std::runtime_error("unknown flag: " + arg);
    }
  }
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_args(argc, argv);
    const double score = simulate_controller(config);
    std::cout << "autotune_metric=" << score << "\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
  return 0;
}
