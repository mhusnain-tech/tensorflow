/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// HLO-level accuracy tests for unary math intrinsics.
//
// These tests operate at the HLO level, compiling and running HLO modules
// through XLA's full compilation pipeline. This means:
//   1. Tests are resilient to changes in the underlying intrinsic
//      implementations (e.g., swapping LLVM intrinsics, changing polynomial
//      approximations, etc.).
//   2. They test the actual end-to-end path that user code follows.
//   3. Accuracy regressions are caught regardless of where in the pipeline
//      the regression was introduced.
//
// Golden baselines are generated offline using mpmath at 50 digits of
// precision (see generate_golden_baselines.py).
//
// NOTE: XLA already has exhaustive tests in xla/tests/exhaustive/ that test
// every representable float value for F32 and smaller types (and sampled
// subsets for F64). Those tests compare against a reference interpreter
// backend. These golden-baseline tests complement the exhaustive suite by:
//   - Comparing against an independent high-precision reference (mpmath),
//     not XLA's own interpreter.
//   - Providing explicit ULP budgets per op that serve as a contract.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/codegen/intrinsic/accuracy/accuracy_budget.h"
#include "xla/codegen/intrinsic/accuracy/golden_baselines.h"
#include "xla/fp_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// ---------------------------------------------------------------------------
// Test fixture: uses PjRt test runner with interpreter reference backend.
// ---------------------------------------------------------------------------

using HloIntrinsicAccuracyTest =
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

// ---------------------------------------------------------------------------
// Accuracy reporting (ULP-based, independent of XLA's ErrorSpec).
// ---------------------------------------------------------------------------

struct AccuracyReport {
  int max_ulp_error = 0;
  double mean_ulp_error = 0.0;
  int count = 0;
  double worst_input = 0.0;
  double worst_expected = 0.0;
  double worst_actual = 0.0;
};

template <typename T>
int UlpDistance(T actual, T expected) {
  if (std::isnan(expected)) {
    return std::isnan(actual) ? 0 : 1000000;
  }
  if (std::isinf(expected)) {
    return (std::isinf(actual) &&
            std::signbit(expected) == std::signbit(actual))
               ? 0
               : 1000000;
  }
  if (std::isnan(actual) || std::isinf(actual)) {
    return 1000000;
  }
  return std::abs(CalculateDistanceInFloats(actual, expected));
}

template <typename T>
AccuracyReport ComputeAccuracyReport(
    const std::vector<codegen::intrinsic::accuracy::RefPoint>& golden,
    const T* results, size_t count) {
  AccuracyReport report;
  int64_t total_ulp = 0;

  for (size_t i = 0; i < count; ++i) {
    T expected = static_cast<T>(golden[i].expected);

    // Skip subnormals in expected values — they are platform-dependent.
    if (std::fpclassify(expected) == FP_SUBNORMAL) continue;

    T actual = results[i];
    int ulp = UlpDistance(actual, expected);
    total_ulp += ulp;
    report.count++;

    if (ulp > report.max_ulp_error) {
      report.max_ulp_error = ulp;
      report.worst_input = golden[i].input;
      report.worst_expected = golden[i].expected;
      report.worst_actual = static_cast<double>(actual);
    }
  }

  if (report.count > 0) {
    report.mean_ulp_error = static_cast<double>(total_ulp) / report.count;
  }
  return report;
}

void LogAccuracyReport(const AccuracyReport& report,
                       absl::string_view test_name) {
  LOG(INFO) << "Accuracy Report for " << test_name << ":\n"
            << "  Tested points: " << report.count << "\n"
            << "  Max ULP Error: " << report.max_ulp_error << "\n"
            << "  Mean ULP Error: " << report.mean_ulp_error << "\n"
            << "  Worst Case: input=" << report.worst_input
            << ", expected=" << report.worst_expected
            << ", actual=" << report.worst_actual;
}

// ---------------------------------------------------------------------------
// HLO module templates.
// ---------------------------------------------------------------------------

std::string MakeUnaryHloModule(absl::string_view op_name,
                               absl::string_view type_str, int64_t count) {
  return absl::StrFormat(R"(
HloModule %s_accuracy_test

ENTRY main {
  input = %s[%d] parameter(0)
  ROOT result = %s[%d] %s(input)
}
)",
                         op_name, type_str, count, type_str, count, op_name);
}

// ---------------------------------------------------------------------------
// Parameterized test infrastructure.
// ---------------------------------------------------------------------------

struct IntrinsicAccuracyTestParam {
  std::string name;
  std::string hlo_op_name;
  PrimitiveType primitive_type;
  const codegen::intrinsic::accuracy::RefPoint* golden_data;
  size_t golden_count;
  int ulp_budget;
};

class HloIntrinsicAccuracyParamTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>,
      public ::testing::WithParamInterface<IntrinsicAccuracyTestParam> {
 public:
  HloIntrinsicAccuracyParamTest() {
    auto* opts = execution_options_.mutable_debug_options();
    opts->set_xla_cpu_enable_fast_math(false);
    opts->set_xla_cpu_enable_fast_min_max(false);
    opts->set_xla_gpu_enable_fast_min_max(false);
  }
};

// Filter golden points: remove those whose input overflows when cast to T.
template <typename T>
std::vector<codegen::intrinsic::accuracy::RefPoint> FilterGoldenForType(
    const codegen::intrinsic::accuracy::RefPoint* data, size_t count) {
  std::vector<codegen::intrinsic::accuracy::RefPoint> filtered;
  filtered.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    T input = static_cast<T>(data[i].input);
    // Skip points where the input overflows to inf during cast.
    if (std::isinf(input) && !std::isinf(data[i].input)) continue;
    filtered.push_back(data[i]);
  }
  return filtered;
}

TEST_P(HloIntrinsicAccuracyParamTest, WithinUlpBudget) {
  const auto& param = GetParam();

  std::string type_str;
  switch (param.primitive_type) {
    case F32:
      type_str = "f32";
      break;
    case F64:
      type_str = "f64";
      break;
    default:
      GTEST_SKIP() << "Unsupported type";
  }

  if (param.primitive_type == F32) {
    auto golden =
        FilterGoldenForType<float>(param.golden_data, param.golden_count);
    int64_t count = golden.size();

    // Build input literal from golden points.
    std::vector<float> inputs(count);
    for (int64_t i = 0; i < count; ++i) {
      inputs[i] = static_cast<float>(golden[i].input);
    }
    auto input_literal = LiteralUtil::CreateR1<float>(inputs);

    // Build and run HLO module.
    std::string hlo = MakeUnaryHloModule(param.hlo_op_name, type_str, count);
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_cpu_enable_fast_math(false);

    TF_ASSERT_OK_AND_ASSIGN(auto result,
                            Execute(std::move(module), {&input_literal}));

    // Compare against golden baselines.
    auto result_data = result.data<float>();
    auto report = ComputeAccuracyReport<float>(golden, result_data.data(),
                                               result_data.size());
    LogAccuracyReport(report, param.name);

    EXPECT_LE(report.max_ulp_error, param.ulp_budget)
        << "Max ULP error " << report.max_ulp_error << " exceeds budget "
        << param.ulp_budget << ". Worst case: input=" << report.worst_input
        << ", expected=" << report.worst_expected
        << ", actual=" << report.worst_actual;
  } else {
    auto golden =
        FilterGoldenForType<double>(param.golden_data, param.golden_count);
    int64_t count = golden.size();

    std::vector<double> inputs(count);
    for (int64_t i = 0; i < count; ++i) {
      inputs[i] = golden[i].input;
    }
    auto input_literal = LiteralUtil::CreateR1<double>(inputs);

    std::string hlo = MakeUnaryHloModule(param.hlo_op_name, type_str, count);
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_cpu_enable_fast_math(false);

        // XXX TODO SEAN: FIGURE OUT HOW TO ADD EXECUTION OPTIONS HERE.
    TF_ASSERT_OK_AND_ASSIGN(auto result,
                            Execute(std::move(module), {&input_literal}));

    auto result_data = result.data<double>();
    auto report = ComputeAccuracyReport<double>(golden, result_data.data(),
                                                result_data.size());
    LogAccuracyReport(report, param.name);

    EXPECT_LE(report.max_ulp_error, param.ulp_budget)
        << "Max ULP error " << report.max_ulp_error << " exceeds budget "
        << param.ulp_budget << ". Worst case: input=" << report.worst_input
        << ", expected=" << report.worst_expected
        << ", actual=" << report.worst_actual;
  }
}

// ---------------------------------------------------------------------------
// Test case registration.
// ---------------------------------------------------------------------------

using namespace codegen::intrinsic::accuracy;

INSTANTIATE_TEST_SUITE_P(
    UnaryIntrinsics, HloIntrinsicAccuracyParamTest,
    ::testing::Values(
        // Tanh
        IntrinsicAccuracyTestParam{"TanhF32", "tanh", F32, kGoldenTanh.data(),
                                   kGoldenTanh.size(), kTanhF32MaxUlp},
        IntrinsicAccuracyTestParam{"TanhF64", "tanh", F64, kGoldenTanh.data(),
                                   kGoldenTanh.size(), kTanhF64MaxUlp},

        // Exp
        IntrinsicAccuracyTestParam{"ExpF32", "exponential", F32,
                                   kGoldenExp.data(), kGoldenExp.size(),
                                   kExpF32MaxUlp},
        IntrinsicAccuracyTestParam{"ExpF64", "exponential", F64,
                                   kGoldenExp.data(), kGoldenExp.size(),
                                   kExpF64MaxUlp},

        // Log1p
        IntrinsicAccuracyTestParam{"Log1pF32", "log-plus-one", F32,
                                   kGoldenLog1p.data(), kGoldenLog1p.size(),
                                   kLog1pF32MaxUlp},
        IntrinsicAccuracyTestParam{"Log1pF64", "log-plus-one", F64,
                                   kGoldenLog1p.data(), kGoldenLog1p.size(),
                                   kLog1pF64MaxUlp},

        // Rsqrt
        IntrinsicAccuracyTestParam{"RsqrtF32", "rsqrt", F32,
                                   kGoldenRsqrt.data(), kGoldenRsqrt.size(),
                                   kRsqrtF32MaxUlp},
        IntrinsicAccuracyTestParam{"RsqrtF64", "rsqrt", F64,
                                   kGoldenRsqrt.data(), kGoldenRsqrt.size(),
                                   kRsqrtF64MaxUlp},

        // Sqrt
        IntrinsicAccuracyTestParam{"SqrtF32", "sqrt", F32, kGoldenSqrt.data(),
                                   kGoldenSqrt.size(), kSqrtF32MaxUlp},
        IntrinsicAccuracyTestParam{"SqrtF64", "sqrt", F64, kGoldenSqrt.data(),
                                   kGoldenSqrt.size(), kSqrtF64MaxUlp},

        // Erf
        IntrinsicAccuracyTestParam{"ErfF32", "erf", F32, kGoldenErf.data(),
                                   kGoldenErf.size(), kErfF32MaxUlp},
        IntrinsicAccuracyTestParam{"ErfF64", "erf", F64, kGoldenErf.data(),
                                   kGoldenErf.size(), kErfF64MaxUlp}),
    [](const ::testing::TestParamInfo<IntrinsicAccuracyTestParam>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla