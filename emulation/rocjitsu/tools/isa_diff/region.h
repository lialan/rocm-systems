// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file region.h
/// @brief A region: a sequence of recorded machine instructions at their addresses.
///
/// @details A region is cut from a code object or written by hand. The executor needs only
/// each instruction's address and machine words, so a region can be re-run without the
/// artifact it came from.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

struct RegionInst {
  uint64_t pc = 0;
  std::vector<uint32_t> words;
  /// Disassembly when the region was recorded. Diagnostics quote it, and the executor reads
  /// what the decoded instruction does not expose from it (the mnemonic before decoding,
  /// stack-frame addressing, op_sel, true16 and SDWA destination halves).
  std::string text;
};

} // namespace rocjitsu::isa_diff
