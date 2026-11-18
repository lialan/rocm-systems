// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target.h
/// @brief The GPU a region runs on: its ISA family, concrete GPU binding, wave size, and
/// the architectural traits the executor depends on.
///
/// @details Targets are the processors `rocjitsu/code/amdgpu_elf.h` names (the place a new
/// GPU is added) whose ISA family has an executable provider in the default ISA target
/// registry. The registry supplies the concrete GPU binding when it has one, so the decoder
/// and the compute unit select target-specific behaviour (for example gfx1250 vs gfx1251).
#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rocjitsu {
class Decoder;
}

namespace rocjitsu::isa_diff {

struct Target {
  std::string name = "gfx950";
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA4;
  /// Concrete GPU binding in the ISA target registry, or ROCJITSU_CODE_TARGET_INVALID when
  /// the family provider has none for this processor.
  rj_code_target_id_t gpu = ROCJITSU_CODE_TARGET_GFX950;
  unsigned wave_size = 64;
  /// GFX generation: 9 (GFX9 and CDNA1-4), 10 (RDNA1-2), 11 (RDNA3/3.5), 12 (RDNA4, CDNA5).
  unsigned major = 9;
  /// gfx90a and later CDNA: one register file that ACCUM_OFFSET splits into VGPRs and AGPRs.
  bool unified_vgprs = true;

  [[nodiscard]] bool wave32() const { return wave_size == 32; }
  /// Addressable SGPRs: s0-s101 on GFX9, s0-s105 from GFX10.
  [[nodiscard]] unsigned sgprs() const { return major >= 10 ? 106 : 102; }
  /// SGPRs an executor wave allocates (GFX9 keeps two above s101 for the frame).
  [[nodiscard]] unsigned wave_sgprs() const { return major >= 10 ? 106 : 104; }
  /// MODE has DX10_CLAMP and IEEE (removed in GFX12).
  [[nodiscard]] bool has_ieee_dx10() const { return major < 12; }
  /// "gfx1100 wave32".
  [[nodiscard]] std::string label() const { return name + " wave" + std::to_string(wave_size); }
  bool operator==(const Target &) const = default;
};

/// @brief An executable target by processor name ("gfx950") or ISA family ID ("cdna4"),
/// with its default wave size.
[[nodiscard]] std::optional<Target> target_by_name(std::string_view name);
/// @brief The target of an AMDGPU ELF machine number (`e_flags & EF_AMDGPU_MACH`).
[[nodiscard]] std::optional<Target> target_by_elf_mach(uint32_t mach);
/// @brief Apply a wave size; nullopt when the target cannot run at it (GFX9/CDNA1-4 are
/// wave64 only, CDNA5 wave32 only).
[[nodiscard]] std::optional<Target> with_wave_size(Target target, unsigned wave_size);
/// @brief Parse "gfx1100", "gfx1100:w64" or "gfx1100:wave64".
[[nodiscard]] std::optional<Target> parse_target(std::string_view text);
/// @brief Comma-separated names of every supported processor.
[[nodiscard]] std::string supported_targets();
/// @brief The decoder of @p target: its concrete GPU binding when the registry has one, else
/// its ISA family; null when the registry has neither.
[[nodiscard]] std::unique_ptr<Decoder> make_decoder(const Target &target);

} // namespace rocjitsu::isa_diff
