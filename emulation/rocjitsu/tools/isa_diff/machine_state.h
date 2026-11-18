// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file machine_state.h
/// @brief Complete initial/final state of one isolated region execution.
#pragma once

#include "json.h"
#include "location.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

/// TRAPSTS.EXCP bits 0-6: the sticky invalid, input-denormal, divide-by-zero, overflow,
/// underflow, inexact and integer-divide-by-zero flags. VALU arithmetic sets them; they are
/// excluded from frame preservation (see differing_frame_fields).
inline constexpr uint32_t kTrapstsArithmeticExceptions = 0x7fu;

/// Wave control state outside the tracked locations. It starts at the post-dispatch values
/// and is captured after a run, so a caller can check that a region preserved it.
struct Frame {
  uint32_t status_except_scc = 0;
  uint32_t mode = 0;
  uint32_t trapsts = 0;
  std::array<uint32_t, 16> ttmp{};
  uint64_t scratch_base = 0;
  uint32_t s102 = 0; ///< GFX9: the two allocated SGPRs beyond s101 (not addressable as sN).
  uint32_t s103 = 0;
  /// vmcnt, lgkmcnt, expcnt, vscnt, dscnt, kmcnt, tensorcnt, asynccnt, samplecnt.
  std::array<uint8_t, 9> wait_counters{};
  bool in_trap_handler = false;
  uint8_t wave_state = 0;
  uint32_t sleep_cycles = 0;

  bool operator==(const Frame &) const = default;
};

/// Bytes a store wrote from a floating-point result: [addr, addr + size) hold one value
/// of the given format (FpTag kFpF16Lo, kFpBF16Lo, kFpF32 or kFpF64Lo).
struct FpUnit {
  uint64_t addr = 0;
  uint8_t size = 0;
  uint8_t format = 0;
  bool operator==(const FpUnit &) const = default;
  auto operator<=>(const FpUnit &) const = default;
};

/// LDS bytes in memory addresses (mem_written, mem_init, mem_dead): this bit plus the byte
/// offset in the wave's LDS window, so LDS and global memory never alias.
inline constexpr uint64_t kLdsTag = uint64_t{1} << 62;

/// A shared-memory dword access in a barrier epoch (regions with s_barrier): what other
/// waves of the workgroup could observe, and when.
struct SyncAccess {
  uint32_t epoch = 0; ///< Barriers executed before the access.
  bool store = false; ///< A store (or an atomic's write); else a load.
  uint64_t addr = 0;  ///< Dword address (LDS: kLdsTag | offset).
  bool operator==(const SyncAccess &) const = default;
  auto operator<=>(const SyncAccess &) const = default;
};

struct MachineState {
  std::array<uint32_t, kNumLocs> loc{}; ///< SCC is 0/1.
  Frame frame;
  /// VGPRs v0..v{vgpr_count-1}, register-major, kWaveLanes lanes each (VALU support).
  unsigned vgpr_count = 0;
  std::vector<uint32_t> vgpr;
  /// Memory model: the seed of the initial memory content (0: the region has no memory
  /// instruction) and, after a run, every byte the run stored, sorted by address.
  uint64_t mem_seed = 0;
  std::vector<std::pair<uint64_t, uint8_t>> mem_written;
  /// Initial content of every 32-bit word the run read (word index, value), sorted: the
  /// memory input, recorded so a run can be replayed exactly.
  std::vector<std::pair<uint64_t, uint32_t>> mem_init;
  /// Stored bytes that are dead where the run left (sorted): a callable function's own
  /// stack frame at its return, a kernel's private memory and LDS at s_endpgm
  /// (RegionRunner::set_memory_dead_at_exit). They are not compared.
  std::vector<uint64_t> mem_dead;
  /// Barriers (s_barrier) the run executed. One wave runs, so a barrier waits for nothing;
  /// it divides the run into epochs, and mem_sync records the shared-memory dwords (LDS,
  /// global memory outside the private segment) accessed in each, sorted.
  uint32_t barriers = 0;
  std::vector<SyncAccess> mem_sync;
  /// Floating-point provenance after a run: an FpTag per VGPR lane (empty when none is
  /// tagged), and the stored byte ranges holding floating-point results (sorted).
  std::vector<uint8_t> vfp;
  std::vector<FpUnit> mem_fp;
  /// FpTag per scalar location (empty when none is tagged).
  std::vector<uint8_t> sfp;
  /// Where the run left the region: 0 when it fell off the end, else the PC a branch
  /// went to outside the region.
  uint64_t exit_pc = 0;

  [[nodiscard]] uint32_t get(unsigned l) const { return loc[l]; }
  void set(unsigned l, uint32_t v) { loc[l] = v & loc_value_mask(l); }
  [[nodiscard]] uint32_t vget(unsigned r, unsigned lane) const {
    return vgpr[size_t{r} * kWaveLanes + lane];
  }
  void vset(unsigned r, unsigned lane, uint32_t v) { vgpr[size_t{r} * kWaveLanes + lane] = v; }
  [[nodiscard]] uint8_t sfp_tag(unsigned l) const { return l < sfp.size() ? sfp[l] : 0; }
  [[nodiscard]] uint8_t fp_tag(unsigned r, unsigned lane) const {
    const size_t i = size_t{r} * kWaveLanes + lane;
    return i < vfp.size() ? vfp[i] : 0;
  }
  void resize_vgprs(unsigned n) {
    vgpr_count = n;
    vgpr.assign(size_t{n} * kWaveLanes, 0);
  }

  [[nodiscard]] Json to_json() const;
  static std::optional<MachineState> from_json(const Json &j, std::string &error);
  bool operator==(const MachineState &) const = default;
};

/// Locations whose values differ.
[[nodiscard]] LocSet differing_locs(const MachineState &a, const MachineState &b);
/// VGPRs that differ on at least one lane (a count mismatch differs everywhere).
[[nodiscard]] VSet differing_vgprs(const MachineState &a, const MachineState &b);
/// Lanes of VGPR @p r that differ, as a 64-bit lane mask.
[[nodiscard]] uint64_t differing_lanes(const MachineState &a, const MachineState &b, unsigned r);
/// Frame field names whose values differ.
[[nodiscard]] std::vector<std::string> differing_frame_fields(const Frame &a, const Frame &b);

/// @brief SplitMix64 step (deterministic, portable).
[[nodiscard]] uint64_t splitmix64(uint64_t x);

/// Initial content of memory: the 32-bit word at byte address 4 * @p word for memory seed
/// @p seed. Every runner that reads the same (seed, word) reads the same value.
using MemoryContent = uint32_t (*)(uint64_t seed, uint64_t word);

/// @brief Default MemoryContent: uniformly distributed words.
[[nodiscard]] uint32_t uniform_memory_word(uint64_t seed, uint64_t word);

} // namespace rocjitsu::isa_diff
