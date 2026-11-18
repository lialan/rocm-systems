// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "region_runner.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/diagnostic.h"

#include "effects.h"
#include "json.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <exception>
#include <map>
#include <span>
#include <unordered_set>

namespace rocjitsu::isa_diff {
namespace {

constexpr uint32_t kVgprFile = 256;   ///< Per-wave VGPR file the CU is configured with.
constexpr uint32_t kMinVgprAlloc = 8; ///< Allocation for a region with no VGPR operand.

std::array<uint8_t, 9> pack_counters(const amdgpu::WaitCounters &c) {
  return {c.vmcnt, c.lgkmcnt,   c.expcnt,   c.vscnt,    c.dscnt,
          c.kmcnt, c.tensorcnt, c.asynccnt, c.samplecnt};
}

void unpack_counters(const std::array<uint8_t, 9> &p, amdgpu::WaitCounters &c) {
  c.vmcnt = p[0];
  c.lgkmcnt = p[1];
  c.expcnt = p[2];
  c.vscnt = p[3];
  c.dscnt = p[4];
  c.kmcnt = p[5];
  c.tensorcnt = p[6];
  c.asynccnt = p[7];
  c.samplecnt = p[8];
}

RunStatus map_error(amdgpu::InstructionExecutionError e) {
  using E = amdgpu::InstructionExecutionError;
  switch (e) {
  case E::None:
    return RunStatus::ExecutionError;
  case E::InvalidInstructionEncoding:
  case E::DecodeFailure:
    return RunStatus::DecodeFailure;
  case E::UnsupportedOperandValue:
    return RunStatus::UnsupportedOperand;
  case E::MissingExecutionBackend:
    return RunStatus::MissingExecutionBackend;
  case E::InvalidRegisterAccess:
    return RunStatus::InvalidRegisterAccess;
  case E::UnimplementedInstruction:
    return RunStatus::Unimplemented;
  case E::IllegalInstruction:
    return RunStatus::IllegalInstruction;
  case E::PluginFailure:
    return RunStatus::ExecutionError;
  }
  return RunStatus::ExecutionError;
}

/// The lowest set bit of @p bits, or @p none when no bit is set.
template <size_t N> unsigned lowest_bit(const std::bitset<N> &bits, unsigned none = N) {
  for (unsigned bit = 0; bit < N; ++bit)
    if (bits.test(bit))
      return bit;
  return none;
}

/// The set bits of @p bits below @p limit, ascending.
template <size_t N> std::vector<unsigned> set_bits(const std::bitset<N> &bits, unsigned limit = N) {
  std::vector<unsigned> out;
  for (unsigned bit = 0; bit < limit; ++bit)
    if (bits.test(bit))
      out.push_back(bit);
  return out;
}

std::vector<uint64_t> sorted(const std::unordered_set<uint64_t> &values) {
  std::vector<uint64_t> out(values.begin(), values.end());
  std::sort(out.begin(), out.end());
  return out;
}

/// The NaN-choice values of one floating-point format (RegionRunner::set_nan_choice): the
/// canonical quiet NaN, all ones, and the key bits a random NaN keeps (the sign and the
/// payload below the quiet bit).
struct NanFormat {
  uint64_t quiet;
  uint64_t ones;
  uint64_t keep;
};
constexpr NanFormat kNanF64{0x7ff8000000000000ull, ~uint64_t{0}, 0x8007ffffffffffffull};
constexpr NanFormat kNanF32{0x7fc00000u, 0xffffffffu, 0x803fffffu};
constexpr NanFormat kNanF16{0x7e00u, 0xffffu, 0x81ffu};
constexpr NanFormat kNanBF16{0x7fc0u, 0xffffu, 0x803fu};

/// NaN alternative @p choice of @p format: 1 canonical, 2 all ones, 3 random from @p key.
uint64_t chosen_nan(unsigned choice, const NanFormat &format, uint64_t key) {
  return choice == 1   ? format.quiet
         : choice == 2 ? format.ones
                       : (key & format.keep) | format.quiet;
}

/// @p word with its 16-bit half @p half (0 low, 1 high) replaced by @p value.
uint32_t with_half(uint32_t word, unsigned half, uint64_t value) {
  return (word & ~(0xffffu << (16 * half))) | (static_cast<uint32_t>(value) << (16 * half));
}

} // namespace

void MemoryImage::begin(uint64_t seed, const std::vector<std::pair<uint64_t, uint32_t>> &preset) {
  if (!started_ || seed != seed_) {
    words_.clear();
    sorted_.clear();
    added_.clear();
    seed_ = seed;
    started_ = true;
  }
  for (const auto &[w, v] : preset)
    if (words_.try_emplace(w, v).second)
      added(w, v);
}

bool MemoryImage::small_case() const {
  return small_scalars_ && (splitmix64(seed_ ^ 0x736d616c6cull) & 3u) != 0;
}

uint32_t MemoryImage::word(uint64_t w, int pointer_half, bool scalar) {
  auto it = words_.find(w);
  if (it != words_.end())
    return it->second;
  uint32_t v = content_(seed_, w);
  if (pointer_half < 0 && scalar && small_case()) {
    const uint64_t h = splitmix64(seed_ ^ (w * 0x9e3779b97f4a7c15ull) ^ 0x736d616c6cull);
    v = static_cast<uint32_t>(h >> 32) & ((1u << (h % 13)) - 1u); // below 2^k, k in 0..12
  }
  if (pointer_half >= 0) {
    // Canonical and far above the code: high half in [0x10, 0x7f], low half 64 KiB aligned.
    const uint64_t h = splitmix64(seed_ ^ (w * 0x9e3779b97f4a7c15ull) ^ 0x7074725f);
    v = pointer_half == 0 ? static_cast<uint32_t>(h) & 0xffff0000u
                          : 0x10u + static_cast<uint32_t>((h >> 32) % 0x70u);
  }
  words_.emplace(w, v);
  added(w, v);
  return v;
}

std::vector<std::pair<uint64_t, uint32_t>> MemoryImage::snapshot() const {
  if (!added_.empty()) {
    std::sort(added_.begin(), added_.end());
    const size_t n = sorted_.size();
    sorted_.insert(sorted_.end(), added_.begin(), added_.end());
    std::inplace_merge(sorted_.begin(), sorted_.begin() + static_cast<std::ptrdiff_t>(n),
                       sorted_.end());
    added_.clear();
  }
  return sorted_;
}

/// Seeds load bytes from the initial memory content before the pipeline reads them, and
/// records the bytes stores write. Observations fire after routing, before the access.
/// Global, buffer, scratch and flat accesses go to GpuMemory; LDS accesses (route LOCAL)
/// to the CU's LDS, whose bytes appear as kLdsTag | offset. Atomics read and write.
class RegionRunner::MemoryModel final : public ExecutionPlugin {
public:
  /// @p mem already holds the region's code in [code_lo, code_hi); a copy is kept to undo
  /// stores into it.
  MemoryModel(amdgpu::GpuMemory *mem, amdgpu::Lds *lds, uint64_t code_lo, uint64_t code_hi)
      : ExecutionPlugin("rj_isa_diff_memory"), mem_(mem), lds_(lds), code_lo_(code_lo),
        code_hi_(code_hi), guard_lo_(code_lo), guard_hi_(code_hi),
        code_(code_hi > code_lo ? code_hi - code_lo : 0) {
    if (!code_.empty())
      mem_->read_block(code_lo_, std::span<uint8_t>(code_));
  }
  /// Loads from [lo, hi) read code (RegionRunner::guard_code).
  void set_code_guard(uint64_t lo, uint64_t hi) {
    guard_lo_ = lo;
    guard_hi_ = hi;
  }
  [[nodiscard]] bool reads_code() const { return reads_code_; }
  bool observes_before_execute_instruction() const override { return false; }
  bool observes_after_execute_instruction() const override { return false; }
  bool observes_async_instruction_issued() const override { return false; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  bool observes_memory_routing() const override { return true; }

  void set_access(MemoryImage *image, uint32_t pointer_lo, uint32_t pointer_hi, bool frame,
                  bool scalar) {
    image_ = image;
    pointer_lo_ = pointer_lo;
    pointer_hi_ = pointer_hi;
    frame_ = frame;
    scalar_ = scalar;
  }

  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &a) override {
    if (a.lds_destination || !a.image_sample_footprints.empty()) {
      unsupported_ = true;
      return;
    }
    const bool local = a.route == amdgpu::MemoryRoute::LOCAL;
    const bool atomic = a.atomic_op != amdgpu::AtomicOp::NONE;
    const bool reads = a.is_load || atomic, writes = !a.is_load || atomic;
    const bool priv = a.decoded_space == amdgpu::DecodedMemorySpace::SCRATCH ||
                      a.scratch_lane_mask != 0 || a.scratch_element_stride_bytes != 0;
    const uint64_t esz = a.element_size_bytes;
    const uint64_t n = std::max<uint32_t>(a.elements_per_lane, 1);
    const uint64_t stride = a.scratch_element_stride_bytes;
    std::vector<uint64_t> &bytes = bytes_;
    for (const auto *set : {&a.addresses, &a.secondary_addresses}) {
      for (size_t lane = 0; lane < set->size() && lane < 64; ++lane) {
        if (!(a.request_lane_mask >> lane & 1u))
          continue;
        const uint64_t base = (*set)[lane];
        // Swizzled scratch and private buffers interleave lanes by dword: element bytes
        // advance by the swizzle stride every 4 bytes.
        const bool swizzled =
            stride != 0 && (a.scratch_lane_mask == 0 || (a.scratch_lane_mask >> lane & 1u));
        bytes.clear();
        for (uint64_t e = 0; e < n; ++e) {
          if (!a.element_lane_masks.empty() &&
              (e >= a.element_lane_masks.size() || !(a.element_lane_masks[e] >> lane & 1u)))
            continue; // out of the buffer's range: not accessed
          for (uint64_t j = 0; j < esz; ++j) {
            const uint64_t logical = e * esz + j;
            const uint64_t addr =
                swizzled ? base + (logical / 4) * stride + logical % 4 : base + logical;
            if (addr < base && !swizzled) {
              wraps_ = true; // past the end of the 64-bit address space
              break;
            }
            bytes.push_back(addr);
          }
        }
        // Shared memory is what other waves of the workgroup see: with barriers, LDS loads
        // and stores and global (non-private) stores keep their epoch. Global loads are not
        // tracked: they are most of a GEMM's traffic, and a wave hands data to the others
        // through LDS or by storing it. Scalar loads read kernel arguments and constants,
        // which register allocation may rematerialize after a barrier.
        if (track_epochs_ && (local || (!priv && writes)) && !scalar_)
          for (uint64_t b : bytes) {
            const uint64_t dword = (local ? kLdsTag : 0) | (b & ~uint64_t{3});
            for (bool store : {false, true})
              if (store ? writes : (reads && local))
                if (sync_.empty() || sync_.back() != SyncAccess{epoch_, store, dword})
                  sync_.push_back({epoch_, store, dword});
          }
        if (local) {
          for (uint64_t b : bytes)
            if (b >= kLdsWindow)
              lds_oob_ = true; // outside the wave's LDS window
          if (lds_oob_)
            return;
          if (reads)
            for (size_t k = 0; k < bytes.size(); ++k)
              if (k == 0 || bytes[k] >> 2 != bytes[k - 1] >> 2) // a word is seeded once
                seed_lds_word(bytes[k] >> 2);
          if (writes)
            for (uint64_t b : bytes)
              lds_stored_.insert(b);
          if (writes && !atomic && !swizzled && !bytes.empty())
            record_fp_units(lane, kLdsTag | bytes.front(), bytes.size());
          continue;
        }
        // Code shares this memory: a load from a guarded code range reads instructions, which
        // differ between two programs, so they would see different inputs.
        if (reads)
          for (size_t i = 0; i < bytes.size(); ++i) {
            const uint64_t b = bytes[i];
            if (b >= guard_lo_ && b < guard_hi_)
              reads_code_ = true;
            if (i > 0 && b >> 2 == bytes[i - 1] >> 2)
              continue; // a word is seeded once, by its first byte
            const uint64_t k = swizzled ? 32 : (b - base) >> 2; // dword of this lane's access
            const int half = k < 32 ? (pointer_lo_ >> k & 1u   ? 0
                                       : pointer_hi_ >> k & 1u ? 1
                                                               : -1)
                                    : -1;
            seed_word(b >> 2, half);
          }
        if (writes) {
          for (uint64_t b : bytes) {
            stored_.insert(b);
            if (b >= guard_lo_ && b < guard_hi_)
              writes_code_ = true;
            if (priv)
              private_.insert(b);
            if (priv && frame_)
              frame_bytes_.insert(b);
          }
          if (atomic || swizzled)
            erase_fp_units(bytes);
          else if (!bytes.empty())
            record_fp_units(lane, bytes.front(), bytes.size());
        }
      }
    }
  }

  /// Undo the previous case (touched bytes back to 0, the unwritten state) and set the
  /// seed of the next one.
  void reset(uint64_t seed) {
    // The region's own code stays: seeding skipped it, and the next case fetches it. A case
    // that stored into it (a MEMORY_FAULT, touches_code) gets its bytes restored, or every
    // later case would fetch the corrupted instructions.
    static constexpr uint8_t kZero[1] = {};
    auto clear = [&](uint64_t b) {
      if (b < code_lo_ || b >= code_hi_)
        mem_->write_block(b, std::span<const uint8_t>(kZero, 1));
      else
        mem_->write_block(b, std::span<const uint8_t>(&code_[b - code_lo_], 1));
    };
    for (uint64_t w : seeded_)
      for (unsigned k = 0; k < 4; ++k)
        clear((w << 2) + k);
    for (uint64_t b : stored_)
      clear(b);
    for (uint64_t w : lds_seeded_)
      for (unsigned k = 0; k < 4; ++k)
        lds_->write8(static_cast<uint32_t>((w << 2) + k), 0);
    for (uint64_t b : lds_stored_)
      lds_->write8(static_cast<uint32_t>(b), 0);
    seeded_.clear();
    stored_.clear();
    lds_seeded_.clear();
    lds_stored_.clear();
    private_.clear();
    frame_bytes_.clear();
    fp_units_.clear();
    store_fp_ = false;
    seed_ = seed;
    unsupported_ = false;
    wraps_ = false;
    lds_oob_ = false;
    reads_code_ = false;
    writes_code_ = false;
    epoch_ = 0;
    sync_.clear();
  }
  /// Regions with a barrier: record the epoch of every shared-memory dword access.
  void track_barrier_epochs() { track_epochs_ = true; }
  /// A barrier ran: later accesses belong to the next epoch.
  void next_epoch() { ++epoch_; }
  /// The shared-memory accesses of this run per epoch, sorted, without duplicates.
  [[nodiscard]] std::vector<SyncAccess> sync() const {
    std::vector<SyncAccess> out = sync_;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }
  [[nodiscard]] uint64_t seed() const { return seed_; }
  [[nodiscard]] bool unsupported() const { return unsupported_; }
  [[nodiscard]] bool wraps() const { return wraps_; }
  [[nodiscard]] bool lds_oob() const { return lds_oob_; }
  /// Floating-point provenance of the next store's data: per lane, the FpTag of each data
  /// dword; @p hi16 for a `_d16_hi` store. Off for every other access.
  void set_store_fp(bool on, bool hi16,
                    const std::array<std::array<uint8_t, 4>, kWaveLanes> &tags) {
    store_fp_ = on;
    store_hi16_ = hi16;
    if (on)
      store_tags_ = tags;
  }
  [[nodiscard]] std::vector<FpUnit> fp_units() const {
    std::vector<FpUnit> out;
    out.reserve(fp_units_.size());
    for (const auto &[addr, unit] : fp_units_)
      out.push_back(unit);
    return out;
  }
  [[nodiscard]] std::vector<std::pair<uint64_t, uint8_t>> written() const {
    std::vector<std::pair<uint64_t, uint8_t>> out;
    const std::vector<uint64_t> stored = sorted(stored_);
    std::vector<uint8_t> run;
    for (size_t first = 0; first < stored.size();) {
      size_t end = first + 1; // one read per run of consecutive bytes
      while (end < stored.size() && stored[end] == stored[end - 1] + 1)
        ++end;
      run.resize(end - first);
      mem_->read_block(stored[first], std::span<uint8_t>(run));
      for (size_t k = 0; k < run.size(); ++k)
        out.emplace_back(stored[first + k], run[k]);
      first = end;
    }
    const auto lds = static_cast<std::ptrdiff_t>(out.size());
    for (uint64_t b : sorted(lds_stored_))
      out.emplace_back(kLdsTag | b, lds_->read8(static_cast<uint32_t>(b)));
    // Sorted by address (MachineState::mem_written): a global address at or above the LDS
    // tag (wild address arithmetic) sorts after LDS bytes.
    std::inplace_merge(out.begin(), out.begin() + lds, out.end());
    return out;
  }
  /// Stored bytes dead at the exit (see RegionRunner::set_memory_dead_at_exit), sorted.
  [[nodiscard]] std::vector<uint64_t> dead(const std::string &kind) const {
    std::vector<uint64_t> out;
    if (kind == "frame")
      out.assign(frame_bytes_.begin(), frame_bytes_.end());
    if (kind == "private_memory" || kind == "private")
      out.assign(private_.begin(), private_.end());
    if (kind == "private")
      for (uint64_t b : lds_stored_)
        out.push_back(kLdsTag | b);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }
  /// The @p bytes of code at @p pc are the region's own (memory instructions are fetched
  /// from this memory; caches written back first).
  [[nodiscard]] bool code_intact(uint64_t pc, size_t bytes) const {
    if (pc < code_lo_ || pc + bytes > code_hi_)
      return true;
    std::vector<uint8_t> now(bytes);
    mem_->read_block(pc, std::span<uint8_t>(now));
    return std::equal(now.begin(), now.end(),
                      code_.begin() + static_cast<std::ptrdiff_t>(pc - code_lo_));
  }
  /// A store landed in the guarded code range, noted as it happens.
  [[nodiscard]] bool touches_code() const { return writes_code_; }

private:
  /// Drop the floating-point units that overlap [lo, hi).
  void erase_fp_units(uint64_t lo, uint64_t hi) {
    constexpr uint64_t kMaxUnit = 8; // an f64
    auto it = fp_units_.lower_bound(lo >= kMaxUnit ? lo - kMaxUnit + 1 : 0);
    while (it != fp_units_.end() && it->first < hi)
      it = lo < it->first + it->second.size ? fp_units_.erase(it) : std::next(it);
  }
  void erase_fp_units(const std::vector<uint64_t> &bytes) {
    for (uint64_t b : bytes)
      erase_fp_units(b, b + 1);
  }
  void add_fp_unit(uint64_t addr, uint8_t size, uint8_t format) {
    fp_units_[addr] = {addr, size, format};
  }
  /// A store replaces the floating-point units it overlaps, and adds those its data
  /// carries: f32 dwords, f64 dword pairs, and f16/bf16 halves.
  void record_fp_units(size_t lane, uint64_t base, uint64_t bytes) {
    erase_fp_units(base, base + bytes);
    if (!store_fp_ || lane >= kWaveLanes)
      return;
    const auto &t = store_tags_[lane];
    auto half = [&](uint64_t addr, uint8_t tag, bool hi) {
      const uint8_t f16 = hi ? kFpF16Hi : kFpF16Lo, bf16 = hi ? kFpBF16Hi : kFpBF16Lo;
      if (tag & f16)
        add_fp_unit(addr, 2, kFpF16Lo);
      else if (tag & bf16)
        add_fp_unit(addr, 2, kFpBF16Lo);
    };
    if (bytes == 2) {
      half(base, t[0], store_hi16_);
      return;
    }
    const uint64_t dwords = std::min<uint64_t>(bytes / 4, 4);
    for (uint64_t k = 0; k < dwords; ++k) {
      const uint64_t addr = base + 4 * k;
      if (t[k] & kFpF32) {
        add_fp_unit(addr, 4, kFpF32);
      } else if ((t[k] & kFpF64Lo) && k + 1 < dwords && (t[k + 1] & kFpF64Hi)) {
        add_fp_unit(addr, 8, kFpF64Lo);
        ++k;
      } else {
        half(addr, t[k], false);
        half(addr + 2, t[k], true);
      }
    }
  }

  void seed_word(uint64_t w, int pointer_half) {
    if (!seeded_.insert(w).second)
      return;
    const uint32_t value =
        image_ ? image_->word(w, pointer_half, scalar_) : uniform_memory_word(seed_, w);
    for (unsigned k = 0; k < 4; ++k) {
      const uint64_t addr = (w << 2) + k;
      if (stored_.count(addr) || (addr >= code_lo_ && addr < code_hi_))
        continue; // a byte this run stored, or the region's own code
      const uint8_t byte = static_cast<uint8_t>(value >> (8 * k));
      mem_->write_block(addr, std::span<const uint8_t>(&byte, 1));
    }
  }
  /// LDS word @p w (offset / 4): seeded from the same image, as word (kLdsTag | 4w) / 4.
  void seed_lds_word(uint64_t w) {
    if (!lds_seeded_.insert(w).second)
      return;
    const uint64_t tagged = (kLdsTag >> 2) | w;
    const uint32_t value = image_ ? image_->word(tagged, -1) : uniform_memory_word(seed_, tagged);
    for (unsigned k = 0; k < 4; ++k) {
      const uint64_t addr = (w << 2) + k;
      if (!lds_stored_.count(addr))
        lds_->write8(static_cast<uint32_t>(addr), static_cast<uint8_t>(value >> (8 * k)));
    }
  }

  amdgpu::GpuMemory *mem_;
  amdgpu::Lds *lds_;
  MemoryImage *image_ = nullptr;
  uint32_t pointer_lo_ = 0, pointer_hi_ = 0;
  bool frame_ = false;  ///< The current instruction addresses the stack frame (no VGPR address).
  bool scalar_ = false; ///< The current instruction is a scalar load (MemoryImage::word).
  uint64_t code_lo_, code_hi_;
  uint64_t guard_lo_, guard_hi_; ///< Loads from here read code (set_code_guard).
  std::vector<uint8_t> code_;    ///< The code bytes in [code_lo_, code_hi_), to undo stores.
  bool reads_code_ = false;      ///< This case loaded from the code range.
  bool writes_code_ = false;     ///< This case stored into the code range.
  uint64_t seed_ = 0;
  bool unsupported_ = false;
  bool wraps_ = false;
  bool lds_oob_ = false;
  std::unordered_set<uint64_t> seeded_;      ///< 4-byte words seeded this run.
  std::unordered_set<uint64_t> stored_;      ///< Bytes stored this run.
  std::unordered_set<uint64_t> lds_seeded_;  ///< LDS words (offset / 4) seeded this run.
  std::unordered_set<uint64_t> lds_stored_;  ///< LDS byte offsets stored this run.
  std::unordered_set<uint64_t> private_;     ///< Stored bytes of private (scratch) accesses.
  std::unordered_set<uint64_t> frame_bytes_; ///< Private bytes stored in stack-frame form.
  std::map<uint64_t, FpUnit> fp_units_;      ///< Stored floating-point values this run, by address.
  std::vector<uint64_t> bytes_;              ///< The bytes of one lane's access (reused).
  bool track_epochs_ = false;                ///< The region has a barrier.
  uint32_t epoch_ = 0;                       ///< Barriers executed so far this run.
  std::vector<SyncAccess> sync_;             ///< Shared-memory dword accesses per epoch.
  bool store_fp_ = false, store_hi16_ = false;
  std::array<std::array<uint8_t, 4>, kWaveLanes> store_tags_{};
};

const char *run_status_name(RunStatus s) {
  switch (s) {
  case RunStatus::Ok:
    return "OK";
  case RunStatus::DecodeFailure:
    return "DECODE_FAILURE";
  case RunStatus::MissingExecutionBackend:
    return "MISSING_EXECUTION_BACKEND";
  case RunStatus::Unimplemented:
    return "UNIMPLEMENTED";
  case RunStatus::UnsupportedOperand:
    return "UNSUPPORTED_OPERAND";
  case RunStatus::InvalidRegisterAccess:
    return "INVALID_REGISTER_ACCESS";
  case RunStatus::MemoryFault:
    return "MEMORY_FAULT";
  case RunStatus::IllegalInstruction:
    return "ILLEGAL_INSTRUCTION";
  case RunStatus::ExecutionError:
    return "EXECUTION_ERROR";
  case RunStatus::HostException:
    return "HOST_EXCEPTION";
  case RunStatus::UndefinedBehavior:
    return "UNDEFINED_BEHAVIOR";
  case RunStatus::PcWritten:
    return "PC_WRITTEN";
  case RunStatus::WaveStateChanged:
    return "WAVE_STATE_CHANGED";
  case RunStatus::StepLimit:
    return "STEP_LIMIT";
  case RunStatus::BudgetExceeded:
    return "BUDGET_EXCEEDED";
  }
  return "?";
}

RegionRunner::RegionRunner() = default;

RegionRunner::~RegionRunner() {
  // Release the wave's register blocks before the CU goes away.
  if (wave_ && !wave_->is_halted())
    wave_->halt(amdgpu::Wavefront::CpCompletionNotice::Suppress);
  steps_.clear();
}

util::FailureOr<std::unique_ptr<RegionRunner>>
RegionRunner::create(const std::vector<RegionInst> &insts, const std::string &label,
                     const Target &target, unsigned vgpr_count, util::DiagnosticEmitter emit) {
  std::unique_ptr<RegionRunner> r(new RegionRunner());
  r->target_ = target;
  if (vgpr_count > kNumVgprs) {
    return emit.emit() << "region needs more than 256 VGPRs and 256 AGPRs";
  }
  r->vgpr_count_ = vgpr_count;
  r->steps_.resize(insts.size());
  bool has_memory = false;
  for (size_t i = 0; i < insts.size(); ++i) {
    const RegionInst &ri = insts[i];
    Step &step = r->steps_[i];
    step.words = ri.words;
    step.pc = ri.pc;
    step.text = ri.text;
    // Classification needs a decode; the memory decision only needs the mnemonic.
    const std::string_view mnemonic = std::string_view(ri.text).substr(0, ri.text.find(' '));
    // Lane permutes issue like LDS instructions (step(): the LDS counter as the hardware
    // keeps it), though they access no memory.
    step.memory = is_region_memory(mnemonic) || is_region_lane_permute(mnemonic);
    step.frame_access = is_frame_access(mnemonic, ri.text);
    has_memory = has_memory || step.memory;
  }
  r->memory_ = std::make_unique<amdgpu::GpuMemory>(label + "_mem");
  r->l2_ = std::make_unique<amdgpu::L2Cache>(label + "_l2");
  r->l2_->set_backing_memory(r->memory_.get()); // global accesses reach this memory
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = target.arch;
  config.target = target.gpu;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = target.wave_sgprs();
  config.vgprs_per_wf = kVgprFile;
  config.lds_size_kb = 64;
  // Waits are not checked here: a region replays the same PCs for every case, which the
  // CU's own wait scoreboard would report as hazards. Memory operations outstanding at the
  // region entry are the caller's to analyse.
  config.memory_wait_diagnostics = amdgpu::MemoryWaitDiagnostics::Off;
  util::StringDiagnostic diag;
  r->cu_ = amdgpu::ComputeUnitCore::create(label, config, r->memory_.get(), r->l2_.get(),
                                           simdojo::ExecMode::FUNCTIONAL, diag.emitter());
  if (!r->cu_) {
    return emit.emit() << "cannot create a " << target.name << " compute unit: " << diag.message();
  }
  if (has_memory) {
    // step() fetches from memory: place the region's words at their PCs, and observe
    // memory accesses to seed loads and record stores.
    uint64_t lo = ~uint64_t{0}, hi = 0;
    for (const RegionInst &ri : insts) {
      r->memory_->write_block(
          ri.pc, std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(ri.words.data()),
                                          ri.words.size() * 4));
      lo = std::min(lo, ri.pc);
      hi = std::max<uint64_t>(hi, ri.pc + ri.words.size() * 4);
    }
    r->code_lo_ = lo;
    r->code_hi_ = hi;
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto *model = new MemoryModel(r->memory_.get(), &r->cu_->lds(), lo, hi);
    if (!group->add(OwnedPlugin(model, PluginDeleter{&delete_execution_plugin}))) {
      return emit.emit() << "cannot attach the memory-model plugin";
    }
    r->mem_ = model;
    r->cu_->set_plugin_group(std::move(group)); // the compute unit keeps the group
  }
  // AGPR locations (kAgprBase + n) are the second half of the wave's 512-register block;
  // without ACCUM_OFFSET a dispatch of N <= 256 VGPRs gives v0..v{N-1} and a0..a{N-1}.
  r->vgpr_alloc_ = vgpr_count == 0          ? kMinVgprAlloc
                   : vgpr_count > kAgprBase ? kNumArchVgprs
                                            : vgpr_count;
  r->wave_ =
      r->cu_->dispatch_wf(0, r->entry_pc(), target.wave_sgprs(), r->vgpr_alloc_, target.wave_size);
  if (!r->wave_) {
    return emit.emit() << "cannot dispatch a " << target.label() << " wave";
  }
  r->configure_wave();
  // Private memory: scratch instructions need a scratch base (FLAT_SCRATCH); it is fixed
  // in the frame like the rest of the dispatch state.
  r->wave_->set_scratch_base(kScratchBase);
  r->decoder_ = make_decoder(target);
  if (!r->decoder_) {
    return emit.emit() << "no " << target.name << " decoder";
  }
  for (size_t i = 0; i < r->steps_.size(); ++i)
    r->pc_index_.emplace(r->steps_[i].pc, i);
  std::vector<Effects> effects(r->steps_.size());
  std::vector<std::vector<SourceOperand>> sources(r->steps_.size());
  for (size_t i = 0; i < r->steps_.size(); ++i) {
    Step &step = r->steps_[i];
    util::StringDiagnostic ddiag;
    // The step's own words: the decoded instruction may keep pointing into them.
    auto decoded = r->decoder_->decode_window(step.words, step.pc, ddiag.emitter());
    if (decoded.failed() || static_cast<size_t>(decoded.value()->size()) != step.words.size() * 4) {
      if (!r->decode_failure_) {
        r->decode_failure_ = i;
        r->decode_error_ = decoded.failed()
                               ? "decode failed: " + ddiag.message()
                               : "decoded size does not match the recorded instruction words";
      }
      continue;
    }
    step.inst = std::move(decoded).value();
    effects[i] = classify(*step.inst, target);
    sources[i] = source_operands(*step.inst, target);
    r->plan_step(step, effects[i], sources[i]);
  }
  if (r->mem_)
    r->plan_pointers(effects, sources);
  // Fix the frame to the post-dispatch values.
  r->baseline_ = r->capture().frame;
  return r;
}

void RegionRunner::plan_step(Step &step, const Effects &fx,
                             const std::vector<SourceOperand> &sources) {
  const Instruction &inst = *step.inst;
  const std::string_view mn = inst.mnemonic();
  step.barrier = fx.barrier;
  step.endpgm = mn == "s_endpgm";
  step.scalar_load = mn.starts_with("s_load");
  if (fx.barrier && mem_)
    mem_->track_barrier_epochs();
  if (fx.branch) {
    step.branch = true;
    // A negative offset (from the next instruction) goes back, at or before the branch:
    // the region may loop.
    if (const auto offset = inst.branch_offset_bytes(); offset && *offset < 0)
      back_edge_ = true;
  }
  for (const SourceOperand &o : sources) {
    if (o.fieldless)
      continue;
    OperandRef ref;
    if (o.constant) {
      ref.kind = OperandRef::Constant;
      ref.value = *o.constant;
    } else if (o.vlocs.any()) {
      ref.kind = OperandRef::Vgpr;
      ref.count = static_cast<unsigned>(o.vlocs.count());
      ref.reg = lowest_bit(o.vlocs);
    } else if (o.locs.any()) {
      ref.kind = OperandRef::Scalar;
      ref.count = static_cast<unsigned>(o.locs.count());
      ref.reg = lowest_bit(o.locs);
    }
    step.srcs.push_back(ref);
  }
  FpPlan &plan = step.fp;
  plan.kind = fp_result_kind(mn, step.text);
  plan.dsts = set_bits(fx.vwrites);
  if (plan.kind == FpResult::CndMask) {
    // The lane-select mask: VCC (fieldless, VOP2) or the last wave-mask source (VOP3): an
    // SGPR pair on wave64, one SGPR on wave32 (where a scalar data source is one SGPR too,
    // so the last one wins).
    const size_t mask_regs = target_.wave32() ? 1 : 2;
    for (const SourceOperand &o : sources)
      if (o.locs.count() == mask_regs && o.vlocs.none() && !o.constant)
        plan.mask_loc = static_cast<int>(lowest_bit(o.locs));
    if (plan.mask_loc < 0)
      plan.kind = FpResult::None;
  }
  if (plan.kind == FpResult::PkMov) {
    const std::string &t = step.text;
    const size_t at = t.find("op_sel:[");
    if (at != std::string::npos && at + 11 <= t.size()) {
      plan.op_sel_lo = t[at + 8] == '1';
      plan.op_sel_hi = t[at + 10] == '1';
    }
  }
  plan.store = step.memory && fx.mem_write;
  plan.skind = fp_scalar_result_kind(mn);
  plan.sdsts = set_bits(fx.writes);
  std::erase(plan.sdsts, kLocScc);
  plan.store_hi16 = mn.find("_d16_hi") != std::string_view::npos;
  if (plan.store) { // a store's data carries its floating-point provenance into memory
    int vgprs = 0;
    for (size_t s = 0; s < step.srcs.size(); ++s)
      if (step.srcs[s].kind == OperandRef::Vgpr && ++vgprs == 2) {
        step.store_data = static_cast<int>(s);
        break;
      }
  }
}

void RegionRunner::plan_pointers(const std::vector<Effects> &effects,
                                 const std::vector<std::vector<SourceOperand>> &sources) {
  constexpr std::pair<int, int> kNone{-1, -1};
  std::vector<std::pair<int, int>> origin(kNumSgprs, kNone); // (load index, dword)
  for (size_t i = 0; i < steps_.size(); ++i) {
    const Step &step = steps_[i];
    if (!step.inst)
      break;
    const Effects &fx = effects[i];
    const std::vector<unsigned> writes = set_bits(fx.writes, kNumSgprs);
    if (step.memory) {
      for (const SourceOperand &so : sources[i]) {
        if (so.locs.count() != 2)
          continue;
        const unsigned lo = lowest_bit(so.locs);
        for (unsigned p : {lo, lo + 1})
          if (p < kNumSgprs && origin[p].first >= 0) {
            const auto [li, k] = origin[p];
            Step &load = steps_[static_cast<size_t>(li)];
            (p == lo ? load.pointer_lo : load.pointer_hi) |= 1u << k;
          }
        break; // the first SGPR pair is the base (sbase / saddr)
      }
      const bool scalar_load = fx.mem_read && step.scalar_load;
      for (size_t idx = 0; idx < writes.size(); ++idx)
        origin[writes[idx]] = scalar_load && idx < 32
                                  ? std::pair<int, int>{static_cast<int>(i), static_cast<int>(idx)}
                                  : kNone;
      continue;
    }
    const SourceOperand *first = nullptr;
    for (const SourceOperand &so : sources[i])
      if (so.locs.any()) {
        first = &so;
        break;
      }
    const unsigned first_lo = first ? lowest_bit(first->locs) : kNumSgprs;
    for (size_t idx = 0; idx < writes.size(); ++idx) {
      const unsigned from = first_lo + static_cast<unsigned>(idx);
      origin[writes[idx]] =
          first && from < kNumSgprs && first->locs.test(from) ? origin[from] : kNone;
    }
  }
}

void RegionRunner::simulate_model_only_for_test(size_t index) {
  if (index < steps_.size())
    steps_[index].model_only = true;
}

void RegionRunner::configure_wave() {
  // One workgroup's LDS window at offset 0 (the CU's LDS): DS addresses outside it are a
  // fault of the case (MemoryModel::lds_oob). Scratch lanes are unbounded.
  wave_->set_lds_base(0);
  wave_->set_lds_size(static_cast<uint32_t>(kLdsWindow));
  wave_->set_scratch_lane_size(0x7fffffffu);
}

void RegionRunner::load(const MachineState &s) {
  if (wave_->state() != amdgpu::WfState::RUNNING) { // a memory fault halted it: re-dispatch
    wave_ = cu_->dispatch_wf(0, entry_pc(), target_.wave_sgprs(), vgpr_alloc_, target_.wave_size);
    configure_wave();
  }
  if (mem_)
    image_->begin(s.mem_seed, s.mem_init);
  amdgpu::Wavefront &wf = *wave_;
  const uint32_t base = wf.sgpr_alloc().base;
  for (unsigned i = 0; i < target_.sgprs(); ++i)
    cu_->write_sgpr(base + i, s.loc[i]);
  if (target_.major == 9) { // the two allocated SGPRs above s101
    cu_->write_sgpr(base + 102, s.frame.s102);
    cu_->write_sgpr(base + 103, s.frame.s103);
  }
  wf.set_vcc_raw(uint64_t{s.loc[kLocVccLo]} | (uint64_t{s.loc[kLocVccHi]} << 32));
  wf.set_exec_raw(uint64_t{s.loc[kLocExecLo]} | (uint64_t{s.loc[kLocExecHi]} << 32));
  wf.set_m0(s.loc[kLocM0]);
  wf.set_status_raw((s.frame.status_except_scc & ~1u) | (s.loc[kLocScc] & 1u));
  wf.set_mode_raw(s.frame.mode);
  wf.set_trapsts(s.frame.trapsts);
  for (uint32_t i = 0; i < 16; ++i)
    wf.set_ttmp(i, s.frame.ttmp[i]);
  wf.set_scratch_base(s.frame.scratch_base);
  unpack_counters(s.frame.wait_counters, wf.wait_counters());
  wf.set_in_trap_handler(s.frame.in_trap_handler);
  if (mem_) {
    // Write back the previous case's stores and drop every cached line (flush_all: the L1
    // scalar/vector caches and the write-back L2), then undo its memory, so this case reads
    // its own input. The L2 keeps its lines when the write-back is incomplete: drop them.
    cu_->flush_all();
    mem_->reset(s.mem_seed);
    l2_->invalidate_all();
  }
  const uint32_t vbase = wf.vgpr_alloc().base;
  for (unsigned r = 0; r < vgpr_count_ && r < s.vgpr_count; ++r)
    for (unsigned lane = 0; lane < target_.wave_size; ++lane)
      cu_->write_vgpr(vbase + r, lane, s.vget(r, lane));
  wf.begin_instruction_execution();
  wf.pc = entry_pc();
  vfp_.assign(size_t{vgpr_count_} * kWaveLanes, 0); // inputs carry no provenance
  sfp_.fill(0);
  nan_produced_ = false;
  barriers_ = 0;
}

MachineState RegionRunner::capture() const {
  const amdgpu::Wavefront &wf = *wave_;
  MachineState s;
  const uint32_t base = wf.sgpr_alloc().base;
  for (unsigned i = 0; i < target_.sgprs(); ++i)
    s.loc[i] = cu_->read_sgpr_storage(base + i);
  if (target_.major == 9) {
    s.frame.s102 = cu_->read_sgpr_storage(base + 102);
    s.frame.s103 = cu_->read_sgpr_storage(base + 103);
  }
  const uint64_t vcc = wf.vcc();
  const uint64_t exec = wf.exec_raw();
  s.loc[kLocVccLo] = static_cast<uint32_t>(vcc);
  s.loc[kLocVccHi] = static_cast<uint32_t>(vcc >> 32);
  s.loc[kLocExecLo] = static_cast<uint32_t>(exec);
  s.loc[kLocExecHi] = static_cast<uint32_t>(exec >> 32);
  s.loc[kLocM0] = wf.m0();
  s.loc[kLocScc] = wf.read_scc() ? 1u : 0u;
  s.frame.status_except_scc = wf.status_raw() & ~1u;
  s.frame.mode = wf.mode_raw();
  s.frame.trapsts = wf.trapsts();
  for (uint32_t i = 0; i < 16; ++i)
    s.frame.ttmp[i] = wf.ttmp(i);
  s.frame.scratch_base = wf.scratch_base();
  s.frame.wait_counters = pack_counters(wf.wait_counters());
  s.frame.in_trap_handler = wf.in_trap_handler();
  s.frame.wave_state = static_cast<uint8_t>(wf.state());
  s.frame.sleep_cycles = wf.sleep_cycles();
  s.resize_vgprs(vgpr_count_);
  const uint32_t vbase = wf.vgpr_alloc().base;
  for (unsigned r = 0; r < vgpr_count_; ++r)
    for (unsigned lane = 0; lane < target_.wave_size; ++lane)
      s.vset(r, lane, cu_->read_vgpr_storage(vbase + r, lane));
  if (mem_) {
    cu_->flush_all(); // stores may still sit in the write-back L2
    s.mem_seed = mem_->seed();
    s.mem_written = mem_->written();
    s.mem_dead = mem_->dead(dead_kind_);
    s.mem_init = image_->snapshot();
    s.mem_sync = mem_->sync();
  }
  s.barriers = barriers_;
  s.exit_pc = exit_pc_;
  if (std::ranges::any_of(vfp_, [](uint8_t t) { return t != 0; }))
    s.vfp = vfp_;
  if (std::ranges::any_of(sfp_, [](uint8_t t) { return t != 0; }))
    s.sfp.assign(sfp_.begin(), sfp_.end());
  if (mem_)
    s.mem_fp = mem_->fp_units();
  return s;
}

uint32_t RegionRunner::read_loc(unsigned l) const {
  const amdgpu::Wavefront &wf = *wave_;
  if (l < kNumSgprs) // s102-s105 do not exist on GFX9
    return l < target_.sgprs() ? cu_->read_sgpr_storage(wf.sgpr_alloc().base + l) : 0u;
  switch (l) {
  case kLocVccLo:
    return static_cast<uint32_t>(wf.vcc());
  case kLocVccHi:
    return static_cast<uint32_t>(wf.vcc() >> 32);
  case kLocExecLo:
    return static_cast<uint32_t>(wf.exec_raw());
  case kLocExecHi:
    return static_cast<uint32_t>(wf.exec_raw() >> 32);
  case kLocM0:
    return wf.m0();
  default:
    return wf.read_scc() ? 1u : 0u;
  }
}

uint32_t RegionRunner::read_vgpr(unsigned r, unsigned lane) const {
  return cu_->read_vgpr_storage(wave_->vgpr_alloc().base + r, lane);
}

std::optional<uint32_t> RegionRunner::operand_value(const OperandRef &op, unsigned lane) const {
  switch (op.kind) {
  case OperandRef::Constant:
    return op.value;
  case OperandRef::Vgpr:
    if (op.reg + std::max(op.count, 1u) > vgpr_count_)
      return std::nullopt;
    return read_vgpr(op.reg, lane);
  case OperandRef::Scalar:
    return read_loc(op.reg);
  default:
    return std::nullopt;
  }
}

void RegionRunner::plan_tags(size_t i, std::vector<std::array<uint8_t, kWaveLanes>> &out) const {
  const FpPlan &p = steps_[i].fp;
  out.resize(p.dsts.size());
  const auto &src = steps_[i].srcs;
  auto src_tag = [&](size_t s, unsigned k, unsigned lane) -> uint8_t {
    if (s >= src.size())
      return 0;
    if (src[s].kind == OperandRef::Scalar) // an SGPR source: the same value in every lane
      return src[s].reg + k < kNumLocs ? sfp_[src[s].reg + k] : 0;
    if (src[s].kind != OperandRef::Vgpr || src[s].reg + k >= vgpr_count_)
      return 0;
    return vfp_[size_t{src[s].reg + k} * kWaveLanes + lane];
  };
  uint64_t mask = 0;
  if (p.kind == FpResult::CndMask)
    mask = uint64_t{read_loc(static_cast<unsigned>(p.mask_loc))} |
           (target_.wave32() ? 0 : uint64_t{read_loc(static_cast<unsigned>(p.mask_loc) + 1)} << 32);
  for (size_t k = 0; k < p.dsts.size(); ++k) {
    const unsigned d = p.dsts[k];
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      const uint8_t cur = d < vgpr_count_ ? vfp_[size_t{d} * kWaveLanes + lane] : 0;
      uint8_t t = 0;
      switch (p.kind) {
      case FpResult::None:
        break;
      case FpResult::F32:
        t = kFpF32;
        break;
      case FpResult::F64:
        t = k % 2 == 0 ? kFpF64Lo : kFpF64Hi;
        break;
      case FpResult::F16Lo:
        t = (cur & (kFpF16Hi | kFpBF16Hi)) | kFpF16Lo;
        break;
      case FpResult::F16Hi:
        t = (cur & (kFpF16Lo | kFpBF16Lo)) | kFpF16Hi;
        break;
      case FpResult::BF16Lo:
        t = (cur & (kFpF16Hi | kFpBF16Hi)) | kFpBF16Lo;
        break;
      case FpResult::BF16Hi:
        t = (cur & (kFpF16Lo | kFpBF16Lo)) | kFpBF16Hi;
        break;
      case FpResult::F16x2:
        t = kFpF16Lo | kFpF16Hi;
        break;
      case FpResult::BF16x2:
        t = kFpBF16Lo | kFpBF16Hi;
        break;
      case FpResult::Move:
        t = src_tag(0, static_cast<unsigned>(k), lane);
        break;
      case FpResult::CndMask:
        t = src_tag((mask >> lane) & 1u ? 1 : 0, 0, lane);
        break;
      case FpResult::PkMov:
        t = k == 0 ? src_tag(0, p.op_sel_lo, lane) : src_tag(1, p.op_sel_hi, lane);
        break;
      case FpResult::Select:
      case FpResult::CMov:
      case FpResult::ReadFirstLane:
      case FpResult::ReadLane:
        break; // scalar-result kinds (plan_stags)
      }
      out[k][lane] = t;
    }
  }
}

void RegionRunner::plan_stags(size_t i, std::vector<uint8_t> &out) const {
  const FpPlan &p = steps_[i].fp;
  out.assign(p.sdsts.size(), 0);
  if (p.sdsts.empty() || p.skind == FpResult::None)
    return;
  const auto &src = steps_[i].srcs;
  auto stag = [&](size_t s, unsigned k) -> uint8_t {
    if (s >= src.size() || src[s].kind != OperandRef::Scalar || src[s].reg + k >= kNumLocs)
      return 0;
    return sfp_[src[s].reg + k];
  };
  auto lane_tag = [&](unsigned lane) -> uint8_t {
    if (src.empty() || src[0].kind != OperandRef::Vgpr || src[0].reg >= vgpr_count_)
      return 0;
    return vfp_[size_t{src[0].reg} * kWaveLanes + (lane % kWaveLanes)];
  };
  const bool scc = read_loc(kLocScc) & 1u;
  for (size_t k = 0; k < p.sdsts.size(); ++k) {
    const auto kk = static_cast<unsigned>(k);
    switch (p.skind) {
    case FpResult::F32:
      out[k] = kFpF32;
      break;
    case FpResult::F16Lo:
      out[k] = kFpF16Lo;
      break;
    case FpResult::F16x2:
      out[k] = kFpF16Lo | kFpF16Hi;
      break;
    case FpResult::Move:
      out[k] = stag(0, kk);
      break;
    case FpResult::Select:
      out[k] = scc ? stag(0, kk) : stag(1, kk);
      break;
    case FpResult::CMov:
      out[k] = scc ? stag(0, kk) : sfp_[p.sdsts[k]];
      break;
    case FpResult::ReadFirstLane: {
      const uint64_t exec = wave_->exec();
      out[k] = lane_tag(exec ? static_cast<unsigned>(std::countr_zero(exec)) : 0u);
      break;
    }
    case FpResult::ReadLane: {
      const auto lane = src.size() > 1 ? operand_value(src[1], 0) : std::nullopt;
      out[k] = lane ? lane_tag(*lane % target_.wave_size) : 0;
      break;
    }
    default:
      break;
    }
  }
}

void RegionRunner::apply_stags(size_t i, const std::vector<uint8_t> &tags) {
  const FpPlan &p = steps_[i].fp;
  for (size_t k = 0; k < p.sdsts.size() && k < tags.size(); ++k)
    sfp_[p.sdsts[k]] = tags[k];
}

void RegionRunner::write_loc(unsigned l, uint32_t v) {
  amdgpu::Wavefront &wf = *wave_;
  if (l < kNumSgprs) {
    if (l < target_.sgprs())
      cu_->write_sgpr(wf.sgpr_alloc().base + l, v);
    return;
  }
  switch (l) {
  case kLocVccLo:
    wf.set_vcc_raw((wf.vcc() & ~uint64_t{0xffffffffu}) | v);
    break;
  case kLocVccHi:
    wf.set_vcc_raw((wf.vcc() & uint64_t{0xffffffffu}) | (uint64_t{v} << 32));
    break;
  case kLocExecLo:
    wf.set_exec_raw((wf.exec_raw() & ~uint64_t{0xffffffffu}) | v);
    break;
  case kLocExecHi:
    wf.set_exec_raw((wf.exec_raw() & uint64_t{0xffffffffu}) | (uint64_t{v} << 32));
    break;
  case kLocM0:
    wf.set_m0(v);
    break;
  default:
    wf.set_status_raw((wf.status_raw() & ~1u) | (v & 1u)); // SCC
    break;
  }
}

void RegionRunner::apply_tags(size_t i, uint64_t exec,
                              const std::vector<std::array<uint8_t, kWaveLanes>> &tags) {
  const FpPlan &p = steps_[i].fp;
  for (size_t k = 0; k < p.dsts.size() && k < tags.size(); ++k) {
    if (p.dsts[k] >= vgpr_count_)
      continue;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      if ((exec >> lane) & 1u)
        vfp_[size_t{p.dsts[k]} * kWaveLanes + lane] = tags[k][lane];
  }
}

void RegionRunner::choose_nans(size_t i, uint64_t exec) {
  const FpPlan &p = steps_[i].fp;
  const bool scalar_fp =
      p.skind == FpResult::F32 || p.skind == FpResult::F16Lo || p.skind == FpResult::F16x2;
  bool vector_fp = true;
  switch (p.kind) {
  case FpResult::None:
  case FpResult::Move:
  case FpResult::CndMask:
  case FpResult::PkMov:
  case FpResult::Select:
  case FpResult::CMov:
  case FpResult::ReadFirstLane:
  case FpResult::ReadLane:
    vector_fp = false; // not a floating-point operation: NaN bits pass through unchanged
    break;
  default:
    break;
  }
  if (!scalar_fp && !vector_fp)
    return;
  // Random NaNs are keyed by the NaN the hardware produced (with the destination component
  // and the lane), not by the instruction's index or position: the same operation on the
  // same inputs draws the same NaN in two programs, whatever order a scheduler gave the
  // floating-point operations and whatever hints, moves or register names lie between.
  auto key_of = [&](uint64_t hw, uint64_t component, uint64_t lane) {
    return splitmix64(nan_seed_ ^ splitmix64(hw ^ 0x6e616e5f6b6579ull) ^ (component << 8) ^ lane);
  };
  if (scalar_fp) {
    for (size_t c = 0; c < p.sdsts.size(); ++c) {
      const unsigned l = p.sdsts[c];
      const uint32_t w = read_loc(l);
      uint32_t out = w;
      bool any = false;
      if (p.skind == FpResult::F32) {
        if (is_nan_f32(w)) {
          any = true;
          out = static_cast<uint32_t>(chosen_nan(nan_choice_, kNanF32, key_of(w, c, 0xffu)));
        }
      } else {
        for (unsigned h = 0; h < (p.skind == FpResult::F16x2 ? 2u : 1u); ++h) {
          const uint32_t half = (w >> (16 * h)) & 0xffffu;
          if (!is_nan_f16(half))
            continue;
          any = true;
          out = with_half(
              out, h,
              chosen_nan(nan_choice_, kNanF16, key_of(half, (uint64_t{c} << 1) | h, 0xffu)));
        }
      }
      if (!any)
        continue;
      nan_produced_ = true;
      if (nan_choice_)
        write_loc(l, out);
    }
  }
  if (!vector_fp)
    return;
  const uint32_t vbase = wave_->vgpr_alloc().base;
  for (size_t k = 0; k < p.dsts.size(); ++k) {
    const unsigned d = p.dsts[k];
    if (d >= vgpr_count_)
      continue;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      if (!((exec >> lane) & 1u))
        continue;
      const uint8_t t = vfp_[size_t{d} * kWaveLanes + lane];
      const uint32_t w = read_vgpr(d, lane);
      if (t & kFpF32) {
        if (!is_nan_f32(w))
          continue;
        nan_produced_ = true;
        if (nan_choice_)
          cu_->write_vgpr(
              vbase + d, lane,
              static_cast<uint32_t>(chosen_nan(nan_choice_, kNanF32, key_of(w, k, lane))));
      } else if ((t & kFpF64Lo) && d + 1 < vgpr_count_) {
        const uint64_t v = uint64_t{w} | (uint64_t{read_vgpr(d + 1, lane)} << 32);
        if (!is_nan_f64(v))
          continue;
        nan_produced_ = true;
        if (nan_choice_) {
          const uint64_t n = chosen_nan(nan_choice_, kNanF64, key_of(v, k, lane));
          cu_->write_vgpr(vbase + d, lane, static_cast<uint32_t>(n));
          cu_->write_vgpr(vbase + d + 1, lane, static_cast<uint32_t>(n >> 32));
        }
      } else {
        uint32_t out = w;
        bool any = false;
        for (unsigned h = 0; h < 2; ++h) {
          const uint32_t half = (w >> (16 * h)) & 0xffffu;
          const bool f16 = t & (h ? kFpF16Hi : kFpF16Lo), bf16 = t & (h ? kFpBF16Hi : kFpBF16Lo);
          if (!((f16 && is_nan_f16(half)) || (bf16 && is_nan_bf16(half))))
            continue;
          any = true;
          out = with_half(out, h,
                          chosen_nan(nan_choice_, f16 ? kNanF16 : kNanBF16,
                                     key_of(half, (uint64_t{k} << 1) | h, lane)));
        }
        if (!any)
          continue;
        nan_produced_ = true;
        if (nan_choice_)
          cu_->write_vgpr(vbase + d, lane, out);
      }
    }
  }
}

void RegionRunner::guard_code(uint64_t lo, uint64_t hi) {
  if (mem_)
    mem_->set_code_guard(lo, hi);
}

void share_code_guard(RegionRunner &a, RegionRunner &b) {
  const auto [alo, ahi] = a.code_range();
  const auto [blo, bhi] = b.code_range();
  const bool ae = alo >= ahi, be = blo >= bhi;
  if (ae && be)
    return;
  const uint64_t lo = ae ? blo : be ? alo : std::min(alo, blo);
  const uint64_t hi = ae ? bhi : be ? ahi : std::max(ahi, bhi);
  a.guard_code(lo, hi);
  b.guard_code(lo, hi);
}

size_t RegionRunner::step_limit(size_t budget) const {
  constexpr size_t kLoopStepFactor = 64, kMaxLoopSteps = 1u << 16;
  if (!back_edge_)
    return budget;
  return std::min(std::max<size_t>(budget, 1) * kLoopStepFactor,
                  std::max<size_t>(budget, kMaxLoopSteps)) *
         loop_scale_;
}

const MachineState &RegionRunner::record_step(size_t i, std::vector<TraceStep> &trace,
                                              const MachineState &prev) const {
  TraceStep entry;
  entry.index = i;
  entry.pc = steps_[i].pc;
  entry.text = steps_[i].text;
  entry.after = capture();
  entry.changed = differing_locs(prev, entry.after);
  entry.vchanged = differing_vgprs(prev, entry.after);
  entry.frame_changed = differing_frame_fields(prev.frame, entry.after.frame);
  trace.push_back(std::move(entry)); // may move prev's storage: compared first
  return trace.back().after;
}

RunOutcome RegionRunner::run(size_t budget, std::vector<TraceStep> *trace) {
  RunOutcome out;
  auto fail = [&out](RunStatus status, size_t index, std::string detail = {}) {
    out.status = status;
    out.failed_index = index;
    out.detail = std::move(detail);
    return std::move(out);
  };
  amdgpu::Wavefront &wf = *wave_;
  MachineState start;
  const MachineState *prev = nullptr; // the state before the next traced step
  if (trace) {
    start = capture();
    prev = &start;
  }
  auto record = [&](size_t i) {
    if (trace)
      prev = &record_step(i, *trace, *prev);
  };
  // Without a backward branch every instruction runs at most once; a loop gets a step
  // limit (STEP_LIMIT past it; see step_limit()).
  const size_t limit = step_limit(budget);
  exit_pc_ = 0;
  bool left = false;
  size_t next = 0;
  for (size_t i = 0; i < steps_.size() && !left; i = next) {
    next = i + 1;
    if (out.executed >= limit)
      return fail(back_edge_ ? RunStatus::StepLimit : RunStatus::BudgetExceeded, i);
    if (decode_failure_ && i >= *decode_failure_)
      return fail(RunStatus::DecodeFailure, *decode_failure_, decode_error_);
    const Step &step = steps_[i];
    if (step.barrier || step.endpgm) {
      // One wave runs: a barrier has no other wave to wait for, and closes a barrier epoch of
      // the shared-memory accesses (MemoryModel). An early s_endpgm ends the wave, which is
      // the region's end exit.
      ++out.executed;
      if (step.barrier) {
        ++barriers_;
        if (mem_)
          mem_->next_epoch();
      } else {
        exit_pc_ = 0;
        left = true;
      }
      record(i);
      continue;
    }
    Instruction *inst = step.inst.get();
    const uint64_t exec_before = wf.exec();
    plan_tags(i, scratch_tags_);
    plan_stags(i, scratch_stags_);
    if (step.memory && inst && inst->execute && !step.model_only) {
      // The full issue path: fetch at the PC, execute, route, complete (functional).
      wf.pc = step.pc;
      if (mem_) {
        mem_->set_access(image_.get(), step.pointer_lo, step.pointer_hi, step.frame_access,
                         step.scalar_load);
        std::array<std::array<uint8_t, 4>, kWaveLanes> lane_tags{};
        if (step.store_data >= 0) {
          const OperandRef &data = step.srcs[static_cast<size_t>(step.store_data)];
          for (unsigned lane = 0; lane < kWaveLanes; ++lane)
            for (unsigned k = 0; k < 4 && k < std::max(data.count, 1u); ++k)
              if (data.reg + k < vgpr_count_)
                lane_tags[lane][k] = vfp_[size_t{data.reg + k} * kWaveLanes + lane];
        }
        mem_->set_store_fp(step.fp.store, step.fp.store_hi16, lane_tags);
        // Seeding writes backing memory when the access is observed (before the pipeline
        // reads it). Write back and drop every cached line first, so the access fills
        // its lines from the seeded memory instead of an earlier, partially seeded fill.
        cu_->flush_all();
        l2_->invalidate_all();
        // step() fetches the instruction from this memory: it must still be the region's.
        // (A case's store into the code is caught after it lands, and reset() restores the
        // bytes; were they still corrupted, a decode failure would halt the wave and look
        // like a memory fault.)
        if (!mem_->code_intact(step.pc, inst->size()))
          return fail(RunStatus::ExecutionError, i,
                      "the region's code at '" + step.text + "' was overwritten");
      }
      const uint64_t ub_before = cu_->isa_diagnostic_count();
      const bool trap_before = wf.in_trap_handler();
      // Translation boundary: RocJITsu reports unrecoverable execution failures as exceptions;
      // here they are a classified outcome of the case, never a crash of the tester.
      try {
        (void)cu_->step();
      } catch (const std::exception &e) {
        return fail(RunStatus::HostException, i, e.what());
      }
      ++out.executed;
      if (wf.instruction_execution_failed())
        return fail(map_error(wf.instruction_execution_error()), i, "memory instruction failed");
      if (mem_ && mem_->unsupported())
        return fail(RunStatus::UnsupportedOperand, i, "LDS-DMA or image memory access");
      // An access to guarded code: a pointer the program could not have (its objects never
      // overlap the code), and two programs' code bytes differ. Outside the model's address
      // domain, like a wrapping access: a fault.
      if (mem_ && (mem_->touches_code() || mem_->reads_code()))
        return fail(RunStatus::MemoryFault, i,
                    std::string(mem_->touches_code() ? "store into" : "load from") +
                        " guarded code at '" + step.text + "'");
      if (mem_ && mem_->lds_oob())
        return fail(RunStatus::MemoryFault, i,
                    "LDS address outside the " + std::to_string(kLdsWindow / 1024) +
                        " KiB window at '" + step.text + "'");
      if (cu_->isa_diagnostic_count() != ub_before)
        return fail(RunStatus::UndefinedBehavior, i,
                    "RocJITsu reported ISA undefined behaviour (see isa-undefined log)");
      if (mem_ && mem_->wraps())
        return fail(RunStatus::MemoryFault, i,
                    "memory access wraps around the 64-bit address space at '" + step.text + "'");
      // An access the hardware traps on (misaligned, memory violation) enters the trap
      // handler: a fault of the case, like a halt.
      if (wf.state() == amdgpu::WfState::HALTED || wf.state() == amdgpu::WfState::MEMORY_TRAP ||
          (wf.in_trap_handler() && !trap_before))
        return fail(RunStatus::MemoryFault, i, "memory access fault at '" + step.text + "'");
      if (wf.state() != amdgpu::WfState::RUNNING)
        return fail(RunStatus::WaveStateChanged, i,
                    "the wave stopped running at a memory instruction (state " +
                        std::to_string(static_cast<unsigned>(wf.state())) + ")");
      if (wf.pc != step.pc + inst->size())
        return fail(RunStatus::PcWritten, i,
                    "PC " + hex64(wf.pc) + " after '" + step.text + "', expected " +
                        hex64(step.pc + inst->size()) + " (wave state " +
                        std::to_string(static_cast<unsigned>(wf.state())) + ", trapsts " +
                        hex32(wf.trapsts()) + ")");
      apply_tags(i, exec_before, scratch_tags_); // load destinations: no provenance
      apply_stags(i, scratch_stags_);
      record(i);
      continue;
    }
    if (!inst->execute || step.model_only)
      return fail(RunStatus::MissingExecutionBackend, i, "instruction has no execution callback");
    wf.pc = step.pc;
    const uint64_t ub_before = cu_->isa_diagnostic_count();
    std::string detail;
    util::Result result = util::Result::failure();
    try { // translation boundary, as for step() above
      result = cu_->execute_instruction(inst, wf, &detail);
    } catch (const std::exception &e) {
      return fail(RunStatus::HostException, i, e.what());
    } catch (...) {
      return fail(RunStatus::HostException, i, "unknown exception");
    }
    ++out.executed;
    if (result.failed())
      return fail(map_error(wf.instruction_execution_error()), i, std::move(detail));
    if (cu_->isa_diagnostic_count() != ub_before)
      return fail(RunStatus::UndefinedBehavior, i,
                  "RocJITsu reported ISA undefined behaviour (see isa-undefined log)");
    if (step.branch) {
      // Taken: RocJITsu leaves PC at target - size; not taken: unchanged. Either way the
      // next PC is PC + size. Falling through past the end, or into the region's own
      // fallthrough PC, is the fallthrough exit (0).
      const uint64_t to = wf.pc + static_cast<uint64_t>(inst->size());
      wf.pc = step.pc;
      const Step &last = steps_.back();
      const uint64_t end_pc = last.pc + (last.inst ? last.inst->size() : 0);
      if (auto it = pc_index_.find(to); it != pc_index_.end()) {
        next = it->second;
      } else {
        exit_pc_ = to == end_pc ? 0 : to;
        left = true;
      }
    } else if (wf.pc != step.pc) {
      return fail(RunStatus::PcWritten, i);
    }
    if (wf.state() != amdgpu::WfState::RUNNING)
      return fail(RunStatus::WaveStateChanged, i);
    apply_tags(i, exec_before, scratch_tags_);
    apply_stags(i, scratch_stags_);
    choose_nans(i, exec_before);
    record(i);
  }
  out.exit_pc = exit_pc_;
  return out;
}

} // namespace rocjitsu::isa_diff
