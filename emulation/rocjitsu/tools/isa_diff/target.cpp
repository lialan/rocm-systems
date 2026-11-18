// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "target.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>

namespace rocjitsu::isa_diff {
namespace {

/// Architectural traits of an ISA family that the registry does not describe.
struct FamilyTraits {
  rj_code_arch_t arch;
  unsigned major;
  bool unified_vgprs;
  unsigned default_wave_size;
  bool wave32_supported;
  bool wave64_supported;
};

constexpr std::array kFamilies = std::to_array<FamilyTraits>({
    {ROCJITSU_CODE_ARCH_CDNA1, 9, false, 64, false, true},
    {ROCJITSU_CODE_ARCH_CDNA2, 9, true, 64, false, true},
    {ROCJITSU_CODE_ARCH_CDNA3, 9, true, 64, false, true},
    {ROCJITSU_CODE_ARCH_CDNA4, 9, true, 64, false, true},
    {ROCJITSU_CODE_ARCH_RDNA1, 10, false, 32, true, true},
    {ROCJITSU_CODE_ARCH_RDNA2, 10, false, 32, true, true},
    {ROCJITSU_CODE_ARCH_RDNA3, 11, false, 32, true, true},
    {ROCJITSU_CODE_ARCH_RDNA3_5, 11, false, 32, true, true},
    {ROCJITSU_CODE_ARCH_RDNA4, 12, false, 32, true, true},
    {ROCJITSU_CODE_ARCH_CDNA5, 12, false, 32, true, false},
});

const FamilyTraits *family(rj_code_arch_t arch) {
  for (const FamilyTraits &traits : kFamilies)
    if (traits.arch == arch)
      return &traits;
  return nullptr;
}

/// The family has an executable provider in this image's registry.
bool executable(rj_code_arch_t arch) {
  const IsaTargetDescriptor *descriptor = default_isa_target_registry().find(arch);
  return descriptor && descriptor->supports_execution;
}

constexpr uint32_t kMaxMachine = EF_AMDGPU_MACH;

} // namespace

std::optional<Target> target_by_elf_mach(uint32_t mach) {
  mach &= EF_AMDGPU_MACH;
  const rj_code_arch_t arch = arch_for_elf_mach(mach);
  const FamilyTraits *traits = family(arch);
  if (!traits || !executable(arch))
    return std::nullopt;
  Target target;
  target.name = elf_mach_name(mach);
  target.arch = arch;
  const IsaGpuTargetDescription *gpu =
      default_isa_target_registry().find_gpu_target_by_elf_machine(mach);
  target.gpu = gpu ? gpu->public_id : ROCJITSU_CODE_TARGET_INVALID;
  target.wave_size = traits->default_wave_size;
  target.major = traits->major;
  target.unified_vgprs = traits->unified_vgprs;
  return target;
}

std::optional<Target> target_by_name(std::string_view name) {
  for (uint32_t mach = 1; mach <= kMaxMachine; ++mach)
    if (arch_for_elf_mach(mach) != ROCJITSU_CODE_ARCH_INVALID && name == elf_mach_name(mach))
      return target_by_elf_mach(mach);
  // An ISA family ID or alias of the registry: the family's representative processor.
  if (const IsaTargetDescriptor *descriptor = default_isa_target_registry().find(name))
    return target_by_elf_mach(elf_mach_for_arch(descriptor->architecture_id));
  return std::nullopt;
}

std::optional<Target> with_wave_size(Target target, unsigned wave_size) {
  const FamilyTraits *traits = family(target.arch);
  if (!traits || (wave_size == 32 && !traits->wave32_supported) ||
      (wave_size == 64 && !traits->wave64_supported) || (wave_size != 32 && wave_size != 64))
    return std::nullopt;
  target.wave_size = wave_size;
  return target;
}

std::optional<Target> parse_target(std::string_view text) {
  const size_t colon = text.find(':');
  auto target = target_by_name(text.substr(0, colon));
  if (!target || colon == std::string_view::npos)
    return target;
  const std::string_view wave = text.substr(colon + 1);
  if (wave == "w32" || wave == "wave32")
    return with_wave_size(*target, 32);
  if (wave == "w64" || wave == "wave64")
    return with_wave_size(*target, 64);
  return std::nullopt;
}

std::string supported_targets() {
  std::string names;
  for (uint32_t mach = 1; mach <= kMaxMachine; ++mach)
    if (target_by_elf_mach(mach))
      names += (names.empty() ? "" : ",") + std::string(elf_mach_name(mach));
  return names;
}

std::unique_ptr<Decoder> make_decoder(const Target &target) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  return target.gpu != ROCJITSU_CODE_TARGET_INVALID ? Decoder::create(registry, target.gpu)
                                                    : Decoder::create(registry, target.arch);
}

} // namespace rocjitsu::isa_diff
