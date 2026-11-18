// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file region_executor_test.cpp
/// @brief Tests for the rj_isa_diff region executor: targets, locations, instruction
/// effects, and RegionRunner execution of recorded instruction words.
///
/// Fixtures are assembled once by fixtures/regen_fixtures.sh (fixtures/executor.asm ->
/// fixtures/executor.inc); the tests use the recorded words and never run an assembler.

#include "effects.h"
#include "json.h"
#include "location.h"
#include "machine_state.h"
#include "region_runner.h"
#include "target.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include "fixtures/executor.inc"
// clang-format on

namespace {

using namespace rocjitsu;
using namespace rocjitsu::isa_diff;

constexpr uint64_t kRegionPc = 0x4000;

LocSet locs(std::string_view text) {
  auto parsed = parse_locs(text);
  EXPECT_TRUE(parsed.has_value()) << text;
  return parsed.value_or(LocSet{});
}

Target target_named(std::string_view text) {
  auto target = parse_target(text);
  EXPECT_TRUE(target.has_value()) << text;
  return target.value_or(Target{});
}

/// Split recorded words into instructions at consecutive addresses from @p pc.
std::vector<RegionInst> region_for(const std::vector<uint32_t> &words, const Target &target = {},
                                   uint64_t pc = kRegionPc) {
  auto decoder = make_decoder(target);
  std::vector<RegionInst> region;
  size_t offset = 0;
  while (offset < words.size()) {
    auto decoded =
        decoder->decode_window(std::span<const uint32_t>(words).subspan(offset), pc + offset * 4);
    if (decoded.failed()) {
      ADD_FAILURE() << "decode failed at word " << offset << " for " << target.name;
      break;
    }
    const size_t count = static_cast<size_t>(decoded.value()->size()) / 4;
    RegionInst inst;
    inst.pc = pc + offset * 4;
    inst.words.assign(words.begin() + static_cast<std::ptrdiff_t>(offset),
                      words.begin() + static_cast<std::ptrdiff_t>(offset + count));
    inst.text = decoded.value()->disassemble();
    region.push_back(std::move(inst));
    offset += count;
  }
  return region;
}

/// Decode @p words as consecutive instructions. They may point into @p words, which must
/// outlive them.
std::vector<std::unique_ptr<Instruction>> decode_all(const std::vector<uint32_t> &words) {
  auto decoder = make_decoder(Target{});
  std::vector<std::unique_ptr<Instruction>> insts;
  for (size_t offset = 0; offset < words.size();) {
    auto decoded = decoder->decode_window(std::span<const uint32_t>(words).subspan(offset),
                                          kRegionPc + offset * 4);
    if (decoded.failed())
      break;
    offset += static_cast<size_t>(decoded.value()->size()) / 4;
    insts.push_back(std::move(decoded).value());
  }
  return insts;
}

std::unique_ptr<RegionRunner> make_runner(const std::vector<RegionInst> &region,
                                          const Target &target = {}, unsigned vgpr_count = 0) {
  util::StringDiagnostic diag;
  auto runner = RegionRunner::create(region, "test", target, vgpr_count, diag.emitter());
  EXPECT_TRUE(runner.succeeded()) << diag.message();
  return runner.succeeded() ? std::move(runner).value() : nullptr;
}

/// The runner's post-dispatch frame, @p vgpr_count VGPRs, and every lane of the wave active.
MachineState initial_state(const RegionRunner &runner, const Target &target,
                           unsigned vgpr_count = 0) {
  MachineState state;
  state.frame = runner.baseline_frame();
  state.resize_vgprs(vgpr_count);
  state.set(kLocExecLo, 0xffffffffu);
  state.set(kLocExecHi, target.wave32() ? 0u : 0xffffffffu);
  return state;
}

/// Load @p state, run up to @p budget instructions, and capture the final state.
std::pair<RunOutcome, MachineState> run_from(RegionRunner &runner, const MachineState &state,
                                             size_t budget = 64) {
  runner.load(state);
  RunOutcome outcome = runner.run(budget);
  return {outcome, runner.capture()};
}

uint32_t f32_bits(float value) { return std::bit_cast<uint32_t>(value); }

// Targets and locations

TEST(IsaDiffTarget, ResolvesProcessorsThroughTheRegistry) {
  const Target gfx950 = target_named("gfx950");
  EXPECT_EQ(gfx950, Target{});
  EXPECT_EQ(gfx950.arch, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(gfx950.gpu, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(gfx950.wave_size, 64u);

  const Target gfx1251 = target_named("gfx1251");
  EXPECT_EQ(gfx1251.arch, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(gfx1251.gpu, ROCJITSU_CODE_TARGET_GFX1251);
  EXPECT_EQ(gfx1251.wave_size, 32u);
  EXPECT_EQ(gfx1251.major, 12u);

  // An executable family without a concrete registry binding for this processor.
  const Target gfx908 = target_named("gfx908");
  EXPECT_EQ(gfx908.arch, ROCJITSU_CODE_ARCH_CDNA1);
  EXPECT_EQ(gfx908.gpu, ROCJITSU_CODE_TARGET_INVALID);
  EXPECT_FALSE(gfx908.unified_vgprs);

  EXPECT_EQ(target_named("cdna4").name, "gfx950"); // a registry family ID
  ASSERT_TRUE(target_by_elf_mach(0x4f));
  EXPECT_EQ(target_by_elf_mach(0x4f)->name, "gfx950");
  EXPECT_FALSE(target_by_elf_mach(0));
  EXPECT_FALSE(target_by_name("gfx9999"));

  EXPECT_EQ(target_named("gfx1100:w64").wave_size, 64u);
  EXPECT_EQ(target_named("gfx1100").wave_size, 32u);
  EXPECT_FALSE(parse_target("gfx950:w32"));
  EXPECT_FALSE(parse_target("gfx1250:wave64"));
  EXPECT_FALSE(parse_target("gfx1100:w16"));

  const std::string all = supported_targets();
  for (std::string_view name : {"gfx908", "gfx90a", "gfx942", "gfx950", "gfx1010", "gfx1030",
                                "gfx1100", "gfx1150", "gfx1200", "gfx1201", "gfx1250", "gfx1251"})
    EXPECT_NE(all.find(name), std::string::npos) << name << " missing from " << all;
}

TEST(IsaDiffLocations, AgprsAreVectorLocationsAfterTheVgprs) {
  EXPECT_EQ(vgpr_name(3), "v3");
  EXPECT_EQ(vgpr_name(kAgprBase + 40), "a40");
  EXPECT_EQ(parse_vgpr_name("a40"), kAgprBase + 40);
  EXPECT_EQ(parse_vgpr_name("v255"), 255u);
  EXPECT_FALSE(parse_vgpr_name("a256"));
  EXPECT_FALSE(parse_vgpr_name("s1"));
  VSet v;
  v.set(1);
  v.set(kAgprBase);
  v.set(kAgprBase + 255);
  EXPECT_EQ(format_vgprs(v), "v1,a0,a255");
  EXPECT_EQ(parse_vgprs("v1,a0,a255"), v);
  EXPECT_EQ(parse_locs(format_locs(locs("s7,vcc_hi,scc"))), locs("s7,vcc_hi,scc"));
}

TEST(IsaDiffFormat, JsonParsesAndPrintsExactly) {
  std::string error;
  EXPECT_FALSE(Json::parse("{\"a\": [1, 2,]}", &error).has_value());
  auto j = Json::parse("{\"s\":\"x\\u0041\\n\",\"n\":-3,\"u\":18446744073709551615}", &error);
  ASSERT_TRUE(j) << error;
  EXPECT_EQ((*j)["s"].as_string(), "xA\n");
  EXPECT_EQ((*j)["n"].as_int(), -3);
  EXPECT_EQ((*j)["u"].as_uint(), UINT64_MAX);
  auto again = Json::parse(j->dump(), &error);
  ASSERT_TRUE(again) << error;
  EXPECT_EQ(again->dump(), j->dump());
}

// Instruction effects

TEST(IsaDiffEffects, ClassifiesSaluOperandsPrecisely) {
  auto insts = decode_all(k_effects);
  ASSERT_EQ(insts.size(), 21u);
  auto fx = [&](size_t i) { return classify(*insts[i], Target{}); };

  Effects add = fx(0); // s_add_u32 s4, s2, s3
  EXPECT_TRUE(add.executable);
  EXPECT_EQ(add.reads, locs("s2,s3"));
  EXPECT_EQ(add.writes, locs("s4,scc"));

  Effects addc = fx(1); // s_addc_u32 s5, s0, s1
  EXPECT_EQ(addc.reads, locs("s0,s1,scc"));
  EXPECT_EQ(addc.writes, locs("s5,scc"));

  Effects csel = fx(2); // s_cselect_b64 s[14:15], s[2:3], vcc
  EXPECT_EQ(csel.reads, locs("s2,s3,vcc_lo,vcc_hi,scc"));
  EXPECT_EQ(csel.writes, locs("s14,s15"));
  EXPECT_EQ(csel.wide_reads, locs("s2,vcc_lo"));

  Effects vcchi = fx(3); // s_mov_b32 vcc_hi, s1: only the high half
  EXPECT_TRUE(vcchi.executable);
  EXPECT_EQ(vcchi.writes, locs("vcc_hi"));

  Effects exec_and = fx(4); // s_and_b64 exec, exec, s[2:3]
  EXPECT_EQ(exec_and.reads, locs("s2,s3,exec_lo,exec_hi"));
  EXPECT_EQ(exec_and.writes, locs("exec_lo,exec_hi,scc"));

  Effects saveexec = fx(5); // s_and_saveexec_b64 s[2:3], s[4:5]
  EXPECT_TRUE(saveexec.executable);
  EXPECT_EQ(saveexec.reads, locs("s4,s5,exec_lo,exec_hi"));
  EXPECT_EQ(saveexec.writes, locs("s2,s3,exec_lo,exec_hi,scc"));

  Effects cmov = fx(6); // s_cmov_b32 s6, s7: a conditional def also reads s6
  EXPECT_EQ(cmov.reads, locs("s6,s7,scc"));
  EXPECT_EQ(cmov.writes, locs("s6"));

  EXPECT_EQ(fx(7).writes, locs("m0"));        // s_mov_b32 m0, s2
  EXPECT_EQ(fx(8).reads, locs("scc"));        // s_mov_b32 s9, src_scc
  EXPECT_EQ(fx(9).writes, locs("s6,s7,scc")); // s_lshl_b64
  EXPECT_EQ(fx(9).wide_reads, locs("s4"));

  for (size_t i : {10u, 11u, 12u}) { // s_cbranch_scc0 / execz / vccnz: the runner follows them
    EXPECT_TRUE(fx(i).executable);
    EXPECT_TRUE(fx(i).branch);
    EXPECT_TRUE(fx(i).control_flow);
    EXPECT_FALSE(fx(i).implicit_special_reads) << i;
  }
  EXPECT_EQ(fx(10).reads, locs("scc"));
  EXPECT_EQ(fx(11).reads, locs("exec_lo,exec_hi"));
  EXPECT_EQ(fx(12).reads, locs("vcc_lo,vcc_hi"));

  Effects getpc = fx(13);
  EXPECT_FALSE(getpc.executable);
  EXPECT_TRUE(getpc.reads_pc);
  EXPECT_TRUE(fx(14).executable); // s_waitcnt
  EXPECT_FALSE(fx(14).implicit_special_reads);
  Effects load = fx(15);
  EXPECT_TRUE(load.executable); // s_load_dword: a region-executable scalar load
  EXPECT_TRUE(load.mem_read);
  EXPECT_FALSE(load.implicit_special_reads); // SMEM: its operands name everything it reads
  EXPECT_TRUE(load.implicit_reads.none());
  EXPECT_EQ(load.reads, locs("s0,s1"));
  // v_mov_b32 v0, s1: reads s1 and EXEC (the lanes it writes) and writes v0.
  Effects vmov = fx(16);
  EXPECT_TRUE(vmov.executable) << vmov.reject_reason;
  EXPECT_TRUE(vmov.valu);
  EXPECT_EQ(vmov.reads, locs("s1,exec_lo,exec_hi"));
  EXPECT_TRUE(vmov.writes.none());
  EXPECT_EQ(format_vgprs(vmov.vwrites), "v0");
  EXPECT_TRUE(vmov.vreads.none());
  EXPECT_FALSE(vmov.implicit_special_reads);
  EXPECT_NE(fx(17).reject_reason.find("trap temporary"), std::string::npos) << fx(17).reject_reason;
  EXPECT_NE(fx(18).reject_reason.find("allowlist"), std::string::npos) << fx(18).reject_reason;
  EXPECT_TRUE(fx(19).unknown_effect); // s_movrels_b32: indirect SGPR
  EXPECT_TRUE(fx(20).control_flow);   // s_endpgm
  EXPECT_FALSE(fx(20).implicit_special_reads);
}

// Execution

TEST(IsaDiffExecutor, ScalarArithmeticCarriesThroughScc) {
  auto runner = make_runner(region_for(k_scalar_arith));
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{});
  state.set(0, 0xffffffffu);
  state.set(1, 1);
  state.set(2, 0xfffffff0u);
  state.set(3, 0x20);
  state.set(20, 0x5a5a5a5au);
  const auto [outcome, final_state] = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  EXPECT_EQ(outcome.executed, 4u);
  EXPECT_EQ(final_state.get(4), 0x10u); // 0xfffffff0 + 0x20, carry out
  EXPECT_EQ(final_state.get(5), 1u);    // 0xffffffff + 1 + carry
  EXPECT_EQ(final_state.get(6), 0x10u); // SCC (carry) selects s4
  EXPECT_EQ(final_state.get(8), 0x40u); // s[4:5] << 2
  EXPECT_EQ(final_state.get(9), 4u);
  EXPECT_EQ(final_state.get(kLocScc), 1u);
  EXPECT_EQ(final_state.get(20), 0x5a5a5a5au);
  EXPECT_EQ(differing_locs(state, final_state), locs("s4,s5,s6,s8,s9,scc"));
  EXPECT_TRUE(differing_frame_fields(state.frame, final_state.frame).empty());
}

TEST(IsaDiffExecutor, VectorResultsFollowExec) {
  constexpr unsigned kVgprs = 5;
  auto runner = make_runner(region_for(k_valu), Target{}, kVgprs);
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{}, kVgprs);
  const uint64_t exec = 0x0000ffffffff00ffull;
  state.set(kLocExecLo, static_cast<uint32_t>(exec));
  state.set(kLocExecHi, static_cast<uint32_t>(exec >> 32));
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    state.vset(0, lane, lane * 3 + 1);
    state.vset(1, lane, 100 - lane);
    for (unsigned r = 2; r < kVgprs; ++r)
      state.vset(r, lane, 0xdeadbeefu);
  }
  const auto [outcome, final_state] = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  const uint64_t vcc =
      uint64_t{final_state.get(kLocVccLo)} | (uint64_t{final_state.get(kLocVccHi)} << 32);
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    const uint32_t a = state.vget(0, lane), b = state.vget(1, lane);
    if ((exec >> lane) & 1u) {
      EXPECT_EQ(final_state.vget(2, lane), a + b) << lane;
      EXPECT_EQ(final_state.vget(3, lane), a > b ? a : b) << lane;
      EXPECT_EQ(final_state.vget(4, lane), a * b) << lane;
      EXPECT_EQ((vcc >> lane) & 1u, a > b ? 1u : 0u) << lane;
    } else {
      for (unsigned r = 2; r < kVgprs; ++r)
        EXPECT_EQ(final_state.vget(r, lane), 0xdeadbeefu) << "v" << r << " lane " << lane;
    }
  }
}

TEST(IsaDiffExecutor, Wave32RunsTheLowLanes) {
  constexpr unsigned kVgprs = 4;
  const Target gfx1100 = target_named("gfx1100");
  auto runner = make_runner(region_for(k_valu_wave32, gfx1100), gfx1100, kVgprs);
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, gfx1100, kVgprs);
  for (unsigned lane = 0; lane < 32; ++lane) {
    state.vset(0, lane, lane * 7);
    state.vset(1, lane, 100);
  }
  const auto [outcome, final_state] = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  for (unsigned lane = 0; lane < 32; ++lane) {
    const uint32_t a = lane * 7, b = 100;
    EXPECT_EQ(final_state.vget(2, lane), a + b) << lane;
    EXPECT_EQ(final_state.vget(3, lane), a > b ? a : b) << lane;
    EXPECT_EQ((final_state.get(kLocVccLo) >> lane) & 1u, a > b ? 1u : 0u) << lane;
  }
  EXPECT_EQ(final_state.get(kLocVccHi), state.get(kLocVccHi)); // one VCC SGPR on wave32
}

TEST(IsaDiffExecutor, EveryTargetRunsScalarAndVectorCode) {
  const std::vector<std::pair<std::string_view, const std::vector<uint32_t> *>> fixtures = {
      {"gfx908", &k_bitwise_gfx908},   {"gfx90a", &k_bitwise_gfx90a},
      {"gfx942", &k_bitwise_gfx942},   {"gfx950", &k_bitwise_gfx950},
      {"gfx1010", &k_bitwise_gfx1010}, {"gfx1030", &k_bitwise_gfx1030},
      {"gfx1100", &k_bitwise_gfx1100}, {"gfx1150", &k_bitwise_gfx1150},
      {"gfx1200", &k_bitwise_gfx1200}, {"gfx1201", &k_bitwise_gfx1201},
      {"gfx1250", &k_bitwise_gfx1250}, {"gfx1251", &k_bitwise_gfx1251},
  };
  constexpr unsigned kVgprs = 4;
  for (const auto &[name, words] : fixtures) {
    for (unsigned wave_size : {32u, 64u}) {
      auto target = with_wave_size(target_named(name), wave_size);
      if (!target)
        continue; // not a wave size of this target
      SCOPED_TRACE(target->label());
      auto runner = make_runner(region_for(*words, *target), *target, kVgprs);
      ASSERT_TRUE(runner);
      MachineState state = initial_state(*runner, *target, kVgprs);
      state.set(2, 0x12345678u);
      state.set(3, 0x0ff00ff0u);
      for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
        state.vset(0, lane, lane * 0x01010101u);
        state.vset(1, lane, ~lane);
      }
      const auto [outcome, final_state] = run_from(*runner, state);
      ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
      EXPECT_EQ(final_state.get(4), 0x12345678u & 0x0ff00ff0u);
      EXPECT_EQ(final_state.get(5), 0x12345678u ^ 0x0ff00ff0u);
      EXPECT_EQ(final_state.get(6), 0x12345678u << 3);
      EXPECT_EQ(final_state.get(kLocScc), 1u);
      for (unsigned lane = 0; lane < target->wave_size; ++lane) {
        EXPECT_EQ(final_state.vget(2, lane), (lane * 0x01010101u) ^ ~lane) << lane;
        EXPECT_EQ(final_state.vget(3, lane), 0x12345678u & (lane * 0x01010101u)) << lane;
      }
    }
  }
}

TEST(IsaDiffExecutor, MfmaAccumulatesIntoAgprs) {
  constexpr unsigned kVgprs = kAgprBase + 4; // v0..v255 and a0..a3
  auto runner = make_runner(region_for(k_mfma), Target{}, kVgprs);
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{}, kVgprs);
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    state.vset(0, lane, f32_bits(1.0f));
    state.vset(1, lane, f32_bits(1.0f));
    for (unsigned acc = 0; acc < 4; ++acc)
      state.vset(kAgprBase + acc, lane, f32_bits(0.5f));
  }
  const auto [outcome, final_state] = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  // Every 4x4x1 block: D = A * B + C = 1 * 1 + 0.5.
  for (unsigned acc = 0; acc < 4; ++acc)
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      EXPECT_EQ(final_state.vget(kAgprBase + acc, lane), f32_bits(1.5f))
          << vgpr_name(kAgprBase + acc) << " lane " << lane;
}

TEST(IsaDiffExecutor, GlobalLoadsReadTheImageAndStoresAreRecorded) {
  constexpr unsigned kVgprs = 3;
  constexpr uint64_t kBase = 0x0000'0010'0000'0000ull;
  auto runner = make_runner(region_for(k_global_memory), Target{}, kVgprs);
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{}, kVgprs);
  state.set(0, static_cast<uint32_t>(kBase));
  state.set(1, static_cast<uint32_t>(kBase >> 32));
  state.mem_seed = 7;
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    state.vset(0, lane, lane * 4);
    state.mem_init.emplace_back((kBase + lane * 4) / 4, 0x1000 + lane);
  }
  const auto [outcome, final_state] = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  std::vector<std::pair<uint64_t, uint8_t>> expected;
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    EXPECT_EQ(final_state.vget(1, lane), 0x1000 + lane) << lane;
    EXPECT_EQ(final_state.vget(2, lane), 0x1001 + lane) << lane;
    for (unsigned byte = 0; byte < 4; ++byte)
      expected.emplace_back(kBase + 256 + lane * 4 + byte,
                            static_cast<uint8_t>((0x1001 + lane) >> (8 * byte)));
  }
  EXPECT_EQ(final_state.mem_written, expected);
  EXPECT_EQ(final_state.mem_init.size(), kWaveLanes); // exactly the words the case read

  // Without preset words, data comes from the image's content function.
  auto image = std::make_shared<MemoryImage>();
  image->set_content([](uint64_t, uint64_t word) { return static_cast<uint32_t>(word * 3); });
  runner->set_memory_image(image);
  state.mem_init.clear();
  state.mem_seed = 8;
  const auto [outcome2, final2] = run_from(*runner, state);
  ASSERT_EQ(outcome2.status, RunStatus::Ok) << outcome2.detail;
  for (unsigned lane = 0; lane < kWaveLanes; ++lane)
    EXPECT_EQ(final2.vget(1, lane), static_cast<uint32_t>((kBase + lane * 4) / 4 * 3)) << lane;
}

TEST(IsaDiffExecutor, LdsRoundTripsAndFaultsOutsideTheWindow) {
  constexpr unsigned kVgprs = 3;
  auto runner = make_runner(region_for(k_lds), Target{}, kVgprs);
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{}, kVgprs);
  state.set(kLocM0, 0xffffffffu);
  state.mem_seed = 3;
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    state.vset(0, lane, lane * 4);
    state.vset(1, lane, lane + 100);
  }
  MachineState outside = state;
  for (unsigned lane = 0; lane < kWaveLanes; ++lane)
    outside.vset(0, lane, static_cast<uint32_t>(RegionRunner::kLdsWindow) + lane * 4);

  for (int round = 0; round < 2; ++round) { // the fault halts the wave; load() re-dispatches it
    const auto [outcome, final_state] = run_from(*runner, state);
    ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      EXPECT_EQ(final_state.vget(2, lane), lane + 100) << lane;
    const RunOutcome fault = run_from(*runner, outside).first;
    EXPECT_EQ(fault.status, RunStatus::MemoryFault) << fault.detail;
    EXPECT_EQ(fault.failed_index, 0u);
  }
}

TEST(IsaDiffExecutor, LoadsFromTheRegionCodeFault) {
  auto runner = make_runner(region_for(k_scalar_load_code));
  ASSERT_TRUE(runner);
  const auto [lo, hi] = runner->code_range();
  EXPECT_EQ(lo, kRegionPc);
  EXPECT_GT(hi, lo);
  runner->guard_code(lo, hi);
  MachineState state = initial_state(*runner, Target{});
  state.set(0, static_cast<uint32_t>(kRegionPc));
  state.mem_seed = 1;
  const RunOutcome outcome = run_from(*runner, state).first;
  EXPECT_EQ(outcome.status, RunStatus::MemoryFault) << outcome.detail;
}

TEST(IsaDiffExecutor, LoopsRunToTheirExitUpToTheStepLimit) {
  auto runner = make_runner(region_for(k_loop));
  ASSERT_TRUE(runner);
  EXPECT_TRUE(runner->loops());
  MachineState state = initial_state(*runner, Target{});
  state.set(3, 5);
  auto [outcome, final_state] = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  EXPECT_EQ(final_state.get(2), 5u);
  EXPECT_EQ(outcome.executed, 1u + 5u * 3u);
  EXPECT_EQ(outcome.exit_pc, 0u);

  state.set(3, 0); // a do-while loop runs once
  std::tie(outcome, final_state) = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  EXPECT_EQ(final_state.get(2), 1u);

  state.set(3, 0xffffffffu);
  std::tie(outcome, final_state) = run_from(*runner, state, 4);
  EXPECT_EQ(outcome.status, RunStatus::StepLimit);
  EXPECT_EQ(outcome.executed, runner->step_limit(4));
}

TEST(IsaDiffExecutor, BranchesOutOfTheRegionAreExits) {
  auto runner = make_runner(region_for(k_branch_exit));
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{});
  state.set(0, 0);
  state.set(1, 41);
  auto [outcome, final_state] = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  EXPECT_EQ(outcome.exit_pc, kRegionPc + 28); // s_cbranch_scc1 at +4, offset 5 dwords
  EXPECT_EQ(final_state.exit_pc, kRegionPc + 28);
  EXPECT_EQ(final_state.get(1), 41u);

  state.set(0, 1);
  std::tie(outcome, final_state) = run_from(*runner, state);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  EXPECT_EQ(outcome.exit_pc, 0u); // fell off the end
  EXPECT_EQ(final_state.get(1), 42u);
}

TEST(IsaDiffExecutor, FailuresAreClassified) {
  // A truncated 64-bit encoding: the decoded size differs from the recorded words.
  std::vector<RegionInst> truncated = region_for(k_global_memory);
  truncated.resize(1);
  truncated[0].words.resize(1);
  auto runner = make_runner(truncated, Target{}, 3);
  ASSERT_TRUE(runner);
  RunOutcome outcome = run_from(*runner, initial_state(*runner, Target{}, 3)).first;
  EXPECT_EQ(outcome.status, RunStatus::DecodeFailure);
  EXPECT_EQ(outcome.failed_index, 0u);

  runner = make_runner(region_for(k_scalar_arith));
  ASSERT_TRUE(runner);
  outcome = run_from(*runner, initial_state(*runner, Target{}), 2).first;
  EXPECT_EQ(outcome.status, RunStatus::BudgetExceeded);
  EXPECT_EQ(outcome.executed, 2u);
  runner->simulate_model_only_for_test(1);
  outcome = run_from(*runner, initial_state(*runner, Target{})).first;
  EXPECT_EQ(outcome.status, RunStatus::MissingExecutionBackend);
  EXPECT_EQ(outcome.failed_index, 1u);

  util::StringDiagnostic diag;
  auto too_many =
      RegionRunner::create(region_for(k_valu), "test", Target{}, kNumVgprs + 1, diag.emitter());
  EXPECT_TRUE(too_many.failed());
  EXPECT_NE(diag.message().find("VGPRs"), std::string::npos) << diag.message();
}

TEST(IsaDiffExecutor, CasesDoNotLeakIntoEachOther) {
  constexpr unsigned kVgprs = 3;
  constexpr uint64_t kBase = 0x0000'0020'0000'0000ull;
  auto runner = make_runner(region_for(k_global_memory), Target{}, kVgprs);
  ASSERT_TRUE(runner);
  auto make_case = [&](uint64_t seed, uint32_t offset) {
    MachineState state = initial_state(*runner, Target{}, kVgprs);
    state.set(0, static_cast<uint32_t>(kBase));
    state.set(1, static_cast<uint32_t>(kBase >> 32));
    state.mem_seed = seed;
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      state.vset(0, lane, offset + lane * 4);
    return state;
  };
  const MachineState first = make_case(11, 0), second = make_case(12, 64);
  const MachineState once = run_from(*runner, first).second;
  (void)run_from(*runner, second);
  const MachineState again = run_from(*runner, first).second;
  EXPECT_EQ(once, again);
  EXPECT_FALSE(once.mem_written.empty());
}

TEST(IsaDiffExecutor, TraceRecordsEveryStep) {
  auto runner = make_runner(region_for(k_scalar_arith));
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{});
  state.set(2, 1);
  state.set(3, 2);
  runner->load(state);
  std::vector<TraceStep> trace;
  RunOutcome outcome = runner->run(64, &trace);
  ASSERT_EQ(outcome.status, RunStatus::Ok) << outcome.detail;
  ASSERT_EQ(trace.size(), 4u);
  EXPECT_EQ(trace[0].pc, kRegionPc);
  EXPECT_EQ(trace[0].changed, locs("s4"));
  EXPECT_EQ(trace[0].after.get(4), 3u);
  EXPECT_EQ(trace[3].after, runner->capture());
}

TEST(IsaDiffFormat, MachineStatesRoundTripThroughJson) {
  constexpr unsigned kVgprs = 3;
  auto runner = make_runner(region_for(k_global_memory), Target{}, kVgprs);
  ASSERT_TRUE(runner);
  MachineState state = initial_state(*runner, Target{}, kVgprs);
  state.set(1, 0x30);
  state.mem_seed = 5;
  for (unsigned lane = 0; lane < kWaveLanes; ++lane)
    state.vset(0, lane, lane * 4);
  const MachineState final_state = run_from(*runner, state).second;
  ASSERT_FALSE(final_state.mem_written.empty());
  std::string error;
  auto parsed = Json::parse(final_state.to_json().dump(), &error);
  ASSERT_TRUE(parsed) << error;
  auto back = MachineState::from_json(*parsed, error);
  ASSERT_TRUE(back) << error;
  EXPECT_EQ(*back, final_state);
}

} // namespace
