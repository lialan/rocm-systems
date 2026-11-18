# rj_isa_diff region executor

`rj_isa_diff` tests whether two AMDGPU instruction sequences behave the same. The
first piece is this directory's region executor: it runs a recorded sequence of
machine instructions (a *region*) on a standalone wave, from a machine state the
caller chooses, and captures the state the region leaves. Later pieces build on
it: input generation and paired differential testing, then code-object analysis
that finds the regions two builds of a kernel differ in.

The executor uses RocJITsu's own instruction execution (`ComputeUnitCore`), so no
instruction semantics are re-implemented here.

## What it runs

- Scalar and vector ALU instructions, including AGPR moves and MFMA.
- Global, scalar, flat, buffer and scratch loads, stores and atomics, LDS
  accesses, `ds_bpermute`/`ds_permute`, LDS transpose loads and the counter
  waits of every generation. Memory is an input: words are generated on first
  read from a seed (or preset), and every stored byte is recorded.
- Direct branches inside the region (taken edges inside it continue there,
  others are exits) and loops, up to a step limit; `s_barrier` and an early
  `s_endpgm`.
- Every target RocJITsu executes (`supported_targets()`): the processors
  `rocjitsu/code/amdgpu_elf.h` names, at each of their wave sizes. The ISA target
  registry supplies the concrete GPU binding when it has one.

A failure is classified (`RunStatus`) and never treated as a no-op: decode
failure, missing execution backend, unimplemented or illegal instruction,
invalid register access, memory fault, ISA undefined behaviour reported by
RocJITsu, PC written, step limit, budget.

## API

```cpp
util::StringDiagnostic diag;
auto created = RegionRunner::create(region, "label", *parse_target("gfx950"),
                                    /*vgpr_count=*/4, diag.emitter());
if (created.failed())
  return diag.message();
RegionRunner &runner = *created.value();
MachineState state;                    // SGPRs, VCC, EXEC, M0, SCC, VGPR lanes, memory
state.frame = runner.baseline_frame(); // the post-dispatch frame
runner.load(state);
RunOutcome outcome = runner.run(/*budget=*/64);
MachineState after = runner.capture();
```

- `region.h`: `RegionInst` (address, machine words, disassembly).
- `machine_state.h`: `MachineState`, its JSON form, and comparisons
  (`differing_locs`, `differing_vgprs`, `differing_frame_fields`).
- `location.h`: tracked scalar locations (s0-s105, VCC/EXEC halves, M0, SCC) and
  vector locations (v0-v255, a0-a255).
- `effects.h`: per-instruction effect summary (reads, writes, executable subset,
  floating-point result kinds). `InstDefUse` reports special registers only for
  field-less operands, so SCC, VCC halves and M0 named through a selector field
  are classified here.
- `target.h`: target lookup by processor name, ISA family ID or ELF machine.
- `json.h`: a small ordered JSON reader/writer (there is no general JSON library
  in the tree), so states stay hand-editable.

## Build and test

```sh
cmake -S . -B build -GNinja -DBUILD_TESTING=ON
ninja -C build rj_isa_diff_tests
ctest --test-dir build -R '^isa_diff\.' --output-on-failure
```

The tests run recorded instruction words from
`tests/isa_diff/fixtures/executor.inc`. To change a fixture, edit
`executor.asm` (each section is `=== name processor`) and run
`tests/isa_diff/fixtures/regen_fixtures.sh [llvm-mc] [llvm-objcopy]`. LLVM is
needed only for that step, never to build or run the tests.

## Limits

- One wave runs: a barrier waits for nothing, and shared memory other waves
  could write is an input.
- The executor allocates up to 256 VGPRs and 256 AGPRs per wave and a 64 KiB LDS
  window; LDS addresses outside it are a memory fault.
- Inactive lanes are not modelled as undefined: their values are whatever the
  state holds.
