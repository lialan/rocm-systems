# Region executor fixtures. Each section is `=== name processor` followed by its
# instructions; regen_fixtures.sh assembles it for that processor into executor.inc.
# The tests use the recorded words only and never run an assembler.

=== effects gfx950
s_add_u32 s4, s2, s3
s_addc_u32 s5, s0, s1
s_cselect_b64 s[14:15], s[2:3], vcc
s_mov_b32 vcc_hi, s1
s_and_b64 exec, exec, s[2:3]
s_and_saveexec_b64 s[2:3], s[4:5]
s_cmov_b32 s6, s7
s_mov_b32 m0, s2
s_mov_b32 s9, src_scc
s_lshl_b64 s[6:7], s[4:5], 3
s_cbranch_scc0 1
s_cbranch_execz 1
s_cbranch_vccnz 1
s_getpc_b64 s[18:19]
s_waitcnt 0
s_load_dword s20, s[0:1], 0x0
v_mov_b32 v0, s1
s_mov_b32 s0, ttmp0
s_getreg_b32 s4, hwreg(HW_REG_MODE)
s_movrels_b32 s0, s1
s_endpgm

=== scalar_arith gfx950
s_add_u32 s4, s2, s3
s_addc_u32 s5, s0, s1
s_cselect_b32 s6, s4, s5
s_lshl_b64 s[8:9], s[4:5], 2

=== valu gfx950
v_add_u32 v2, v0, v1
v_cmp_gt_u32 vcc, v0, v1
v_cndmask_b32 v3, v1, v0, vcc
v_mul_lo_u32 v4, v0, v1

=== valu_wave32 gfx1100
v_add_nc_u32 v2, v0, v1
v_cmp_gt_u32 vcc_lo, v0, v1
v_cndmask_b32 v3, v1, v0, vcc_lo

=== mfma gfx950
v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[0:3]

=== global_memory gfx950
global_load_dword v1, v0, s[0:1]
s_waitcnt vmcnt(0)
v_add_u32 v2, 1, v1
global_store_dword v0, v2, s[0:1] offset:256

=== lds gfx950
ds_write_b32 v0, v1
s_waitcnt lgkmcnt(0)
ds_read_b32 v2, v0
s_waitcnt lgkmcnt(0)

=== scalar_load_code gfx950
s_load_dword s4, s[0:1], 0x0
s_waitcnt lgkmcnt(0)

=== loop gfx950
s_mov_b32 s2, 0
s_add_u32 s2, s2, 1
s_cmp_lt_u32 s2, s3
s_cbranch_scc1 -3

=== branch_exit gfx950
s_cmp_eq_u32 s0, 0
s_cbranch_scc1 5
s_add_u32 s1, s1, 1

=== bitwise_gfx908 gfx908
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx90a gfx90a
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx942 gfx942
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx950 gfx950
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1010 gfx1010
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1030 gfx1030
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1100 gfx1100
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1150 gfx1150
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1200 gfx1200
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1201 gfx1201
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1250 gfx1250
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0

=== bitwise_gfx1251 gfx1251
s_and_b32 s4, s2, s3
s_xor_b32 s5, s2, s3
s_lshl_b32 s6, s2, 3
v_xor_b32 v2, v0, v1
v_and_b32 v3, s2, v0
