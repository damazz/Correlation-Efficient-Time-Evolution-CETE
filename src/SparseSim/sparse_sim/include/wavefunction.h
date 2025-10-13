#ifndef WAVEFUNCTION_H
#define WAVEFUNCTION_H

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef klib_unused
#undef klib_unused
#endif
#define klib_unused

#include "khash.h"
#include "pauli.h"

// Struct for a single slater determinant
typedef struct SlaterDeterminantC {
  unsigned int N;      // Number of orbitals
  double complex coef; // Complex coefficient in front of the slater determinant
  unsigned int *orbitals; // Dynamically allocated array for orbital occupations
  unsigned int
      encoding; // Decimal encoding of orbital occupations for hashing purposes
} SlaterDeterminantC;

// Slater determinant function declarations
SlaterDeterminantC *slater_determinant_init_c(unsigned int N,
                                              double complex coef,
                                              unsigned int orbitals[]);
void free_slater_determinant_c(SlaterDeterminantC *sdet);
char *slater_determinant_to_string_c(SlaterDeterminantC *sdet, char bra_or_ket);
SlaterDeterminantC *
slater_determinant_scalar_multiplication_c(SlaterDeterminantC *sdet,
                                           double complex scalar);
SlaterDeterminantC *slater_determinant_adjoint_c(SlaterDeterminantC *sdet);
double slater_determinant_comparison_c(SlaterDeterminantC *bra,
                                       SlaterDeterminantC *ket);
double complex slater_dermininant_multiplication_c(SlaterDeterminantC *bra,
                                                   SlaterDeterminantC *ket);
SlaterDeterminantC *
slater_determinant_pauli_string_multiplication_c(PauliStringC *pString,
                                                 SlaterDeterminantC *sdet);

// Struct for slater determinant khash hashmap
KHASH_MAP_INIT_INT(slater_hash, SlaterDeterminantC *)

// Struct for a wavefunction containing multiple slater determinants
typedef struct WavefunctionC {
  unsigned int N; // Number of orbitals
  unsigned int s; // Current number of slater determinants
  khash_t(slater_hash) *
      slater_determinants; // Hashmap of slater determinant elements
} WavefunctionC;

// Wavefunction function declarations
WavefunctionC *wavefunction_init_c(unsigned int N);
void free_wavefunction_c(WavefunctionC *wfn);
char *wavefunction_to_string_c(WavefunctionC *wfn, char bra_or_ket);
double wavefunction_norm_c(WavefunctionC *wfn);
WavefunctionC *wavefunction_scalar_multiplication_c(WavefunctionC *wfn,
                                                    double complex scalar);
WavefunctionC *wavefunction_adjoint_c(WavefunctionC *wfn);
double complex wavefunction_multiplication_c(WavefunctionC *bra,
                                             WavefunctionC *ket);
void wavefunction_append_slater_determinant_c(WavefunctionC *wfn,
                                              SlaterDeterminantC *sdet);
WavefunctionC *wavefunction_pauli_string_multiplication_c(PauliStringC *pString,
                                                          WavefunctionC *wfn);
WavefunctionC *wavefunction_pauli_sum_multiplication_c(PauliSumC *pSum,
                                                       WavefunctionC *wfn);
WavefunctionC *wavefunction_pauli_string_evolution_c(PauliStringC *pString,
                                                     WavefunctionC *wfn,
                                                     double complex epsilon);
WavefunctionC *wavefunction_pauli_sum_evolution_c(PauliSumC *pSum,
                                                  WavefunctionC *wfn,
                                                  double complex epsilon);
WavefunctionC *wavefunction_remove_global_phase_c(WavefunctionC *wfn);
WavefunctionC *wavefunction_remove_near_zero_terms_c(WavefunctionC *wfn,
                                                     double cutoff);

static inline khiter_t kh_begin_slater_hash(khash_t(slater_hash) * h) {
  (void)h;
  return kh_begin(h);
}
static inline khiter_t kh_end_slater_hash(khash_t(slater_hash) * h) {
  (void)h;
  return kh_end(h);
}
static inline int kh_exist_slater_hash(khash_t(slater_hash) * h, khiter_t k) {
  return kh_exist(h, k);
}
static inline SlaterDeterminantC *kh_value_slater_hash(khash_t(slater_hash) * h,
                                                       khiter_t k) {
  return kh_val(h, k);
}

#endif // WAVEFUNCTION_H
