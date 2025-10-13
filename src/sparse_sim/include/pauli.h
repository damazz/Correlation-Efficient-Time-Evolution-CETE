#ifndef PAULI_H
#define PAULI_H

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "khash.h"

#ifdef klib_unused
#undef klib_unused
#endif
#define klib_unused

// Struct for a single pauli string
typedef struct PauliStringC {
  unsigned int N;       // Number of qubits
  double complex coef;  // Coefficient in front of the Pauli string
  unsigned int *paulis; // Dynamically allocated array for Pauli operators
  uint64_t
      encoding; // Decimal encoding of orbital occupations for indexing purposes
} PauliStringC;

// Pauli String function declarations
PauliStringC *pauli_string_init_as_chars_c(unsigned int N, double complex coef,
                                           char paulis[]);
PauliStringC *pauli_string_init_as_ints_c(unsigned int N, double complex coef,
                                          unsigned int paulis[]);
void free_pauli_string_c(PauliStringC *pString);
char *pauli_string_to_string_no_coef_c(PauliStringC *pString);
char *pauli_string_to_string_c(PauliStringC *pString);
PauliStringC *pauli_string_scalar_multiplication_c(PauliStringC *pString,
                                                   double complex scalar);
PauliStringC *pauli_string_adjoint_c(PauliStringC *pString);
double pauli_string_comparison_c(PauliStringC *left, PauliStringC *right);
PauliStringC *pauli_string_multiplication_c(PauliStringC *left,
                                            PauliStringC *right);

// Struct for pauli string khash hashmap
KHASH_MAP_INIT_INT64(pauli_hash, PauliStringC *)

// Struct for pauli sum with multiple pauli strings
typedef struct PauliSumC {
  unsigned int N;                      // Number of qubits
  unsigned int p;                      // Current number of Pauli strings
  khash_t(pauli_hash) * pauli_strings; // Hashmap of Pauli string elements
} PauliSumC;

// Pauli sum function declarations
PauliSumC *pauli_sum_init_c(unsigned int N);
void free_pauli_sum_c(PauliSumC *pSum);
char *pauli_sum_to_string_c(PauliSumC *pSum);
void pauli_sum_append_pauli_string_c(PauliSumC *pSum, PauliStringC *pString);
PauliSumC *pauli_sum_scalar_multiplication_c(PauliSumC *pSum,
                                             complex double scalar);
PauliSumC *pauli_sum_adjoint_c(PauliSumC *pSum);
PauliSumC *pauli_sum_multiplication_c(PauliSumC *left, PauliSumC *right);
PauliSumC *pauli_sum_addition_c(PauliSumC *left, PauliSumC *right);
PauliStringC **get_pauli_strings_c(PauliSumC *pSum);

static inline khiter_t kh_begin_pauli_hash(khash_t(pauli_hash) * h) {
  (void)h;
  return kh_begin(h);
}
static inline khiter_t kh_end_pauli_hash(khash_t(pauli_hash) * h) {
  (void)h;
  return kh_end(h);
}
static inline int kh_exist_pauli_hash(khash_t(pauli_hash) * h, khiter_t k) {
  return kh_exist(h, k);
}
static inline PauliStringC *kh_value_pauli_hash(khash_t(pauli_hash) * h,
                                                khiter_t k) {
  return kh_val(h, k);
}

#endif // PAULI_H
