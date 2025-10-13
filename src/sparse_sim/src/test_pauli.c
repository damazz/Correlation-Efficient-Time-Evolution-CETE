#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "pauli.h"

void test_pauli_string_initialization_scaling_freeing(void);
void test_pauli_string_multiplication(void);
void test_pauli_sum_initialization_scaling_freeing(void);
void test_pauli_sum_adjoint(void);
void test_pauli_sum_multiplication(void);
void test_pauli_sum_addition(void);
void test_pauli_encoding(unsigned int N);
void test_pauli_sum_get_pauli_strings(void);

void test_pauli_string_initialization_scaling_freeing(void) {
  unsigned int paulis0[] = {0, 1, 0, 1};
  char paulis1[] = {'I', 'X', 'I', 'X'};
  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliStringC *pString2;
  PauliStringC *pString3;
  char *pString0_str;
  char *pString1_str;
  char *pString2_str;
  char *pString3_str;

  pString0 = pauli_string_init_as_ints_c(4, (double complex)1.0, paulis0);
  pString1 = pauli_string_init_as_chars_c(4, (double complex)I, paulis1);
  pString0_str = pauli_string_to_string_c(pString0);
  printf("%s with encoding %llu\n", pString0_str, pString0->encoding);

  pString1_str = pauli_string_to_string_c(pString1);

  pString2 = pauli_string_scalar_multiplication_c(pString1, 2);
  pString2_str = pauli_string_to_string_c(pString2);
  printf("2 * %s = %s\n", pString1_str, pString2_str);

  pString3 = pauli_string_adjoint_c(pString2);
  pString3_str = pauli_string_to_string_c(pString3);
  printf("Adjoint of %s = %s\n", pString2_str, pString3_str);

  free(pString0_str);
  free(pString1_str);
  free(pString2_str);
  free(pString3_str);

  free_pauli_string_c(pString0);
  free_pauli_string_c(pString1);
  free_pauli_string_c(pString2);
  free_pauli_string_c(pString3);
}

void test_pauli_string_multiplication(void) {
  unsigned int paulis0[] = {0, 2, 0, 1};
  char paulis1[] = {'I', 'X', 'I', 'I'};
  char *pString0_str;
  char *pString1_str;
  char *pString2_str;
  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliStringC *pString2;

  pString0 = pauli_string_init_as_ints_c(4, (double complex)1, paulis0);
  pString1 = pauli_string_init_as_chars_c(4, (double complex)I, paulis1);
  pString2 = pauli_string_multiplication_c(pString0, pString1);

  pString0_str = pauli_string_to_string_c(pString0);
  pString1_str = pauli_string_to_string_c(pString1);
  pString2_str = pauli_string_to_string_c(pString2);

  printf("%s * %s = %s\n", pString0_str, pString1_str, pString2_str);

  free(pString0_str);
  free(pString1_str);
  free(pString2_str);
  free_pauli_string_c(pString0);
  free_pauli_string_c(pString1);
  free_pauli_string_c(pString2);
}

void test_pauli_sum_initialization_scaling_freeing(void) {
  unsigned int paulis0[] = {0, 2, 0, 1};
  char paulis1[] = {'I', 'X', 'I', 'I'};
  char *pSum_str;
  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliSumC *pSum;

  pString0 = pauli_string_init_as_ints_c(4, (double complex)1, paulis0);
  pString1 = pauli_string_init_as_chars_c(4, (double complex)I, paulis1);

  pSum = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum, pString0);
  pauli_sum_append_pauli_string_c(pSum, pString1);

  pSum_str = pauli_sum_to_string_c(pSum);
  printf("%s contains %u pStrings of 2\n", pSum_str, pSum->p);

  free(pSum_str);
  free_pauli_sum_c(pSum);
}

void test_pauli_sum_adjoint(void) {
  unsigned int paulis0[] = {0, 2, 0, 1};
  unsigned int paulis2[] = {0, 2, 1, 1};
  char paulis1[] = {'I', 'X', 'X', 'I'};
  char paulis3[] = {'I', 'X', 'I', 'I'};

  char *pSum0_str;
  char *pSum1_str;

  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliStringC *pString2;
  PauliStringC *pString3;

  PauliSumC *pSum0;
  PauliSumC *pSum1;

  pString0 = pauli_string_init_as_ints_c(4, (double complex)1, paulis0);
  pString1 = pauli_string_init_as_chars_c(4, (double complex)I, paulis1);
  pString2 = pauli_string_init_as_ints_c(4, (double complex)1, paulis2);
  pString3 = pauli_string_init_as_chars_c(4, (double complex) - I, paulis3);

  pSum0 = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum0, pString0);
  pauli_sum_append_pauli_string_c(pSum0, pString1);
  pauli_sum_append_pauli_string_c(pSum0, pString2);
  pauli_sum_append_pauli_string_c(pSum0, pString3);

  pSum1 = pauli_sum_adjoint_c(pSum0);

  pSum0_str = pauli_sum_to_string_c(pSum0);
  pSum1_str = pauli_sum_to_string_c(pSum1);

  printf("(%s)^dagger = (%s)\n", pSum0_str, pSum1_str);

  free(pSum0_str);
  free(pSum1_str);

  free_pauli_sum_c(pSum0);
  free_pauli_sum_c(pSum1);
}

void test_pauli_sum_multiplication(void) {
  unsigned int paulis0[] = {0, 2, 0, 1};
  unsigned int paulis2[] = {0, 2, 0, 1};
  char paulis1[] = {'I', 'X', 'I', 'I'};
  char paulis3[] = {'I', 'X', 'I', 'I'};

  char *pSum0_str;
  char *pSum1_str;
  char *pSum2_str;

  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliStringC *pString2;
  PauliStringC *pString3;

  PauliSumC *pSum0;
  PauliSumC *pSum1;
  PauliSumC *pSum2;

  pString0 = pauli_string_init_as_ints_c(4, (double complex)1, paulis0);
  pString1 = pauli_string_init_as_chars_c(4, (double complex)I, paulis1);
  pString2 = pauli_string_init_as_ints_c(4, (double complex)1, paulis2);
  pString3 = pauli_string_init_as_chars_c(4, (double complex)I, paulis3);

  pSum0 = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum0, pString0);
  pauli_sum_append_pauli_string_c(pSum0, pString1);

  pSum1 = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum1, pString2);
  pauli_sum_append_pauli_string_c(pSum1, pString3);

  pSum2 = pauli_sum_multiplication_c(pSum0, pSum1);

  pSum0_str = pauli_sum_to_string_c(pSum0);
  pSum1_str = pauli_sum_to_string_c(pSum1);
  pSum2_str = pauli_sum_to_string_c(pSum2);

  printf("(%s) * (%s) = (%s)\n", pSum0_str, pSum1_str, pSum2_str);

  free(pSum0_str);
  free(pSum1_str);
  free(pSum2_str);

  free_pauli_sum_c(pSum0);
  free_pauli_sum_c(pSum1);
  free_pauli_sum_c(pSum2);
}

void test_pauli_sum_addition(void) {
  unsigned int paulis0[] = {0, 2, 0, 1};
  unsigned int paulis2[] = {0, 2, 0, 1};
  char paulis1[] = {'I', 'X', 'I', 'I'};
  char paulis3[] = {'I', 'X', 'I', 'I'};

  char *pSum0_str;
  char *pSum1_str;
  char *pSum2_str;

  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliStringC *pString2;
  PauliStringC *pString3;

  PauliSumC *pSum0;
  PauliSumC *pSum1;
  PauliSumC *pSum2;

  pString0 = pauli_string_init_as_ints_c(4, (double complex)1, paulis0);
  pString1 = pauli_string_init_as_chars_c(4, (double complex)I, paulis1);
  pString2 = pauli_string_init_as_ints_c(4, (double complex)1, paulis2);
  pString3 = pauli_string_init_as_chars_c(4, (double complex) - I, paulis3);

  pSum0 = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum0, pString0);
  pauli_sum_append_pauli_string_c(pSum0, pString1);

  pSum1 = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum1, pString2);
  pauli_sum_append_pauli_string_c(pSum1, pString3);

  pSum2 = pauli_sum_addition_c(pSum0, pSum1);

  pSum0_str = pauli_sum_to_string_c(pSum0);
  pSum1_str = pauli_sum_to_string_c(pSum1);
  pSum2_str = pauli_sum_to_string_c(pSum2);

  printf("(%s) + (%s) = (%s)\n", pSum0_str, pSum1_str, pSum2_str);

  free(pSum0_str);
  free(pSum1_str);
  free(pSum2_str);

  free_pauli_sum_c(pSum0);
  free_pauli_sum_c(pSum1);
  free_pauli_sum_c(pSum2);
}

void test_pauli_encoding(unsigned int N) {
  uint64_t total_strings = 1ULL << (2 * N);
  char pauli_chars[] = {'I', 'X', 'Y', 'Z'};

  printf("Testing %llu possible Pauli strings for %u qubits...\n",
         total_strings, N);

  char *paulis = (char *)malloc(N * sizeof(char));
  if (!paulis) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return;
  }

  int pass = 1;
  for (uint64_t i = 0; i < total_strings; i++) {
    uint64_t value = i;
    for (unsigned int j = 0; j < N; j++) {
      paulis[N - j - 1] = pauli_chars[value % 4];
      value /= 4;
    }

    PauliStringC *pString =
        pauli_string_init_as_chars_c(N, 1.0 + 0.0 * (double complex)I, paulis);
    if (!pString) {
      fprintf(stderr, "Error: Failed to create Pauli string.\n");
      continue;
    }

    if (pString->encoding != i) {
      printf("Incorrect encoding at %llu != %llu!\n", pString->encoding, i);
      char *p_str = pauli_string_to_string_c(pString);
      printf("%s\n", p_str);
      free(p_str);
      pass = 0;
    }

    free_pauli_string_c(pString);
    if (!pass) {
      break;
    }
  }
  free(paulis);
  if (pass) {
    printf("Pauli encoding test completed successfully for %u qubits.\n", N);
  }
}

void test_pauli_sum_get_pauli_strings(void) {
  unsigned int paulis0[] = {3, 2, 0, 1};
  unsigned int paulis2[] = {2, 2, 0, 1};
  char paulis1[] = {'I', 'X', 'Y', 'I'};
  char paulis3[] = {'I', 'X', 'I', 'I'};

  char *pSum0_str;

  PauliStringC *pString0;
  PauliStringC *pString1;
  PauliStringC *pString2;
  PauliStringC *pString3;

  PauliSumC *pSum0;

  pString0 = pauli_string_init_as_ints_c(4, (double complex)1, paulis0);
  pString1 = pauli_string_init_as_chars_c(4, (double complex)I, paulis1);
  pString2 = pauli_string_init_as_ints_c(4, (double complex)1, paulis2);
  pString3 = pauli_string_init_as_chars_c(4, (double complex) - I, paulis3);

  pSum0 = pauli_sum_init_c(4);
  pauli_sum_append_pauli_string_c(pSum0, pString0);
  pauli_sum_append_pauli_string_c(pSum0, pString1);
  pauli_sum_append_pauli_string_c(pSum0, pString2);
  pauli_sum_append_pauli_string_c(pSum0, pString3);

  PauliStringC **pauli_strings = get_pauli_strings_c(pSum0);
  if (!pauli_strings) {
    fprintf(stderr, "Error: Failed to retrieve Pauli strings.\n");
    return;
  }

  pSum0_str = pauli_sum_to_string_c(pSum0);
  printf("%s contains %u Pauli strings:\n", pSum0_str, pSum0->p);
  free(pSum0_str);

  for (unsigned int i = 0; i < pSum0->p; i++) {
    char *p_str = pauli_string_to_string_c(pauli_strings[i]);
    printf("Pauli string %u: %s\n", i, p_str);
    free(p_str);
  }

  free(pauli_strings);
  free_pauli_sum_c(pSum0);
}

int main(void) {
  printf("\nTesting PauliString initializaiton, scaling, and freeing\n");
  test_pauli_string_initialization_scaling_freeing();
  printf("\nTesting PauliString Multiplication\n");
  test_pauli_string_multiplication();
  printf("\nTesting PauliSum initialization, scaling, and freeing\n");
  test_pauli_sum_initialization_scaling_freeing();
  printf("\nTesting PauliSum Adjoint\n");
  test_pauli_sum_adjoint();
  printf("\nTesting PauliSum Multiplication\n");
  test_pauli_sum_multiplication();
  printf("\nTesting PauliSum Addition\n");
  test_pauli_sum_addition();
  // printf("\nTesting Pauli String Encoding\n");
  // test_pauli_encoding(5);
  printf("\nTesting Pauli Sum Retrieval\n");
  test_pauli_sum_get_pauli_strings();
}
