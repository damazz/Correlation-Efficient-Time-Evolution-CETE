#include "pauli.h"

// Deprecated

PauliStringC *pauli_string_init_as_chars_c(unsigned int N, double complex coef,
                                           char paulis[]) {
  PauliStringC *pString = (PauliStringC *)malloc(sizeof(PauliStringC));
  pString->N = N;
  pString->coef = coef;

  pString->paulis = (unsigned int *)malloc(N * sizeof(unsigned int));
  pString->encoding = 0;
  for (unsigned int i = 0; i < N; i++) {
    switch (paulis[i]) {
    case 'I':
      pString->paulis[i] = 0;
      break;
    case 'X':
      pString->paulis[i] = 1;
      break;
    case 'Y':
      pString->paulis[i] = 2;
      break;
    case 'Z':
      pString->paulis[i] = 3;
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator '%c'.\n", paulis[i]);
      free(pString->paulis);
      free(pString);
      return NULL;
    }
    pString->encoding =
        pString->encoding * (uint64_t)4 + (uint64_t)pString->paulis[i];
  }
  return pString;
}

PauliStringC *pauli_string_init_as_ints_c(unsigned int N, double complex coef,
                                          unsigned int paulis[]) {
  PauliStringC *pString = (PauliStringC *)malloc(sizeof(PauliStringC));
  pString->N = N;
  pString->coef = coef;

  pString->paulis = (unsigned int *)malloc(N * sizeof(unsigned int));
  pString->encoding = 0;
  for (unsigned int i = 0; i < N; i++) {
    pString->paulis[i] = paulis[i];
    pString->encoding =
        pString->encoding * (uint64_t)4 + (uint64_t)pString->paulis[i];
  }
  return pString;
}

void free_pauli_string_c(PauliStringC *pString) {
  free(pString->paulis);
  free(pString);
}

char *pauli_string_to_string_no_coef_c(PauliStringC *pString) {

  size_t buffer_size = pString->N // for paulis
                       + 1;       // for terminator

  // Allocate memory for the buffer
  char *buffer = (char *)malloc(buffer_size);
  if (!buffer) return NULL;

  char pauli_as_char[] = {'I', 'X', 'Y', 'Z'};

  // Append the Pauli operators
  for (unsigned int i = 0; i < pString->N; i++) {
    buffer[i] = pauli_as_char[pString->paulis[i]];
  }
  buffer[pString->N] = '\0';

  return buffer;
}

char *pauli_string_to_string_c(PauliStringC *pString) {

  size_t buffer_size = 40           // for coef
                       + pString->N // for paulis
                       + 1;         // for terminator

  // Allocate memory for the buffer
  char *buffer = (char *)malloc(buffer_size);

  char pauli_as_char[] = {'I', 'X', 'Y', 'Z'};

  // Write the coefficient part to the buffer
  snprintf(buffer, buffer_size, "(%.4lf + %.4lfi)", creal(pString->coef),
           cimag(pString->coef));

  // Append the Pauli operators
  for (unsigned int i = 0; i < pString->N; i++) {
    char pauli_char[2]; // To hold a single Pauli operator (1 digit + null
                        // terminator)
    pauli_char[0] = pauli_as_char[pString->paulis[i]];
    pauli_char[1] = '\0';
    strcat(buffer, pauli_char);
  }

  return buffer;
}

PauliStringC *pauli_string_scalar_multiplication_c(PauliStringC *pString,
                                                   double complex scalar) {
  PauliStringC *new_pString = pauli_string_init_as_ints_c(
      pString->N, pString->coef * scalar, pString->paulis);
  return new_pString;
}

PauliStringC *pauli_string_adjoint_c(PauliStringC *pString) {
  double complex coef = pString->coef;
  PauliStringC *new_pString = pauli_string_init_as_ints_c(
      pString->N,
      (double complex)(creal(coef) + ((double complex) - I) * cimag(coef)),
      pString->paulis);
  return new_pString;
}

double pauli_string_comparison_c(PauliStringC *left, PauliStringC *right) {
  double comparison = 1;

  if (left->N != right->N) {
    fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
    return 0;
  }

  if (left->encoding != right->encoding) {
    comparison = 0;
  }
  return comparison;
}

PauliStringC *pauli_string_multiplication_c(PauliStringC *left,
                                            PauliStringC *right) {
  double complex coef;
  unsigned int N;
  unsigned int caylay_pauli[4][4] = {
      {0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};

  double complex caylay_coef[4][4] = {
      {1.0, 1.0, 1.0, 1.0},
      {1.0, 1, (double complex)I, (double complex) - I},
      {1.0, (double complex) - I, 1.0, (double complex)I},
      {1.0, (double complex)I, (double complex) - I, 1.0}};

  unsigned int *paulis;

  PauliStringC *product;

  if (left->N != right->N) {
    fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
    return NULL;
  }

  coef = left->coef * right->coef;

  N = left->N;

  paulis = (unsigned int *)malloc(N * sizeof(unsigned int));

  for (unsigned int i = 0; i < N; i++) {
    paulis[i] = caylay_pauli[left->paulis[i]][right->paulis[i]];
    coef = coef * caylay_coef[left->paulis[i]][right->paulis[i]];
  }

  product = pauli_string_init_as_ints_c(N, coef, paulis);
  free(paulis);
  return product;
}

PauliSumC *pauli_sum_init_c(unsigned int N) {
  PauliSumC *pSum = (PauliSumC *)malloc(sizeof(PauliSumC));
  pSum->N = N;
  pSum->p = 0;
  pSum->pauli_strings = kh_init(pauli_hash);
  return pSum;
}

void free_pauli_sum_c(PauliSumC *pSum) {
  if (pSum == NULL)
    return;

  khiter_t k;
  for (k = kh_begin(pSum->pauli_strings); k != kh_end(pSum->pauli_strings);
       ++k) {
    if (kh_exist(pSum->pauli_strings, k)) {
      free_pauli_string_c((PauliStringC *)kh_value(pSum->pauli_strings, k));
    }
  }

  kh_destroy(pauli_hash, pSum->pauli_strings);
  free(pSum);
}

char *pauli_sum_to_string_c(PauliSumC *pSum) {
  if (!pSum) {
    fprintf(stderr, "Error: Received NULL PauliSum.\n");
    return NULL;
  }

  if (pSum->p == 0) {
    char *empty_str = (char *)malloc(1);
    if (!empty_str) {
      fprintf(stderr, "Error: Memory allocation failed.\n");
      return NULL;
    }
    empty_str[0] = '\0';
    return empty_str;
  }

  size_t buffer_size = 1; // Space for null terminator
  char **pauli_strings = (char **)malloc(pSum->p * sizeof(char *));
  if (!pauli_strings) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return NULL;
  }

  // Iterate through the hash table
  khiter_t k;
  unsigned int index = 0;
  for (k = kh_begin(pSum->pauli_strings); k != kh_end(pSum->pauli_strings);
       ++k) {
    if (kh_exist(pSum->pauli_strings, k)) {
      PauliStringC *pString = (PauliStringC *)kh_value(pSum->pauli_strings, k);
      pauli_strings[index] = pauli_string_to_string_c(pString);
      if (!pauli_strings[index]) {
        fprintf(stderr, "Error: Failed to allocate memory for Pauli string.\n");
        // Free previously allocated strings before exiting
        for (unsigned int j = 0; j < index; j++) {
          free(pauli_strings[j]);
        }
        free(pauli_strings);
        return NULL;
      }
      buffer_size += strlen(pauli_strings[index]) + 3; // +3 for " + "
      index++;
    }
  }

  // Allocate buffer for the final string
  char *buffer = (char *)malloc(buffer_size);
  if (!buffer) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    for (unsigned int i = 0; i < index; i++) {
      free(pauli_strings[i]);
    }
    free(pauli_strings);
    return NULL;
  }

  // Initialize the buffer with the first Pauli string
  snprintf(buffer, buffer_size, "%s", pauli_strings[0]);

  // Append remaining Pauli strings
  for (unsigned int i = 1; i < index; i++) {
    strncat(buffer, " + ", buffer_size - strlen(buffer) - 1);
    strncat(buffer, pauli_strings[i], buffer_size - strlen(buffer) - 1);
  }

  // Free intermediate Pauli strings
  for (unsigned int i = 0; i < index; i++) {
    free(pauli_strings[i]);
  }
  free(pauli_strings);

  return buffer;
}

void pauli_sum_append_pauli_string_c(PauliSumC *pSum, PauliStringC *pString) {
  int ret;
  khiter_t k = kh_put(pauli_hash, pSum->pauli_strings, pString->encoding, &ret);

  if (ret == 0) {
    PauliStringC *existing_pString =
        (PauliStringC *)kh_value(pSum->pauli_strings, k);
    existing_pString->coef += pString->coef;

    if (fabs(creal(existing_pString->coef)) < 1e-12 &&
        fabs(cimag(existing_pString->coef)) < 1e-12) {
      kh_del(pauli_hash, pSum->pauli_strings, k);
      free_pauli_string_c(existing_pString);
      pSum->p--;
    }

    free_pauli_string_c(pString);
  } else {
    kh_value(pSum->pauli_strings, k) = pString;
    pSum->p++;
  }
}

PauliSumC *pauli_sum_scalar_multiplication_c(PauliSumC *pSum,
                                             double complex scalar) {
  if (!pSum) {
    fprintf(stderr, "Error: NULL PauliSum input to scalar multiplication.\n");
    return NULL;
  }

  PauliSumC *new_pSum = pauli_sum_init_c(pSum->N);
  if (!new_pSum) {
    fprintf(stderr, "Error: Failed to allocate new PauliSum.\n");
    return NULL;
  }

  khiter_t k;
  for (k = kh_begin(pSum->pauli_strings); k != kh_end(pSum->pauli_strings);
       ++k) {
    if (kh_exist(pSum->pauli_strings, k)) {
      PauliStringC *original_pString =
          (PauliStringC *)kh_value(pSum->pauli_strings, k);
      PauliStringC *scaled_pString =
          pauli_string_scalar_multiplication_c(original_pString, scalar);
      if (!scaled_pString) {
        fprintf(stderr,
                "Error: Scalar multiplication failed for Pauli string.\n");
        free_pauli_sum_c(new_pSum);
        return NULL;
      }
      pauli_sum_append_pauli_string_c(new_pSum, scaled_pString);
    }
  }

  return new_pSum;
}

PauliSumC *pauli_sum_adjoint_c(PauliSumC *pSum) {
  if (!pSum) {
    fprintf(stderr, "Error: NULL PauliSum input to adjoint.\n");
    return NULL;
  }

  PauliSumC *new_pSum = pauli_sum_init_c(pSum->N);
  if (!new_pSum) {
    fprintf(stderr, "Error: Failed to allocate new PauliSum.\n");
    return NULL;
  }

  khiter_t k;
  for (k = kh_begin(pSum->pauli_strings); k != kh_end(pSum->pauli_strings);
       ++k) {
    if (kh_exist(pSum->pauli_strings, k)) {
      PauliStringC *original_pString =
          (PauliStringC *)kh_value(pSum->pauli_strings, k);
      PauliStringC *adjoint_pString = pauli_string_adjoint_c(original_pString);
      if (!adjoint_pString) {
        fprintf(stderr,
                "Error: Adjoint computation failed for Pauli string.\n");
        free_pauli_sum_c(new_pSum);
        return NULL;
      }
      pauli_sum_append_pauli_string_c(new_pSum, adjoint_pString);
    }
  }

  return new_pSum;
}

PauliSumC *pauli_sum_multiplication_c(PauliSumC *left, PauliSumC *right) {
  if (!left || !right) {
    fprintf(stderr, "Error: NULL PauliSum input to multiplication.\n");
    return NULL;
  }
  if (left->N != right->N) {
    fprintf(stderr, "Error: Pauli sums have different number of qubits.\n");
    return NULL;
  }

  PauliSumC *new_pSum = pauli_sum_init_c(left->N);
  if (!new_pSum) {
    fprintf(stderr, "Error: Failed to allocate new PauliSum.\n");
    return NULL;
  }

  khiter_t k1, k2;
  for (k1 = kh_begin(left->pauli_strings); k1 != kh_end(left->pauli_strings);
       ++k1) {
    if (kh_exist(left->pauli_strings, k1)) {
      PauliStringC *left_pString =
          (PauliStringC *)kh_value(left->pauli_strings, k1);

      for (k2 = kh_begin(right->pauli_strings);
           k2 != kh_end(right->pauli_strings); ++k2) {
        if (kh_exist(right->pauli_strings, k2)) {
          PauliStringC *right_pString =
              (PauliStringC *)kh_value(right->pauli_strings, k2);

          PauliStringC *product_pString =
              pauli_string_multiplication_c(left_pString, right_pString);
          if (!product_pString) {
            fprintf(stderr,
                    "Error: Multiplication failed for Pauli strings.\n");
            free_pauli_sum_c(new_pSum);
            return NULL;
          }
          pauli_sum_append_pauli_string_c(new_pSum, product_pString);
        }
      }
    }
  }

  return new_pSum;
}

PauliSumC *pauli_sum_addition_c(PauliSumC *left, PauliSumC *right) {

  if (!left || !right) {
    fprintf(stderr, "Error: NULL PauliSum input to multiplication.\n");
    return NULL;
  }
  if (left->N != right->N) {
    fprintf(stderr, "Error: Pauli sums have different number of qubits.\n");
    return NULL;
  }

  PauliSumC *new_pSum = pauli_sum_init_c(left->N);
  if (!new_pSum) {
    fprintf(stderr, "Error: Failed to allocate new PauliSum.\n");
    return NULL;
  }

  khiter_t k;
  for (k = kh_begin(left->pauli_strings); k != kh_end(left->pauli_strings);
       ++k) {
    if (kh_exist(left->pauli_strings, k)) {
      PauliStringC *pString = (PauliStringC *)kh_value(left->pauli_strings, k);
      PauliStringC *pString_copy = pauli_string_init_as_ints_c(
          pString->N, pString->coef, pString->paulis);
      pauli_sum_append_pauli_string_c(new_pSum, pString_copy);
    }
  }

  for (k = kh_begin(right->pauli_strings); k != kh_end(right->pauli_strings);
       ++k) {
    if (kh_exist(right->pauli_strings, k)) {
      PauliStringC *pString = (PauliStringC *)kh_value(right->pauli_strings, k);
      PauliStringC *pString_copy = pauli_string_init_as_ints_c(
          pString->N, pString->coef, pString->paulis);
      pauli_sum_append_pauli_string_c(new_pSum, pString_copy);
    }
  }

  return new_pSum;
}

PauliStringC **get_pauli_strings_c(PauliSumC *pSum) {
  if (!pSum) {
    fprintf(stderr, "Error: Received NULL PauliSum.\n");
    return NULL;
  }

  PauliStringC **pauli_strings =
      (PauliStringC **)malloc(pSum->p * sizeof(PauliStringC *));
  if (!pauli_strings) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return NULL;
  }

  khiter_t k;
  unsigned int index = 0;
  for (k = kh_begin(pSum->pauli_strings); k != kh_end(pSum->pauli_strings);
       ++k) {
    if (kh_exist(pSum->pauli_strings, k)) {
      PauliStringC *pString = kh_value(pSum->pauli_strings, k);
      PauliStringC *new_pString = pauli_string_init_as_ints_c(
          pString->N, pString->coef, pString->paulis);
      pauli_strings[index++] = new_pString;
    }
  }

  return pauli_strings;
}
