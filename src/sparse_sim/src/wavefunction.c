#include "wavefunction.h"

void free_slater_determinant_c(SlaterDeterminantC *sdet) {
  free(sdet->orbitals);
  free(sdet);
}

SlaterDeterminantC *slater_determinant_init_c(const unsigned int N,
                                              double complex coef,
                                              unsigned int orbitals[]) {
  SlaterDeterminantC *sdet;
  unsigned int encoding;
  unsigned int i;

  sdet = (SlaterDeterminantC *)malloc(sizeof(SlaterDeterminantC));
  if (!sdet) {
    fprintf(stderr, "Malloc failed for SlaterDeterminant\n");
    return NULL;
  }

  sdet->N = N;
  sdet->coef = coef;

  sdet->orbitals = (unsigned int *)malloc(N * sizeof(unsigned int));
  if (!sdet->orbitals) {
    fprintf(stderr, "Malloc failed for orbitals\n");
    free(sdet);
    return NULL;
  }

  encoding = 0;
  for (i = 0; i < N; i++) {
    sdet->orbitals[i] = orbitals[i];
    if (orbitals[i]) {
      encoding += 1 << i;
    }
  }
  sdet->encoding = encoding;

  return sdet;
}

char *slater_determinant_to_string_c(SlaterDeterminantC *sdet,
                                     char bra_or_ket) {
  size_t buffer_size;
  char *buffer;
  unsigned int i;
  char orbital_char[2]; // To hold a single orbital (1 digit + null terminator)

  // Calculate buffer size
  buffer_size = 40        // for coef
                + 1       // for |
                + sdet->N // for orbitals
                + 1       // for >
                + 1;      // for terminator

  // Allocate memory for the buffer
  buffer = (char *)malloc(buffer_size);
  if (!buffer) {
    fprintf(stderr, "Malloc failed for slater_determinant_to_string_c\n");
    return NULL;
  }

  // Write the coefficient part to the buffer
  snprintf(buffer, buffer_size, "(%.4lf + %.4lfi)", creal(sdet->coef),
           cimag(sdet->coef));

  // Append the `|` character
  if (bra_or_ket == 'b') {
    strcat(buffer, "<");
  } else if (bra_or_ket == 'k') {
    strcat(buffer, "|");
  } else {
    fprintf(stderr, "Error: Invalid bra_or_ket character.\n");
    free(buffer);
    return NULL;
  }

  // Append the orbital occupations
  for (i = 0; i < sdet->N; i++) {
    snprintf(orbital_char, sizeof(orbital_char), "%u", sdet->orbitals[i]);
    strcat(buffer, orbital_char);
  }

  // Append the `>` character
  if (bra_or_ket == 'b') {
    strcat(buffer, "|");
  } else if (bra_or_ket == 'k') {
    strcat(buffer, ">");
  }

  return buffer;
}

SlaterDeterminantC *
slater_determinant_scalar_multiplication_c(SlaterDeterminantC *sdet,
                                           double complex scalar) {
  SlaterDeterminantC *new_sdet =
      slater_determinant_init_c(sdet->N, sdet->coef * scalar, sdet->orbitals);
  return new_sdet;
}

SlaterDeterminantC *slater_determinant_adjoint_c(SlaterDeterminantC *sdet) {
  double complex coef = sdet->coef;
  SlaterDeterminantC *new_sdet = slater_determinant_init_c(
      sdet->N,
      (double complex)(creal(coef) + -1 * cimag(coef) * (double complex)I),
      sdet->orbitals);
  return new_sdet;
}

double slater_determinant_comparison_c(SlaterDeterminantC *bra,
                                       SlaterDeterminantC *ket) {
  if (bra->N != ket->N) {
    fprintf(stderr,
            "Error: Slater determinants have different number of orbitals.\n");
    return 0;
  }
  return (double)(bra->encoding == ket->encoding);
}

double complex slater_dermininant_multiplication_c(SlaterDeterminantC *bra,
                                                   SlaterDeterminantC *ket) {
  return slater_determinant_comparison_c(bra, ket) * bra->coef * ket->coef;
}

SlaterDeterminantC *
slater_determinant_pauli_string_multiplication_c(PauliStringC *pString,
                                                 SlaterDeterminantC *sdet) {

  unsigned int N;
  unsigned int *new_orbitals;
  double complex new_coef;
  SlaterDeterminantC *new_sdet;

  if (pString->N != sdet->N) {
    fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
    return NULL;
  }

  N = sdet->N;
  new_orbitals = (unsigned int *)malloc(sdet->N * sizeof(int));
  new_coef = pString->coef * sdet->coef;

  for (unsigned int i = 0; i < N; i++) {
    switch (pString->paulis[i]) {
    case 0: // I
      new_orbitals[i] = sdet->orbitals[i];
      break;
    case 1: // X
      new_orbitals[i] = sdet->orbitals[i] ^ 1;
      break;
    case 2: // Y
      new_orbitals[i] = sdet->orbitals[i] ^ 1;
      new_coef =
          new_coef * ((double complex)I) * (1 - (int)(sdet->orbitals[i] << 1));
      break;
    case 3: // Z
      new_orbitals[i] = sdet->orbitals[i];
      new_coef = new_coef * (1 - (int)(sdet->orbitals[i] << 1));
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator '%u'.\n",
              pString->paulis[i]);
      return NULL;
    }
  }
  new_sdet = slater_determinant_init_c(N, new_coef, new_orbitals);
  free(new_orbitals);
  return new_sdet;
}

void free_wavefunction_c(WavefunctionC *wfn) {
  if (wfn == NULL)
    return;

  khiter_t k;
  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      free_slater_determinant_c(
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k));
    }
  }

  kh_destroy(slater_hash, wfn->slater_determinants);
  free(wfn);
}

WavefunctionC *wavefunction_init_c(unsigned int N) {
  WavefunctionC *wfn = (WavefunctionC *)malloc(sizeof(WavefunctionC));
  wfn->s = 0;
  wfn->slater_determinants = kh_init(slater_hash);
  wfn->N = N;
  return wfn;
}

void wavefunction_append_slater_determinant_c(WavefunctionC *wfn,
                                              SlaterDeterminantC *sdet) {
  int ret;
  khiter_t k =
      kh_put(slater_hash, wfn->slater_determinants, sdet->encoding, &ret);

  if (ret == 0) {
    SlaterDeterminantC *existing_sdet =
        (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
    existing_sdet->coef += sdet->coef;

    if (fabs(creal(existing_sdet->coef)) < 1e-8 &&
        fabs(cimag(existing_sdet->coef)) < 1e-8) {
      kh_del(slater_hash, wfn->slater_determinants, k);
      free_slater_determinant_c(existing_sdet);
      wfn->s--;
    }

    free_slater_determinant_c(sdet);
  } else {
    kh_value(wfn->slater_determinants, k) = sdet;
    wfn->s++;
  }
}

double wavefunction_norm_c(WavefunctionC *wfn) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return 0.0;
  }

  double norm = 0.0;
  khiter_t k;

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      double complex coef = sdet->coef;
      norm += creal(coef) * creal(coef) + cimag(coef) * cimag(coef);
    }
  }

  return sqrt(norm);
}

WavefunctionC *wavefunction_scalar_multiplication_c(WavefunctionC *wfn,
                                                    double complex scalar) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return NULL;
  }

  WavefunctionC *new_wfn = wavefunction_init_c(wfn->N);
  khiter_t k;

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      SlaterDeterminantC *new_sdet =
          slater_determinant_scalar_multiplication_c(sdet, scalar);

      int ret;
      khiter_t new_k = kh_put(slater_hash, new_wfn->slater_determinants,
                              new_sdet->encoding, &ret);
      kh_value(new_wfn->slater_determinants, new_k) = new_sdet;
      new_wfn->s++;
    }
  }

  return new_wfn;
}

WavefunctionC *wavefunction_adjoint_c(WavefunctionC *wfn) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return NULL;
  }

  WavefunctionC *new_wfn = wavefunction_init_c(wfn->N);
  khiter_t k;

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      SlaterDeterminantC *new_sdet = slater_determinant_adjoint_c(sdet);

      int ret;
      khiter_t new_k = kh_put(slater_hash, new_wfn->slater_determinants,
                              new_sdet->encoding, &ret);
      kh_value(new_wfn->slater_determinants, new_k) = new_sdet;
      new_wfn->s++;
    }
  }

  return new_wfn;
}

double complex wavefunction_multiplication_c(WavefunctionC *bra,
                                             WavefunctionC *ket) {
  if (!bra || !ket) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return 0.0 + 0.0 * (double complex)I;
  }

  double complex product = 0.0 + 0.0 * (double complex)I;
  khiter_t k1, k2;

  for (k1 = kh_begin(bra->slater_determinants);
       k1 != kh_end(bra->slater_determinants); ++k1) {
    if (kh_exist(bra->slater_determinants, k1)) {
      SlaterDeterminantC *sdet_bra =
          (SlaterDeterminantC *)kh_value(bra->slater_determinants, k1);
      unsigned int encoding = sdet_bra->encoding;

      k2 = kh_get(slater_hash, ket->slater_determinants, encoding);
      if (k2 != kh_end(ket->slater_determinants)) {
        SlaterDeterminantC *sdet_ket =
            (SlaterDeterminantC *)kh_value(ket->slater_determinants, k2);
        product += slater_dermininant_multiplication_c(sdet_bra, sdet_ket);
      }
    }
  }

  return product;
}

char *wavefunction_to_string_c(WavefunctionC *wfn, char bra_or_ket) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return NULL;
  }

  if (wfn->s == 0) {
    char *empty_str = (char *)malloc(1);
    if (!empty_str)
      return NULL;
    empty_str[0] = '\0';
    return empty_str;
  }

  size_t buffer_size = 1; // Start with space for null terminator
  char *buffer = (char *)malloc(buffer_size);
  if (!buffer) {
    fprintf(stderr, "Error: Memory allocation failed.\n");
    return NULL;
  }
  buffer[0] = '\0';

  khiter_t k;
  int first_entry = 1; // Flag to track first element (to avoid extra " + ")

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      char *sdet_str = slater_determinant_to_string_c(sdet, bra_or_ket);
      if (!sdet_str) {
        fprintf(stderr, "Error: Failed to allocate memory for Slater "
                        "determinant string.\n");
        free(buffer);
        return NULL;
      }

      // Expand buffer size to accommodate new string
      size_t new_size = buffer_size + strlen(sdet_str) +
                        (first_entry ? 0 : 3); // Extra 3 for " + "
      char *new_buffer = (char *)realloc(buffer, new_size);
      if (!new_buffer) {
        fprintf(stderr, "Error: Memory reallocation failed.\n");
        free(buffer);
        free(sdet_str);
        return NULL;
      }
      buffer = new_buffer;
      buffer_size = new_size;

      // Append " + " if it's not the first entry
      if (!first_entry) {
        strcat(buffer, " + ");
      }
      strcat(buffer, sdet_str);

      free(sdet_str);
      first_entry = 0;
    }
  }

  return buffer;
}

WavefunctionC *wavefunction_pauli_string_multiplication_c(PauliStringC *pString,
                                                          WavefunctionC *wfn) {
  if (!wfn || !pString) {
    fprintf(stderr, "Error: Received NULL wavefunction or Pauli string.\n");
    return NULL;
  }
  if (wfn->N != pString->N) {
    fprintf(stderr, "Error: Wavefunction and Pauli string have different "
                    "number of qubits.\n");
    return NULL;
  }

  WavefunctionC *new_wfn = wavefunction_init_c(wfn->N);
  khiter_t k;

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      SlaterDeterminantC *new_sdet =
          slater_determinant_pauli_string_multiplication_c(pString, sdet);

      if (new_sdet) {
        wavefunction_append_slater_determinant_c(new_wfn, new_sdet);
      }
    }
  }

  return new_wfn;
}

WavefunctionC *wavefunction_pauli_sum_multiplication_c(PauliSumC *pSum,
                                                       WavefunctionC *wfn) {
  if (!wfn || !pSum) {
    fprintf(stderr, "Error: Received NULL wavefunction or Pauli sum.\n");
    return NULL;
  }
  if (wfn->N != pSum->N) {
    fprintf(stderr, "Error: Wavefunction and Pauli sum have different number "
                    "of qubits.\n");
    return NULL;
  }

  WavefunctionC *new_wfn = wavefunction_init_c(wfn->N);
  if (!new_wfn) {
    fprintf(stderr, "Error: Failed to allocate new wavefunction.\n");
    return NULL;
  }

  khiter_t kp, ks;
  int success = 0;

  for (kp = kh_begin(pSum->pauli_strings); kp != kh_end(pSum->pauli_strings);
       ++kp) {
    if (kh_exist(pSum->pauli_strings, kp)) {
      PauliStringC *pString = (PauliStringC *)kh_value(pSum->pauli_strings, kp);

      for (ks = kh_begin(wfn->slater_determinants);
           ks != kh_end(wfn->slater_determinants); ++ks) {
        if (kh_exist(wfn->slater_determinants, ks)) {
          SlaterDeterminantC *sdet =
              (SlaterDeterminantC *)kh_value(wfn->slater_determinants, ks);
          SlaterDeterminantC *new_sdet =
              slater_determinant_pauli_string_multiplication_c(pString, sdet);

          if (!new_sdet) {
            fprintf(stderr, "Error: Pauli string multiplication failed for a "
                            "Slater determinant.\n");
            continue;
          }

          wavefunction_append_slater_determinant_c(new_wfn, new_sdet);
          success = 1;
        }
      }
    }
  }

  if (!success) {
    fprintf(stderr, "Warning: No valid Slater determinants were processed.\n");
  }

  return new_wfn;
}

static inline SlaterDeterminantC *
evolution_helper_cosh(PauliStringC *pString, SlaterDeterminantC *sdet,
                      double complex epsilon) {
  unsigned int N = sdet->N;
  double complex x = pString->coef * epsilon;
  // double complex approx_cosh = 1 + x * x * (0.5 + x * x / 24.0);
  double complex new_coef = ccosh(x) * sdet->coef;
  return slater_determinant_init_c(N, new_coef, sdet->orbitals);
}

static inline SlaterDeterminantC *
evolution_helper_sinh(PauliStringC *pString, SlaterDeterminantC *sdet,
                      double complex epsilon) {

  unsigned int N;
  unsigned int *new_orbitals;
  double complex new_coef;
  SlaterDeterminantC *new_sdet;

  if (pString->N != sdet->N) {
    fprintf(stderr, "Error: Pauli strings have different number of qubits.\n");
    return NULL;
  }

  N = sdet->N;
  new_orbitals = (unsigned int *)malloc(sdet->N * sizeof(int));
  double complex x = pString->coef * epsilon;
  // double complex sinh = x * (1 + x * x * (1.0 / 6.0 + x * x / 120.0));
  new_coef = csinh(x) * sdet->coef;

  for (unsigned int i = 0; i < N; i++) {
    switch (pString->paulis[i]) {
    case 0: // I
      new_orbitals[i] = sdet->orbitals[i];
      break;
    case 1: // X
      new_orbitals[i] = sdet->orbitals[i] ^ 1;
      break;
    case 2: // Y
      new_orbitals[i] = sdet->orbitals[i] ^ 1;
      new_coef =
          new_coef * (double complex)I * (1 - (int)(sdet->orbitals[i] << 1));
      break;
    case 3: // Z
      new_orbitals[i] = sdet->orbitals[i];
      new_coef = new_coef * (1 - (int)(sdet->orbitals[i] << 1));
      break;
    default:
      fprintf(stderr, "Error: Invalid Pauli operator '%u'.\n",
              pString->paulis[i]);
      return NULL;
    }
  }
  new_sdet = slater_determinant_init_c(N, new_coef, new_orbitals);
  free(new_orbitals);
  return new_sdet;
}

WavefunctionC *wavefunction_pauli_string_evolution_c(PauliStringC *pString,
                                                     WavefunctionC *wfn,
                                                     double complex epsilon) {
  if (!wfn || !pString) {
    fprintf(stderr, "Error: Received NULL wavefunction or Pauli string.\n");
    return NULL;
  }
  if (wfn->N != pString->N) {
    fprintf(stderr, "Error: Wavefunction and Pauli string have different "
                    "number of qubits.\n");
    return NULL;
  }

  WavefunctionC *new_wfn = wavefunction_init_c(wfn->N);
  khiter_t k;

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);

      SlaterDeterminantC *new_sdet_cosh =
          evolution_helper_cosh(pString, sdet, epsilon);
      SlaterDeterminantC *new_sdet_sinh =
          evolution_helper_sinh(pString, sdet, epsilon);

      if (new_sdet_cosh) {
        wavefunction_append_slater_determinant_c(new_wfn, new_sdet_cosh);
      }
      if (new_sdet_sinh) {
        wavefunction_append_slater_determinant_c(new_wfn, new_sdet_sinh);
      }
    }
  }

  return new_wfn;
}

WavefunctionC *wavefunction_pauli_sum_evolution_c(PauliSumC *pSum,
                                                  WavefunctionC *wfn,
                                                  double complex epsilon) {
  if (!wfn || !pSum) {
    fprintf(stderr, "Error: Received NULL wavefunction or Pauli sum.\n");
    return NULL;
  }

  if (wfn->s == 0 || pSum->p == 0) {
    fprintf(stderr, "Error: Received empty wavefunction or Pauli sum.\n");
    return NULL;
  }

  WavefunctionC *new_wfn = NULL;
  khiter_t kp;
  int first_iteration = 1;

  for (kp = kh_begin(pSum->pauli_strings); kp != kh_end(pSum->pauli_strings);
       ++kp) {
    if (kh_exist(pSum->pauli_strings, kp)) {
      PauliStringC *pString = (PauliStringC *)kh_value(pSum->pauli_strings, kp);

      new_wfn = wavefunction_pauli_string_evolution_c(pString, wfn, epsilon);
      if (!new_wfn) {
        fprintf(stderr, "Error: Pauli string evolution failed.\n");
        continue;
      }

      if (!first_iteration) {
        free_wavefunction_c(wfn);
      }
      wfn = new_wfn;
      first_iteration = 0;
    }
  }

  return wfn;
}

static inline SlaterDeterminantC *
wavefunction_get_min_encoding_sdet_c(WavefunctionC *wfn) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return NULL;
  }

  SlaterDeterminantC *min_sdet = NULL;
  unsigned int min_encoding = UINT_MAX;
  khiter_t k;

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      if (sdet->encoding < min_encoding) {
        min_encoding = sdet->encoding;
        min_sdet = sdet;
      }
    }
  }

  return min_sdet;
}

WavefunctionC *wavefunction_remove_global_phase_c(WavefunctionC *wfn) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return NULL;
  }

  SlaterDeterminantC *min_sdet = wavefunction_get_min_encoding_sdet_c(wfn);
  double phase = carg(min_sdet->coef);
  double complex invert = cexp(-1 * (double complex)I * phase);
  return wavefunction_scalar_multiplication_c(wfn, invert);
}

WavefunctionC *wavefunction_remove_near_zero_terms_c(WavefunctionC *wfn,
                                                     double cutoff) {
  if (!wfn) {
    fprintf(stderr, "Error: Received NULL wavefunction.\n");
    return NULL;
  }

  WavefunctionC *new_wfn = wavefunction_init_c(wfn->N);
  khiter_t k;

  for (k = kh_begin(wfn->slater_determinants);
       k != kh_end(wfn->slater_determinants); ++k) {
    if (kh_exist(wfn->slater_determinants, k)) {
      SlaterDeterminantC *sdet =
          (SlaterDeterminantC *)kh_value(wfn->slater_determinants, k);
      if (fabs(creal(sdet->coef)) > cutoff ||
          fabs(cimag(sdet->coef)) > cutoff) {
        SlaterDeterminantC *new_sdet =
            slater_determinant_init_c(sdet->N, sdet->coef, sdet->orbitals);
        wavefunction_append_slater_determinant_c(new_wfn, new_sdet);
      }
    }
  }

  return new_wfn;
}
