#ifndef INVERT_MATRIX_H
#define INVERT_MATRIX_H

void invert_cpu(float* data, int actualsize, float* determinant);
int invert_matrix(float* a, int n, float* determinant);
#endif

