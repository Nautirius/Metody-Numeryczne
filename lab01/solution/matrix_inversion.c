// #include "/usr/include/gsl/include/gsl/gsl_math.h"
// #include "/usr/include/gsl/include/gsl/gsl_linalg.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <stdio.h>
#include <stdlib.h>

int print_matrix(FILE *f, const gsl_matrix *m)
{
    int status, n = 0;

    for (size_t i = 0; i < m->size1; i++)
    {
        for (size_t j = 0; j < m->size2; j++)
        {
            printf("%.6g\t", gsl_matrix_get(m, i, j));
            if ((status = fprintf(f, "%.6g\t", gsl_matrix_get(m, i, j))) < 0)
                return -1;
            n += status;
        }
        printf("\n");

        if ((status = fprintf(f, "\n")) < 0)
            return -1;
        n += status;
    }

    return n;
}
void find_diagonal(const gsl_matrix *m, double *diag)
{
    for (size_t i = 0; i < m->size1; i++)
        diag[i] = gsl_matrix_get(m, i, i);
}

int main()
{
    // inicjalizacja
    int m = 4;
    int n = 4;
    gsl_matrix *A = gsl_matrix_calloc(n, m);
    gsl_matrix *A_COPY = gsl_matrix_calloc(n, m);

    int RO = 2;

    // wypelnianie macierzy wartosciami
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double value = 1. / (double)(i + j + RO);
            gsl_matrix_set(A, i, j, value);
            gsl_matrix_set(A_COPY, i, j, value);
        }
    }

    //
    // szukamy rozkladu LU
    // int gsl_linalg_LU_decomp(gsl_matrix *a, gsl permutation *p, int *signum);

    gsl_permutation *p = gsl_permutation_calloc(m);
    int sig = 1;
    gsl_linalg_LU_decomp(A, p, &sig);

    //
    // macierz dolna - na diagonali jedynki
    // szukanie diagonali
    double *diag = calloc(A->size1, sizeof(double));
    find_diagonal(A, diag);

    FILE *fp = fopen("wyniki.txt", "w");

    double det = 1;
    fprintf(fp, "Diagonala macierzy U: ");
    for (int i = 0; i < m; i++)
    {
        fprintf(fp, "%.6f\t", diag[i]);
        det *= diag[i];
    }
    fprintf(fp, "\n\nWyznacznik macierzy A: %g\n", det);

    //
    // wyznaczanie macierzy odwrotnej
    // int gsl_linalg_LU_solve(gsl_matrix *A, gsl_permutation *p, gsl_vector *b, gsl_vector *x);

    gsl_vector *b1 = gsl_vector_calloc(m);
    gsl_vector_set(b1, 0, 1.);
    gsl_vector *b2 = gsl_vector_calloc(m);
    gsl_vector_set(b2, 1, 1.);
    gsl_vector *b3 = gsl_vector_calloc(m);
    gsl_vector_set(b3, 2, 1.);
    gsl_vector *b4 = gsl_vector_calloc(m);
    gsl_vector_set(b4, 3, 1.);

    gsl_vector *x1 = gsl_vector_calloc(m);
    gsl_vector *x2 = gsl_vector_calloc(m);
    gsl_vector *x3 = gsl_vector_calloc(m);
    gsl_vector *x4 = gsl_vector_calloc(m);

    gsl_linalg_LU_solve(A, p, b1, x1);
    gsl_linalg_LU_solve(A, p, b2, x2);
    gsl_linalg_LU_solve(A, p, b3, x3);
    gsl_linalg_LU_solve(A, p, b4, x4);

    gsl_matrix *SOLUTION = gsl_matrix_calloc(n, m);

    for (int i = 0; i < n; i++)
    {
        gsl_matrix_set(SOLUTION, i, 0, gsl_vector_get(x1, i));
        gsl_matrix_set(SOLUTION, i, 1, gsl_vector_get(x2, i));
        gsl_matrix_set(SOLUTION, i, 2, gsl_vector_get(x3, i));
        gsl_matrix_set(SOLUTION, i, 3, gsl_vector_get(x4, i));
    }

    fprintf(fp, "\nMacierz odwrotna:\n");
    print_matrix(fp, SOLUTION);

    //
    // liczenie iloczynu macierzy
    gsl_matrix *PRODUCT = gsl_matrix_calloc(n, m);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += gsl_matrix_get(A_COPY, i, k) * gsl_matrix_get(SOLUTION, k, j);
            }
            gsl_matrix_set(PRODUCT, i, j, sum);
        }
    }
    fprintf(fp, "\nWynik iloczynu:\n");
    print_matrix(fp, PRODUCT);

    //
    // wyznaczanie wskaźnika uwarunkowania macierzy
    double norm1 = 0.0, norm2 = 0.0;
    double h1, h2;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            h1 = fabs(gsl_matrix_get(A_COPY, i, j));
            h2 = fabs(gsl_matrix_get(SOLUTION, j, i));
            if (h1 > norm1)
                norm1 = h1;
            if (h2 > norm2)
                norm2 = h2;
        }
    }
    double cond = norm1 * norm2;
    fprintf(fp, "\nWskaźnik uwarunkowania:\n%f", cond);
    fclose(fp);

    //
    // zwalnianie pamięci
    gsl_matrix_free(PRODUCT);
    gsl_matrix_free(SOLUTION);
    gsl_vector_free(b1);
    gsl_vector_free(x1);

    free(diag);
    gsl_permutation_free(p);
    gsl_matrix_free(A_COPY);
    gsl_matrix_free(A);

    return 0;
}