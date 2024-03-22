#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <stdio.h>
#include <stdlib.h>

void print_matrix(const gsl_matrix *M, FILE *fp)
{
    for (size_t row = 0; row < M->size1; ++row)
    {
        for (size_t col = 0; col < M->size2; ++col)
        {
            printf("%f\t", gsl_matrix_get(M, row, col));
            fprintf(fp, "%f\t", gsl_matrix_get(M, row, col));
        }
        printf("\n");
        fprintf(fp, "\n");
    }
}
void print_vector(const gsl_vector *V, FILE *fp)
{
    for (size_t col = 0; col < V->size; ++col)
    {
        printf("%f ", gsl_vector_get(V, col));
        fprintf(fp, "%f ", gsl_vector_get(V, col));
    }
}
double vector_vector_product(gsl_vector *V1, gsl_vector *V2)
{
    double sum = 0.0;
    for (size_t col = 0; col < V1->size; col++)
    {
        sum += gsl_vector_get(V1, col) * gsl_vector_get(V2, col);
    }
    return sum;
}
void fill_matrix_A(gsl_matrix *A)
{
    // wymiary macierzy
    int dim = A->size1;

    for (size_t row = 0; row < dim; row++)
    {
        for (size_t col = 0; col < dim; col++)
        {
            gsl_matrix_set(A, row, col, pow(2 + abs(row - col), (-abs(row - col)) / 2.0));
        }
    }
}
void matrix_vector_product(gsl_matrix *M, gsl_vector *V, gsl_vector *P)
{
    for (size_t row = 0; row < M->size1; ++row)
    {
        double sum = 0.0;
        for (size_t col = 0; col < M->size2; ++col)
        {
            sum += gsl_matrix_get(M, row, col) * gsl_vector_get(V, col);
        }
        gsl_vector_set(P, row, sum);
    }
}
void matrix_matrix_product(gsl_matrix *A, gsl_matrix *B, gsl_matrix *P)
{
    for (size_t row = 0; row < A->size1; ++row)
    {
        for (size_t col = 0; col < A->size2; ++col)
        {
            double sum = 0.0;

            for (size_t k = 0; k < A->size2; ++k)
            {
                sum += gsl_matrix_get(A, row, k) * gsl_matrix_get(B, k, col);
            }
            gsl_matrix_set(P, row, col, sum);
        }
    }
}
void tensor_product(gsl_vector *V, gsl_matrix *P)
{
    for (size_t row = 0; row < V->size; row++)
    {
        for (size_t col = 0; col < V->size; col++)
        {
            gsl_matrix_set(P, row, col, gsl_vector_get(V, row) * gsl_vector_get(V, col));
        }
    }
}

void solve_task(int n, int M, FILE *fp)
{
    printf("Rozwiązanie dla: n=%d, M=%d\n\n", n, M);
    fprintf(fp, "Rozwiązanie dla: n=%d, M=%d\n\n", n, M);

    gsl_matrix *A = gsl_matrix_calloc(n, n);
    fill_matrix_A(A);
    gsl_matrix *W0 = gsl_matrix_calloc(n, n);
    gsl_matrix_memcpy(W0, A);

    gsl_vector *x0 = gsl_vector_calloc(n);
    gsl_vector *x0_copy = gsl_vector_calloc(n);
    gsl_matrix *x0x0 = gsl_matrix_calloc(n, n);

    gsl_matrix *EigenValuesMatrix = gsl_matrix_calloc(n, n);

    for (int i = 0; i < n; i++)
    {
        // ustawianie początkowych wartości x0
        for (size_t col = 0; col < x0->size; col++)
            gsl_vector_set(x0, col, 1);

        double lambda;
        for (int j = 0; j < M; j++)
        {
            // gsl_vector_memcpy(x0_copy, x0);
            matrix_vector_product(W0, x0, x0_copy);

            lambda = vector_vector_product(x0_copy, x0) / vector_vector_product(x0, x0);

            for (size_t col = 0; col < x0->size; col++)
            {
                double val = gsl_vector_get(x0_copy, col) / sqrt(vector_vector_product(x0_copy, x0_copy));
                gsl_vector_set(x0, col, val);
            }
        }

        printf("Wartość własna %d: %lf\n", i + 1, lambda);
        fprintf(fp, "Wartość własna %d: %lf\n", i + 1, lambda);

        printf("Wektor własny:\n");
        fprintf(fp, "Wektor własny:\n");
        print_vector(x0, fp);
        printf("\n\n");
        fprintf(fp, "\n\n");

        // dodawanie do macierzy wektorów własnych
        for (size_t row = 0; row < W0->size1; row++)
        {
            gsl_matrix_set(EigenValuesMatrix, row, i, gsl_vector_get(x0, row));
        }

        // redukcja macierzy W0
        tensor_product(x0, x0x0);

        for (size_t row = 0; row < W0->size1; row++)
        {
            for (size_t col = 0; col < W0->size2; col++)
            {
                gsl_matrix_set(W0, row, col, gsl_matrix_get(W0, row, col) - lambda * gsl_matrix_get(x0x0, row, col));
            }
        }
    }

    printf("\nMacierz wektorów własnych:\n");
    fprintf(fp, "\nMacierz wektorów własnych:\n");
    print_matrix(EigenValuesMatrix, fp);
    printf("\n");
    fprintf(fp, "\n");

    // obliczanie D=X^tAX
    gsl_matrix *D = gsl_matrix_calloc(n, n);
    gsl_matrix *D_h = gsl_matrix_calloc(n, n);
    gsl_matrix_transpose(EigenValuesMatrix);
    matrix_matrix_product(EigenValuesMatrix, A, D);
    gsl_matrix_transpose(EigenValuesMatrix);
    matrix_matrix_product(D, EigenValuesMatrix, D_h);
    printf("Macierz D:\n");
    fprintf(fp, "Macierz D:\n");
    print_matrix(D, fp);
    printf("\n\n");
    fprintf(fp, "\n\n");

    gsl_matrix_free(D);
    gsl_matrix_free(D_h);

    gsl_matrix_free(A);
    gsl_matrix_free(W0);
    gsl_vector_free(x0);
    gsl_vector_free(x0_copy);
    gsl_matrix_free(x0x0);
    gsl_matrix_free(EigenValuesMatrix);
}

int main()
{
    FILE *fp = fopen("wyniki.txt", "w");

    solve_task(7, 12, fp);

    solve_task(7, 120, fp);

    fclose(fp);

    return 0;
}