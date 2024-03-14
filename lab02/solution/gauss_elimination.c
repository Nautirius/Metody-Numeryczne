#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <stdio.h>
#include <stdlib.h>

void print_matrix(const gsl_matrix *M)
{
    printf("\nMacierz:\n");
    for (size_t row = 0; row < M->size1; ++row)
    {
        for (size_t col = 0; col < M->size2; ++col)
        {
            printf("\t%3.1f", gsl_matrix_get(M, row, col));
        }
        printf("\n");
    }
}

void print_vector(const gsl_vector *V)
{
    printf("\nWektor:\n");
    for (size_t col = 0; col < V->size; ++col)
    {
        printf("\t%3.1f", gsl_vector_get(V, col));
    }
    printf("\n");
}

void fill_matrix(gsl_matrix *M, double *values)
{
    for (size_t row = 0; row < M->size1; ++row)
    {
        for (size_t col = 0; col < M->size2; ++col)
        {
            gsl_matrix_set(M, row, col, values[M->size2 * row + col]);
        }
    }
}

void fill_vector(gsl_vector *V, double *values)
{
    for (size_t col = 0; col < V->size; ++col)
    {
        gsl_vector_set(V, col, values[col]);
    }
}

void gauss_elim_solve(gsl_matrix *A, gsl_vector *V)
{
    //
    // rozwiązywanie układu
    //
    // zerowanie poddiagonali
    // iterowanie po kolumnach
    for (int i = 0; i < A->size2 - 1; i++)
    {
        // dla kazdej kolumny pobieram wartosc a_nn
        double a = gsl_matrix_get(A, i, i);

        // dzielimy wiersz przez a_nn
        for (int j = 0; j < A->size2; j++)
        {
            gsl_matrix_set(A, i, j, gsl_matrix_get(A, i, j) / a);
        }
        gsl_vector_set(V, i, gsl_vector_get(V, i) / a);

        // zerowanie poddiagonali
        // iterowanie po wierszach
        for (int j = i + 1; j < A->size1; j++)
        {
            double mnoznik = gsl_matrix_get(A, j, i);

            // iterowanie po kolumnach
            for (int k = 0; k < A->size2; k++)
            {
                double gora = gsl_matrix_get(A, i, k);
                gsl_matrix_set(A, j, k, gsl_matrix_get(A, j, k) - gora * mnoznik);
            }
            gsl_vector_set(V, j, gsl_vector_get(V, j) - mnoznik * gsl_vector_get(V, i));
        }
    }
    // jedynkownaie ostatniego wiersza
    gsl_vector_set(V, V->size - 1, gsl_vector_get(V, V->size - 1) / gsl_matrix_get(A, A->size1 - 1, A->size2 - 1));
    gsl_matrix_set(A, A->size1 - 1, A->size2 - 1, 1);

    printf("\nPo zerowaniu dołu:");
    print_matrix(A);
    print_vector(V);

    //
    // zerowanie naddiagonali
    // iterowanie po kolumnach
    for (int j = A->size2 - 1; j >= 0; j--)
    {
        double mnoznik = gsl_vector_get(V, j);

        // iterowanie po wierszach
        for (int i = j - 1; i >= 0; i--)
        {
            double wyraz = gsl_matrix_get(A, i, j);
            gsl_vector_set(V, i, gsl_vector_get(V, i) - wyraz * mnoznik);
            gsl_matrix_set(A, i, j, mnoznik - mnoznik);
        }
    }

    printf("\nPo zerowaniu góry:");
    print_matrix(A);
    print_vector(V);
}

void matrix_vector_multiplication(gsl_matrix *M, gsl_vector *V, gsl_vector *C)
{
    for (size_t row = 0; row < M->size1; ++row)
    {
        double sum = 0.0;
        for (size_t col = 0; col < M->size2; ++col)
        {
            sum += gsl_matrix_get(M, row, col) * gsl_vector_get(V, col);
        }
        gsl_vector_set(C, row, sum);
    }
}

double calculate_deviation(gsl_vector *V1, gsl_vector *V2)
{
    double sum = 0.0;
    for (size_t row = 0; row < V1->size; ++row)
    {
        double diff = gsl_vector_get(V2, row) - gsl_vector_get(V1, row);
        sum += diff * diff;
    }
    return sqrt(sum) / 5.0;
}

int main()
{
    FILE *fp = fopen("wyniki.txt", "w");
    int m = 5, n = 5;
    double a_data[] = {
        0, 1, 6, 9, 10,
        2, 1, 6, 9, 10,
        1, 6, 6, 8, 6,
        5, 9, 10, 7, 10,
        3, 4, 9, 7, 9};
    double b_data[] = {10, 2, 9, 9, 3};

    gsl_matrix *A = gsl_matrix_calloc(m, n);
    gsl_matrix *A_CP = gsl_matrix_calloc(m, n);
    gsl_vector *b = gsl_vector_calloc(n);
    gsl_vector *b_cp = gsl_vector_calloc(n);
    gsl_vector *c = gsl_vector_calloc(n);

    for (double q = 0.01; q < 3.0; q += 0.01)
    {
        // zmiana wartości w macierzy A i wektorze b
        fill_matrix(A, a_data);
        fill_matrix(A_CP, a_data);
        fill_vector(b, b_data);
        fill_vector(b_cp, b_data);
        gsl_matrix_set(A, 0, 0, 2.0 * q);
        gsl_matrix_set(A_CP, 0, 0, 2.0 * q);

        printf("\n\nMacierz dla q=%f:\n", q);
        print_matrix(A);
        print_vector(b);

        // rozwiązanie układu
        gauss_elim_solve(A, b);

        // mnożenie A*b
        matrix_vector_multiplication(A_CP, b, c);
        printf("\nWektor C:");
        print_vector(c);

        // obliczanie odchylenia między wektorami b i c
        double o = calculate_deviation(b_cp, c);
        printf("\nOdchylenie między wektorami b i c: %g\n", o);

        fprintf(fp, "%.2f,%g\n", q, o);
    }

    fclose(fp);
    gsl_matrix_free(A);
    gsl_matrix_free(A_CP);
    gsl_vector_free(b);
    gsl_vector_free(b_cp);
    gsl_vector_free(c);

    return 0;
}