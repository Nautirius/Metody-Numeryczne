#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <stdio.h>
#include <stdlib.h>

double lambda_f(double x)
{
    // cegła ogniotrwała
    if (x <= 40.0)
    {
        return 0.4;
    }
    // cegła o średniej ogniotrwałości
    else if (x <= 70.0)
    {
        return 0.2;
    }
    // cegła niskotemperaturowa
    else
    {
        return 0.1;
    }
}

double vector_vector_product(gsl_vector *V1, gsl_vector *V2)
{
    double sum = 0.0;
    for (int col = 0; col < V1->size; col++)
    {
        sum += gsl_vector_get(V1, col) * gsl_vector_get(V2, col);
    }
    return sum;
}

void matrix_vector_product(gsl_matrix *M, gsl_vector *V, gsl_vector *C)
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

void fill_matrix_A(gsl_matrix *A)
{
    // wymiary macierzy
    int dim = A->size1;
    // wielkość kroku (razem 100 cm cegły)
    double step = 100.0 / (dim + 1);
    // połowa kroku do obliczania wartości pośrednich
    double halfStep = step / 2;
    // zmienna pomocnicza przechowująca wartość kolejnego węzła
    double lambdaValue = 0.0;

    for (int row = 0; row < dim; row++)
    {
        for (int col = 0; col < dim; col++)
        {
            // diagonala
            if (row == col)
            {
                lambdaValue = -lambda_f(step * row + halfStep) - lambda_f(step * (row + 1) + halfStep);
                gsl_matrix_set(A, row, col, lambdaValue);
            }
            // jeden nad diagonalą
            else if (row == col + 1)
            {
                lambdaValue = lambda_f(step * row + halfStep);
                gsl_matrix_set(A, row, col, lambdaValue);
            }
            // jeden pod diagonalą
            else if (row == col - 1)
            {
                lambdaValue = lambda_f(step * col + halfStep);
                gsl_matrix_set(A, row, col, lambdaValue);
            }
        }
    }
}

void fill_vector_b(gsl_vector *t)
{
    int dim = t->size;
    // wielkość kroku (razem 100 cm cegły)
    double step = 100.0 / (dim + 1);
    // połowa kroku do obliczania wartości pośrednich
    double halfStep = step / 2;
    // zmienna pomocnicza przechowująca wartość kolejnego węzła
    double lambdaValue;
    // temperatury na obu końcach pieca
    double T_L = 1000;
    double T_R = 100;

    // temperatura przy lewej granicy
    lambdaValue = -lambda_f(0.0 + halfStep);
    gsl_vector_set(t, 0, T_L * lambdaValue);

    // temperatury po środku do ustalenia
    for (int col = 1; col < dim - 1; col++)
    {
        gsl_vector_set(t, col, 0);
    }

    // temperatura przy prawej granicy
    lambdaValue = -lambda_f(step * (dim - 1) + halfStep);
    gsl_vector_set(t, dim - 1, T_R * lambdaValue);
}

void solve(gsl_matrix *A, gsl_vector *t, gsl_vector *b)
{
    int dim = t->size;
    // wektor r
    gsl_vector *r = gsl_vector_calloc(dim);
    // wektor przechowujący wynik mnożenia A*t
    gsl_vector *A_t = gsl_vector_calloc(dim);
    // wektor przechowujący wynik mnożenia r*A
    gsl_vector *r_A = gsl_vector_calloc(dim);
    // zmienna przechowująca wynik iloczynu skalarnego r*r
    double r_r_product = 0.0;
    // zmienna przechowująca wynik iloczynu r*A*r
    double r_A_r_product = 0.0;
    // zmienna przechowująca wartość alfy
    double alpha = 0.0;
    // zmienna przechowująca wartość normy wektora t
    double norm = 0.0;

    // maksymalnie 60000 iteracji
    for (int i = 0; i < 60000; i++)
    // for (int i = 0; i < 10; i++)
    {
        // liczenie iloczynu A*t
        matrix_vector_product(A, t, A_t);

        // liczenie r = b - A*t
        for (int col = 0; col < dim; col++)
        {
            gsl_vector_set(r, col, gsl_vector_get(b, col) - gsl_vector_get(A_t, col));
        }

        // liczenie iloczynu skalarnego r*r
        r_r_product = vector_vector_product(r, r);
        // sprawdzanie czy norma euklidesowa reszty ||r|| < 10e-6
        if (sqrt(r_r_product) < 0.000001)
            return;

        // liczenie iloczynu r*A
        matrix_vector_product(A, r, r_A);

        // liczenie iloczynu r_A*r
        r_A_r_product = vector_vector_product(r_A, r);

        alpha = r_r_product / r_A_r_product;

        // liczenie następnej watrości t_k+1 = t_k + alpha*r_k
        for (int col = 0; col < dim; col++)
        {
            gsl_vector_set(t, col, gsl_vector_get(t, col) + alpha * gsl_vector_get(r, col));
        }

        norm = vector_vector_product(t, t);
        printf("%d\t%f\t%f\n", i, r_r_product, norm);
    }

    // zwalnianie pamięci
    gsl_vector_free(r);
    gsl_vector_free(A_t);
    gsl_vector_free(r_A);
}

int main()
{
    gsl_matrix *A = gsl_matrix_calloc(9, 9);
    fill_matrix_A(A);
    gsl_vector *b = gsl_vector_calloc(9);
    fill_vector_b(b);
    gsl_vector *t = gsl_vector_calloc(9);
    for (size_t col = 0; col < t->size; col++)
    {
        gsl_vector_set(t, col, 0);
    }

    solve(A, t, b);

    // wypisywanie wektora z rozkładem temperatury
    printf("\nRozkład temperatury w piecu: \n");
    for (size_t col = 0; col < t->size; col++)
    {
        printf("%.2f ", gsl_vector_get(t, col));
    }
    printf("\n");

    // zapisywanie wyniku do pliku
    FILE *fp = fopen("wyniki.txt", "w");

    fclose(fp);

    // zwalnianie pamięci
    gsl_matrix_free(A);
    gsl_vector_free(b);
    gsl_vector_free(t);

    return 0;
}