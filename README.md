Лабораторная работа № 2: Структура данных 
Задание:
Перемножить 2 квадратные матрицы размера 2048x2048 с элементами типа double.

Исходные матрицы генерируются в программе (случайным образом либо по определенной формуле) либо считываются из заранее подготовленного файла.

Оценить сложность алгоритма по формуле c = 2 n3, где n - размерность матрицы.

Оценить производительность в MFlops, p = c/t*10-6, где t - время в секундах работы алгоритма.

Выполнить 3 варианта перемножения и их анализ и сравнение:

1-й вариант перемножения - по формуле из линейной алгебры.

2-й вариант перемножения - результат работы функции cblas_dgemm из библиотеки BLAS (рекомендуемая реализация из Intel MKL)

3-й вариант перемножения - оптимизированный алгоритм по вашему выбору, написанный вами, производительность должна быть не ниже 30% от 2-го варианта

Реализация:
Листинг прграммы:
```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>
#include <mkl.h>

using namespace std;
using namespace std::chrono;

const int N = 2048;
const double OPS = 2.0 * N * N * N;

void fill_matrix(double* matrix) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.1, 1.0);
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dis(gen);
    }
}

void clear_matrix(double* matrix) {
    for (int i = 0; i < N * N; ++i) matrix[i] = 0.0;
}

void print_stats(string name, double t) {
    double p = (OPS / t) * 1e-6;
    cout << left << setw(35) << name
         << "Время: " << setw(10) << t << " сек. | "
         << "Производительность: " << p << " MFlops" << endl;
}

int main() {
    system("chcp 65001");

    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    fill_matrix(A);
    fill_matrix(B);

    cout << "Размер матрицы: " << N << "x" << N << endl;
    cout << "Сложность (c): " << OPS << endl;
    cout << "------------------------------------------------------------" << endl;

    clear_matrix(C);
    auto start = high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    auto end = high_resolution_clock::now();
    print_stats("1. Линейная алгебра (IJK):", duration<double>(end - start).count());

    clear_matrix(C);
    start = high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    end = high_resolution_clock::now();
    print_stats("2. Функция cblas_dgemm (MKL):", duration<double>(end - start).count());

    clear_matrix(C);
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double temp = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += temp * B[k * N + j];
            }
        }
    }
    end = high_resolution_clock::now();
    print_stats("3. Оптимизированный (IKJ + OMP):", duration<double>(end - start).count());

    cout << "------------------------------------------------------------" << endl;
    cout << "Автор: Щербаков Борис Владимирович" << endl;
    cout << "Группа: 090301-ПОВа-о25" << endl;

    delete[] A; delete[] B; delete[] C;
    return 0;
}
```
Результат выполнения программы:
<img width="746" height="245" alt="image" src="https://github.com/user-attachments/assets/8c7c3040-0524-4ba3-b890-4cc624900d60" />
