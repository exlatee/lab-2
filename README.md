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
#include <cmath>
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


void check_correctness(const double* ref, const double* test, int size, double eps = 1e-5) {
    double max_diff = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = std::abs(ref[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    cout << "   -> Проверка: ";
    if (max_diff <= eps) {
        cout << "\033[32mУСПЕШНО\033[0m";
    } else {
        cout << "\033[31mОШИБКА\033[0m";
    }
    cout << " (Макс. разность элементов: " << max_diff << ")" << endl << endl;
}

int main() {
    system("chcp 65001");
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];
    double* C_ref = new double[N * N]; // Матрица для эталонного результата

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


    for (int i = 0; i < N * N; ++i) C_ref[i] = C[i];
    cout << "   -> Результат сохранен как эталонный." << endl << endl;


    clear_matrix(C);
    start = high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    end = high_resolution_clock::now();
    print_stats("2. Функция cblas_dgemm (MKL):", duration<double>(end - start).count());
    check_correctness(C_ref, C, N * N);


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
    check_correctness(C_ref, C, N * N);

    cout << "------------------------------------------------------------" << endl;
    cout << "Автор: Щербаков Борис Владимирович" << endl;
    cout << "Группа: 090301-ПОВа-о25" << endl;

    delete[] A; delete[] B; delete[] C; delete[] C_ref;
    return 0;
}
```
Результат выполнения программы:
<img width="747" height="398" alt="image" src="https://github.com/user-attachments/assets/3ed9e553-23be-4b96-b46c-0e73b4e661e0" />

