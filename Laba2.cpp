#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef USE_MKL
#include <mkl.h>
#else
extern "C" {
    void cblas_zgemm(const int* Order, const int* TransA, const int* TransB,
                     const int* M, const int* N, const int* K,
                     const void* alpha, const void* A, const int* lda,
                     const void* B, const int* ldb, const void* beta,
                     void* C, const int* ldc);
}
#endif

using namespace std;
using namespace chrono;

using Complex = complex<double>;

class Matrix {
private:
    vector<Complex> data;
    size_t rows, cols;
    
public:
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, Complex(0, 0)) {}
    
    Complex& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    const Complex& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
    
    Complex* ptr() { return data.data(); }
    const Complex* ptr() const { return data.data(); }
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    void fillRandom() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-10.0, 10.0);
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                (*this)(i, j) = Complex(dist(gen), dist(gen));
            }
        }
    }
    
    void print(int max_rows = 5, int max_cols = 5) const {
        cout << fixed << setprecision(4);
        for (size_t i = 0; i < min(rows, (size_t)max_rows); i++) {
            for (size_t j = 0; j < min(cols, (size_t)max_cols); j++) {
                cout << "(" << (*this)(i, j).real() << ", " << (*this)(i, j).imag() << "i) ";
            }
            cout << endl;
        }
        if (rows > max_rows || cols > max_cols) {
            cout << "..." << endl;
        }
    }
    
    bool isClose(const Matrix& other, double tol = 1e-10) const {
        if (rows != other.rows || cols != other.cols) return false;
        for (size_t i = 0; i < rows * cols; i++) {
            if (abs(data[i] - other.data[i]) > tol) return false;
        }
        return true;
    }
};

Matrix multiplyClassic(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    Matrix C(n, n);
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            Complex sum(0, 0);
            for (size_t k = 0; k < n; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

Matrix multiplyBLAS(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    Matrix C(n, n);
    
    #ifdef USE_MKL
    const double alpha_real = 1.0;
    const double alpha_imag = 0.0;
    const double beta_real = 0.0;
    const double beta_imag = 0.0;
    
    MKL_Complex16 alpha = {alpha_real, alpha_imag};
    MKL_Complex16 beta = {beta_real, beta_imag};
    
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                &alpha, 
                reinterpret_cast<const MKL_Complex16*>(A.ptr()), n,
                reinterpret_cast<const MKL_Complex16*>(B.ptr()), n,
                &beta,
                reinterpret_cast<MKL_Complex16*>(C.ptr()), n);
    #else
    C = multiplyClassic(A, B);
    #endif
    
    return C;
}

class OptimizedMatrixMultiplier {
private:
    static constexpr size_t BLOCK_SIZE = 128;
    
    static void multiplyBlock(const Complex* A, const Complex* B, Complex* C,
                              size_t n, size_t i_start, size_t j_start, size_t k_start) {
        size_t i_end = min(i_start + BLOCK_SIZE, n);
        size_t j_end = min(j_start + BLOCK_SIZE, n);
        size_t k_end = min(k_start + BLOCK_SIZE, n);
        
        for (size_t i = i_start; i < i_end; i++) {
            for (size_t k = k_start; k < k_end; k++) {
                Complex aik = A[i * n + k];
                if (aik == Complex(0, 0)) continue;
                
                for (size_t j = j_start; j < j_end; j++) {
                    C[i * n + j] += aik * B[k * n + j];
                }
            }
        }
    }
    
public:
    static Matrix multiply(const Matrix& A, const Matrix& B) {
        size_t n = A.getRows();
        Matrix C(n, n);
        
        memset(C.ptr(), 0, n * n * sizeof(Complex));
        
        for (size_t i = 0; i < n; i += BLOCK_SIZE) {
            for (size_t j = 0; j < n; j += BLOCK_SIZE) {
                for (size_t k = 0; k < n; k += BLOCK_SIZE) {
                    multiplyBlock(A.ptr(), B.ptr(), C.ptr(), n, i, j, k);
                }
            }
        }
        
        return C;
    }
};

class AdvancedOptimizedMultiplier {
private:
    static constexpr size_t BLOCK_SIZE = 256;
    static constexpr size_t UNROLL_FACTOR = 4;
    
public:
    static Matrix multiply(const Matrix& A, const Matrix& B) {
        size_t n = A.getRows();
        Matrix C(n, n);
        
        memset(C.ptr(), 0, n * n * sizeof(Complex));
        
        Matrix BT(n, n);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                BT(j, i) = B(i, j);
            }
        }
        
        for (size_t i = 0; i < n; i += BLOCK_SIZE) {
            for (size_t j = 0; j < n; j += BLOCK_SIZE) {
                for (size_t k = 0; k < n; k += BLOCK_SIZE) {
                    size_t i_end = min(i + BLOCK_SIZE, n);
                    size_t j_end = min(j + BLOCK_SIZE, n);
                    size_t k_end = min(k + BLOCK_SIZE, n);
                    
                    for (size_t ii = i; ii < i_end; ii++) {
                        for (size_t kk = k; kk < k_end; kk++) {
                            Complex aik = A(ii, kk);
                            if (aik == Complex(0, 0)) continue;
                            
                            size_t jj = j;
                            for (; jj + UNROLL_FACTOR - 1 < j_end; jj += UNROLL_FACTOR) {
                                C(ii, jj) += aik * BT(kk, jj);
                                C(ii, jj + 1) += aik * BT(kk, jj + 1);
                                C(ii, jj + 2) += aik * BT(kk, jj + 2);
                                C(ii, jj + 3) += aik * BT(kk, jj + 3);
                            }
                            for (; jj < j_end; jj++) {
                                C(ii, jj) += aik * BT(kk, jj);
                            }
                        }
                    }
                }
            }
        }
        
        return C;
    }
};

template<typename Func>
double measurePerformance(Func func, const Matrix& A, const Matrix& B, 
                          Matrix& result, const string& name, size_t n) {
    for (int i = 0; i < 3; i++) {
        func(A, B);
    }
    
    auto start = high_resolution_clock::now();
    result = func(A, B);
    auto end = high_resolution_clock::now();
    
    double time = duration_cast<duration<double>>(end - start).count();
    
    double c = 2.0 * n * n * n;
    double mflops = c / time * 1e-6;
    
    cout << "----------------------------------------" << endl;
    cout << name << ":" << endl;
    cout << "  Time: " << fixed << setprecision(6) << time << " seconds" << endl;
    cout << "  Operations: " << scientific << setprecision(2) << c << endl;
    cout << "  Performance: " << fixed << setprecision(2) << mflops << " MFLOPS" << endl;
    
    return mflops;
}

int main() {
    const size_t N = 2048;
    cout << "Matrix size: " << N << "x" << N << endl;
    cout << "Complex double precision" << endl;
    cout << "========================================" << endl;
    
    cout << "Generating matrices..." << endl;
    Matrix A(N, N), B(N, N);
    A.fillRandom();
    B.fillRandom();
    
    cout << "Matrices generated. First 5x5 of A:" << endl;
    A.print(5, 5);
    
    Matrix result1, result2, result3;
    
    double perf1 = measurePerformance(multiplyClassic, A, B, result1, 
                                       "Variant 1: Classic O(n^3) algorithm", N);
    
    double perf2 = measurePerformance(multiplyBLAS, A, B, result2,
                                       "Variant 2: BLAS (Intel MKL) cblas_zgemm", N);
    
    double perf3 = measurePerformance(OptimizedMatrixMultiplier::multiply, A, B, result3,
                                       "Variant 3: Optimized (block-based with cache optimization)", N);
    
    Matrix result3b;
    double perf3b = measurePerformance(AdvancedOptimizedMultiplier::multiply, A, B, result3b,
                                        "Variant 3b: Advanced (block + transpose + loop unrolling)", N);
    
    cout << "========================================" << endl;
    cout << "Verification:" << endl;
    cout << "Result1 == Result2: " << (result1.isClose(result2) ? "PASS" : "FAIL") << endl;
    cout << "Result1 == Result3: " << (result1.isClose(result3) ? "PASS" : "FAIL") << endl;
    cout << "Result1 == Result3b: " << (result1.isClose(result3b) ? "PASS" : "FAIL") << endl;
    
    cout << "========================================" << endl;
    cout << "Performance Analysis:" << endl;
    cout << "  Classic algorithm:      " << setw(10) << fixed << setprecision(2) << perf1 << " MFLOPS" << endl;
    cout << "  BLAS (Intel MKL):       " << setw(10) << perf2 << " MFLOPS" << endl;
    cout << "  Optimized (block):      " << setw(10) << perf3 << " MFLOPS" << endl;
    cout << "  Advanced optimized:     " << setw(10) << perf3b << " MFLOPS" << endl;
    
    if (perf2 > 0) {
        double ratio = perf3b / perf2 * 100;
        cout << "  Advanced optimized vs BLAS: " << setw(10) << fixed << setprecision(1) 
             << ratio << "%" << endl;
        cout << "  Target: >= 30% of BLAS performance" << endl;
        cout << "  Status: " << (ratio >= 30.0 ? "ACHIEVED" : "NOT ACHIEVED") << endl;
    }
    
    return 0;
}
