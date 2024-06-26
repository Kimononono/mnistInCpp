#include <cstddef>
#include <vector> 


template <typename T> using Matrix = std::vector<std::vector<T>>;

template <typename T> Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b);
template <typename T> Matrix<T> crossMultiply(const Matrix<T>& a, const Matrix<T>& b);
template <typename T> Matrix<T> add(const Matrix<T>& a, const Matrix<T>& b);
template <typename T> Matrix<T> subtract(const Matrix<T>& a, const Matrix<T>& b);
template <typename T> Matrix<T> transpose(const Matrix<T>& a);
template <typename T, typename F> Matrix<T> applyFunction(const Matrix<T>& a, F func);



template <typename T> std::pair<size_t, size_t> shape(const Matrix<T>& a);

