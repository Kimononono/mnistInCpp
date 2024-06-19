#include "math.cpp"
#include <utility>
#include <vector>
template <typename T> using MatrixData = std::vector<std::vector<T>>;


template <typename T>
class Matrix {
    public: 
        int rows;
        int cols;
        MatrixData<T> data;
        Matrix(int rows, int cols) {
            this->rows = rows;
            this->cols = cols;
            this->data = MatrixData<T>(rows, std::vector<T>(cols));
        }
        Matrix(MatrixData<T>& m) {
            auto [ rows, cols ] = shape(m);
            this->rows = rows;
            this->cols = cols;
            this->data = m;

        }
        std::pair<int, int> shape() {
            return std::make_pair(rows, cols);
        }
        Matrix<T> dot(MatrixData<T>& b) {
            return Matrix(dot(this->data, b));
        }
        Matrix<T> add(MatrixData<T>& b) {
            return Matrix(add(this->data, b));
        }
        Matrix<T> subtract(MatrixData<T>& b) {
            return Matrix(subtract(this->data, b));
        }
        Matrix<T> multiplyElements(MatrixData<T>& b) {
            return Matrix(multiply(this->data, b));
        }
        Matrix<T> transpose() {
            return Matrix(transpose(this->data));
        }
        Matrix<T> applyFunction(T (*f)(T)) {
            return Matrix(applyFunction(this->data, f));
        }
        void print() {
            printMatrix(this->data);
        }
};
