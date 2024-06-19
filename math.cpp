#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector> 
#include <random>    
#include <cmath>     
#include <algorithm> 

template <typename T> using Matrix = std::vector<std::vector<T>>;
template <typename T> std::pair<size_t, size_t> shape(const Matrix<T>& a) {
    return { a.size(), a[0].size() }; 
}
float getRandomFloat() {
    static std::random_device rd;   
    static std::mt19937 gen(rd());  
    static std::uniform_real_distribution<float> dis(-1.0, 1.0); 
    return dis(gen);
}
template <typename T>
Matrix<T> createMatrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    Matrix<T> result(rows, std::vector<T>(cols));
    // Initialize matrix with some values (optional)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = getRandomFloat();
        }
    }
    return result;
}

template <typename T> Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b) {
    if (a[0].size() != b.size()) {
        std::cout << "a[0].size() = " << a[0].size() << std::endl;

        std::string errorMessage = "Dimensions don't match for dot product: "
                                   "Matrix A shape: (" + std::to_string(a.size()) + ", " + std::to_string(a[0].size()) + ") "
                                   "Matrix B shape: (" + std::to_string(b.size()) + ", " + std::to_string(b[0].size()) + ") "
                                   + std::to_string(a[0].size()) + " != " + std::to_string(b.size());
        throw std::invalid_argument(errorMessage);
    }

    int resultRows = a.size();
    int resultCols = b[0].size();
    Matrix<T> result = createMatrix<T>(resultRows, resultCols);

    for (int i = 0; i < resultRows; ++i) {
        for (int j = 0; j < resultCols; ++j) {
            result[i][j] = 0; // Initialize the result cell
            for (int k = 0; k < a[0].size(); ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

template <typename T> Matrix<T> add(const Matrix<T>& a, const Matrix<T>& b) {
    if (shape(a) != shape(b) ) throw std::invalid_argument("Matrices must have same size");
    auto [ rows, cols ] = shape(a);
    Matrix<T> result = createMatrix<T>(rows, cols);
    for( int i = 0; i < a.size() ; i++ ) {
        for( int j = 0; j < a.size() ; j++ ) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}


template <typename T>
Matrix<T> subtract(const Matrix<T>& a, const Matrix<T>& b) {
    if (shape(a) != shape(b)) throw std::invalid_argument("Matrices must have the same size");
    
    auto [rows, cols] = shape(a);
    Matrix<T> result = createMatrix<T>(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    
    return result;
}

template <typename T> Matrix<T> multiply(const Matrix<T>& a, const Matrix<T>& b) {
    if (shape(a) != shape(b) ) throw std::invalid_argument("Matrices must have same size");
    auto [ rows, cols ] = shape(a);
    Matrix<T> result = createMatrix<T>(rows, cols);
    for( int i = 0; i < a.size() ; i++ ) {
        for( int j = 0; j < a.size() ; j++ ) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}


template <typename T> Matrix<T> transpose(const Matrix<T>& a) {
    auto [ rows, cols ] = shape(a);
    Matrix<T> result = createMatrix<T>(cols, rows);
    for(int i = 0; i < a.size(); i++) {
    for(int j = 0; j < a[0].size(); j++) {
        // check if out of bounds
        if (j >= result.size() || i >= result[0].size()) {
            throw std::invalid_argument("Index out of bounds");
        }
        result[j][i] = a[i][j];
    }
    }
    return result;
}

template <typename T, typename F> Matrix<T> applyFunction(const Matrix<T>& a, F func) {
    auto [ rows, cols ] = shape(a);
    Matrix<T> result = createMatrix<T>(rows, cols);
    for(int i = 0; i < a.size(); i++) {
    for(int j = 0; j < a[0].size(); j++) {
        result[i][j] = func(a[i][j]);
    }
    }
    return result;
}


float dReLU(float in) {
    return in > 0 ? 1.0f : 0.0f;
};

float ReLU(float in) {
    return in > 0 ? in : 0.0f;
};
Matrix<float> softmax(const Matrix<float>& input) {
    Matrix<float> output(input.size(), std::vector<float>(input[0].size()));
    
    for (size_t i = 0; i < input.size(); ++i) {
        float maxVal = *std::max_element(input[i].begin(), input[i].end());

        float sum = 0.0f;
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] = std::exp(input[i][j] - maxVal); 
            sum += output[i][j];
        }
        
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] /= sum;
        }
    }
    
    return output;
}

