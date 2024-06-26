#include "math.cpp"
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

template <typename T> using Matrix = std::vector<std::vector<T>>;

class Module {
    public: 
        Matrix<float> forward(Matrix<float> input);
        float backward(float loss);
};
        


void printMatrix(Matrix<float> m) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[0].size(); j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
};

std::string printDimensions(const Matrix<float>& matrix) {
    if (matrix.empty()) {
        return "(0, 0)";
    } else if (matrix[0].empty()) {
        return "(" + std::to_string(matrix.size()) + ", 0)";
    }
    return "(" + std::to_string(matrix.size()) + ", " + std::to_string(matrix[0].size()) + ")";
}
std::string to_string(const std::pair<int, int>& p) {
    return "(" + std::to_string(p.first) + ", " + std::to_string(p.second) + ")";
}

class LinearReLU : public Module {
    public:
        Matrix<float> weights;
        Matrix<float> bias;
        Matrix<float> last_input;
        LinearReLU(int in_size, int out_size) {
            this->weights = createMatrix<float>(in_size, out_size);
            this->bias = createMatrix<float>(1, out_size);
            //bias = {{0.0f, 0.0f, 0.0f}};
        };
        Matrix<float> forward(Matrix<float>& input) {
            this->last_input = input;
            Matrix<float> z = add(dot(input, weights), bias);
            return applyFunction(z, ReLU);
        };
        std::tuple<Matrix<float>, Matrix<float>, Matrix<float>> backward(Matrix<float>& dA) {
            Matrix<float> z = dot(last_input, weights);
            z = applyFunction(z, dReLU);
            z = multiply(dA, z);
            Matrix<float> dZ = multiply(dA, applyFunction(dot(last_input, weights), dReLU));  // dZ: (m x out_size)
            Matrix<float> dW = dot(transpose(last_input), dZ);  // dW: (in_size x out_size)
            Matrix<float> dB = dZ;  // dB: (m x out_size)
            Matrix<float> dL = transpose(dot(weights, transpose(dZ)));  // dL: (m x in_size)

            //this->weights = subtract(weights, applyFunction(dW, [](float x) { return 0.001f * x; }));
            //this->bias = subtract(bias, applyFunction(dB, [](float x) { return 0.001f * x; }));

            return std::make_tuple(dL, dW, dB);
        }
};

class LinearSoftmax : public Module {
    public:
        Matrix<float> weights;
        Matrix<float> bias;
        Matrix<float> last_input;
        LinearSoftmax(int in_size, int out_size) {
            this->weights = createMatrix<float>(in_size, out_size);
            this->bias = createMatrix<float>(1, out_size);
            //bias = {{0.0f, 0.0f}};
        };
        Matrix<float> forward(Matrix<float>& input) {
            this->last_input = input;
            return softmax(add(dot(input, weights), bias));
        };
        std::tuple<Matrix<float>, Matrix<float>, Matrix<float>> backward(Matrix<float>& dA) {
   
            Matrix<float> dZ = dA;  // dZ: (m x out_size)
            Matrix<float> dW = dot(transpose(last_input), dZ);  // dW: (in_size x out_size)
            Matrix<float> dB = dZ;  // dB: (m x out_size)
            Matrix<float> dL = transpose(dot(weights, transpose(dZ)));  // dL: (m x in_size)

            Matrix<float> oldWeights = weights;
            //this->weights = subtract(weights, applyFunction(dW, [](float x) { return 0.001f * x; }));
            //this->bias = subtract(bias, applyFunction(dB, [](float x) { return 0.001f * x; }));

            return std::make_tuple(dL, dW, dB);
        };
};



int main() {
    try{ 
        LinearReLU l1(2, 2);
        LinearSoftmax l2(2, 2);

        // Training '&' logic
        Matrix<float> possibleInputs[4];
        possibleInputs[0] = {{1.0f, 0.0f}};
        possibleInputs[1] = {{0.0f, 1.0f}};
        possibleInputs[2] = {{1.0f, 1.0f}};
        possibleInputs[3] = {{0.0f, 0.0f}};
        Matrix<float> possibleOutputs[4];
        possibleOutputs[0] = {{0.0f, 1.0f}};
        possibleOutputs[1] = {{0.0f, 1.0f}};
        possibleOutputs[2] = {{1.0f, 0.0f}};
        possibleOutputs[3] = {{0.0f, 1.0f}};

        // Easy intialization with correct sizes
        Matrix<float> output = l1.forward(possibleInputs[0]);
        Matrix<float> output2 = l2.forward(output);
        Matrix<float> loss = subtract(output2, possibleOutputs[0]);
        
        Matrix<float> dA;
        Matrix<float> dW;
        Matrix<float> dB;

        std::tie(dA, dW, dB) = l2.backward(loss);
        

        //Matrix<float> dZ =l2.backward(loss);
        std::tie(dA, dW, dB) = l1.backward(dA);
        l1.weights = subtract(l1.weights, applyFunction(dW, [](float x) { return 0.001f * x; }));
        l1.bias = subtract(l1.bias, applyFunction(dB, [](float x) { return 0.001f * x; }));
        //Matrix<float> dZ2 = l1.backward(dZ);

        for (int i = 0; i < 400000; i++) {
            std::cout << "Iteration: " << i << std::endl;
            for (int m = 0; m < 6; m++) {
                int j = m;
                if(m>=4) {
                    j = 2;
                } 
                output = l1.forward(possibleInputs[j]);
                output2 = l2.forward(output);
                loss = subtract(output2, possibleOutputs[j]);
                std::cout << "Loss: " << std::endl;
                printMatrix(loss);
                std::cout << "-------------" << std::endl;
                std::tie(dA, dW, dB) = l2.backward(loss);
                l2.weights = subtract(l2.weights, applyFunction(dW, [](float x) { return 0.0001f * x; }));
                l2.bias = subtract(l2.bias, applyFunction(dB, [](float x) { return 0.0001f * x; }));
                std::tie(dA, dW, dB) = l1.backward(dA);

                l1.weights = subtract(l1.weights, applyFunction(dW, [](float x) { return 0.0001f * x; }));
                l1.bias = subtract(l1.bias, applyFunction(dB, [](float x) { return 0.0001f * x; }));
            }

            if (i % 1 == 0) {
                std::cout << "Weights: " << std::endl;
                printMatrix(l1.weights);
                std::cout << "Weights2: " << std::endl;
                printMatrix(l2.weights);
            }
        }

        // EVALUATE
        for (int i = 0; i < 4; i++) {
            Matrix<float> output = l1.forward(possibleInputs[i]);
            Matrix<float> output2 = l2.forward(output);
            std::cout << "Expected: " << std::endl;
            printMatrix(possibleOutputs[i]);
            std::cout << "Actual: " << std::endl;
            printMatrix(output2);
        }

        /*Matrix<float> input = {{1.0f, 2.0f}};
        Matrix<float> output = l1.forward(input);
        printMatrix(output);
        Matrix<float> output2 = l2.forward(output);
        printMatrix(output2);
        Matrix<float> expected = {{0.5f, 0.5f}};
        std::cout << "Expected: " << std::endl;
        Matrix<float> loss = subtract(output2, expected);
        std::cout << "Loss: " << std::endl;
        printMatrix(loss);
        Matrix<float> dZ = l2.backward(loss);
        printMatrix(dZ);
        Matrix<float> dZ2 = l1.backward(dZ);
        printMatrix(dZ2);*/

    } catch (std::invalid_argument& e) {
        std::cout << e.what() << std::endl;
    };
    return 0;
}
