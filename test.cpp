
#include <vector>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

class Example {
public:
    Example(int in_size, int out_size) {
        // Initialize the weights matrix with dimensions in_size x out_size
        this->weights = Matrix<float>(in_size, std::vector<float>(out_size));
    }

private:
    Matrix<float> weights;
};

int main() {
    int in_size = 5;
    int out_size = 4;

    // Create an instance of Example
    Example example(in_size, out_size);

    return 0;
}
