#include <iostream>

// Define a template struct that takes a variable number of integers
template <int... Ns>
struct NumberList {};

// Function to print the numbers in the NumberList
template <int... Ns>
void printNumberList(NumberList<Ns...>) {
    ((std::cout << Ns << " "), ...);
}

int main() {
    // Define a NumberList with numbers 2, 8, and 3
    NumberList<2, 8, 3> myNumberList;

    // Print the numbers
    printNumberList(myNumberList);

    return 0;
}
