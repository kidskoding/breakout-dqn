#include <iostream>
#include <torch/serialize/input-archive.h>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}
