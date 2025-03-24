#pragma once

#include <torch/torch.h>

struct DQN : torch::nn::Module {
    torch::nn::Linear input{nullptr}, hidden{nullptr}, output{nullptr};

    DQN(int64_t states, int64_t actions) {
        input = register_module("input", torch::nn::Linear(states, 64));
        hidden = register_module("hidden", torch::nn::Linear(64, 64));
        output = register_module("output", torch::nn::Linear(64, actions));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(input(x));
        x = torch::relu(hidden(x));
        return output(x);
    }
};
