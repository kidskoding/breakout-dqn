#include "dqn.hpp"
#include <memory>
#include <torch/torch.h>

void train(std::shared_ptr<DQN> model);

int main() {
    DQN model = DQN(4, 3);
    auto model_ptr = std::make_shared<DQN>(model);
    train(model_ptr);
}

void train(std::shared_ptr<DQN> model) {
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

    torch::Tensor inputs = torch::rand({10, 4});
    torch::Tensor targets = torch::rand({10, 3});

    torch::nn::MSELoss loss_fn;

    int episodes = 100;
    for(int i = 0; i < episodes; i++) {
        torch::Tensor outputs = model->forward(inputs);
        torch::Tensor loss = loss_fn(outputs, targets);

        loss.backward();
        optimizer.step();

        optimizer.zero_grad();

        std::cout << "Episode " << i << ": Loss = " << loss.item<float>() << "\n";
    }
}
