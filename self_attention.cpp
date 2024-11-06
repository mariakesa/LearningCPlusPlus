#include <iostream>
// Adjust this include path if needed based on your setup
#include </home/maria/LearningCPlusPlus/libs/eigen-3.4.0/Eigen/Dense>
#include <vector>

int main() {
    // Dimensions for batch, sequence length, and embedding size
    int B = 2; // Batch size
    int T = 3; // Sequence length
    int C = 4; // Embedding dimension

    // Create a batch of matrices (B x T x C)
    std::vector<Eigen::MatrixXd> tensor(B, Eigen::MatrixXd(T, C));

    // Initialize each matrix in the batch with random values
    for (int i = 0; i < B; ++i) {
        tensor[i] = Eigen::MatrixXd::Random(T, C);
    }

    // Print out each matrix in the batch
    for (int i = 0; i < B; ++i) {
        std::cout << "Batch " << i << ":\n" << tensor[i] << "\n\n";
    }

    return 0;
}
