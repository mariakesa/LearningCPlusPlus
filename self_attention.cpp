#include <iostream>
// Adjust this include path if needed based on your setup
#include </home/maria/LearningCPlusPlus/libs/eigen-3.4.0/Eigen/Dense>
#include <vector>

class SelfAttention {
public:
    // Constructor with input and output dimensions
    SelfAttention(int embed_dim) : embed_dim(embed_dim) {
        // Initialize the weight matrix for Q, K, and V with random values
        weights = Eigen::MatrixXd::Random(embed_dim, 3 * embed_dim);
    }

    // Method to print weights (for testing)
    void print_weights() const {
        std::cout << "Weight matrix for QKV:\n" << weights << "\n\n";
    }

    Eigen::MatrixXd get_QKV(const Eigen::MatrixXd& x) const {
        // Compute Q, K, and V by multiplying with the weight matrix
        Eigen::MatrixXd QKV = x * weights;
        return QKV;
    }

    std::vector<Eigen::MatrixXd> split_QKV(const Eigen::MatrixXd& QKV) const {
        // Get number of rows (T) from QKV matrix
        int T = QKV.rows();

        // Initialize each matrix with dimensions (T, embed_dim)
        std::vector<Eigen::MatrixXd> QKV_split(3, Eigen::MatrixXd(T, embed_dim));

        // Extract Q, K, and V using block operations
        QKV_split[0] = QKV.block(0, 0, T, embed_dim);                  // Q
        QKV_split[1] = QKV.block(0, embed_dim, T, embed_dim);           // K
        QKV_split[2] = QKV.block(0, 2 * embed_dim, T, embed_dim);       // V

        return QKV_split;
    }

private:
    int embed_dim;  
    Eigen::MatrixXd weights;      // Weight matrix for Q, K, V
};



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

        int embed_dim = 4;  // Example embedding dimension (C)

    // Create a SelfAttention object
    SelfAttention attention(embed_dim);

    // Print the initialized weight matrix
    attention.print_weights();

    Eigen::MatrixXd mat;

    mat=attention.get_QKV(tensor[0]);

    std::cout << "QKV matrix:\n" << mat << "\n\n";

    std::vector<Eigen::MatrixXd> QKV_split = attention.split_QKV(mat);

    std::cout << "Q matrix:\n" << QKV_split[0] << "\n\n";

    return 0;
}
