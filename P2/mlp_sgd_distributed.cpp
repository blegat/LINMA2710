#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <unistd.h>

#include "globals.hpp"
#include "matrix.hpp"
#include "distributedmatrix.hpp"

class Node
{
public:
    int rows, cols;
    AbstractMatrix* values;
    AbstractMatrix* grads;
    std::function<void()> backward_op;
    std::vector<Node*> dependencies;

public:
    Node(int m, int n) : rows(m), cols(n)
    {
        values = new Matrix(m, n);
        grads = new Matrix(m, n);
    }

    ~Node() {
        delete values;
        delete grads;
    }

    Node(const Matrix &values) : rows(values.numRows()), cols(values.numCols())
    {
        this->values = new Matrix(values);
        this->grads = new Matrix(values.numRows(), values.numCols());
    }
    
    Node(const DistributedMatrix& dMatrix) : 
                                         rows(dMatrix.numRows()),
                                         cols(dMatrix.numCols())
    {
        values = new DistributedMatrix(dMatrix);
        
        // Create a zero matrix with same dimensions for gradients
        int numProcs;
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
        Matrix tempGrads(dMatrix.numRows(), dMatrix.numCols());
        tempGrads.fill(0.0);
        grads = new DistributedMatrix(tempGrads, numProcs);
    }

    // Copy constructor
    Node(const Node& other) : 
                            rows(other.rows), 
                            cols(other.cols),
                            values(other.values), 
                            grads(other.grads),
                            backward_op(other.backward_op),
                            dependencies(other.dependencies) {}

    // Matrix multiplication with a regular Matrix on the left and distributed on the right
    Node *operator*(Node &other)
    {
        if (cols != other.rows)
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        
        Matrix* this_values = dynamic_cast<Matrix*>(this->values);
        DistributedMatrix* other_values = dynamic_cast<DistributedMatrix*>(other.values);

        Node *result = new Node(multiply(*this_values, *other_values));
        push_node(result);

        // Store shared pointers
        auto this_grads = this->grads;
        auto other_grads = other.grads;
        auto result_grads = result->grads;

        result->dependencies.push_back(this);
        result->dependencies.push_back(&other);

        result->backward_op = [this_values, this_grads, other_values, other_grads, result_grads]()
        {
            // dL/dA = dL/dC * B^T
            DistributedMatrix* result_grads_ = dynamic_cast<DistributedMatrix*>(result_grads); 
            Matrix* this_grads_ = dynamic_cast<Matrix*>(this_grads);
            *this_grads_ = *this_grads_ + result_grads_->multiplyTransposed(*other_values);
            // dL/dB = A^T * dL/dC
            // Add to existing gradients
            DistributedMatrix* other_grads_ = dynamic_cast<DistributedMatrix*>(other_grads);
            *other_grads_ = DistributedMatrix::applyBinary(*other_grads_, multiply(this_values->transpose(), (*result_grads_)), 
                                              [](double a, double b) { return a + b; });
        };

        return result;
    }

    // Apply an element-wise function to the distributed matrix
    Node* apply(std::function<double(double)> func, 
                          std::function<double(double)> func_derivative = nullptr)
    {
        DistributedMatrix* values_ = dynamic_cast<DistributedMatrix*>(values);
        Node* result = new Node(values_->apply(func));
        push_node(result);

        result->dependencies.push_back(this);

        // Store needed pointers for backpropagation
        auto values_ptr = values_;
        auto this_grads = grads;
        auto result_grads = result->grads;

        if (func_derivative)
        {
            result->backward_op = [values_ptr, this_grads, result_grads, func_derivative]()
            {
                // Apply the derivative function and multiply by the incoming gradient
                auto derivative_func = [func_derivative](double x) { return func_derivative(x); };
                
                // Create a matrix with derivatives
                DistributedMatrix derivatives = values_ptr->apply(derivative_func);
                
                // Multiply element-wise with incoming gradients
                DistributedMatrix* result_grads_ = dynamic_cast<DistributedMatrix*>(result_grads);
                DistributedMatrix gradient = DistributedMatrix::applyBinary(*result_grads_, derivatives, 
                                                              [](double a, double b) { return a * b; });
                
                // Add to existing gradients
                DistributedMatrix* this_grads_ = dynamic_cast<DistributedMatrix*>(this_grads);
                *this_grads_ = DistributedMatrix::applyBinary(*this_grads_, gradient, 
                                                  [](double a, double b) { return a + b; });
            };
        }

        return result;
    }

    void backward()
    {
        if (backward_op)
            backward_op();
        for (const auto& dep : dependencies)
        {
            dep->backward();
        }
    }

    void zero_grad()
    {
        Matrix* grads_ = dynamic_cast<Matrix*>(grads);
        grads_->fill(0.);
    }

    double get(int i, int j) const
    {
        return values->get(i, j);
    }

    void set(int i, int j, double value)
    {
        values->set(i, j, value);
    }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

// Binary cross entropy for distributed matrices
inline Node* binary_cross_entropy(Node& predictions, Node& targets)
{
    if (predictions.rows != targets.rows || predictions.cols != targets.cols)
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    // Create a new distributed node with same dimensions as predictions
    DistributedMatrix* prediction_values = dynamic_cast<DistributedMatrix*>(predictions.values);
    Node* loss = new Node(*prediction_values);
    push_node(loss);

    loss->dependencies.push_back(&predictions);

    // Apply BCE loss computation element-wise
    auto bce_func = [](double pred, double target) {
        return -target * std::log(pred + 1e-12) - (1 - target) * std::log(1 - pred + 1e-12);
    };
    
    // Apply the loss function element-wise
    DistributedMatrix* target_values = dynamic_cast<DistributedMatrix*>(targets.values);
    DistributedMatrix* loss_values = dynamic_cast<DistributedMatrix*>(loss->values);
    *loss_values = DistributedMatrix::applyBinary(*prediction_values, *target_values, bce_func);
    
    // Set up backward operation to modify the original predictions
    loss->backward_op = [&predictions, &targets]()
    {
        // BCE gradient computation: (-target/(pred+ε) + (1-target)/(1-pred+ε))
        auto bce_grad_func = [](double pred, double target) {
            return (-target / (pred + 1e-12) + (1 - target) / (1 - pred + 1e-12));
        };
        
        // Apply gradient computation element-wise
        DistributedMatrix* prediction_values = dynamic_cast<DistributedMatrix*>(predictions.values);
        DistributedMatrix* target_values = dynamic_cast<DistributedMatrix*>(targets.values);
        DistributedMatrix gradient = DistributedMatrix::applyBinary(*prediction_values, *target_values, bce_grad_func);
        
        // Add to existing gradients
        DistributedMatrix* prediction_grads = dynamic_cast<DistributedMatrix*>(predictions.grads);
        (*prediction_grads) = DistributedMatrix::applyBinary(*prediction_grads, gradient, 
                                                 [](double a, double b) { return a + b; });
        
    };

    return loss;
}

// Helper functions
inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_derivative(double x)
{
    double s = sigmoid(x);
    return s * (1 - s);
}

// Distributed dataset structure
struct Dataset
{
    DistributedMatrix X;
    DistributedMatrix Y;
    Dataset(DistributedMatrix A, DistributedMatrix B) : X(A), Y(B)
    {}
};

// MLP implementation with distributed data processing
class MLP
{
private:
    Node W1, W2;
    double learning_rate;
    int rank, numProcesses;

public:
    MLP(int input_size, int hidden_size, int output_size, double lr)
        : W1(hidden_size, input_size),
          W2(output_size, hidden_size),
          learning_rate(lr),
          rank(0),
          numProcesses(0)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
        if (rank==0) {
            initialize(W1);
            initialize(W2);
        }
        synchronize();
    }

    void synchronize() {
        Matrix* W1_values = dynamic_cast<Matrix*>(W1.values);
        Matrix* W2_values = dynamic_cast<Matrix*>(W2.values);
        sync_matrix(W1_values, rank, 0);
        sync_matrix(W2_values, rank, 0);
    }

    static void initialize(Node &matrix)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        int fan_in = matrix.getCols();
        int fan_out = matrix.getRows();
        double stddev = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<> dis(0.0, stddev);
        for (int i = 0; i < matrix.getRows(); ++i)
        {
            for (int j = 0; j < matrix.getCols(); ++j)
            {
                matrix.values->set(i, j, dis(gen));
            }
        }
    }

    Node *forward(Node &input)
    {
        
        Node *z1 = W1 * input;
        Node *a1 = z1->apply(sigmoid, sigmoid_derivative);
        Node *z2 = W2 * *a1;
        return z2->apply(sigmoid, sigmoid_derivative);
    }

    void train(const Dataset& data, int epochs)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {   
            Node input = Node(data.X);
            Node target = Node(data.Y);

            // Forward pass
            Node* output = forward(input);

            // Compute loss
            Node* losses = binary_cross_entropy(*output, target);
            
            // Set gradient to 1 to start backpropagation
            Matrix gradMatrix(losses->rows, losses->cols);
            gradMatrix.fill(1.0 / (losses->rows * losses->cols)); // Normalize by size
            *losses->grads = DistributedMatrix(gradMatrix, numProcesses);
            
            // Compute total loss for reporting
            DistributedMatrix* losses_values = dynamic_cast<DistributedMatrix*>(losses->values);
            double loss = losses_values->sum()/(losses->rows * losses->cols);

            // Backward pass
            losses->backward();

            // Update weights and biases using learning_rate (all processes already have a synchronized gradient)
            Matrix* W2_values = dynamic_cast<Matrix*>(W2.values);
            Matrix* W2_grads = dynamic_cast<Matrix*>(W2.grads);
            Matrix* W1_values = dynamic_cast<Matrix*>(W1.values);
            Matrix* W1_grads = dynamic_cast<Matrix*>(W1.grads);
            W2_values->sub_mul(learning_rate, *W2_grads);
            W1_values->sub_mul(learning_rate, *W1_grads);

            // Reset gradients for the next iteration (done on all processes)
            W1.zero_grad();
            W2.zero_grad();
            
            clear_nodes();
            
            
            if (rank == 0 && ((epoch + 1) % 100 == 0)) {
                std::cout << "Epoch " << epoch + 1 << " completed. Average loss: " 
                          << loss << std::endl;
            }
        }
    }
};
