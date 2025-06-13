#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "globals.hpp"
#include "matrix.hpp"

class Node
{
public: // We made it public for simplicity. Students: ignore this, we are lazy :-)
    int rows, cols;
    std::shared_ptr<Matrix> values;
    std::shared_ptr<Matrix> grads;
    std::function<void()> backward_op;
    std::vector<Node *> dependencies;

public:
    Node(int m, int n) : rows(m), cols(n)
    {
        values = std::make_shared<Matrix>(m, n);
        grads = std::make_shared<Matrix>(m, n);
    }

    Node(const Matrix &values) : rows(values.numRows()), cols(values.numCols())
    {
        this->values = std::make_shared<Matrix>(values);
        this->grads = std::make_shared<Matrix>(values.numRows(), values.numCols());
    }

    // Copy constructor
    Node(const Node &other) : rows(other.rows), cols(other.cols),
                              values(other.values), grads(other.grads),
                              backward_op(other.backward_op),
                              dependencies(other.dependencies) {}

    Node *operator*(Node &other)
    {
        if (cols != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        Node *result = new Node((*(this->values)) * (*other.values));
        push_node(result);

        // Store shared pointers
        auto this_values = this->values;
        auto this_grads = this->grads;
        auto other_values = other.values;
        auto other_grads = other.grads;
        auto result_grads = result->grads;

        result->dependencies.push_back(this);
        result->dependencies.push_back(&other);

        result->backward_op = [this_values, this_grads, other_values, other_grads, result_grads]()
        {
            // dL/dA = dL/dC * B^T
            *this_grads = *this_grads + (*result_grads) * other_values->transpose();
            // dL/dB = A^T * dL/dC
            *other_grads = *other_grads + this_values->transpose() * (*result_grads);
        };

        return result;
    }

    Node *operator+(Node &other)
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }
        Node *result = new Node((*(this->values)) + (*other.values));
        push_node(result);

        // Store shared pointers
        auto this_grads = this->grads;
        auto other_grads = other.grads;
        auto result_grads = result->grads;

        result->dependencies.push_back(this);
        result->dependencies.push_back(&other);

        result->backward_op = [this_grads, other_grads, result_grads]()
        {
            *this_grads = *this_grads + *result_grads;
            *other_grads = *other_grads + *result_grads;
        };

        return result;
    }

    Node *operator-(Node &other)
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }
        Node *result = new Node((*(this->values)) - (*other.values));
        push_node(result);

        // Store shared pointers
        auto this_grads = this->grads;
        auto other_grads = other.grads;
        auto result_grads = result->grads;

        result->dependencies.push_back(this);
        result->dependencies.push_back(&other);

        result->backward_op = [this_grads, other_grads, result_grads]()
        {
            *this_grads = *this_grads + *result_grads;
            *other_grads = *other_grads + *result_grads;
        };

        return result;
    }

    // Rest of the class implementation (apply, transpose, backward, etc.) remains the same...
    Node *apply(std::function<double(double)> func, std::function<double(double)> func_derivative = nullptr)
    {
        Node *result = new Node((*(this->values)).apply(func));
        push_node(result);

        result->dependencies.push_back(this);

        auto this_values = this->values;
        auto this_grads = this->grads;
        auto result_values = result->values;
        auto result_grads = result->grads;

        if (func_derivative)
        {
            result->backward_op = [this_values, this_grads, result_grads, func_derivative,
                                   rows = this->rows, cols = this->cols]()
            {
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < cols; ++j)
                    {
                        double curr_grad = this_grads->get(i, j) +
                                           result_grads->get(i, j) * func_derivative(this_values->get(i, j));
                        this_grads->set(i, j, curr_grad);
                    }
                }
            };
        }

        return result;
    }

    Node *transpose()
    {
        Node *result = new Node((*(this->values)).transpose());
        push_node(result);

        auto this_grads = this->grads;
        auto result_grads = result->grads;

        result->dependencies.push_back(this);

        result->backward_op = [this_grads, result_grads]()
        {
            *this_grads = (*this_grads).transpose();
        };

        return result;
    }

    void backward()
    {
        if (backward_op)
            backward_op();
        for (const auto &dep : dependencies)
        {
            dep->backward();
        }
    }

    void zero_grad()
    {
        grads->fill(0.);
    }

    double get(int i, int j) const
    {
        return values->get(i, j);
    }

    void set(int i, int j, double value)
    {
        return values->set(i, j, value);
    }

    int getRows() const { return rows; }
    int getCols() const { return cols; }

    void print_grads()
    {
        std::cout << "Gradients:" << std::endl;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                std::cout << grads->get(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
};

inline Node *binary_cross_entropy(Node &predictions, Node &targets)
{
    if (predictions.rows != targets.rows || predictions.cols != targets.cols)
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    Node *loss = new Node(1, 1); // BCE loss is a scalar
    push_node(loss);

    loss->dependencies.push_back(&predictions);

    double total_loss = 0.0;
    for (int i = 0; i < predictions.rows; ++i)
    {
        for (int j = 0; j < predictions.cols; ++j)
        {
            double pred = predictions.values->get(i, j);
            double target = targets.values->get(i, j);
            total_loss += -target * std::log(pred + 1e-12) - (1 - target) * std::log(1 - pred + 1e-12);
        }
    }

    loss->values->set(0, 0, total_loss / (predictions.rows * predictions.cols));

    // Set up backward operation to modify the original predictions
    loss->backward_op = [&predictions, &targets]()
    {
        for (int i = 0; i < predictions.rows; ++i)
        {
            for (int j = 0; j < predictions.cols; ++j)
            {
                double pred = predictions.values->get(i, j);
                double target = targets.values->get(i, j);
                double grad = (-target / (pred + 1e-12) + (1 - target) / (1 - pred + 1e-12)) / (predictions.rows * predictions.cols);
                predictions.grads->set(i, j, predictions.grads->get(i, j) + grad);
            }
        }
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

// Dataset already split in batches.
// `x` contains a vector of batches of data, each column containing one input.
// `y` contains the corresponding outputs.
struct Dataset
{
    std::vector<Node> X;
    std::vector<Node> Y;
};

// MLP implementation
class MLP
{
private:
    Node W1, b1, W2, b2;
    double learning_rate;

public:
    MLP(int input_size, int hidden_size, int output_size, double lr)
        : W1(hidden_size, input_size), b1(hidden_size, 1),
          W2(output_size, hidden_size), b2(output_size, 1),
          learning_rate(lr)
    {
        // The bias `b1` and `b2` are initialized to zero by the constructor
        // which is appropriate. For the weight matrices, we want some
        // appropriately sampled random numbers so we call `initialize`.
        initialize(W1);
        initialize(W2);
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
        Node *z1 = *(W1 * input) + b1;
        Node *a1 = z1->apply(sigmoid, sigmoid_derivative);
        Node *z2 = *(W2 * *a1) + b2;
        return z2->apply(sigmoid, sigmoid_derivative);
    }

    void train(const Dataset &data, int epochs)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (size_t i = 0; i < data.X.size(); ++i)
            {
                Node input = data.X[i];
                Node target = data.Y[i];

                // Forward pass
                Node *output = forward(input);

                // Compute error
                // error = output - target;
                Node *error = binary_cross_entropy(*output, target);
                error->grads->set(0, 0, 1.);

                // Backward pass
                error->backward();

                // Update weights and biases using learning_rate
                W2.values->sub_mul(learning_rate, *W2.grads);
                b2.values->sub_mul(learning_rate, *b2.grads);
                W1.values->sub_mul(learning_rate, *W1.grads);
                b1.values->sub_mul(learning_rate, *b1.grads);

                // Reset gradients for the next iteration
                W1.zero_grad();
                b1.zero_grad();
                W2.zero_grad();
                b2.zero_grad();

                clear_nodes();
            }
            // std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
        }
    }
};
