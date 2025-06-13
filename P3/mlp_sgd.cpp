#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <functional>
#include <cmath>
#include <stdexcept>


#include "globals.hpp"
#include "matrix_opencl.hpp"

// --- Node Class using MatrixCL ---
class Node
{
public: // We made it public for simplicity.
    int rows, cols;
    std::shared_ptr<MatrixCL> values;
    std::shared_ptr<MatrixCL> grads;
    std::function<void()> backward_op;
    std::vector<Node *> dependencies;

    // Keep track of the context and queue needed by MatrixCL
    cl::Context context_;
    cl::CommandQueue queue_;

public:
    // Constructor requiring context and queue
    Node(int m, int n, cl::Context context, cl::CommandQueue queue)
        : rows(m), cols(n), context_(context), queue_(queue)
    {
        values = std::make_shared<MatrixCL>(m, n, context, queue);
        grads = std::make_shared<MatrixCL>(m, n, context, queue);
        grads->fill(0.0f); // Initialize gradients to zero
    }

    // Constructor from existing MatrixCL (shares context/queue)
    Node(const MatrixCL &initial_values)
        : rows(initial_values.numRows()), cols(initial_values.numCols()),
          context_(initial_values.getContext()), queue_(initial_values.getQueue())
    {
        this->values = std::make_shared<MatrixCL>(initial_values); // Copy constructor
        this->grads = std::make_shared<MatrixCL>(rows, cols, context_, queue_);
        this->grads->fill(0.0f); // Initialize gradients to zero
    }

    // Copy constructor
    Node(const Node &other) : rows(other.rows), cols(other.cols),
                              values(other.values), grads(other.grads),
                              backward_op(other.backward_op),
                              dependencies(other.dependencies),
                              context_(other.context_), queue_(other.queue_) {}

    // Destructor
    ~Node() = default;

    // --- Operations using MatrixCL ---

    Node *operator*(Node &other)
    {
        if (cols != other.rows)
        {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        // Perform multiplication using MatrixCL's operator*
        MatrixCL result_values = (*(this->values)) * (*other.values);
        Node *result = new Node(result_values);
        push_node(result);

        // Store shared pointers needed for backward pass
        auto this_values = this->values;
        auto this_grads = this->grads;
        auto other_values = other.values;
        auto other_grads = other.grads;
        auto result_grads = result->grads; // Gradient of the *output* of this op

        result->dependencies.push_back(this);
        result->dependencies.push_back(&other);

        // Backward operation for multiplication
        result->backward_op = [this_values, this_grads, other_values, other_grads, result_grads]() mutable
        {
            // dL/dA = dL/dC * B^T
            MatrixCL grad_a_update = (*result_grads) * other_values->transpose();
            *this_grads = *this_grads + grad_a_update;

            // dL/dB = A^T * dL/dC
            MatrixCL grad_b_update = this_values->transpose() * (*result_grads);
            *other_grads = *other_grads + grad_b_update;
        };

        return result;
    }

    Node *operator+(Node &other)
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }
        MatrixCL result_values = (*(this->values)) + (*other.values);
        Node *result = new Node(result_values);
        push_node(result);

        auto this_grads = this->grads;
        auto other_grads = other.grads;
        auto result_grads = result->grads;

        result->dependencies.push_back(this);
        result->dependencies.push_back(&other);

        // Backward operation for addition
        result->backward_op = [this_grads, other_grads, result_grads]() mutable
        {
            // dL/dA = dL/dC * 1 = dL/dC
            *this_grads = *this_grads + *result_grads;
            // dL/dB = dL/dC * 1 = dL/dC
            *other_grads = *other_grads + *result_grads;
        };

        return result;
    }

    // Use MatrixCL's specific sigmoid operation
    Node *sigmoid()
    {
        // Apply sigmoid using MatrixCL method
        MatrixCL result_values = this->values->sigmoid();
        Node *result = new Node(result_values);
        push_node(result);

        result->dependencies.push_back(this);

        // Capture necessary values for backward pass
        auto this_values = this->values;   // Input values to sigmoid (z)
        auto this_grads = this->grads;     // Gradient accumulator for sigmoid input (dL/dz)
        auto result_grads = result->grads; // Gradient of the loss w.r.t sigmoid output (dL/da)

        // Backward operation using MatrixCL's sigmoid_backward
        result->backward_op = [this_values, this_grads, result_grads]() mutable
        {
            this_grads->sigmoid_backward(*this_values, *result_grads);
        };

        return result;
    }

    Node *transpose()
    {
        MatrixCL result_values = this->values->transpose();
        Node *result = new Node(result_values);
        push_node(result);

        auto this_grads = this->grads;
        auto result_grads = result->grads;

        result->dependencies.push_back(this);

        // Backward operation for transpose
        result->backward_op = [this_grads, result_grads]() mutable
        {
            *this_grads = *this_grads + result_grads->transpose();
        };

        return result;
    }

    // Recursive backward pass trigger
    void backward()
    {
        // Execute the operation's specific backward step
        if (backward_op)
        {
            backward_op();
        }
        for (Node *dep : dependencies)
        {
            if (dep) { // Basic null check
                 dep->backward();
            }
        }
    }

    // Zero out gradients using MatrixCL::fill
    void zero_grad()
    {
        if (grads)
        {
            grads->fill(0.0f);
        }
    }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    cl::Context getContext() const { return context_; }
    cl::CommandQueue getQueue() const { return queue_; }


    // Print gradients (requires host transfer)
    void print_grads()
    {
        if (!grads) return;
        std::cout << "Gradients (" << rows << "x" << cols << "):" << std::endl;
        try {
            std::vector<float> host_grads = grads->copyToHost();
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    std::cout << host_grads[i * cols + j] << " ";
                }
                std::cout << std::endl;
            }
        } catch (const cl::Error& err) {
            std::cerr << "OpenCL error copying gradients to host: " << err.what() << " (" << err.err() << ")" << std::endl;
        } catch (const std::exception& e) {
             std::cerr << "Error copying gradients to host: " << e.what() << std::endl;
        }
    }

     // Print values (requires host transfer)
    void print_values()
    {
        if (!values) return;
        std::cout << "Values (" << rows << "x" << cols << "):" << std::endl;
         try {
            std::vector<float> host_values = values->copyToHost();
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    std::cout << host_values[i * cols + j] << " ";
                }
                std::cout << std::endl;
            }
        } catch (const cl::Error& err) {
            std::cerr << "OpenCL error copying values to host: " << err.what() << " (" << err.err() << ")" << std::endl;
        } catch (const std::exception& e) {
             std::cerr << "Error copying values to host: " << e.what() << std::endl;
        }
    }
};

// --- Loss Function using MatrixCL ---

inline Node *binary_cross_entropy(Node &predictions, Node &targets)
{
    if (predictions.rows != targets.rows || predictions.cols != targets.cols)
    {
        throw std::invalid_argument("BCE: Predictions and targets must have the same dimensions.");
    }
    if (!predictions.values || !targets.values || !predictions.grads) {
         throw std::runtime_error("BCE: Invalid node values or grads pointers.");
    }

    // Use MatrixCL's binary_cross_entropy method for the forward pass.
    // It should return a 1x1 MatrixCL containing the scalar loss.
    MatrixCL loss_value_matrix = predictions.values->binary_cross_entropy(*targets.values);

    // Create a Node to hold the loss value.
    Node *loss_node = new Node(loss_value_matrix);
    push_node(loss_node); // Manage node lifecycle

    loss_node->dependencies.push_back(&predictions);
    // Targets are constants in this context, typically no dependency added.

    // Store pointers needed for the backward pass
    auto pred_values = predictions.values;
    auto pred_grads = predictions.grads; // Gradients w.r.t predictions (dL/dPred)
    auto target_values = targets.values;
    auto loss_grads = loss_node->grads; // Gradient *of* the loss node (dL/dL), assumed to be 1 here.

    // Set up backward operation using MatrixCL's bce_backward
    loss_node->backward_op = [pred_values, pred_grads, target_values, loss_grads]() mutable
    {
        pred_grads->binary_cross_entropy_backward(*pred_values, *target_values); // Note does not use loss grads, directly backward from the prediction and targets
    };

    return loss_node;
}


// --- Dataset Structure ---
struct Dataset
{
    MatrixCL X;
    MatrixCL Y;
    Dataset(MatrixCL A, MatrixCL B) : X(A), Y(B)
    {}
};

// --- MLP Implementation using MatrixCL ---
class MLP
{
private:
    Node W1, W2;
    float learning_rate; // Use float
    cl::Context context_; // Store context
    cl::CommandQueue queue_; // Store queue

public:
    // Constructor requires OpenCL context and queue
    MLP(int input_size, int hidden_size, int output_size, float lr,
        cl::Context context, cl::CommandQueue queue)
        : // Initialize Nodes with context and queue
          W1(hidden_size, input_size, context, queue),
          W2(output_size, hidden_size, context, queue),
          learning_rate(lr),
          context_(context), queue_(queue)
    {
        // Initialize weights using the adapted initialize method
        initialize(W1);
        initialize(W2);
    }

    // Initialize weights using Xavier/Glorot initialization on the host, then transfer
    static void initialize(Node &matrix_node)
    {
        std::vector<float> host_data(matrix_node.rows * matrix_node.cols);
        std::random_device rd;
        std::mt19937 gen(rd());
        int fan_in = matrix_node.getCols();
        int fan_out = matrix_node.getRows();
        float stddev = std::sqrt(2.0f / (float)(fan_in + fan_out));
        std::normal_distribution<float> dis(0.0f, stddev); // Use float distribution

        for (int i = 0; i < matrix_node.rows; ++i)
        {
            for (int j = 0; j < matrix_node.cols; ++j)
            {
                host_data[i * matrix_node.cols + j] = dis(gen);
            }
        }

        // Re-create the MatrixCL with initial data from host
        matrix_node.values = std::make_shared<MatrixCL>(
            matrix_node.rows, matrix_node.cols,
            matrix_node.getContext(), matrix_node.getQueue(),
            &host_data // Pass pointer to host data
        );
        // Ensure gradients are reset after potentially re-creating values matrix
         matrix_node.zero_grad();
    }

    // Forward pass using MatrixCL operations
    Node *forward(Node &input)
    {
        // z1 = W1 * input
        Node *z1= W1 * input;

        // a1 = sigmoid(z1)
        Node *a1 = z1->sigmoid(); // Use the Node's sigmoid method

        // z2 = W2 * a1
        Node *z2 = W2 * *a1;

        // output = sigmoid(z2)
        Node *output = z2->sigmoid();

        return output;
    }

    // Training loop
    void train(const Dataset &data, int epochs)
    {
        std::cout << "Starting training for " << epochs << " epochs..." << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            // Get current batch
            Node input = Node(data.X);
            Node target = Node(data.Y);

            // --- Forward Pass ---
            Node *output = forward(input);

            // --- Compute Loss ---
            Node *loss_node = binary_cross_entropy(*output, target);

            // --- Backward Pass ---
            // Trigger backpropagation starting from the loss node
            loss_node->backward();

            // --- Update Weights and Biases ---
            // Use MatrixCL's sub_mul for update: W = W - lr * grad(W)
            W2.values->sub_mul(learning_rate, *W2.grads);
            W1.values->sub_mul(learning_rate, *W1.grads);

            // --- Reset Gradients for Next Iteration ---
            W1.zero_grad();
            W2.zero_grad();

            // --- Track loss ---
            float loss = 0.0f;
            try {
                std::vector<float> current_loss_vec = loss_node->values->copyToHost();

                double total_loss_sum = 0.0;
                for (float term : current_loss_vec) {
                    total_loss_sum += term;
                }
                
                loss = total_loss_sum/current_loss_vec.size();
                } catch (const cl::Error& err) {
                std::cerr << "OpenCL error getting loss value: " << err.what() << " (" << err.err() << ")" << std::endl;
                } catch (const std::exception& e) {
                std::cerr << "Error getting loss value: " << e.what() << std::endl;
            }
            if ((epoch+1)%100==0) {
                std::cout << "Epoch " << epoch+1 << "/" << epochs
                        << " completed. Average Loss: " << (loss)
                        << std::endl;
            }


            // --- Clean up compute graph nodes ---
            clear_nodes();

        } // End epoch loop
        std::cout << "Training finished." << std::endl;
    }
};