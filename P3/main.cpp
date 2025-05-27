#include "matrix_opencl.hpp"
#include "mlp_sgd.cpp" // Note: Including .cpp is generally discouraged, prefer linking .o files. Included here to match original structure.
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cmath>   // For std::exp, std::log
#include <limits> // For std::numeric_limits

// Helper function to print a matrix (copies to host first)
void printMatrix(const std::string& label, const MatrixCL& mat) {
    std::cout << label << " (" << mat.numRows() << "x" << mat.numCols() << "):\n";
    try {
        std::vector<float> host_data = mat.copyToHost();
        for (int i = 0; i < mat.numRows(); ++i) {
            std::cout << "  [";
            for (int j = 0; j < mat.numCols(); ++j) {
                std::cout << " " << host_data[i * mat.numCols() + j];
            }
            std::cout << " ]\n";
        }
         std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error printing matrix: " << e.what() << std::endl;
    }
}

// Helper function for approximate float comparison
bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Helper function to verify matrix contents
bool verifyMatrix(const std::string& label, const MatrixCL& mat, const std::vector<float>& expected, float epsilon = 1e-5f) {
    std::cout << "Verifying " << label << "..." << std::endl;
    if (static_cast<size_t>(mat.numRows() * mat.numCols()) != expected.size()) {
        std::cerr << "Verification failed: Dimension mismatch for " << label << ". Got "
                  << mat.numRows() << "x" << mat.numCols() << ", expected " << expected.size() << " elements." << std::endl;
        return false;
    }
    try {
        std::vector<float> actual = mat.copyToHost();
        bool match = true;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!approxEqual(actual[i], expected[i], epsilon)) {
                std::cerr << "Verification failed for " << label << " at index " << i
                          << ". Got " << actual[i] << ", expected " << expected[i] << std::endl;
                match = false;
                // Don't break, report all mismatches if desired, or break here for efficiency
                 break;
            }
        }
        if (match) {
            std::cout << label << " verified successfully." << std::endl;
        } else {
             std::cout << label << " verification failed." << std::endl;
        }
        return match;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error verifying matrix " << label << ": " << e.what() << std::endl;
        return false;
    }
}


void test_mlp_training(cl::Context context, cl::CommandQueue queue)
{
    std::cout << "\n--- Starting MLP Training Test ---" << std::endl;

    // Create a simple XOR dataset
    // X: 3 features x 4 samples (adding a bias feature '1.0f')
    std::vector<float> x_host_data = {
        0.0f, 0.0f, 1.0f, 1.0f, // Feature 1
        0.0f, 1.0f, 0.0f, 1.0f, // Feature 2
        1.0f, 1.0f, 1.0f, 1.0f  // Bias Feature
    };
    // Y: 1 output x 4 samples
    std::vector<float> y_host_data = {
        0.0f,       // Output 0 for 0,0
        1.0f,       // Output 1 for 0,1
        1.0f,       // Output 1 for 1,0
        0.0f        // Output 0 for 1,1
    };

    // Create MatrixCL objects on the device for the single batch
    // Dimensions: X(features, samples), Y(output_size, samples)
    MatrixCL batch_x_mat(3, 4, context, queue, &x_host_data); // 3 features, 4 samples
    MatrixCL batch_y_mat(1, 4, context, queue, &y_host_data); // 1 output, 4 samples

    std::cout << "Created X batch matrix on device:" << std::endl;
    printMatrix("Batch X", batch_x_mat);
    std::cout << "Created Y batch matrix on device:" << std::endl;
    printMatrix("Batch Y", batch_y_mat);

    // Distribute the data
    Dataset data = Dataset(batch_x_mat, batch_y_mat);

    // Create and train the model
    // MLP(input_size, hidden_size, output_size, learning_rate, context, queue)
    MLP model(3, 128, 1, 1.0f, context, queue);

    model.train(data, 2000); // Keep epochs as before

    std::cout << "--- MLP Training Completed ---" << std::endl;

    // Optional: Test the trained model (Forward pass only)
    std::cout << "--- Testing Trained MLP ---" << std::endl;
    Node final_input_node(batch_x_mat); // Wrap test input in a Node
    Node* final_output_node = model.forward(final_input_node);
    printMatrix("MLP Output on Training Data", *final_output_node->values);
    clear_nodes(); // Clean up graph nodes created by forward pass

}


int main() {
    try {
        // 1. --- OpenCL Setup ---
        std::cout << "--- OpenCL Setup ---" << std::endl;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found." << std::endl;
            return 1;
        }
        cl::Platform platform = platforms.front();
        std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cout << "No GPU found, trying CPU..." << std::endl;
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty()) {
                std::cerr << "No OpenCL devices found." << std::endl;
                return 1;
            }
        }
        cl::Device device = devices.front();
        std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::Context context(device);
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE); // Keep profiling enabled

        std::vector<cl::Device> devices_to_init = {device};
        try {
            MatrixCL::initializeKernels(context, devices_to_init);
            std::cout << "Kernel initialization successful." << std::endl;
        } catch (const std::exception& e) {
            // Catching std::exception here because initializeKernels wraps cl::Error
            std::cerr << "FATAL ERROR during kernel initialization: " << e.what() << std::endl;
            // If the error was a BuildError, the log should have been printed
            // by the loadAndBuildProgram function within initializeKernels.
            return 1;
        }

        // 2. --- Basic Matrix Operations Test ---
        std::cout << "\n--- Basic Matrix Operations Test ---" << std::endl;

        std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // 2x3
        std::vector<float> dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}; // 2x3
        std::vector<float> dataC = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}; // 3x3

        MatrixCL matA(2, 3, context, queue, &dataA);
        MatrixCL matB(2, 3, context, queue, &dataB);
        MatrixCL matC(3, 3, context, queue, &dataC);
        MatrixCL matD(2, 3, context, queue); // Initialized to 0 by constructor

        printMatrix("Matrix A (original)", matA);
        printMatrix("Matrix B", matB);
        printMatrix("Matrix C (3x3)", matC);
        printMatrix("Matrix D (initially zero)", matD);

        // Test fill
        matD.fill(5.5f);
        printMatrix("Matrix D after fill(5.5)", matD);
        assert(verifyMatrix("Matrix D fill", matD, {5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f}));

        // Test Copy Constructor
        MatrixCL matA_copy(matA);
        printMatrix("Matrix A Copy (via copy constructor)", matA_copy);
        assert(verifyMatrix("Matrix A Copy Ctor", matA_copy, dataA));

        // Test Copy Assignment Operator
        MatrixCL matD_assigned(1, 1, context, queue); // Create with different dimensions
        matD_assigned = matD;
        printMatrix("Matrix D Assigned (via assignment operator)", matD_assigned);
        assert(verifyMatrix("Matrix D Assignment Op", matD_assigned, {5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f}));


        // Test Addition
        MatrixCL matAdd = matA + matB;
        printMatrix("Matrix A + B", matAdd);
        assert(verifyMatrix("Matrix A + B", matAdd, {8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f}));

        // Test Transpose
        MatrixCL matATrans = matA.transpose();
        printMatrix("Matrix A Transposed", matATrans); // Should be 3x2
        assert(verifyMatrix("Matrix A Transposed", matATrans, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));

        // Test Matrix Multiplication: A(2x3) * C(3x3) -> Result(2x3)
        // Expected: [ (1*1+2*4+3*7) (1*2+2*5+3*8) (1*3+2*6+3*9) ] = [ 30 36 42 ]
        //           [ (4*1+5*4+6*7) (4*2+5*5+6*8) (4*3+5*6+6*9) ] = [ 66 81 96 ]
        MatrixCL matMul = matA * matC;
        printMatrix("Matrix A * C", matMul);
        assert(verifyMatrix("Matrix A * C", matMul, {30.0f, 36.0f, 42.0f, 66.0f, 81.0f, 96.0f}));


        // Test sub_mul: matA_copy = matA_copy - 2.0 * matB
        // matA_copy starts as {1, 2, 3, 4, 5, 6}
        // 2.0 * matB = {14, 16, 18, 20, 22, 24}
        // Expected result = {-13, -14, -15, -16, -17, -18}
        printMatrix("Matrix A Copy before sub_mul", matA_copy);
        matA_copy.sub_mul(2.0f, matB);
        printMatrix("Matrix A Copy after sub_mul(2.0, B)", matA_copy);
        assert(verifyMatrix("Matrix A Copy sub_mul", matA_copy, {-13.0f, -14.0f, -15.0f, -16.0f, -17.0f, -18.0f}));


        // 3. --- Neural Network Related Operations Test ---
        std::cout << "\n--- Neural Network Operations Test ---" << std::endl;
        std::vector<float> dataSigmoidInput = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}; // 1x5
        MatrixCL matSigmoidInput(1, 5, context, queue, &dataSigmoidInput);
        printMatrix("Matrix Sigmoid Input", matSigmoidInput);

        // Test sigmoid()
        MatrixCL matSigmoidOutput = matSigmoidInput.sigmoid();
        printMatrix("Matrix Sigmoid Output", matSigmoidOutput);
        auto sigmoid = [](float x){ return 1.0f / (1.0f + std::exp(-x)); };
        assert(verifyMatrix("Matrix Sigmoid Output Verify", matSigmoidOutput,
                     {sigmoid(-2.0f), sigmoid(-1.0f), sigmoid(0.0f), sigmoid(1.0f), sigmoid(2.0f)}));

        // Test sigmoid_backward()
        // grad_acc starts as 0. grad_acc += output_gradient * sigmoid'(input)
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        MatrixCL matGradAcc(1, 5, context, queue); // Initialize gradients accumulator to 0
        matGradAcc.fill(0.0f);
        std::vector<float> dataOutputGrad = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f}; // 1x5
        MatrixCL matOutputGrad(1, 5, context, queue, &dataOutputGrad);
        printMatrix("Matrix Gradient Accumulator (before)", matGradAcc);
        printMatrix("Matrix Output Gradient (for sigmoid backward)", matOutputGrad);

        matGradAcc.sigmoid_backward(matSigmoidInput, matOutputGrad);
        printMatrix("Matrix Gradient Accumulator (after sigmoid_backward)", matGradAcc);

        std::vector<float> expectedSigmoidGrad(5);
        for(size_t i = 0; i < dataSigmoidInput.size(); ++i) {
            float s_in = sigmoid(dataSigmoidInput[i]);
            float s_prime = s_in * (1.0f - s_in);
            expectedSigmoidGrad[i] = dataOutputGrad[i] * s_prime;
        }
        assert(verifyMatrix("Matrix Sigmoid Backward Verify", matGradAcc, expectedSigmoidGrad));


        // Test binary_cross_entropy()
        std::vector<float> dataPreds = {0.1f, 0.8f, 0.3f, 0.9f}; // 1x4 Predictions
        std::vector<float> dataTargets = {0.0f, 1.0f, 0.0f, 1.0f}; // 1x4 Targets
        MatrixCL matPreds(1, 4, context, queue, &dataPreds);
        MatrixCL matTargets(1, 4, context, queue, &dataTargets);
        printMatrix("Matrix Predictions (for BCE)", matPreds);
        printMatrix("Matrix Targets (for BCE)", matTargets);

        MatrixCL matBCE = matPreds.binary_cross_entropy(matTargets); // Returns 1x4 elementwise BCE loss
        printMatrix("Matrix BCE Loss (1x4)", matBCE);

        // Recompute the four pointwise BCE terms (no averaging):
        float eps = 1e-8f;
        auto safe_log = [&](float x){ return std::log(std::max(x, eps)); };
        float t0 = -(0.0f * safe_log(0.1f) + 1.0f * safe_log(0.9f));  // sample 0
        float t1 = -(1.0f * safe_log(0.8f) + 0.0f * safe_log(0.2f));  // sample 1
        float t2 = -(0.0f * safe_log(0.3f) + 1.0f * safe_log(0.7f));  // sample 2
        float t3 = -(1.0f * safe_log(0.9f) + 0.0f * safe_log(0.1f));  // sample 3
        std::vector<float> expectedBCE = { t0, t1, t2, t3 };
        assert(verifyMatrix("Matrix BCE Loss Verify", matBCE, expectedBCE));


        // Test binary_cross_entropy_backward()
        // grad_acc += (1/N) * (-target/pred + (1-target)/(1-pred)) * output_gradient_of_loss (assume 1)
        MatrixCL matBCEGradAcc(1, 4, context, queue);
        matBCEGradAcc.fill(0.0f); // Start with zero gradients
        printMatrix("Matrix Grad Acc (before BCE backward)", matBCEGradAcc);

        matBCEGradAcc.binary_cross_entropy_backward(matPreds, matTargets);
        printMatrix("Matrix Grad Acc (after BCE backward)", matBCEGradAcc);

        float inv_N = 1.0f / 4.0f;
        // Match kernel's potential clamping/epsilon use if necessary for verification
        auto grad_term = [&](float pred, float targ) {
            float denom1 = std::max(pred, eps);
            float denom2 = std::max(1.0f - pred, eps);
             return -(targ / denom1 - (1.0f - targ) / denom2);
        };
        float bce_grad1 = inv_N * grad_term(0.1f, 0.0f); // inv_N * (1 / 0.9)
        float bce_grad2 = inv_N * grad_term(0.8f, 1.0f); // inv_N * (-1 / 0.8)
        float bce_grad3 = inv_N * grad_term(0.3f, 0.0f); // inv_N * (1 / 0.7)
        float bce_grad4 = inv_N * grad_term(0.9f, 1.0f); // inv_N * (-1 / 0.9)
        assert(verifyMatrix("Matrix BCE Backward Verify", matBCEGradAcc, {bce_grad1, bce_grad2, bce_grad3, bce_grad4}));


        // 4. --- Run MLP Training Test ---
        test_mlp_training(context, queue);


    } catch (const cl::BuildError& err) { // Catch specific build error first
        std::cerr << "OpenCL Build Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog()) {
            std::cerr << "Device " << pair.first.getInfo<CL_DEVICE_NAME>() << " Build Log:" << std::endl;
            std::cerr << pair.second << std::endl;
        }
        return 1;
    } catch (const cl::Error& err) { // Catch other OpenCL errors
        std::cerr << "OpenCL Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) { // Catch standard exceptions
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll OpenCL Matrix and MLP tests completed successfully." << std::endl;
    return 0;
}