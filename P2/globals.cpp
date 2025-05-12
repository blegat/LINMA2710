#include "globals.hpp"
#include "mlp_sgd_distributed.cpp"

std::vector<Node *> all_nodes;

// Function to push a Node* into the vector (initializing if nullptr)
void push_node(Node *node)
{
    all_nodes.push_back(node);
}

// Function to clear the vector and free allocated memory
void clear_nodes()
{
    for (Node *node : all_nodes)
    {
        delete node; // Free dynamically allocated memory
    }
    all_nodes.clear();
}