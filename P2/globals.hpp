#ifndef GLOBALS_H
#define GLOBALS_H
#include <vector>

class Node;

extern std::vector<Node *> all_nodes;

void push_node(Node *node);
void clear_nodes();

#endif // GLOBALS_H