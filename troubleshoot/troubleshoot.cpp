#include <cstdlib>
#include <iostream>

int main()
{
    int i = 0;
    int *a = new int[2];
    a[-1] = 1000000;
    std::cout << i << std::endl;
    a[i] = 0;
    return EXIT_SUCCESS;
}
