# Project 1

This project is the first part of a three-part series focusing on matrix multiplication and its optimization. In this initial phase, we implement a basic Matrix class with fundamental operations (addition, subtraction, and multiplication) in C++. The following parts will explore optimizing the multiplication operation using parallel computing on CPU (with MPI) and GPU architectures.

For this first project, you will need to implement your own benchmark to test the performance of your implementation. The benchmark should measure the execution time of matrix multiplication operations for different matrix sizes.

## Requirements

This project must be implemented using only the C++ Standard Library. No external libraries or dependencies are allowed. This ensures portability and helps focus on fundamental implementation concepts.

## Tasks
The `Matrix`class should implement the following member functions (or *methods*)
- a constructor `Matrix(int rows, int cols)`
- a copy constructor `Matrix(const Matrix &other)`
- the numRows function `int numRows() const`
- the numCols function `int numCols() const`
- the get function `double get(int i, int j) const`
- the set function `void set(int i, int j, double value)`
- the fill function `void fill(double value)`
- the addition operator `Matrix operator+(const Matrix &other) const`
- the subtraction operator `Matrix operator-(const Matrix &other) const`
- the multiplication operator `Matrix operator*(const Matrix &other) const`
- the scalar multiplication operator `Matrix operator*(double scalar) const`
- the transpose function `Matrix transpose() const`
- the apply function `Matrix apply(const std::function<double(double)> &func) const`
- the sub_mul function `void sub_mul(double scalar, const Matrix &other)`

The implementation details are provided in the `matrix.hpp`file.
## Questions

1. Assume I use the copy constructor `Matrix(const Matrix& other)` to copy a matrix. Then, I modify an element of the copied matrix. What happens to the original matrix ?
2. How do row-major and column-major storage patterns affect the performance of matrix operations (especially matrix multiplication) ?
3. Explain why the `Matrix` class does not need an explicitly defined destructor `~Matrix()`.

## Guidelines

 - **Deadline**: The deadline is the Thursday 6th March 23h59.
 - **Fraud**:  As always for this course, you must do all the writing (report, code) <ins>individually</ins>. Never share your production. However, you are allowed, and even encouraged, to exchange ideas on how to address the assignment.
 - **Plagiarism**: As always, you must cite all your sources.
 - **Report Submission**: Using the Moodle assignment activity, submit your report in a file called `Report_Project_1_FirstName_LastName.pdf`. The report should be short (maximum 2 pages) and should include answers to the questions and a benchmark with an associated analysis (does it follows what you expected ?).
 - **Code Submission**: On Inginious, submit your files `matrix.cpp`, `matrix.hpp` containing your implementation of the `Matrix` class. You are allowed to make as many submission as you need, only the last submission will be taken into account. You are advised to verify that your submission passes the tests in Inginious early before the deadline. Note that, even if submitting the code on Inginious is mandatory, the Inginious automatic grading has no influence to the final grading. The tests on Inginious are similar to those included in `tests.cpp`. Since these tests are minimalist, passing them is a necessary but not sufficient condition for having a correct code.
 - **LLM**: The use of artificial intelligence tools is permitted provided that you mention both the AI tool used and the prompts/commands used in the comments of your submitted code and in your report.
 - **Language**: English is the default language. However, since the course is French-friendly, French is accepted without penalty.
 - **Questions**: If you have any questions, please contact the TA: `benoit.loucheur@uclouvain.be` and `brieuc.pinon@uclouvain.be`.