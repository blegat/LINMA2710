# Project 3

This project focuses on performing matrix-matrix multiplication with OpenCL. Your code will thus be able to be compiled for many devices, including GPUs, and allows you to leverage their efficiency for parallel computations.

## Requirements

This project must be implemented using only the C++ Standard Library and OpenCL. No other libraries or dependencies are allowed. This ensures portability.

## Tasks

The `MatrixCL` class should implement the methods as defined in the `matrix_opencl.hpp` file (and tested in the `main.cpp` file).

A MatrixCL object stores:
 - rows_ and cols_ for the number of rows/columns in the matrix
 - context_ a reference to an OpenCL context allowing you to manage memory on the device (e.g. GPU)
 - queue_ a reference to an OpenCL CommandQueue to launch the execution of kernels on the device
 - buffer_ an OpenCL buffer storing the matrix elements on the device

The contructor initialize a buffer on the device filled with zeroes or provided initial data. All matrix operations can then be performed directly using the device memory (no back and forth between the host and device memory).

In OpenCL, the code executed on the device must be compiled during the program execution. In order to avoid compiling this code at each function call, the codes are compiled one time at initialization with 'initializeKernels', and stored in 'kernels_', which is an object shared by all the MatrixCL objects. 

## Questions

1. Implement 2 versions of the matrix-matrix multiplication: a simple and a faster one. Provide your two OpenCL kernel codes in the report and explain them briefly.
2. Profile and analyze your two implementations on a GPU using a profiler such as *tau2* ([comments on *tau2*](https://github.com/blegat/LINMA2710/tree/main/examples)) or [*tracy*](https://github.com/wolfpld/tracy/tree/master/examples/OpenCLVectorAdd). The GPU can be on your computer or on the Manneback cluster, note that *tau2* is installed on the cluster. It may also be useful to query the [profiling info](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetEventProfilingInfo.html) as detailed during the lecture (don't forget to enable profiling in the queues with `CL_QUEUE_PROFILING_ENABLE`).
3. Measure also the impact of the kernel implementation to the power consumption of the GPU. To measure the power consumption, different tools are available, [codecarbon](https://github.com/mlco2/codecarbon) is an example. More details will be given during the lecture in S12.

## Guidelines

 - **Deadline**: The deadline is the Friday 16th May 23h59.
 - **Fraud**:  As always for this course, you must do all the writing (report, code) <ins>individually</ins>. Never share your production. However, you are allowed, and even encouraged, to exchange ideas on how to address the assignment.
 - **Plagiarism**: As always, you must cite all your sources.
 - **Report Submission**: Using the Moodle assignment activity, submit your report in a file called `Report_Project_3_FirstName_LastName.pdf`. The report should be short (maximum 2 pages for the text; the codes, tables and images are not counted in the 2 pages limit) and should include answers to the questions.
 - **Code Submission**: On [Inginious](https://inginious.info.ucl.ac.be/course/LINMA2710/project_opencl), submit your files `matrix_opencl.cpp`, `matrix_opencl.hpp` containing your implementation of the `MatrixCL` class. You are allowed to make as many submission as you need, only the last submission will be taken into account. You are advised to verify that your submission passes the tests in Inginious early before the deadline. Note that, even if submitting the code on Inginious is mandatory, the Inginious automatic grading has no influence to the final grading. The tests on Inginious are similar to those included in `main.cpp`. Since these tests are minimalist, passing them is a necessary but not sufficient condition for having a correct code.
 - **LLM**: The use of artificial intelligence tools is permitted provided that you mention both the AI tool used and the prompts/commands used in the comments of your submitted code and in your report.
 - **Language**: English is the default language. However, since the course is French-friendly, French is accepted without penalty.
 - **Questions**: If you have any questions, please contact the TA: `benoit.loucheur@uclouvain.be` and `brieuc.pinon@uclouvain.be`.
