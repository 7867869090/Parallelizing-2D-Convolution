# Parallelizing-2D-Convolution
This project explores the optimization of 2D convolution — a core operation in image processing and deep learning — by implementing and comparing multiple execution strategies: sequential (CPU-based), multithreaded, and GPU-accelerated using CUDA.

The primary goal is to analyze the performance benefits and trade-offs of each method. The sequential implementation uses basic nested loops in Python, providing a baseline for comparison. The multithreaded version divides the image into horizontal chunks processed concurrently using Python’s threading module, demonstrating basic parallelism on the CPU. The GPU implementation leverages PyCUDA to launch CUDA kernels, enabling massively parallel execution of the convolution operation on a GPU.

By applying all three techniques to synthetic 2D matrices (simulating image data), the project measures execution times and visualizes speed-up factors. Results are validated across all implementations to ensure correctness. Through this comparative study, we aim to provide insights into how parallel computing techniques can be applied to improve the efficiency of computationally expensive operations like convolution.

This project is suitable for students, educators, and developers interested in:

Parallel and Distributed Computing (PDC)

GPU Programming with CUDA

Performance benchmarking

Image processing fundamentals

✅ Features:
Implementations of 2D convolution using three approaches: sequential, multithreaded, and GPU-based

Execution time benchmarking and comparison

CUDA kernel for efficient GPU convolution via PyCUDA

Easy-to-read and well-commented Jupyter Notebook

Modular design for extension and experimentation

🧰 Tech Stack:
Python 3

NumPy

PyCUDA

Threading

Jupyter Notebook

▶️ How to Run:
Install dependencies (see requirements.txt)

Open the notebook in Jupyter

Run each section and observe timing results

Analyze and compare the performance of each approach


