# mamu
This project is aiming to develop a software whcih can be used to estimate the quantitative genetic parameter. This software will use opencl and GPU solve the Matrix multiplication and inverse.
AI Reml average information restricted maxium likelihood algorithm will be used.

For now the C matrix generation was completed. Following functions will be added in future:

1. Generate the A (relationship) Matrix basing on pedigree.
2. Better matrix inverse methods.(Present method cost many time on data switch between host and device)
3. Support the reading of multi factors.
4. ... will be added in future.

本软件主要用于估计数量性状的遗传参数。矩阵的乘法运算和求逆将通过opencl 调用 GPU 实现。在计算遗传参数时将使用平均信息法。
