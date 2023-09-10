__kernel void mmmKernel(__global float4 *matrixA,
                              __global float4 *matrixB,
                              __global float4* matrixC,
                              int widthA,__local float4* tempA)
{
  
     int index = get_global_id(0);
     matrixC[index] = matrixA[index] + matrixB[index];

}