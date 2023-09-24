__kernel void mamu(__global int* a,
                                    __global int* b,
                                    __global int* c,
                                    const int M, 
                                    const int N, 
                                    const int K){
    
    /**
     * Get work-item identifiers.
     **/
    
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int index = (rowIndex * N) + colIndex;

    /**
     * Compute element c[rowIndex, colIndex].
     **/

    int sum = 0;
    for(int k = 0; k < K; k++){
        sum += a[rowIndex*K + k] * b[k*N + colIndex];
    }
    c[index] = sum;
}

__kernel void inverse(__global int* a,
                                    __global int* b,
                                    const int M, 
                                    const int N){
  int colid = get_global_id(0);

        b[colid]=a[colid]+a[colid];

                                    }
