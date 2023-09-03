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

__kernel void inverse(__global float* a,
                                    __global float* b,
                                    const int M, 
                                    const int N){
  int colid = get_global_id(0);

  
 for(int j=0;j<M;j++){
    float mid = a[j*N+j];
         for(int i=0;i<N;i++){
         
             b[i+N*j]=a[i+N*j]/mid;
             a[i+N*j]=b[j*N+j];
                
            }
            if(1+j < M){
            for(int k=1+j;k<M;k++){
                   float f=a[j+k*N];
                    b[k*N+colid]=a[k*N+colid]-a[j*N+colid]*f;
                    a[k*N+colid]=b[k*N+colid];
                }}

        }
	
    }
 	



