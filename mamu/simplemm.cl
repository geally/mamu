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
    float mid = a[j*M+j];
         for(int i=0;i<M;i++){
         if(i!=j){
         b[i+M*j]=a[i+M*j]/mid;
             a[i+M*j]=b[i+M*j];
         } else{a[j*M+j]=1/mid;
         b[j*M+j]=1/mid;} 
         }

    for(int k=0;k<M;k++){
            if(k!=j){
                    b[colid*M+k]=a[colid*M+k]-a[colid*M+j]*a[j*M+k];
            }else{
                b[colid*M+k]= -a[colid*M+k]/mid;
            }
            
                   
     }
   for(int l=0;l<M;l++){
        for(int m=0;m<M;m++){
        if(l!=j && m !=j){
         a[l*M+m]=b[l*M+m];
         b[l*M+m]=a[l*M+m];
        }
        else if(l!=j && m ==j){a[l*M+m]=b[l*M+m];
        b[l*M+m]=a[l*M+m];}
        else {b[l*M+m]=a[l*M+m];}
        }
  
    } 
 }      
	
}
 	



