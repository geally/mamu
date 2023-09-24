
__kernel void mamu(__global float* a,
                                    __global float* b,
                                    __global float* c,
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

    float sum = 0;
    for(int k = 0; k < K; k++){
        sum += a[rowIndex*K + k] * b[k*N + colIndex];
    }
    c[index] = sum;
   // printf("%f",c[index]);
}


 

__kernel void inverse2(__global float* a,
                                    __global float* b,
                                    const int M, 
                                    const int N){
  int colid = get_global_id(0);
  int rowid = get_global_id(1);

if(colid!=N){
            if(rowid!=N){
                    b[colid*M+rowid]=a[colid*M+rowid]+a[colid*M+N]*a[N*M+rowid]/a[N*M+N];
            }    else { b[colid*M+rowid] = a[colid*M+rowid]; } 
     
}else{
        b[colid*M+rowid]=a[colid*M+rowid];
	
  } 
  barrier(CLK_GLOBAL_MEM_FENCE);
  //printf("%i",N); 
}	


__kernel void pretreat(__global float* a, __global float* b,
                                    const int M,const int i){
int colid = get_global_id(0);
		float mid = a[i * M + i];
        for(int j=0; j<  M;j++){
            b[colid * M + j]=a[colid * M + j];
			if (colid != i) {
				b[i * M + colid] = a[i * M + colid] / mid;
				b[colid * M + i] = -a[colid * M + i] / mid;
			}
			else {
				b[i * M + i] = 1 / a[i * M + i];
			}  } 
            barrier(CLK_GLOBAL_MEM_FENCE);
          // printf("%f",mid);       
}





__kernel void refmatrix(__global int * y, __global int * uniquey, __global int* out, const int N ){
	int lengthx =get_global_id(0) ; 
	int lengthy = get_global_id(1); //uniquey.size()
	out[lengthx+lengthy*N]=uniquey[lengthy];
   // printf("%i",out[lengthx+lengthy*N]);

}

__kernel void constructmatrix(__global int * y, __global int * uniquey,__global float* out, __global int *ref,const int N ){
	int lengthx = get_global_id(0); 
	int lengthy = get_global_id(1);//uniquey.size()

	if(y[lengthx]==ref[lengthx+N*lengthy]){out[lengthx+N*lengthy]=1;}
	else{out[lengthx+N*lengthy]=0;}
       // printf("%i",out[lengthx+N*lengthy]);

}


__kernel void cbind(__global float* a,
                                    __global float* b,
                                    __global float* c,
                                    const int M, 
                                    const int N, 
                                    const int K){
    
    /**
     * Get work-item identifiers.
     **/
    
    int x = get_global_id(0);
    int y = get_global_id(1);
   
   if(x <M){
    c[y*(M+N)+x]=a[y*M+x];
    }else{
       c[y*(M+N)+x]=b[y*N+x-M];
        
    }
    //printf("%f",c[K+x]);
    
}

__kernel void rbind(__global float* a,
                                    __global float* b,
                                    __global float* c,
                                    const int M, 
                                    const int N, 
                                    const int K){
    
    /**
     * Get work-item identifiers.
     **/
    
    int x = get_global_id(0);
    int y = get_global_id(1);
   
   if(y <M){
    c[y*K+x]=a[y*K+x];
    }else{
       c[y*K+x]=b[(y-M)*K+x];
        
    }
   // printf("%f",c[y*K+x]);
    
}

__kernel void transpose(__global float* a,
                                    __global float* b,
                                    const int M, 
                                    const int N){
    
    /**
     * Get work-item identifiers.
     **/
    
    int x = get_global_id(0);
    int y = get_global_id(1);
   
   b[N*y+x]=a[x*M+y];
  // printf("%i",a[x*N+y]);
    
}



__kernel void trimax(__global float* a,
                                    __global float* b,
                                    __global float* c,
                                    const int M, 
                                    const int N){

    __local float* temp;
    __local float* ta  ;                           
    int x = get_global_id(0);
    int y = get_global_id(1);

  //    ndrange_t ndrange;

      ta[N*y+x]=a[x*N+y];
      //printf("%f",c[N*y+x]);
    
  //   enqueue_kernel(get_default_queue(),CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange,^{mamu(ta, b, c,M,N,N);});
   
   
}