#define TILEX 4
#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2


/* Output tile size : 4x4 = Each thread computes 16 float values*/
/* Required global threads = (widthC / 4, heightC / 4) */
/* This kernel runs on 7xx and CPU as they don't have hardware local memory */
__kernel void mmmKernel(__global float4 *matrixA,
                        __global float4 *matrixB,
                        __global float4* matrixC,
            uint widthA, uint widthB)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));


    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);

    /* Vectorization of input Matrices reduces their width by a factor of 4 */
    widthB /= 4;

    for(int i = 0; i < widthA; i=i+4)
    {
        float4 tempA0 = matrixA[i/4 + (pos.y << TILEY_SHIFT) * (widthA / 4)];
        float4 tempA1 = matrixA[i/4 + ((pos.y << TILEY_SHIFT) + 1) * (widthA / 4)];
        float4 tempA2 = matrixA[i/4 + ((pos.y << TILEY_SHIFT) + 2) * (widthA / 4)];
        float4 tempA3 = matrixA[i/4 + ((pos.y << TILEY_SHIFT) + 3) * (widthA / 4)];

        //Matrix B is not transposed 
        float4 tempB0 = matrixB[pos.x + i * widthB];	
        float4 tempB1 = matrixB[pos.x + (i + 1) * widthB];
        float4 tempB2 = matrixB[pos.x + (i + 2) * widthB];
        float4 tempB3 = matrixB[pos.x + (i + 3) * widthB];

        sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
        sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
        sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
        sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

        sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
        sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
        sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
        sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

        sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
        sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
        sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
        sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

        sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
        sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
        sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
        sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;
    }
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 0) * widthB] = sum0;
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 1) * widthB] = sum1;
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 2) * widthB] = sum2;
    matrixC[pos.x + ((pos.y <<  TILEY_SHIFT) + 3) * widthB] = sum3;
}

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

}

__kernel void constructmatrix(__global int * y, __global int * uniquey,__global int* out, __global int *ref,const int N ){
	int lengthx = get_global_id(0); 
	int lengthy = get_global_id(1);//uniquey.size()

	if(y[lengthx]==ref[lengthx+N*lengthy]){out[lengthx+N*lengthy]=1;}
	else{out[lengthx+N*lengthy]=0;}


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
    printf("%f",c[y*K+x]);
    
}