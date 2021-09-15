#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstring>
#include <ctime>

// Device Kernel
__global__ void VecAdd(float *A, float *B, float *C, int N)
{
   
   int i = blockDim.x * blockIdx.x + threadIdx.x;
	C[i]=0.0; 
    if (i < N)
        C[i] = A[i] + B[i];
}

//Host Function
void cpu_VecAdd(float *A, float *B,float* C, int N) {
	
     for(int i = 0; i < N; i++) {
        C[i]= A[i]+B[i];
    }
   
   }  
          
// Host code
int main()
{
    int N = 512*1000;
    size_t size = N * sizeof(float);
    FILE *f1;
	f1=fopen("VectAdd1.txt","w");
    FILE *f2;
	f2=fopen("VectAdd2.txt","w");

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    // Initialize input vectors
  
 
    for(int i = 0; i < N; i++) {
        h_A[i] = i;
	h_B[i] = i;
	h_C[i]=0.0;
    }
    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);
printf("\nData Parallel Model\n");
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	//printf("\nAfter HostToDevice Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));
    
    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
 clock_t start,stop;
    double time,time1;
    start = std::clock();

    VecAdd<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, N);

		
	stop = std::clock();
	time = ((double)(stop - start))/CLOCKS_PER_SEC;
	//printf("\nAfter global call Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));

    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
//printf("\nAfter DeviceToHost Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));	
  for(int i=0;i<N;i++)
	{
		fprintf(f1,"%f ",h_C[i]);              //if correctly computed, then all values must be N
		fprintf(f1,"\n");
	}
  printf("\n\nExecution Time of parallel Implementation= %lf (ms)\n",time);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
printf("_______________________________________________________________________	");
clock_t start1,stop1;
start1 = std::clock();

  cpu_VecAdd(h_A,h_B,h_C,N);
 stop1 = std::clock();
time1 = ((double)(stop1-start1))/CLOCKS_PER_SEC;
  for(int i=0;i<N;i++)
	{
		fprintf(f2,"%f ",h_C[i]);              //if correctly computed, then all values must be N
		fprintf(f2,"\n");
	}
 //long int Stime=stop - start;
  printf("\n\nExecution Time of Sequential Implementation= %lf (ms)\n",time1 );
cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
int cores;
if (prop.major==2) //Fermi
	cores=(prop.minor==1) ? prop.multiProcessorCount*48 : prop.multiProcessorCount*32;
  printf("No. of cores:%d\n",cores);
  printf("\n\n Total cost=Execution Time * Number ofprocessors used\n\t\t=%f",time*cores);
  printf("\n\n Efficiency=WCSA / WCPA\n\n=%f",time1/time);
    free(h_C);        
    free(h_A);
    free(h_B);
    // Free host memory
 
}
