#include <cuda.h>
#include<stdio.h>
//#include<conio.h>

__global__ void matrix_mul(float *ad,float *bd,float *cd,int N)
{
        float pvalue=0;
        
        //find Row and Column corresponding to a data element for each thread
        int Row = blockIdx.y * blockDim.y + threadIdx.y;

        //calculate dot product of Row of First Matrix and Vector
        for(int i=0;i< N;++i)
        {
                pvalue += ad[Row+i] * bd[i];
        }

        //store dot product at corresponding positon in resultant Matrix
        cd[Row] = pvalue;

}
int main()
{
	int N = 100,i,j;				//N == size of square matrix
	
	float *a,*b;
	float *ad,*bd,*cd,*c;

    //open a file for outputting the result
    FILE *f;
	f=fopen("Parallel Multiply.txt","w");

	size_t size=sizeof(float)* N * N;
	size_t sizev=sizeof(float)* N;

    //allocate host side memory
	a=(float*)malloc(size);
	b=(float*)malloc(sizev);
	c=(float*)malloc(sizev);

	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			a[i*N+j]=1.0;   //(float)(i*N+j);		//initializing each value with its own index
		}
		b[i]=1.0;
	}

    //allocate device memory
	cudaMalloc(&ad,size);
	//printf("\nAfter cudaMalloc for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMalloc(&bd,sizev);
	//printf("\nAfter cudaMalloc bd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMalloc(&cd,sizev);
	//printf("\nAfter cudaMalloc cd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
	//copy value from host to device
    cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(bd,b,sizev,cudaMemcpyHostToDevice);
	printf("\nAfter HostToDevice Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));

    //calculate execution configuration
    dim3 blocksize(16,16);		        //each block contains 16 * 16 (=256) threads 
	dim3 gridsize(N/8,N/8);			//creating just sufficient no of blocks
    
    //GPU timer code
    float time;
    cudaEvent_t start,stop;			
	cudaEventCreate(&start);		
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	matrix_mul <<< gridsize, blocksize >>> (ad, bd, cd, N);
	
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);			//time taken in kernel call calculated
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    //copy back results
	cudaMemcpy(c,cd,sizeof(float)* N,cudaMemcpyDeviceToHost);
	printf("\nAfter DeviceToHost Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    //output results in output_file
	fprintf(f,"Array A was---\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			fprintf(f,"%f ",a[i*N+j]);
		fprintf(f,"\n");
	}
	fprintf(f,"\n Vector B was---\n");
	for(i=0;i<N;i++)
	{
		fprintf(f,"%f ",b[j]);
		fprintf(f,"\n");
	}
	fprintf(f,"\nMultiplication of A and B gives C----\n");
	printf("\nMultiplication of Matrix A and Vector B gives C----\n");
	for(i=0;i<N;i++)
	{
		fprintf(f,"%f ",c[i]); 
		printf("\n%f ",c[i]);              //if correctly computed, then all values must be N
		fprintf(f,"\n");
	}
	printf("\nYou can see output in Parallel Mutiply.txt file in project directory");
    printf("\n\nTime taken is %f (ms)\n",time);
    fprintf(f,"\n\nTime taken is %f (ms)\n",time);
	fclose(f);

	cudaThreadExit();
	//cudaFree(ad); cudaFree(bd); cudaFree (cd);
	free(a);free(b);free(c);
//	_getch();
    return 1;
}
