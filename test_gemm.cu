#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define OFFSET(row, col, rowLength) (row * rowLength + col)

void randMatrix(float *a, int row, int col){
        srand((unsigned int)time(NULL));
        float min = -10.0f;
        float max = 10.0f;

        for(int i = 0; i < row * col; i++)
        {       
                a[i] = min + (float)(rand()) / RAND_MAX * (max - min);
        }

        return;
}


void compareMatrix(float *a, float *b, int row, int col){
        for(int i = 0; i < row * col; i++)
        {
                if(a[i] - b[i] > 0.5f || a[i] - b[i] < -0.5f)
                        {
                                printf("The matrix diff too much \n");
                                return;
                        }
        }
        printf("The matirx are same \n");
        return;
}

void CPUgemm(float *a, float *b, float *c, int m, int k, int n){
        for(int i = 0; i < m; i++)
        {
                for(int j = 0; j < n; j++)
                {
                        for(int q = 0; q < k; q++)
                        {
                                c[OFFSET(i , j, n)] += a[OFFSET(i, q, k)] * b[OFFSET(q, j, n)]; 
                        }
                }
        }
        return;
}

//basic version--use global memory
__global__ void globalGemm(float *a, float *b, float *c, int M, int K, int N){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        float temp =0.0f;
        for(int k = 0; k < K; k++)
        {
                temp += a[k + y * K] * b[k * N + x];
        }

        c[y * N + x] = temp;
}

//shared memory version
__global__ void sharedGemm(float *a, float *b, float *c, int M, int K, int N, int BM,
                           int BN , int BK){
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;

        __shared__ tempA[BM * BK];
        __shared__ tempB[BK * BN];

}

int main()
{
        int M = 16;
        int N = 8;
        int m = 128;
        int n = 64;
        int k = 128;

        const int mem_size_a = m * k * sizeof(float);
        const int mem_size_b = k * n * sizeof(float);
        const int mem_size_c = m * n * sizeof(float);

        float * host_a = (float*) malloc(mem_size_a);
        float * host_b = (float*) malloc(mem_size_b);
        float * host_c = (float*) malloc(mem_size_c);
        float * host_c_cpu = (float*) malloc(mem_size_c);

        randMatrix(host_a, m, k);
        randMatrix(host_b, k, n);

        CPUgemm(host_a, host_b, host_c_cpu, m, k, n);

        float * device_a = NULL;
        float * device_b = NULL;
        float * device_c = NULL;

        cudaMalloc((void**)&device_a, mem_size_a);
        cudaMalloc((void**)&device_b, mem_size_b);
        cudaMalloc((void**)&device_c, mem_size_c);

        cudaMemcpy(device_a, host_a, mem_size_a, cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, mem_size_b, cudaMemcpyHostToDevice);


        dim3 blockDim(N,M);
        dim3 gridDim((n + N - 1) / N, (m + M - 1) / M);
        globalGemm<<<gridDim, blockDim>>>(device_a, device_b, device_c, m, k, n);

        cudaMemcpy(host_c, device_c, mem_size_c, cudaMemcpyDeviceToHost);

        compareMatrix(host_c_cpu, host_c, m, n);
}
