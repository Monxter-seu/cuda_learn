#include <stdio.h>
#include <float>
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
__global__ globalGemm(float *a, float *b, float *c, int M, int K, int N){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        float * a_block = a + blockIdx.y * blockDim.y * K;
        float * b_block = b + blockIdx.x * blockDim.x;

        float temp =0.0f;
        for(int k = 0; k < K; k++)
        {
                temp += a_block[k + threadIdx.y * K] * b_block[threadIdx.x + k * K];
        }

        c[y * K + x] = temp;
}

int main()
{
        int M = 16;
        int N = 16;
        int m = 128;
        int n = 128;
        int k = 128;

        const int mem_size_a = m * k * sizeof(float);
        const int mem_size_b = k * n * sizeof(float);
        const int mem_size_c = m * n * sizeof(float);

        float * host_a = (float*) malloc(mem_size_a);
        float * host_b = (float*) malloc(mem_size_b);
        float * host_c = (float*) malloc(mem_size_c);
        float * host_c_cpu = (float*) malloc(mem_size_c);

        randMatrix(host_a, m, n);
        randMatrix(host_b, m, n);

        cpuGemm(host_a, host_b, host_c_cpu, m, k, n);

        float * device_a = NULL;
        float * device_b = NULL;
        float * device_c = NULL;

        cudaMalloc((void**)&device_a, mem_size_a);
        cudaMalloc((void**)&device_b, mem_size_b);
        cudaMalloc((void**)&device_c, mem_size_c);

        cudaMemcpy(device_a, host_a, mem_size_a, cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, mem_size_b, cudaMemcpyHostToDevice);


        dim3 blockDim(M,N);
        dim3 gridDim((m + M - 1) / M, (n + N - 1) / N);
        globalGemm<<<gridDim, blockDim>>(device_a, device_b, device_c, m, k, n);

        cudaMemcpy(host_c, device_c, mem_size_c, cudaMemcpyDeviceToHost);

        compareMatrix(host_c_cpu, host_c, m, n);
}
