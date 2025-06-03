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

        for(int i=0; i < row * col; i++)
        {       
                a[i] = min + (float)(rand()) / RAND_MAX * (max - min);
        }

        return;
}


void compareMatrix(float *a, float *b, int row, int col){
        for(int i=0; i < row * col; i++)
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

int main()
{
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

        float * device_a = NULL;
        float * device_b = NULL;
        float * device_c = NULL;

        cudaMalloc((void**)&device_a, mem_size_a);
        cudaMalloc((void**)&device_b, mem_size_b);
        cudaMalloc((void**)&device_c, mem_size_c);

        

}
