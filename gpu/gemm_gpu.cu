// #include <cublas_v2.h>

#include "../include/utils.h"
#include <cuda_runtime.h>

#define NUM_RUNS 10

#define O1_BLOCK_SIZE 16
#define O2_TILE_WIDTH 16
#define O3_TILE_WIDTH 32

#define CUDA_CHECK(func)                                                     	   \
	do {                                                                           \
		cudaError_t status = (func);                                               \
		if (status != cudaSuccess) {                                               \
			printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,   \
				cudaGetErrorString(status), status);                               \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

#define CHECK(name) \
	float *d_Aref_ ## name, *d_Bref_ ## name, *d_Cref_ ## name; \
	std::cerr << "checking " << #name << std::endl; \
	CUDA_CHECK(cudaMalloc(&d_Aref_ ## name, Ref::M * Ref::K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Bref_ ## name, Ref::K * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Cref_ ## name, Ref::M * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_Aref_ ## name, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_Bref_ ## name, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	float* d_Cref_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < Ref::M; i++) { \
		for (int j = 0; j < Ref::N; j++) { \
			d_Cref_INI_ ## name[i * Ref::N + j] = 0; \
		} \
	} \
	CUDA_CHECK(cudaMemcpy(d_Cref_ ## name, d_Cref_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	name(d_Aref_ ## name, d_Bref_ ## name, d_Cref_ ## name, Ref::M, Ref::N, Ref::K); \
	cudaError_t err_c_ ## name = cudaGetLastError(); \
	if (err_c_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_ ## name) << std::endl; \
	} \
	CUDA_CHECK(cudaMemcpy(refC, d_Cref_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost)); \
	if (!ref.checkRef(refC)){ \
		std::cerr << "check ref failed!" << std::endl; \
	};

#define TIME(name) \
	float *d_A_ ## name, *d_B_ ## name, *d_C_ ## name; \
	CUDA_CHECK(cudaMalloc(&d_A_ ## name, M * K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_B_ ## name, K * N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_C_ ## name, M * N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_A_ ## name, A, M * K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_B_ ## name, B, K * N * sizeof(float), cudaMemcpyHostToDevice)); \
	cudaEvent_t start_ ## name, end_ ## name; \
	cudaEventCreate(&start_ ## name); \
	cudaEventCreate(&end_ ## name); \
	float* d_C_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < M; i++) { \
		for (int j = 0; j < N; j++) { \
			d_C_INI_ ## name[i * N + j] = 0; \
		} \
	} \
	for (int i = 0; i < 2; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
	} \
	cudaError_t err_t_ ## name = cudaGetLastError(); \
	if (err_t_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_t_ ## name) << std::endl; \
	} \
	float milliseconds_ ## name = 0; \
	for (int i = 0; i < NUM_RUNS; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		cudaDeviceSynchronize(); \
		cudaEventRecord(start_ ## name); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
		cudaEventRecord(end_ ## name); \
		cudaEventSynchronize(end_ ## name); \
		float milliseconds_ ## i = 0; \
		cudaEventElapsedTime(&milliseconds_ ## i, start_ ## name, end_ ## name); \
		milliseconds_ ## name += milliseconds_ ## i; \
	} \
	cudaMemcpy(C, d_C_ ## name, M * N * sizeof(float), cudaMemcpyDeviceToHost); \
	std::cout << "Time taken for GEMM (GPU, " << #name <<"): " << milliseconds_ ## name / (float)NUM_RUNS << "ms" << std::endl; \
	cudaFree(d_A_ ## name); \
	cudaFree(d_B_ ## name); \
	cudaFree(d_C_ ## name);

__global__ void gemm_gpu_o0_kernel(float* A, float* B, float *C, int M, int N, int K) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < K; k++) {
					C[i * N + j]  += A[i * K + k]  * B[k * N + j];
				}
			}
		}
    }
}

void gemm_gpu_o0(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(1);
	dim3 gridSize(1);
	gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// The scafolding for optimized GEMM implementations
__global__ void gemm_gpu_o1_kernel(float* A, float* B, float *C, int M, int N, int K) {
	int row = blockIdx.x * O1_BLOCK_SIZE + threadIdx.x;
	int col = blockIdx.y * O1_BLOCK_SIZE + threadIdx.y;

	if (row < M && col < N) {
		float out = 0;
		for(int i=0; i<K; i++) {
			out += A[row * K + i] * B[col + i * N];
		}
		C[row * N + col] = out;
	} 
}

void gemm_gpu_o1(float* A, float* B, float* C, int M, int N, int K)
{
	dim3 blockSize(O1_BLOCK_SIZE, O1_BLOCK_SIZE, 1);
	dim3 gridSize(ceil((float)M / O1_BLOCK_SIZE), ceil((float)N / O1_BLOCK_SIZE), 1);

	gemm_gpu_o1_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

__global__ void gemm_gpu_o2_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float tileA[O2_TILE_WIDTH][O2_TILE_WIDTH];
	__shared__ float tileB[O2_TILE_WIDTH][O2_TILE_WIDTH];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.x * O2_TILE_WIDTH + tx;
	int col = blockIdx.y * O2_TILE_WIDTH + ty;

	for(int k=0; k<ceil((float)K / O2_TILE_WIDTH); k++) {
		int rowA = row;
		int colA = k * O2_TILE_WIDTH + ty;

		int rowB = k * O2_TILE_WIDTH + tx;
		int colB = col;

		if (rowA < M && colA < K) {
			tileA[tx][ty] = A[rowA * K + colA];
		} else {
			tileA[tx][ty] = 0.0;
		}

		if (rowB < K && colB < N) {
			tileB[tx][ty] = B[rowB * N + colB];
		} else {
			tileB[tx][ty] = 0.0;
		}
		__syncthreads();

		float out = 0;
		for(int q=0; q<O2_TILE_WIDTH; q++) {
			out += tileA[tx][q] * tileB[q][ty];
		}
		 __syncthreads();

		if (row < M && col < N) {
			C[row * N + col] = out;
		}
	}
}

void gemm_gpu_o2(float* A, float* B, float* C, int M, int N, int K)
{
	dim3 blockSize(O2_TILE_WIDTH, O2_TILE_WIDTH, 1);
	dim3 gridSize(ceil((float)M / O2_TILE_WIDTH), ceil((float)N / O2_TILE_WIDTH));
	gemm_gpu_o2_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

__global__ void gemm_gpu_o3_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float tileA[O3_TILE_WIDTH][O3_TILE_WIDTH];
	__shared__ float tileB[O3_TILE_WIDTH][O3_TILE_WIDTH];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.x * O3_TILE_WIDTH + tx;
	int col = blockIdx.y * O3_TILE_WIDTH + ty;

	for(int k=0; k<ceil((float)K / O3_TILE_WIDTH); k++) {
		int rowA = row;
		int colA = k * O3_TILE_WIDTH + ty;

		int rowB = k * O3_TILE_WIDTH + tx;
		int colB = col;

		if (rowA < M && colA < K) {
			tileA[tx][ty] = A[rowA * K + colA];
		} else {
			tileA[tx][ty] = 0.0;
		}

		if (rowB < K && colB < N) {
			tileB[tx][ty] = B[rowB * N + colB];
		} else {
			tileB[tx][ty] = 0.0;
		}
		__syncthreads();

		float out = 0;
		for(int q=0; q<O3_TILE_WIDTH; q++) {
			out += tileA[tx][q] * tileB[q][ty];
		}
		 __syncthreads();

		if (row < M && col < N) {
			C[row * N + col] = out;
		}
	}
}
void gemm_gpu_o3(float* A, float* B, float* C, int M, int N, int K)
{
	dim3 blockSize(O3_TILE_WIDTH, O3_TILE_WIDTH, 1);
	dim3 gridSize(ceil((float)M / O3_TILE_WIDTH), ceil((float)N / O3_TILE_WIDTH));
	gemm_gpu_o3_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// __host__ void gemm_gpu_ec(float* A, float* B, float* C, int M, int N, int K)
// {
// 	cublasHandle_t handle;
//     cublasCreate(&handle);
    
//     float alpha = 1.0f, beta = 1.0f;
//     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//                 N, M, K, 
//                 &alpha, 
//                 B, N, 
//                 A, K, 
//                 &beta, 
//                 C, N);
    
//     cublasDestroy(handle);
// }

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
		return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	// int runs = atoi(argv[3]);
	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	/// GPU Implementation
        // Check if implementation is correct
	auto ref = Ref();
	float* refC = new float[Ref::M * Ref::N]();
 	CHECK(gemm_gpu_o0)
	CHECK(gemm_gpu_o1)
	CHECK(gemm_gpu_o2)
	CHECK(gemm_gpu_o3)
	// CHECK(gemm_gpu_ec)

	// Actual run
 	TIME(gemm_gpu_o0)
	TIME(gemm_gpu_o1)
	TIME(gemm_gpu_o2)
	TIME(gemm_gpu_o3)
	// TIME(gemm_gpu_ec)

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}