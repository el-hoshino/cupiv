#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
 
#define DEFAULT_WIDTH		1024
#define DEFAULT_HEIGHT		1024
#define DEFAULT_INTWNDW		32
#define DEFAULT_INTWNDH		32
#define DEFAULT_SRHWNDL		12
#define DEFAULT_SRHWNDR		12
#define DEFAULT_SRHWNDA		12
#define DEFAULT_SRHWNDB		12
 
 
__host__	void getData(int size, unsigned char *data0, unsigned char *data1, char *path0, char *path1);
 
__global__	void vectorComputation(unsigned char *f, unsigned char *g, int width, int totalVectorsX, int totalVectorsY, char m, char n, char l, char r, char a, char b, char ww, char wh, char fw, char fh, char stepX, char stepY, float *R_posSubX, float *R_posSubY, float *startX, float *startY, float *endX, float *endY);
 
__device__	void getData0intoSharedMemory(unsigned char *f, unsigned char *sh_f, int width, char m, char n, char l, char a, int startPixelX, int startPixelY, char stepX, char stepY);
 
__device__	void getData1intoSharedMemory(unsigned char *g, unsigned char *sh_g, int width, char fw, char fh, int startPixelX, int startPixelY, char stepX, char stepY);
 
__device__	void calculateR(unsigned char *sh_f, unsigned char *sh_g, float *R, char *R_posX, char *R_posY, char m, char n, char ww, char wh, char fw, char fh);
 
__device__	void bitonicSort(float *R, char *R_posX, char *R_posY);
 
__device__	void swapFloat(float *x, float *y);
 
__device__	void swapChar(char *x, char *y);
 
__device__	void subPixelAccuracy(int vectorID, float *R, char *R_posX, char *R_posY, float *subXN, float *subXD, float *subYN, float *subYD, float *R_posSubX, float *R_posSubY, char l, char a, float *endX, float *endY);
 
__global__ void vectorCorrection(int totalVectorsX, float *R_posSubX, float *R_posSubY, float *endX, float *endY);
 
__host__	void printSummary(int width, int height, char intWndW, char intWndH, char srhWndL, char srhWndR, char srhWndA, char srhWndB, char stepX, char stepY, int totalVectorsX, int totalVectorsY, int totalVectors, int Dbx, int Dby, int Dgx, int Dgy, float h_time, float d_time);
 
__host__	void putData(float *startX, float *startY, float *endX, float *endY, int totalVectors, char *path);
 
__host__	void putDataVTK(int height, float *startX, float *startY, float *endX, float *endY, int number, int numberX, int numberY, char *pathVTK);
 
int main(int argc, char *argv[]) {
	int width = DEFAULT_WIDTH, height = DEFAULT_HEIGHT;
	int size = width * height;
 
	unsigned char *h_data0, *h_data1, *d_data0, *d_data1;
	char intWndW = (char)DEFAULT_INTWNDW,
		 intWndH = (char)DEFAULT_INTWNDH,
		 srhWndL = (char)DEFAULT_SRHWNDL,
		 srhWndR = (char)DEFAULT_SRHWNDR,
		 srhWndA = (char)DEFAULT_SRHWNDA,
		 srhWndB = (char)DEFAULT_SRHWNDB,
		 srhWndW = srhWndL + srhWndR + 1,
		 srhWndH = srhWndA + srhWndB + 1,
		 srhFldW = srhWndL + srhWndR + intWndW,
		 srhFldH = srhWndL + srhWndR + intWndH;
 
	char stepX, stepY;
	float *h_startX, *h_startY, *d_startX, *d_startY;
	float *h_endX, *h_endY, *d_endX, *d_endY;
	float *R_posSubX, *R_posSubY;
	int totalVectorsX, totalVectorsY, totalVectors;
 
	char *inputFileName0 = "2ms2000018",
		 *inputFileName1 = "2ms2000019",
		 *inputFileExtension = "raw",
		 *outputFileName = "result",
		 *outputFileExtension = "dat";
 
	char inputFilepath0[64], inputFilepath1[64], outputFilepath[64], outputFilePathVTK[64];
 
	clock_t h_start, h_end, d_start, d_end;
	float h_time, d_time;
 
	cudaError_t cudaError;
 
	dim3 Dg, Db;
	int Dgx, Dgy, Dbx, Dby;
 
	h_data0 = (unsigned char *)malloc(sizeof(unsigned char) * size);
	h_data1 = (unsigned char *)malloc(sizeof(unsigned char) * size);
	cudaMalloc((void **)&d_data0, sizeof(unsigned char) * size);
	cudaMalloc((void **)&d_data1, sizeof(unsigned char) * size);
 
	stepX = (char)(intWndW / 2);
	stepY = (char)(intWndH / 2);
 
	totalVectorsX = (int)((width - srhFldW) / stepX) + 1;
	totalVectorsY = (int)((height - srhFldH) / stepY) + 1;
	totalVectors = totalVectorsX * totalVectorsY;
 
	h_startX = (float *)malloc(sizeof(float) * totalVectors);
	h_startY = (float *)malloc(sizeof(float) * totalVectors);
	h_endX = (float *)malloc(sizeof(float) * totalVectors);
	h_endY = (float *)malloc(sizeof(float) * totalVectors);
	cudaMalloc((void **)&d_startX, sizeof(float) * totalVectors);
	cudaMalloc((void **)&d_startY, sizeof(float) * totalVectors);
	cudaMalloc((void **)&d_endX, sizeof(float) * totalVectors);
	cudaMalloc((void **)&d_endY, sizeof(float) * totalVectors);
 
	cudaMalloc((void **)&R_posSubX, sizeof(float) * totalVectors * 5);
	cudaMalloc((void **)&R_posSubY, sizeof(float) * totalVectors * 5);
 
	sprintf(inputFilepath0, "%s.%s", inputFileName0, inputFileExtension);
	sprintf(inputFilepath1, "%s.%s", inputFileName1, inputFileExtension);
	sprintf(outputFilepath, "%s.%s", outputFileName, outputFileExtension);
	sprintf(outputFilePathVTK, "%s-vtk.vtk", outputFileName);
 
	Db.x = 32;
	Db.y = 32;
	Dg.x = totalVectorsX;
	Dg.y = totalVectorsY;
	Dbx = Db.x; Dby = Db.y;
	Dgx = Dg.x; Dgy = Dg.y;
 
	getData(size, h_data0, h_data1, inputFilepath0, inputFilepath1);
 
	h_start = clock();
	cudaMemcpy(d_data0, h_data0, sizeof(unsigned char) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_data1, h_data1, sizeof(unsigned char) * size, cudaMemcpyHostToDevice);
 
	d_start = clock();
 
	vectorComputation<<<Dg, Db>>>(d_data0, d_data1, width, totalVectorsX, totalVectorsY, intWndW, intWndH, srhWndL, srhWndR, srhWndA, srhWndB, srhWndW, srhWndH, srhFldW, srhFldH, stepX, stepY, R_posSubX, R_posSubY, d_startX, d_startY, d_endX, d_endY);
	cudaThreadSynchronize();
 
	vectorCorrection<<<Dg, Db>>>(totalVectorsX, R_posSubX, R_posSubY, d_endX, d_endY);
	cudaThreadSynchronize();
 
	d_end = clock();
 
	cudaMemcpy(h_startX, d_startX, sizeof(float) * totalVectors, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_startY, d_startY, sizeof(float) * totalVectors, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_endX, d_endX, sizeof(float) * totalVectors, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_endY, d_endY, sizeof(float) * totalVectors, cudaMemcpyDeviceToHost);	
	h_end = clock();
 
	h_time = (float)(h_end - h_start) / CLOCKS_PER_SEC;
	d_time = (float)(d_end - d_start) / CLOCKS_PER_SEC;
	printSummary(width, height, intWndW, intWndH, srhWndL, srhWndR, srhWndA, srhWndB, stepX, stepY, totalVectorsX, totalVectorsY, totalVectors, Dbx, Dby, Dgx, Dgy, h_time, d_time);
 
	putData(h_startX, h_startY, h_endX, h_endY, totalVectors, outputFilepath);
	putDataVTK(height, h_startX, h_startY, h_endX, h_endY, totalVectors, totalVectorsX, totalVectorsY, outputFilePathVTK);
 
	free(h_data0);
	free(h_data1);
	cudaFree(d_data0);
	cudaFree(d_data1);
 
	free(h_startX);
	free(h_startY);
	free(h_endX);
	free(h_endY);
	cudaFree(d_startX);
	cudaFree(d_startY);
	cudaFree(d_endX);
	cudaFree(d_endY);
 
	cudaFree(R_posSubX);
	cudaFree(R_posSubY);
 
	cudaError = cudaGetLastError();
 
	if(cudaError != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n\n", cudaGetErrorString(cudaError));
		exit(-1);
	} else {
		printf("CUDA success\n\n");
	}
 
	return 0;
}
 
__host__		void getData(int size, unsigned char *data0, unsigned char *data1, char *path0, char *path1) {
	int i;
	FILE *fp;
 
	fp = fopen(path0, "rb");
	if (fp == NULL) {
		printf("%s file open error\n", path0);
	}
 
	for (i = 0; i < size; i++) {
		data0[i] = fgetc(fp);
	}
 
	fclose(fp);
 
	fp = fopen(path1, "rb");
	if (fp == NULL) {
		printf("%s file open error\n", path1);
	}
 
	for (i = 0; i < size; i++) {
		data1[i] = fgetc(fp);
	}
 
	fclose(fp);
}
 
__global__	void vectorComputation(unsigned char *f, unsigned char *g, int width, int totalVectorsX, int totalVectorsY, char m, char n, char l, char r, char a, char b, char ww, char wh, char fw, char fh, char stepX, char stepY, float *R_posSubX, float *R_posSubY, float *startX, float *startY, float *endX, float *endY) {
	int vectorID;
	int startPixelX, startPixelY;
 
	__shared__ unsigned char
		sh_f[2304], sh_g[12100];
 
	__shared__ float
		R[4096];
 
	__shared__ char
		R_posX[4096], R_posY[4096];
 
	__shared__ float
		subXN[5], subXD[5], subYN[5], subYD[5];
 
	vectorID = blockIdx.y * gridDim.x + blockIdx.x;
 
	startPixelX = stepX * blockIdx.x;
	startPixelY = stepY * blockIdx.y;
 
	startX[vectorID] = startPixelX + l + (m / 2) + (float)0.5;
	startY[vectorID] = startPixelY + a + (n / 2) + (float)0.5;
	endX[vectorID] = 0;
	endY[vectorID] = 0;
 
	getData0intoSharedMemory(f, sh_f, width, m, n, l, a, startPixelX, startPixelY, stepX, stepY);
 
	getData1intoSharedMemory(g, sh_g, width, fw, fh, startPixelX, startPixelY, stepX, stepY);
 
		__syncthreads();
 
	calculateR(sh_f, sh_g, R, R_posX, R_posY, m, n, ww, wh, fw, fh);
 
		__syncthreads();
 
	bitonicSort(R, R_posX, R_posY);
 
		__syncthreads();
 
	subPixelAccuracy(vectorID, R, R_posX, R_posY, subXN, subXD, subYN, subYD, R_posSubX, R_posSubY, l, a, endX, endY);
 
		__syncthreads();
 
}
 
__device__ void getData0intoSharedMemory(unsigned char *f, unsigned char *sh_f, int width, char m, char n, char l, char a, int startPixelX, int startPixelY, char stepX, char stepY) {
	char j, i;
 
	for (j = threadIdx.y; j < n; j += blockDim.y) {
		for (i = threadIdx.x; i < m; i += blockDim.x) {
			sh_f[j * m + i] = f[(startPixelY + a + j) * width + (startPixelX + l + i)];
		}
	}
 
}
 
__device__ void getData1intoSharedMemory(unsigned char *g, unsigned char *sh_g, int width, char fw, char fh, int startPixelX, int startPixelY, char stepX, char stepY) {
	char j, i;
 
	for (j = threadIdx.y; j < fh; j += blockDim.y) {
		for (i = threadIdx.x; i < fw; i += blockDim.x) {
			sh_g[j * fw + i] = g[(startPixelY + j) * width + (startPixelX + i)];
		}
	}
 
}
 
__device__ void calculateR(unsigned char *sh_f, unsigned char *sh_g, float *R, char *R_posX, char *R_posY, char m, char n, char ww, char wh, char fw, char fh) {
	char i, j, zeta, eta;
	float fValue, gValue;
	float A, B, C;
	int theta;
 
	for (eta = threadIdx.y; eta < wh; eta += blockDim.y) {
		for (zeta = threadIdx.x; zeta < ww; zeta += blockDim.x) {
			theta = eta * ww + zeta;
			A = 0; B = 0; C = 0;
 
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++) {
 
					fValue = (float)sh_f[j * m + i];
					gValue = (float)sh_g[(eta + j) * fw + (zeta + i)];
 
					A += fValue * gValue;
					B += fValue * fValue;
					C += gValue * gValue;
				}
			}
			R[theta] = sqrt((A * A) / (B * C));
			R_posX[theta] = zeta;
			R_posY[theta] = eta;
		}
	}
 
	for (theta = (threadIdx.y * blockDim.x + threadIdx.x) + (ww * wh); theta < 4096; theta += blockDim.x * blockDim.y) {
		R[theta] = 0;
		R_posX[theta] = -128;
		R_posY[theta] = -128;
	}
 
}
 
__device__ void bitonicSort(float *R, char *R_posX, char *R_posY) {
 
	int i, j, k, frac, xtemp, x, group, el0, el1;
	for (i = 0; 1 << i <= 2048; i++) {
		for (j = i; j >= 0; j--) {
 
			if (j == i) {
				for (k = threadIdx.y * blockDim.x + threadIdx.x; k < 2048; k += blockDim.x * blockDim.y) {
					frac = (1 << (j + 1)) - 1;
					xtemp = k >> j;
					x = xtemp << j;
 
					group = (x << 2) + frac;
					el0 = (x << 1) + (k & ((1 << j) - 1));
					el1 = group - el0;
 
					if (R[el0] < R[el1]) {
						swapFloat(&R[el0], &R[el1]);
						swapChar(&R_posX[el0], &R_posX[el1]);
						swapChar(&R_posY[el0], &R_posY[el1]);
					}
				}
 
			} else {
				for (k = threadIdx.y * blockDim.x + threadIdx.x; k < 2048; k += blockDim.x * blockDim.y) {
					xtemp = k >> j;
					x = xtemp << j;
 
					el0 = (x << 1) + (k & ((1 << j) - 1));
					el1 = el0 + (1 << j);
 
					if (R[el0] < R[el1]) {
						swapFloat(&R[el0], &R[el1]);
						swapChar(&R_posX[el0], &R_posX[el1]);
						swapChar(&R_posY[el0], &R_posY[el1]);
					}
				}
 
			}
			__syncthreads();
 
		}
	}
 
}
 
__device__ void swapFloat(float *x, float *y) {
 
	float t;
	 t = *x;
	*x = *y;
	*y =  t;
}
 
__device__ void swapChar(char *x, char *y) {
 
	char t;
	 t = *x;
	*x = *y;
	*y =  t;
}
 
__device__	void subPixelAccuracy(int vectorID, float *R, char *R_posX, char *R_posY, float *subXN, float *subXD, float *subYN, float *subYD, float *R_posSubX, float *R_posSubY, char l, char a, float *endX, float *endY) {
 
	int i, goal;
 
	if (threadIdx.y == 0 && threadIdx.x < 5) {
		subXN[threadIdx.x] = 0;
	} else if (threadIdx.y == 1 && threadIdx.x < 5) {
		subXD[threadIdx.x] = 0;
	} else if (threadIdx.y == 2 && threadIdx.x < 5) {
		subYN[threadIdx.x] = 0;
	} else if (threadIdx.y == 3 && threadIdx.x < 5) {
		subYD[threadIdx.x] = 0;
	}
 
	__syncthreads();
 
	for (goal = 0; goal < 5; goal++) {
		for (i = threadIdx.y * blockDim.x + threadIdx.x; i < 4096; i += blockDim.x * blockDim.y) {
			if ((R_posX[i] == R_posX[goal]) && (R_posY[i] == R_posY[goal])) {
				subXD[goal] -= (R[i] + R[i]);
				subYD[goal] -= (R[i] + R[i]);
			} else if ((R_posX[i] == R_posX[goal] - 1) && (R_posY[i] == R_posY[goal])) {
				subXN[goal] -= R[i];
				subXD[goal] += R[i];
			} else if ((R_posX[i] == R_posX[goal] + 1) && (R_posY[i] == R_posY[goal])) {
				subXN[goal] += R[i];
				subXD[goal] += R[i];
			} else if ((R_posX[i] == R_posX[goal]) && (R_posY[i] == R_posY[goal] - 1)) {
				subYN[goal] -= R[i];
				subYD[goal] += R[i];
			} else if ((R_posX[i] == R_posX[goal]) && (R_posY[i] == R_posY[goal] + 1)) {
				subYN[goal] += R[i];
				subYD[goal] += R[i];
			}
		}
	}
 
	__syncthreads();
 
	if (threadIdx.y == 0 && threadIdx.x < 5) {
 
		float subX;
		subX = subXN[threadIdx.x] / (subXD[threadIdx.x] + subXD[threadIdx.x]);
		if (subX > 1 || subX < -1) {
			subX = 0;
		}
		R_posSubX[vectorID * 5 + threadIdx.x] = (float)R_posX[threadIdx.x] + subX - (float)l;
 
	} else if (threadIdx.y == 1 && threadIdx.x < 5) {
		float subY;
		subY = subYN[threadIdx.x] / (subYD[threadIdx.x] + subYD[threadIdx.x]);
		if (subY > 1 || subY < -1) {
			subY = 0;
		}
		R_posSubY[vectorID * 5 + threadIdx.x] = (float)R_posY[threadIdx.x] + subY - (float)a;
 
	}
 
	__syncthreads();
 
	if (threadIdx.y == 0 && threadIdx.x == 0) {
		endX[vectorID] = R_posSubX[vectorID * 5];
 
	} else if (threadIdx.y == 1 && threadIdx.x == 0) {
		endY[vectorID] = R_posSubY[vectorID * 5];
 
	}
	__syncthreads();
}
 
__global__	void vectorCorrection(int totalVectorsX, float *R_posSubX, float *R_posSubY, float *endX, float *endY) {
 
	if ((blockIdx.x != 0) && (blockIdx.x != gridDim.x - 1) && (blockIdx.y != 0) && (blockIdx.y != gridDim.y - 1) && (threadIdx.y == 0) && (threadIdx.x == 0)) {
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			int i, j, k, vectorID;
			float averageX, averageY, dX, dY, X, Y;
 
			vectorID = blockIdx.y * gridDim.x + blockIdx.x;
			averageX = 0;
			averageY = 0;
			X = R_posSubX[vectorID * 5];
			Y = R_posSubY[vectorID * 5];
 
			for (j = -1; j <= 1; j++) {
				for (i = -1; i <= 1; i++) {
					if ((j != 0) || (i != 0)) {
						k = j * totalVectorsX + i;
						averageX += R_posSubX[(vectorID + k) * 5];
						averageY += R_posSubY[(vectorID + k) * 5];
					}
				}
			}
			averageX /= 8;
			averageY /= 8;
 
			for (i = 1; i < 5; i++) {
				dX = R_posSubX[vectorID * 5 + i] - averageX;
				dY = R_posSubY[vectorID * 5 + i] - averageY;
				if (sqrt(dX * dX + dY * dY) < sqrt((X - averageX) * (X - averageX) + (Y - averageY) * (Y - averageY))) {
					X = R_posSubX[vectorID * 5 + i];
					Y = R_posSubY[vectorID * 5 + i];
				}
 
			}
 
			if (sqrt((X - averageX) * (X - averageX) + (Y - averageY) * (Y - averageY)) > 4) {
				X = averageX;
				Y = averageY;
			}
 
			endX[vectorID] = X;
			endY[vectorID] = Y;
 
		}
 
	}
}
 
__host__		void printSummary(int width, int height, char intWndW, char intWndH, char srhWndL, char srhWndR, char srhWndA, char srhWndB, char stepX, char stepY, int totalVectorsX, int totalVectorsY, int totalVectors, int Dbx, int Dby, int Dgx, int Dgy, float h_time, float d_time) {
	int srhWndW, srhWndH, threadsX, threadsY, threads;
	float overlap;
 
	srhWndW = 1 + srhWndL + srhWndR;
	srhWndH = 1 + srhWndA + srhWndB;
	threadsX = Dgx * Dbx;
	threadsY = Dgy * Dby;
	threads = threadsX * threadsY;
	overlap = 1 - ((float)stepX / (float)intWndW);
 
	printf(
		"Summary:\n"
		"Image size:                  %d * %d\n"
		"Interrogation window wize:   %d * %d\n"
		"Search window size:          %d * %d\n"
		"\n"
		"Steps:                       %d * %d\n"
		"Overlap:                     %.2f%% \n"
		"\n"
		"Vector size:                 %d * %d\n"
		"Total vectors                %d\n"
		"\n"
		"Grid size:                   %d * %d\n"
		"Block size:                  %d * %d\n"
		"Thread size:                 %d * %d\n"
		"Total threads:               %d\n"
		"\n"
		"Processing time (with memcpy):	%f sec\n"
		"Processing time (without memcpy): %f sec\n"
		"\n"
		, width, height
		, (int)intWndW, (int)intWndH
		, (int)srhWndW, (int)srhWndH
 
		, (int)stepX, (int)stepY
		, (overlap * 100)
 
		, totalVectorsX, totalVectorsY
		, totalVectors
 
		, Dgx, Dgy
		, Dbx, Dby
		, threadsX, threadsY
		, threads
 
		, h_time
		, d_time
	);
}
 
__host__	void putData(float *startX, float *startY, float *endX, float *endY, int number, char *path) {
	int i;
 
	FILE *fp;
 
	fp = fopen(path, "w");
	if (fp == NULL) {
		fprintf(stderr, "%s file open error\n", path);
	}
 
	for (i = 0; i < number; i++) {	
 
		fprintf(fp, "%.1f	%.1f	%.1f	%.1f\n", startX[i], startY[i], endX[i] * 2, endY[i] * 2);
	}
 
	fclose(fp);
 
}
 
/*vtk output*/
__host__	void putDataVTK(int height, float *startX, float *startY, float *endX, float *endY, int number, int numberX, int numberY, char *pathVTK) {
	int i;
	FILE *fp;
 
	fp = fopen(pathVTK, "w");
	if (fp == NULL) {
		fprintf(stderr, "%s file open error.\n", pathVTK);
	}
 
	fprintf(fp, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET STRUCTURED_GRID\nDIMENSIONS %d %d 1\nPOINTS %d float\n", numberX, numberY, number);
	for (i = 0; i < number; i++) {
		fprintf(fp,"%.1f %.1f %d\n", startX[i], height - startY[i], 0);
	}
 
	fprintf(fp, "POINT_DATA %d\nSCALARS velocity float\nLOOKUP_TABLE default\n", number);
	for (i = 0; i < number; i++) {
		fprintf(fp,"%.10f\n", sqrt((endX[i] * endX[i]) + (endY[i] * endY[i])));
	}
 
	fprintf(fp, "VECTORS velocity float\n");
	for (i = 0; i < number; i++) {
		fprintf(fp,"%.10f %.10f %.10f\n", endX[i], -endY[i], .0);
	}
 
}