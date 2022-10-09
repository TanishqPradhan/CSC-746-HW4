#include <iostream>

#include <string.h>

#include <stdlib.h>

#include <omp.h>

#include "likwid-stuff.h"



const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";





/* This routine performs a dgemm operation

 *  C := C + A * B

 * where A, B, and C are n-by-n matrices stored in column-major format.

 * On exit, A and B maintain their input values. */

void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 

{

   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled



   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,

   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)

   // after the matrix multiply code but before the end of the parallel code block.



   std::cout << "Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here " << std::endl;

   

   #pragma omp parallel

   {

   LIKWID_MARKER_START(MY_MARKER_REGION_NAME);

   

   double* A_Copy = new double[block_size*block_size];

   double* B_Copy = new double[block_size*block_size];

   double* C_Copy = new double[block_size*block_size];

   

   int nblocks = n/block_size;

   

   #pragma omp for

   for (int i = 0; i<nblocks; i++)

   {

   	for (int j=0; j<nblocks; j++)

   	{

   		//Copy C into C_Copy

   		for (int copy_j = 0; copy_j<block_size; copy_j++)

   		{

   			for (int copy_i = 0; copy_i<block_size; copy_i++)

   			{

   				C_Copy[copy_j*block_size + copy_i] = C[(j*block_size + copy_j)*n + i*block_size + copy_i];

   			}

   		}

   		for (int k =0; k<nblocks; k++)

   		{

   			//Copy for A & B

   			for (int copy_j = 0; copy_j<block_size; copy_j++)

   			{

   				for (int copy_i = 0; copy_i<block_size; copy_i++)

   				{

   					A_Copy[copy_j*block_size + copy_i] = A[(k*block_size + copy_j)*n + i*block_size + copy_i];

   					B_Copy[copy_j*block_size + copy_i] = B[(j*block_size + copy_j)*n + k*block_size + copy_i];

   				}

   			}

   			

   			for(int y = 0; y<block_size; y++)

   			{

   				for (int z = 0; z<block_size; z++)

   				{

   					for (int x = 0; x<block_size; x++)

   					{

   						C_Copy[y*block_size + x] += A_Copy[z*block_size + x] * B_Copy[y*block_size + z];

   					}

   				}

   			}

   		}

   		

   		//Write back to C

   		for (int copy_j = 0; copy_j<block_size; copy_j++)

   		{

   			for (int copy_i = 0; copy_i<block_size; copy_i++)

   			{

   				C[(j*block_size + copy_j)*n + i*block_size + copy_i] = C_Copy[copy_j*block_size + copy_i];

   			}

   		}

   	}

   }

   delete[] A_Copy;

   delete[] B_Copy;

   delete[] C_Copy;

   

   LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);

   }

   

   

}