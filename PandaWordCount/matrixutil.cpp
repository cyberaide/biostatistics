/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: matrixutil.cpp 
	First Version:	2012-07-01 V0.1
	Current Version: V0.3	
	Last Updates:   2012-8-29

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */


#ifndef __MAP_CPP__
#define __MAP_CPP__

#include <stdio.h>

void cpu_2d_blocked_matrix(float *A, float *B, float *C, int wA,int row_id,int col_id, int bz){
			
    int rowId = row_id;		
    int colId = col_id;		
							
	int wB = wA;			
	int BLOCK_SIZE = 50;					
							
    float Csub = 0.0;	
    int aBase = (rowId)*wA*BLOCK_SIZE;
	int bBase = (colId)*wB*BLOCK_SIZE;
	int aBegin = aBase;
	int bBegin = bBase;
	int aEnd,bEnd;
	for (int step = 0; step < wA/BLOCK_SIZE; step++){
		for (int n=0;n<BLOCK_SIZE;n++){
				
			aEnd = aBegin + BLOCK_SIZE;
			//int aBegin = aBase + n*wA + step*BLOCK_SIZE;	
			//int aEnd = aBegin + BLOCK_SIZE;
				
			for (int step2 = 0; step2< wB/BLOCK_SIZE; step2++){
				for (int n2=0; n2<BLOCK_SIZE; n2++){
					
				bEnd = bBegin + BLOCK_SIZE;
				//int bBegin = bBase + n2*wB + step2*BLOCK_SIZE;
				//int bEnd = bBegin + BLOCK_SIZE;

				for (int i=aBegin;i<aEnd;i++)
				for (int j=bBegin;j<bEnd;j++){
					Csub += A[i]*B[j];		
				}//for
				//int x = (rowId)*BLOCK_SIZE + n;
				//int y = (colId)*BLOCK_SIZE + n;
				bBegin += wB;
				}
				bBegin += BLOCK_SIZE;
			}	

				aBegin += wA;
		//C[x][y] += Csub;
		}//int	
		aBegin += BLOCK_SIZE;
	}//for		
	//printf("Csub:%f\n",Csub);
	//int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    //C[c + wB * ty + tx] = Csub;
}


void cpu_1d_blocked_matrix(float *A, float *B, float *C, int m, int start_row_id_id, int end_id, int bz){

	int i,j,k;
    int start_row_idpoint = start_row_id_id;
	int endpoint = end_id;

    int aHeight = endpoint - start_row_idpoint;
    int aHeightBlocks = aHeight/bz;
    int aLastBlockHeight = aHeight - (aHeightBlocks*bz);
    if (aLastBlockHeight>0){
                aHeightBlocks++;
    }//if
    int bWidthBlocks = m/bz;
    int bLastBlockWidth = m - (bWidthBlocks*bz);
    if (bLastBlockWidth>0){
                bWidthBlocks++;
    }//if

	int commBlocks = m/bz;
    int commLastBlockWidth = m - (commBlocks*bz);
    if (commLastBlockWidth >0){
                commBlocks++;
    }//fi

    int aBlockHeight = bz;
    int bBlockWidth = bz;
    int commBlockWidth = bz;
    int ib,jb,kb;
    	
	for (ib=0;ib<aHeightBlocks;ib++){
                if (aLastBlockHeight>0 && ib==(aHeightBlocks-1)){
                        aBlockHeight = aLastBlockHeight;
                }//if

                bBlockWidth = bz;
                for (jb=0; jb<bWidthBlocks;jb++){
                        if (bLastBlockWidth>0&&jb==(bWidthBlocks-1))
                                bBlockWidth = bLastBlockWidth;

                        commBlockWidth = bz;
                        for (kb =0;kb<commBlocks;kb++){
                        if (commLastBlockWidth>0 && kb==(commBlocks-1))
                                commBlockWidth = commLastBlockWidth;
                        for (i = start_row_idpoint+ib*bz;i<start_row_idpoint+(ib*bz)+aBlockHeight;i++){
                                for (k = kb*bz;k<(kb*bz)+commBlockWidth;k++){
                                        for (j= jb*bz;j<(jb*bz)+bBlockWidth;j++){
                                        C[i*m+j]+=A[i*m+k]*B[k*m+j];
                                        }//for
                                }
                        }//for
                        }//for
                }//for
        }//for
}




#endif //__MAP_CPP__