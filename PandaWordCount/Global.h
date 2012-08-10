/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: Global.h 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __GLOBAL_H__
#define __GLOBAL_H__

typedef struct
{
        int point_id;
        int dim;
        int K;
        int* ptrClusterId;
        int start;
        int end;
        int i;

} KM_KEY_T;

typedef struct
{
        int* ptrPoints;
        int* ptrClusters;
        int* ptrChange;

        float *d_tempClusters;
        float *d_tempDenominators;
        float *d_Clusters;
        float *d_Points;

} KM_VAL_T;


#endif
