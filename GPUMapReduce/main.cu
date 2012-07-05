/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: main.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */


#include "Panda.h"
#include "Global.h"
#include <ctype.h>

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

#define __OUTPUT__

void validate(char* h_filebuf, Spec_t* spec, int num)
{
	char* key = (char*)spec->outputKeys;
	char* val = (char*)spec->outputVals;
	int4* offsetSizes = (int4*)spec->outputOffsetSizes;
	int2* range = (int2*)spec->outputKeyListRange;

	printf("# of words:%d\n", spec->outputDiffKeyCount);
	if (num > spec->outputDiffKeyCount) num = spec->outputDiffKeyCount;
	for (int i = 0; i < num; i++)
	{
		int keyOffset = offsetSizes[range[i].x].x;
		int valOffset = offsetSizes[range[i].x].z;
		char* word = key + keyOffset;
		int wordsize = *(int*)(val + valOffset);
		printf("%s - size: %d - count: %d\n", word, wordsize, range[i].y - range[i].x);
	}//for
}//void

//-----------------------------------------------------------------------
//usage: WordCount datafile
//param: datafile 
//-----------------------------------------------------------------------
int main( int argc, char** argv) 
{
	if (argc != 2)
	{
		printf("usage: %s [data file]\n", argv[0]);
		exit(-1);	
	}
	
	Spec_t *spec = GetDefaultSpec();
	d_global_state *d_g_state = GetDGlobalState();//configuration for Panda MapRedduce
	
	spec->workflow = MAP_GROUP;
	
#ifdef __OUTPUT__
	spec->outputToHost = 1;
#endif

	printf(":%s\n",argv[0]);
	TimeVal_t allTimer;
	startTimer(&allTimer);
	
	TimeVal_t preTimer;
	startTimer(&preTimer);
	
	FILE* fp = fopen(argv[1], "r");
	fseek(fp, 0, SEEK_END);
	
	char str[256];
	FILE *myfp;
	myfp = fopen(argv[1], "r");
	int iKey = 0;
	while(fgets(str,sizeof(str),myfp) != NULL)
    {
		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
		printf("%s\t len:%d\n", str,strlen(str));
		AddMapInputRecord2(d_g_state, &iKey, str, sizeof(int), strlen(str));
		iKey++;
    }//while
	//d_g_state->h_num_input_record = iKey;
	fclose(myfp);

	int fileSize = ftell(fp) + 1;
	rewind(fp);
	char* h_filebuf = (char*)malloc(fileSize);
	char* d_filebuf = NULL;
	fread(h_filebuf, fileSize, 1, fp);
	checkCudaErrors(cudaMalloc((void**)&d_filebuf, fileSize));	
	fclose(fp);
	
	WC_KEY_T key;
	key.file = d_filebuf;
	
	for (int i = 0; i < fileSize; i++)
		h_filebuf[i] = toupper(h_filebuf[i]);
	
	//This is not the right approach to assign map tasks. 
	//need to be changed in future 7/2/2012
	//there is no string split in CUDA 4.x

	WC_VAL_T val;
	int offset = 0;
	char* p = h_filebuf;
	char* start = h_filebuf;
	while (1)
	{
		int blockSize = 32;
		if (offset + blockSize > fileSize) blockSize = fileSize - offset;
		p += blockSize;
		for (; *p >= 'A' && *p <= 'Z'; p++);
			
		if (*p != '\0') 
		{
			*p = '\0'; 
			++p;
			blockSize = (int)(p - start);
			val.line_offset = offset;
			val.line_size = blockSize;
			
			AddMapInputRecord(spec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));	
			offset += blockSize;
			start = p;
		}
		else
		{
			*p = '\0'; 
			blockSize = (int)(fileSize - offset);
			val.line_offset = offset;
			val.line_size = blockSize;
			AddMapInputRecord(spec, &key, &val, sizeof(WC_KEY_T), sizeof(WC_VAL_T));	
			break;
		}//else
	}
	
	
	(cudaMemcpy(d_filebuf, h_filebuf, fileSize, cudaMemcpyHostToDevice));	
	endTimer("preprocess", &preTimer);
	//----------------------------------------------
	//map/reduce
	//----------------------------------------------
	printf("before MapReduce ->h_num_input_record:%d\n",d_g_state->h_num_input_record);
	MapReduce(spec, d_g_state);
	
	endTimer("all", &allTimer);
	//----------------------------------------------
	//further processing
	//----------------------------------------------
#ifdef __OUTPUT__
	(cudaMemcpy(h_filebuf, d_filebuf, fileSize, cudaMemcpyDeviceToHost));	
	validate(h_filebuf, spec, 10);
#endif
	//----------------------------------------------
	//finish
	//----------------------------------------------
	FinishMapReduce(spec);
	cudaFree(d_filebuf);
	
	//handle the buffer different
	//free(h_filebuf);
	//handle the buffer different

	return 0;
}
