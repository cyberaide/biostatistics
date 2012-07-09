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
	
	//configuration for Panda MapRedduce
	d_global_state *d_g_state = GetDGlobalState();
	//spec->workflow = MAP_GROUP;
	
	printf("start %s ...\n",argv[0]);
	TimeVal_t allTimer;
	startTimer(&allTimer);
	TimeVal_t preTimer;
	startTimer(&preTimer);
	
	char str[256];
	FILE *myfp;
	myfp = fopen(argv[1], "r");
	int iKey = 0;
	while(fgets(str,sizeof(str),myfp) != NULL)
    {
		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
		//printf("%s\t len:%d\n", str,strlen(str));
		AddMapInputRecord2(d_g_state, &iKey, str, sizeof(int), strlen(str));
		iKey++;
    }//while
	fclose(myfp);

	printf("input:%s  \ttotal lines:%d\n",argv[1],iKey);
	
	//----------------------------------------------
	//map/reduce
	//----------------------------------------------
	MapReduce2(d_g_state);
	
	endTimer("all", &allTimer);

	//----------------------------------------------
	//finish
	//----------------------------------------------
	FinishMapReduce2(d_g_state);
	//cudaFree(d_filebuf);
	
	//handle the buffer different
	//free(h_filebuf);
	//handle the buffer different

	return 0;
}
