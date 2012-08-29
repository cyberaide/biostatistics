/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	Time: 2012-07-01
	Files:PandaLib.cu PandaScan.cu PandaSort.cu PandaUtils.cu 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */
 

Build notes for VS:

BUG:



NOTES:
1)	do not include map.cu and reduce.cu when building source files, as they are already included in PandaLib.cu
2)	Visual Studio Intellisense does not autolist all the members: 
	C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\IDE\devenv.exe /resetsettings
3)	remove inline when build source code in Linux


TODO List:

1)Reduce: Emit					DONE
2)Map (void *key, void*val);	DONE
3)Reduce (void *key, void*val);	DONE
4)Colecent memory access;
5)Many Map tasks Map to one Mapper	DONE
6)Shared memory for intermediate data	DONE
7)Shared memory for reduce data

