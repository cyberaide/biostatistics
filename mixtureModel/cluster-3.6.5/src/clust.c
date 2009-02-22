/*
* All questions regarding the software should be addressed to
* 
*       Prof. Charles A. Bouman
*       Purdue University
*       School of Electrical and Computer Engineering
*       1285 Electrical Engineering Building
*       West Lafayette, IN 47907-1285
*       USA
*       +1 765 494 0340
*       +1 765 494 3358 (fax)
*       email:  bouman@ecn.purdue.edu
*       http://www.ece.purdue.edu/~bouman
* 
* Copyright (c) 1995 The Board of Trustees of Purdue University.
*
* Permission to use, copy, modify, and distribute this software and its
* documentation for any purpose, without fee, and without written agreement is
* hereby granted, provided that the above copyright notice and the following
* two paragraphs appear in all copies of this software.
*
* IN NO EVENT SHALL PURDUE UNIVERSITY BE LIABLE TO ANY PARTY FOR DIRECT,
* INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
* USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF PURDUE UNIVERSITY HAS
* BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* PURDUE UNIVERSITY SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
* AND PURDUE UNIVERSITY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
* UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/


#include "clust_defs.h"
#include "alloc_util.h"
#include "clust_io.h"
#include "clust_util.h"
#include "subcluster.h"


double AverageVariance(struct ClassSig *Sig, int nbands);


int main(argc, argv) 
int  argc;
char *argv[];
{
    FILE *fp,*info_fp;

    int i,j,k;
    int init_num_of_subclasses,max_num;
    int vector_dimension;
    int nclasses;
    int num_of_samples;
    char option1[16];
    int  option2;
    double Rmin;
    char fname[512];
    struct SigSet S;
    struct ClassSig *Sig;

    /* set level of diagnostic printing */
    clusterMessageVerboseLevel = 2;

    /* print usage message if arguments are not valid */
    if((argc!=4) && (argc!=5) && (argc!=6)) {
      fprintf(stderr,"\n\nUsage: %s #_subclasses info_file output_params [option1 option2]\n\n",argv[0] );
      fprintf(stderr,"    #_subclasses - initial number of clusters for each class\n\n");
      fprintf(stderr,"    info_file - name of file which contains the following information:\n");
      fprintf(stderr,"      <# of classes>\n");
      fprintf(stderr,"      <data vector length>\n");
      fprintf(stderr,"      <class 1 data file name> <# of data vectors in class 1>\n");
      fprintf(stderr,"      <class 2 data file name> <# of data vectors in class 2>\n");
      fprintf(stderr,"                     .                        .\n");
      fprintf(stderr,"                     .                        .\n");
      fprintf(stderr,"                     .                        .\n");
      fprintf(stderr,"      <last class data file name> <# of data vectors in last class>\n\n");
      fprintf(stderr,"    output_params - name of file containing output clustering");
      fprintf(stderr," parameters\n\n");
      fprintf(stderr,"    option1 - (optional) controls clustering model\n");
      fprintf(stderr,"      full - (default) use full convariance matrices\n");
      fprintf(stderr,"      diag - use diagonal convariance matrices\n\n");
      fprintf(stderr,"    option2 - (optional) controls number of clusters\n");
      fprintf(stderr,"      0 - (default) estimate number of clusters\n");
      fprintf(stderr,"      n - use n clusters in mixture model with n<#_subclasses\n");
      exit(1);
    } 

    /* read number of initial subclasses to use */
    sscanf(argv[1],"%d",&init_num_of_subclasses);

/* Set option 1 */
    if(argc==4) {
      sprintf(option1,"full") ;
    }

    if((argc==5) || (argc==6)) {
      /* set default option 1 */
      sscanf(argv[4], "%s", option1) ;
      if((strcmp(option1, "full")!=0) && (strcmp(option1, "diag")!=0)) {
        fprintf(stderr,"\nInvalid option1: %s\n\n",option1);
        fprintf(stderr,"There are 2 valid assumptions:\n");
        fprintf(stderr,"    full - default option which allows full convariance matrices\n");
        fprintf(stderr,"    diag - use diagonal convariance matrices\n\n");
        exit(1);
      }
    }
          
/* Set option 2 */
    if((argc==4) || (argc==5)) {
      option2 = 0;
    }
          
    if(argc==6) {
      /* set default option 2 */
      sscanf(argv[5], "%d", &option2) ;
      if( (option2<0) || (option2>init_num_of_subclasses) ) {
        fprintf(stderr,"\nInvalid option2: %d \n\n",option2);
        fprintf(stderr,"There are 2 valid assumptions:\n");
        fprintf(stderr,"      0 - (default) estimate number of clusters\n");
        fprintf(stderr,"      n - use n clusters in mixture model with n<#_subclasses\n\n");
        exit(1);
      }
    }


    /* open information file */
    if((info_fp=fopen(argv[2],"r"))==NULL) {
      fprintf(stderr,"can't open information file");
      exit(1);
    }

    /* read number of classes from info file */
    fscanf(info_fp,"%d\n",&nclasses);

    /* read vector dimension from info file */
    fscanf(info_fp,"%d\n",&vector_dimension);


    /* Initialize SigSet data structure */
    I_InitSigSet (&S);
    I_SigSetNBands (&S, vector_dimension);
    I_SetSigTitle (&S, "test signature set");


    /* Allocate memory for cluster signatures */
    for(k=0; k<nclasses; k++) {
      Sig = I_NewClassSig(&S);
      I_SetClassTitle (Sig, "test class signature");
      for(i=0; i<init_num_of_subclasses; i++)
        I_NewSubSig (&S, Sig);
    }

    /* Read data for each class */
    for(k=0; k<nclasses; k++) {
      /* read class k data file name */
      fscanf(info_fp,"%s",fname);

      /* read number of samples for class k */
      fscanf(info_fp,"%d\n",&num_of_samples);

      Sig = &(S.ClassSig[k]);

      I_AllocClassData (&S, Sig, num_of_samples);

      /* Read Data */
      if((fp=fopen(fname,"r"))==NULL) {
        fprintf(stderr,"can't open data file %s", fname);
        exit(1);
      }

      for(i=0; i<Sig->ClassData.npixels; i++) {
        for(j=0; j<vector_dimension; j++) {
          fscanf(fp,"%lf",&(Sig->ClassData.x[i][j]) );
        }
        fscanf(fp,"\n");
      }
      fclose(fp);

      /* Set unity weights and compute SummedWeights */
      Sig->ClassData.SummedWeights = 0.0;
      for(i=0; i<Sig->ClassData.npixels; i++) {
        Sig->ClassData.w[i] = 1.0;
        Sig->ClassData.SummedWeights += Sig->ClassData.w[i];
      }
    }
    fclose(info_fp);


    /* Compute the average variance over all classes */
    Rmin = 0;
    for(k=0; k<nclasses; k++) {
      Sig = &(S.ClassSig[k]);
      Rmin += AverageVariance(Sig, vector_dimension);
    }
    Rmin = Rmin/(COVAR_DYNAMIC_RANGE*nclasses);

    /* Perform clustering for each class */
    for(k=0; k<nclasses; k++) {

      Sig = &(S.ClassSig[k]);

      if(1<=clusterMessageVerboseLevel) {
        fprintf(stdout,"Start clustering class %d\n\n",k);
      }

      if(strcmp(option1, "diag")==0) {
        /* assume covariance matrices to be diagonal */
        subcluster(&S,k,option2,(int)CLUSTER_DIAG,Rmin,&max_num);

      } else {
        /* no assumption for covariance matrices */
        subcluster(&S,k,option2,(int)CLUSTER_FULL,Rmin,&max_num);
      }

      if(2<=clusterMessageVerboseLevel) {
        fprintf(stdout,"Maximum number of subclasses = %d\n",max_num);
      }

      I_DeallocClassData(&S, Sig);
    }

    /* Write out result to output parameter file */
    if((fp=fopen(argv[3],"w"))==NULL) {
      fprintf(stderr,"can't open parameter output file");
      exit(1);
    }
    I_WriteSigSet(fp, &S);
    fclose(fp);

    /* De-allocate cluster signature memory */
    I_DeallocSigSet(&S); 

    return(0);
}


double AverageVariance(struct ClassSig *Sig, int nbands)
{
     int     i,b1;
     double  *mean,**R,Rmin;

     /* Compute the mean of variance for each band */
     mean = G_alloc_vector(nbands);
     R = G_alloc_matrix(nbands,nbands);

     for(b1=0; b1<nbands; b1++) {
       mean[b1] = 0.0;
       for(i=0; i<Sig->ClassData.npixels; i++) {
         mean[b1] += (Sig->ClassData.x[i][b1])*(Sig->ClassData.w[i]);
       }
       mean[b1] /= Sig->ClassData.SummedWeights;
     }

     printf("Variances: ");
     for(b1=0; b1<nbands; b1++) {
       R[b1][b1] = 0.0;
       for(i=0; i<Sig->ClassData.npixels; i++) {
         R[b1][b1] += (Sig->ClassData.x[i][b1])*(Sig->ClassData.x[i][b1])*(Sig->ClassData.w[i]);
       }
       R[b1][b1] /= Sig->ClassData.SummedWeights;
       R[b1][b1] -= mean[b1]*mean[b1];
       printf("%f ",R[b1][b1]);
     }
     printf("\n");

     /* Compute average of diagonal entries */
     Rmin = 0.0;
     for(b1=0; b1<nbands; b1++) 
       Rmin += R[b1][b1];

   printf("Total variance: %f\n",Rmin);
     Rmin = Rmin/(nbands);

     G_free_vector(mean);
     G_free_matrix(R);

     return(Rmin);
}

