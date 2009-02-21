#include "invert_matrix.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

static float double_abs(float x);

static int 
ludcmp(float *a,int n,int *indx,double *d);

static void 
lubksb(float *a,int n,int *indx,float *b);

int invert_matrix(float* a, int n, double* determinant) {
    int  i,j,f,g;
    
    float* y = (float*) malloc(sizeof(float)*n*n);
    float* col = (float*) malloc(sizeof(float)*n);
    int* indx = (int*) malloc(sizeof(int)*n);

    *determinant = 0.0;
    if(ludcmp(a,n,indx,determinant)) {
    
        printf("\n\nR matrix after LU decomposition:\n");
        for(i=0; i<n; i++) {
            for(j=0; j<n; j++) {
                printf("%.2f ",a[i*n+j]);
            }
            printf("\n");
        }
        
      for(j=0; j<n; j++) {
        *determinant *= a[j*n+j];
      }
      
      printf("determinant: %E\n",*determinant);
      
      for(j=0; j<n; j++) {
        for(i=0; i<n; i++) col[i]=0.0;
        col[j]=1.0;
        lubksb(a,n,indx,col);
        for(i=0; i<n; i++) y[i*n+j]=col[i];
      } 

      for(i=0; i<n; i++)
      for(j=0; j<n; j++) a[i*n+j]=y[i*n+j];
      
      printf("\n\nMatrix at end of clust_invert function:\n");
      for(f=0; f<n; f++) {
          for(g=0; g<n; g++) {
              printf("%.2f ",a[f*n+g]);
          }
          printf("\n");
      }
      free(y);
      free(col);
      free(indx);
      return(1);
    }
    else {
        *determinant = 0.0;
        free(y);
        free(col);
        free(indx);
        return(0);
    } 
}

static float double_abs(float x)
{
       if(x<0) x = -x;
       return(x);
}

#define TINY 1.0e-20

static int 
ludcmp(float *a,int n,int *indx,double *d)
{
    int i,imax,j,k;
    float big,dum,sum,temp;
    float *vv;

    vv= (float*) malloc(sizeof(float)*n);
    
    *d=1.0;
    
    for (i=0;i<n;i++) 
    {
        big=0.0;
        for (j=0;j<n;j++)
            if ((temp=fabs(a[i*n+j])) > big)
                big=temp;
        if (big == 0.0)
            return 0; /* Singular matrix  */
        vv[i]=1.0/big;
    }
    
    int f,g;
    
    for (j=0;j<n;j++) 
    {   
        for (i=0;i<j;i++) 
        {
            sum=a[i*n+j];
            for (k=0;k<i;k++)
                sum -= a[i*n+k]*a[k*n+j];
            a[i*n+j]=sum;
        }
        
        big=0.0;
        for (i=j;i<n;i++) 
        {
            sum=a[i*n+j];
            for (k=0;k<j;k++)
                sum -= a[i*n+k]*a[k*n+j];
            a[i*n+j]=sum;
            if ( (dum=vv[i]*fabs(sum)) >= big) 
            {
                big=dum;
                imax=i;
            }
        }
        
        if (j != imax) 
        {
            for (k=0;k<n;k++) 
            {
                dum=a[imax*n+k];
                a[imax*n+k]=a[j*n+k];
                a[j*n+k]=dum;
            }
            *d = -(*d);
            vv[imax]=vv[j];
        }
        indx[j]=imax;
        
        /*
        printf("\n\nMatrix after %dth iteration of LU decomposition:\n",j);
        for(f=0; f<n; f++) {
            for(g=0; g<n; g++) {
                printf("%.2f ",a[f][g]);
            }
            printf("\n");
        }
        printf("imax: %d\n",imax);
        */

        /* Change made 3/27/98 for robustness */
        if ( (a[j*n+j]>=0)&&(a[j*n+j]<TINY) ) a[j*n+j]= TINY;
        if ( (a[j*n+j]<0)&&(a[j*n+j]>-TINY) ) a[j*n+j]= -TINY;

        if (j != n-1) 
        {
            dum=1.0/(a[j*n+j]);
            for (i=j+1;i<n;i++)
                a[i*n+j] *= dum;
        }
    }
    free(vv);
    return(1);
}

#undef TINY

static void 
lubksb(float *a,int n,int *indx,float *b)
{
    int i,ii,ip,j;
    float sum;

    ii = -1;
    for (i=0;i<n;i++)
    {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if (ii >= 0)
            for (j=ii;j<i;j++)
                sum -= a[i*n+j]*b[j];
        else if (sum)
            ii=i;
        b[i]=sum;
    }
    for (i=n-1;i>=0;i--)
    {
        sum=b[i];
        for (j=i+1;j<n;j++)
            sum -= a[i*n+j]*b[j];
        b[i]=sum/a[i*n+i];
    }
}