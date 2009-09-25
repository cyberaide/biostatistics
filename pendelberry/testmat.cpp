// new test program to check matrix inversion template code
//#include <stdio.h>
//#include <stdlib.h>
#include <iostream.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_h"
void dumpMatrixValues(matrix <double> M)  {
  bool xyz;
  double rv;
  for (int i=0; i < M.getactualsize(); i++)
    {
    cout << "i=" << i << ": ";
    for (int j=0; j<M.getactualsize(); j++)
      {
        M.getvalue(i,j,rv,xyz);
        cout << rv << " ";
      }
    cout << endl;
    }
};
int main (int argc, char** argv)  {
  cout << "hello world; this is a test of matrix inversion." << endl;
  matrix <double> M1(200,200);  // for test we create & invert this matrix
  matrix <double> M2(4,3);      // this will be a copy of original M1
  matrix <double> M3(3,2);      // this will contain the product
  int k = 0;
  rand();  // eliminates the first (= zero) call
  for (int i=0; i < M1.getactualsize(); i++)  // define random values for initial matrix
    for (int j=0; j<M1.getactualsize(); j++)
      {
        M1.setvalue(i,j,-22+(100. * rand())/RAND_MAX);
        k++;
      }
  cout << "original matrix (size " << M1.getactualsize() << " x " <<
      M1.getactualsize() << ") created and filled with random values."<< endl;
//  dumpMatrixValues(M1);
  M2.copymatrix(M1);
  M1.invert();  // invert the matrix
  cout << "DONE with matrix inversion!" << endl;
//  dumpMatrixValues(M1);
  M3.settoproduct(M1,M2);
  cout << "product computed; original x inverse:" << endl;
  M3.comparetoidentity();
//  dumpMatrixValues(M3);
  M3.settoproduct(M2,M1);
  cout << "product computed; inverse x original:" << endl;
  M3.comparetoidentity();
//  dumpMatrixValues(M3);
}
