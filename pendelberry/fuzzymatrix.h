#include "clusteringutils.h"

#define MAXITER 100000

void setCenters(Params* p); 
void setScatterMatrices(Params* p);
void computeGeneralFormula_eq31(Params* p, double beta, double tau, double THETA);
void printCenters(Params* p);

