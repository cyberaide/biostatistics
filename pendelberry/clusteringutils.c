#include "clusteringutils.h"

void printCenters(Params* p) {
    for(int c=0;c<p->numClusters;c++) {
        cout << "Cluster Center #" << c << ": ";
        for(int d=0; d<p->numDimensions;d++){
            cout << p->centers[c*p->numDimensions+d] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printScatters(Params* p) {
    for(int c=0; c < p->numClusters; c++) {
        cout << "Scatter #" << c << ": " << endl;
        for(int i=0; i < p->numDimensions; i++) {
            for(int j=0; j < p->numDimensions; j++) {
                cout << " " << p->scatters[c*p->numDimensions*p->numDimensions+i*p->numDimensions+j];
            }
            cout << endl;
        }
    }
}

void initRand() {
    //int seed = (int)time(0) * (int)getpid();
    srand((int)getpid());   // REMOVED for testing
    
}

double randdouble() {
    return rand() / (double(RAND_MAX) + 1);
}

double randdoubleRange(double min, double max) {
    if (min > max) {
        return randdouble() * (min - max) + max;
    }
    else {
        return randdouble() * (max - min) + min;
    }
}

int randInt(int max) {
    return int(rand() % max) + 1;
}

int randIntRange(int min, int max) {
    if (min > max) {
        return max + int(rand() % (min - max));
    }
    else {
        return min + int(rand() % (max - min));
    }
}

int getRandIndex(int w, int h) {
    return (randIntRange(0, w) - 1) + (randIntRange(0, h) - 1) * w;
}

// What does this do?
// Is it convex hull or the cluster
bool contains(Params* p, double points[]) {
    int count = 1;

    for (int i = 0; i < p->numClusters; i++) {
        for (int j = 0; j < p->numDimensions; j++) {
            if (p->centers[j + i * p->numDimensions] == points[j]) {
                count++;
            }
        }
    }

    if (count == p->numDimensions) {
        return true;
    }
    else {
        return false;
    }
}


// Get the coordinates of a point
void getPoints(Params* p, double points[], int i) {
    for (int j = 0; j < p->numDimensions; j++) {
        points[j] = p->data[j + i * p->numDimensions];
    }
}

// Initializes and allocates memory for all arrays of the Params structure
// Requires numDimensions, numClusters, numEvents to be defined (by readData)
void allocateParamArrays(Params* p) {
    p->centers = new double[p->numClusters*p->numDimensions];
    p->membership = new double[p->numClusters*p->numEvents];
    p->membership2 = new double[p->numClusters*p->numEvents];
    p->scatters = new double[p->numClusters*p->numDimensions*p->numDimensions];
    p->scatter_inverses = new double[p->numClusters*p->numDimensions*p->numDimensions];
    p->determinants = new double[p->numClusters];
    p->Ti = new int[p->numEvents*p->numClusters];
    p->n = new double[p->numClusters];
    p->A_t = new double[p->numClusters];
}

// Read in the file named "f"
int readData(char* f, Params* p) {
    string line1;
    ifstream file(f);           // input file stream
    vector<string> lines;
    int dim = 0;
    char* temp;

    // Read in all of the lines
    if (file.is_open()) {
        while(!file.eof()) {
            getline(file, line1);

            if (!line1.empty()) {
                lines.push_back(line1);
            }
        }

        file.close();
    }
    else {
        cout << "Unable to read the file " << f << endl;
        return -1;
    }

    line1 = lines[0];
    temp = strtok((char*)line1.c_str(), DELIMITER);

    while(temp != NULL) {
        dim++;
        temp = strtok(NULL, DELIMITER);
    }

    // Assume first line is header, delete it
    lines.erase(lines.begin());

    p->numDimensions = dim;
    p->numEvents = (int)lines.size();

    cout << "Number of Dimensions: " << p->numDimensions << endl;
    cout << "Number of Events: " << p->numEvents << endl;

    p->data = (double*)malloc(sizeof(double) * p->numDimensions * p->numEvents);

    for (int i = 0; i < p->numEvents; i++) {
        temp = strtok((char*)lines[i].c_str(), DELIMITER);

        for (int j = 0; j < p->numDimensions && temp != NULL; j++) {
            p->data[j + i * p->numDimensions] = atof(temp);
            temp = strtok(NULL, DELIMITER);
        }
    }
    return 0;
}

void writeData(Params* p, const char* f) {
    ofstream file;
    ofstream summary;
    int precision = 5;

    file.open(f);

    for (int i = 0; i < p->numEvents; i++) {
        for (int j = 0; j < p->numDimensions; j++) {
            file << fixed << setprecision(precision) << p->data[j + i * p->numDimensions]; 
            if(j < p->numDimensions-1) {
                file << ",";
            }
        }
        file << "\t";
        for (int j = 0; j < p->numClusters; j++) {
            file << fixed << setprecision(precision) << p->membership[j + i * p->numClusters];
            if(j < p->numClusters-1) {
                file << ",";
            }
        }
        file << endl;
    }

    file.close();
    
    summary.open("output.summary");
    for (int t = 0; t < p->numClusters; t++) {
        summary << "Cluster #" << t << endl;
        summary << "Probability: " << p->n[t]/p->numEvents << endl;
        summary << "N: " << p->n[t] << endl;
        summary << "Means: ";
        for(int d=0; d < p->numDimensions; d++) {
            summary << p->centers[t*p->numDimensions+d] << " ";
        }
        summary << endl << endl;
        summary << "R Matrix:" << endl;
        for(int i=0; i< p->numDimensions; i++) {
            for(int j=0; j< p->numDimensions; j++) {
                summary << p->scatters[t*p->numDimensions*p->numDimensions+ i*p->numDimensions + j] << " ";
            }
            summary << endl;
        } 
        summary << endl << endl;
    }
}

int clusterColor(double i, int nc) {
    return (int)((i / nc) * 256);
}

string generateOutputFileName(int nc) {
    string output;
    time_t rawtime;
    struct tm *timeinfo;
    int i;
    char ch[50];

    output = "output";
    //output = "../output/output";

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    i = 1 + timeinfo->tm_mon;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = timeinfo->tm_mday;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = 1900 + timeinfo->tm_year;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = timeinfo->tm_hour;
    sprintf(ch, "-%d", i);
    output.append(ch);

    i = timeinfo->tm_min;
    sprintf(ch, "-%d", i);
    output.append(ch);

    sprintf(ch, "_%dclusters.results", nc);
    output.append(ch);

    return output;
}

/*
 *  * Inverts a square matrix (stored as a 1D double array)
 *   * 
 *    * actualsize - the dimension of the matrix
 *     *
 *      * written by Mike Dinolfo 12/98
 *       * version 1.0
 *        */
void invert_cpu(double* data, int actualsize, double* determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    *determinant = 1.0;

    /*printf("\n\nR matrix before inversion:\n");
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            printf("%.4f ",data[i*n+j]);
        }
        printf("\n");
    }*/
    
  if (actualsize <= 0) return;  // sanity check
  if (actualsize == 1) return;  // must be of dimension >= 2
  for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
  for (int i=1; i < actualsize; i++)  { 
    for (int j=i; j < actualsize; j++)  { // do a column of L
      double sum = 0.0;
      for (int k = 0; k < i; k++)  
          sum += data[j*maxsize+k] * data[k*maxsize+i];
      data[j*maxsize+i] -= sum;
      }
    if (i == actualsize-1) continue;
    for (int j=i+1; j < actualsize; j++)  {  // do a row of U
      double sum = 0.0;
      for (int k = 0; k < i; k++)
          sum += data[i*maxsize+k]*data[k*maxsize+j];
      data[i*maxsize+j] = 
         (data[i*maxsize+j]-sum) / data[i*maxsize+i];
      }
    }
    
    for(int i=0; i<actualsize; i++) {
        *determinant *= data[i*n+i];
    }
    
  for ( int i = 0; i < actualsize; i++ )  // invert L
    for ( int j = i; j < actualsize; j++ )  {
      double x = 1.0;
      if ( i != j ) {
        x = 0.0;
        for ( int k = i; k < j; k++ ) 
            x -= data[j*maxsize+k]*data[k*maxsize+i];
        }
      data[j*maxsize+i] = x / data[j*maxsize+j];
      }
  for ( int i = 0; i < actualsize; i++ )   // invert U
    for ( int j = i; j < actualsize; j++ )  {
      if ( i == j ) continue;
      double sum = 0.0;
      for ( int k = i; k < j; k++ )
          sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
      data[i*maxsize+j] = -sum;
      }
  for ( int i = 0; i < actualsize; i++ )   // final inversion
    for ( int j = 0; j < actualsize; j++ )  {
      double sum = 0.0;
      for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
          sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
      data[j*maxsize+i] = sum;
      }
      
    /*
      printf("\n\nR matrix after inversion:\n");
      for(int i=0; i<n; i++) {
          for(int j=0; j<n; j++) {
              printf("%.4f ",data[i*n+j]);
          }
          printf("\n");
      }
    */
 }
