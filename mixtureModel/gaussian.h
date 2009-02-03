/*
 * Parameters file for gaussian mixture model based clustering application
 */

#define MAX_CLUSTERS	64

int validateArguments(int argc, char** argv, int* num_clusters, FILE** infile, FILE** outfile);
void printUsage(char** argv);
