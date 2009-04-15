#! /bin/sh

NUM_CLUSTERS=$1
NUM_THREADS=$2
NUM_DATA_POINTS=$3
NUM_DIMENSIONS=$4
STEP_SIZE=$5
DIST_MEASURE=$6

cp cmedoids.in.h cmedoids.h
perl -pi -e "s/#define NUM_CLUSTERS VALUE/#define NUM_CLUSTERS $NUM_CLUSTERS/g" cmedoids.h
perl -pi -e "s/#define NUM_THREADS VALUE/#define NUM_THREADS $NUM_THREADS/g" cmedoids.h
perl -pi -e "s/#define NUM_DATA_POINTS VALUE/#define NUM_DATA_POINTS $NUM_DATA_POINTS/g" cmedoids.h
perl -pi -e "s/#define NUM_DIMENSIONS VALUE/#define NUM_DIMENSIONS $NUM_DIMENSIONS/g" cmedoids.h
perl -pi -e "s/#define STEP_SIZE VALUE/#define STEP_SIZE $STEP_SIZE/g" cmedoids.h
perl -pi -e "s/#define DIST_MEASURE VALUE/#define DIST_MEASURE $DIST_MEASURE/g" cmedoids.h
