#! /bin/sh

NUM_CLUSTERS=$1
ALL_DIMENSIONS=$2
NUM_EVENTS=$3
DISTANCE_MEASURE=$4
FUZZINESS=$5
K1=$6
K2=$7
K3=$8
MEMBER_THRESH=$9
TABU_ITER=${10}
TABU_TENURE=${11}
VOLUME_TYPE=${12}
VOLUME_INC_PARAMS=${13}
THRESHOLD=${14}
NUM_THREADS=${15}

cp cmeans.in.h cmeans.h
perl -pi -e "s/#define NUM_CLUSTERS VALUE/#define NUM_CLUSTERS $NUM_CLUSTERS/g" cmeans.h
perl -pi -e "s/#define ALL_DIMENSIONS VALUE/#define ALL_DIMENSIONS $ALL_DIMENSIONS/g" cmeans.h
perl -pi -e "s/#define NUM_EVENTS VALUE/#define NUM_EVENTS $NUM_EVENTS/g" cmeans.h
perl -pi -e "s/#define DISTANCE_MEASURE VALUE/#define DISTANCE_MEASURE $DISTANCE_MEASURE/g" cmeans.h
perl -pi -e "s/#define FUZZINESS VALUE/#define FUZZINESS $FUZZINESS/g" cmeans.h
perl -pi -e "s/#define K1 VALUE/#define K1 $K1/g" cmeans.h
perl -pi -e "s/#define K2 VALUE/#define K2 $K2/g" cmeans.h
perl -pi -e "s/#define K3 VALUE/#define K3 $K3/g" cmeans.h
perl -pi -e "s/#define MEMBER_THRESH VALUE/#define MEMBER_THRESH $MEMBER_THRESH/g" cmeans.h
perl -pi -e "s/#define TABU_ITER VALUE/#define TABU_ITER $TABU_ITER/g" cmeans.h
perl -pi -e "s/#define TABU_TENURE VALUE/#define TABU_TENURE $TABU_TENURE/g" cmeans.h
perl -pi -e "s/#define VOLUME_TYPE VALUE/#define VOLUME_TYPE $VOLUME_TYPE/g" cmeans.h
perl -pi -e "s/const float VOLUME_INC_PARAMS/const float VOLUME_INC_PARAMS[] = {$VOLUME_INC_PARAMS};/g" cmeans.h
perl -pi -e "s/#define THRESHOLD VALUE/#define THRESHOLD $THRESHOLD/g" cmeans.h
perl -pi -e "s/#define NUM_THREADS VALUE/#define NUM_THREADS $NUM_THREADS/g" cmeans.h


