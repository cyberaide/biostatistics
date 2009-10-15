#!/usr/bin/env python
"""
Configures the c-means header file (cmeans.in.h), compiles, executes 
based on the input parameters
"""

import sys
import os

# Delimiter used in the input files
DELIMITER = ","


def usage():
    print "Usage: ./cmeans INPUT_FILE NUM_CLUSTERS [FUZZINESS=2] [THRESHOLD=0.0001] [DISTANCE_MEASURE=0] [K1=1.0] [K2=0.01] [K3=1.5] [MEMBER_THRESHOLD=0.05] [TABU_ITER=100] [TABU_TENURE=5] [VOLUME_TYPE=0]"
    print
    print " INPUT_FILE and NUM_CLUSTERS are required."
    print " The rest of the parameters can be specified in NAME=VALUE form"
    print " Alternatively, the parameters can be provided positionally if all are provided"

def parseInputArgs():
    args = sys.argv
    num_args = len(args)
    
    params = {
        'INPUT_FILE' : '',
        'NUM_CLUSTERS' : 0,
        'FUZZINESS' : 2,
        'THRESHOLD' : 0.0001,
        'DISTANCE_MEASURE': 0,
        'K1': 1.0,
        'K2': 0.01,
        'K3': 1.5,
        'MEMBER_THRESHOLD': 0.05,
        'TABU_ITER': 100,
        'TABU_TENURE': 5,
        'VOLUME_TYPE': 0,
    }

    if num_args == 13:
        params['INPUT_FILE'] = args[1]
        params['NUM_CLUSTERS'] = args[2]
        params['FUZZINESS'] = args[3]
        params['THRESHOLD'] = args[4]
        params['DISTANCE_MEASURE'] = args[5]
        params['K1'] = args[6]
        params['K2'] = args[7]
        params['K3'] = args[8]
        params['MEMBER_THRESHOLD'] = args[9]
        params['TABU_ITER'] = args[10]
        params['TABU_TENURE'] = args[11]
        params['VOLUME_TYPE'] = args[12]
    elif num_args == 3:
        params['INPUT_FILE'] = args[1]
        params['NUM_CLUSTERS'] = args[2]
    elif 3 < num_args < 13:
        params['INPUT_FILE'] = args[1]
        params['NUM_CLUSTERS'] = args[2]
        for arg in args[3:]:
            try:
                key,val = arg.split("=")
                key = key.upper()
                assert key in params.keys()
                if key in ['K1','K2','K3','MEMBER_THRESHOLD','THRESHOLD']:
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except AssertionError:
                print "Error: Found invalid parameter '%s'" % key
            except ValueError:
                print "Error: Invalid value '%s' for parameter '%s'" % (val,key)
                print
                usage()
                sys.exit(1)
    else:
        print "Invalid command line arguments."
        print
        usage()
        sys.exit(1)

    if os.path.exists(params['INPUT_FILE']):
        input = open(params['INPUT_FILE'])
        line = input.readline() # Read the header line
        num_dimensions = len(line.split(DELIMITER)) 
        num_events = 0
        for line in input:
            num_events += 1
        params['NUM_DIMENSIONS'] = num_dimensions
        params['NUM_EVENTS'] = num_events
    else:
        print "Invalid input file."
        sys.exit(1)
   
    num_threads = 384
    if 8 < num_dimensions <= 10:
        num_threads = 320;
    elif 10 < num_dimensions <= 12:
        num_threads = 256
    elif 12 < num_dimensions <= 16:
        num_threads = 192
    elif 16 < num_dimensions <= 24:
        num_threads = 128
    else:
        num_threads = 64
    params['NUM_THREADS'] = num_threads

    return params    

if __name__ == '__main__':
    params = parseInputArgs()
    from pprint import pprint
    pprint(params)
     
    header_file = open("cmeans.in.h",'r').read()
    new_header = open("cmeans.h",'w')

    for key,val in params.items():
        header_file = header_file.replace("$%s$" % key,str(val))

    # Output new header file with command line params
    new_header.write(header_file)
    new_header.close()

    # Compile the program
    print "CWD:",os.getcwd()
    os.system('make clean') # cpp files won't get recompiled if cmeans.h changes for some reason...
    error = os.system('make')

    # Run the cmeans program
    if not error:
        cmd = '../../bin/linux/release/cmeans "%s"' % params['INPUT_FILE']
        print cmd
        os.system(cmd)
