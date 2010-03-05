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
    print "Usage: ./cmeans.py INPUT_FILE NUM_CLUSTERS [THRESHOLD=0.0001] [MIN_ITERS=0] [MAX_ITERS=100] [DEVICE=0] [FUZZINESS=2] [DISTANCE_MEASURE=0] [K1=1.0] [K2=0.01] [K3=1.5] [MEMBER_THRESHOLD=0.05] [TABU_ITER=100] [TABU_TENURE=5] [MDL=0] [CPU_ONLY=0]"
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
        'THRESHOLD' : 0.0001,
        'MIN_ITERS' : 0,
        'MAX_ITERS' : 100,
        'DEVICE' : 0,
        'FUZZINESS' : 2,
        'DISTANCE_MEASURE': 0,
        'K1': 1.0,
        'K2': 0.01,
        'K3': 1.5,
        'MEMBER_THRESHOLD': 0.05,
        'TABU_ITER': 100,
        'TABU_TENURE': 5,
        'MDL': 0,
        'CPU_ONLY': 0,
    }
    num_params = len(params.keys())
    if num_args == num_params:
        params['INPUT_FILE'] = args[1]
        params['NUM_CLUSTERS'] = args[2]
        params['THRESHOLD'] = args[3]
        params['MIN_ITERS'] = args[4]
        params['MAX_ITERS'] = args[5]
        params['DEVICE'] = args[6]
        params['FUZZINESS'] = args[7]
        params['DISTANCE_MEASURE'] = args[8]
        params['K1'] = args[9]
        params['K2'] = args[10]
        params['K3'] = args[11]
        params['MEMBER_THRESHOLD'] = args[12]
        params['TABU_ITER'] = args[13]
        params['TABU_TENURE'] = args[14]
        params['MDL'] = args[15]
        params['CPU_ONLY'] = args[16]
    elif num_args == 3:
        params['INPUT_FILE'] = args[1]
        params['NUM_CLUSTERS'] = args[2]
    elif 3 < num_args < num_params:
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
        if params['INPUT_FILE'].lower().endswith(".bin"):
            input = open(params['INPUT_FILE'],'rb')
            import struct
            params['NUM_EVENTS'] = struct.unpack('i',input.read(4))[0]
            params['NUM_DIMENSIONS'] = struct.unpack('i',input.read(4))[0]
        else:
            input = open(params['INPUT_FILE'])
            line = input.readline() # Read the header line
            num_dimensions = len(line.split(DELIMITER))
            num_events = 0
            for line in input:
                num_events += 1
            params['NUM_DIMENSIONS'] = num_dimensions
            params['NUM_EVENTS'] = num_events

        params['NUM_EVENTS'] -= params['NUM_EVENTS'] % 16
        print "%d events removed to ensure memory alignment" % (params['NUM_EVENTS'] % 16)
    else:
        print "Invalid input file."
        sys.exit(1)
   
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
        cmd = '../../bin/linux/release/cmeansSingleGPU "%s"' % params['INPUT_FILE']
        print cmd
        os.system(cmd)
