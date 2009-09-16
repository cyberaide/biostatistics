# change this to your program name
EXE_NAME = fuzzymatrix
# list your object files
OBJS = fuzzymatrix.o
CC = gcc
FLAGS = -g -Wall

# No need to alter anything below this

.c.o:
	$(CC) $< -c $(FLAGS) $(INC)

$(EXE_NAME): $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS) $(INC) $(LIBS) -lm

