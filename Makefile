# CC = mpic++

# all:
# 	$(CC) -O3 -o final bfs.cpp
# 	time mpirun -np 1 ./final --taskid=2 --inputpath="test/test4/test-input-4.gra" --headerpath="test/test4/test-header-4.dat" --outputpath="output/naya1.txt" --verbose=0 --startk=2 --endk=3 --p=3
# time mpirun -n 8 ./a3 --taskid=2 --inputpath=test/test$(TEST)/test-input-$(TEST).gra --headerpath=test/test$(TEST)/test-header-$(TEST).dat --outputpath=test/test$(TEST)/test-output-$(TEST).txt --verbose=1 --startk=1 --endk=2 --p=20

CC=mpic++
FLAGS=-std=c++17 -O3 -g -fopenmp
TEST=0
inputpath="./test/test$(TEST)/test-input-$(TEST).gra"
headerpath="./test/test$(TEST)/test-header-$(TEST).dat"
outputpath="./test/test$(TEST)/test-output-$(TEST).txt"

sources= bfs.cpp
objects=$(sources:.cpp=.o)

a3:$(objects)
	$(CC) $^ $(FLAGS) -o $@

run:a3
	time mpirun -n 2 ./a3 --taskid=2 --inputpath=$(inputpath) --headerpath=$(headerpath) --outputpath=$(outputpath) --verbose=1 --endk=3 --p=4

%.o: %.cpp
	$(CC) $(FLAGS) -c $<

clean:
	rm *.o a3