CC = g++ -fcoroutines -fPIC -std=c++20 -fconcepts-diagnostics-depth=4 -ggdb
all:process1 process2

process1:process1.cpp
	$(CC) process1.cpp -o process1

process2:process2.cpp
	$(CC) process2.cpp -o process2

process3:process3.cpp
	$(CC) process3.cpp -o process3

clean:
	-rm process