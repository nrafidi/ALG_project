all: runexp

runexp: runexp.o scd.o data_proc.o
	gcc -o runexp.o scd.o data_proc.o -lm -fopenmp

runexp.o: runexp.c scd.h data_proc.h
	gcc -c runexp.c

scd.o: scd.c scd.h
	gcc -c scd.c 

data_proc.o: data_proc.c data_proc.h
	gcc -c data_proc.c

clean:
	rm runexp *.o *~

