#include <stdio.h>
#include <stdlib.h>
#include <string.h>


double * readX(void)
{
  float n;
  double *x;
  char read = 'r';  
  int i = 0;
  const char* mode = &read;

  FILE * fid = fopen("testX.txt", mode);
  
  /* if (fid != NULL)
    printf("sup kirstin\n");
  else
  perror("fopen");*/

  while (fscanf(fid, "%f%*c", &n) > 0)
    {     
      //testing
      printf("%f\n", n);
      x[i] = n;
      i++;
    }
  return x;
}

int main(void)
{
  //string fname = "testX.txt";
  double *x = readX();
}
