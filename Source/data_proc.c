#include <stdio.h>
#include "data_proc.h"

int readX(double *x, char *fname)
{
  double n;
  int i,j;
  char read = 'r';  
  const char* mode = &read;

  FILE * fid = fopen(fname, mode);
  
  i = 0;
  while (fscanf(fid, "%lf%*c", &n) > 0)
    {     
      x[i] = n;
      i++;
    } 
  return 0;
}

int readY(int *y, char *fname)
{
  int n, i, j;
  char read = 'r';  
  const char* mode = &read;

  FILE * fid = fopen(fname, mode);

  i = 0;
  while (fscanf(fid, "%d%*c", &n) > 0)
    {     
      y[i] = n;
      i++;
    } 

  return 0;
}


/* MAIN for testing
int main(void)
{
  int j, k;
  int num_feats = 2;
  int num_samp = 3;
  char *fname = "testX.txt";
  char *fYname = "testY.txt";

  //Client must malloc and free these
  double *x = malloc(num_feats*num_samp*sizeof(double));
  int *y = malloc(num_samp*sizeof(int));

  //Test readX
  readX(x, fname);
  printf("Reprinting what was read by readX:\n");
  for (j = 0; j < num_feats; j++)
    {
      for (k = 0; k < num_samp; k++)
	{
	  printf("%f, ", x[num_samp*j + k]);
	}
      printf("\n");
    }

  //Test readY
  readY(y, fYname);
  printf("Reprinting what was read by readY:\n");
  for (k = 0; k < num_samp; k++)
    printf("%d, ", y[k]);
  printf("\n");
  free(x);
  free(y);
}
*/
