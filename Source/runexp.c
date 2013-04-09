#include "scd.h"
#include "data_proc.h"
#include <time.h>
#include <stdlib.h>
#include <float.h>

double test_w(double* w, double* x, int* y, int num_feats, int num_samp)
{
  int i,j;
  int num_wrong;
  # pragma omp parallel for
  for (i = 0; i < num_samp; i++)
    {
      double dot = 0;
      # pragma omp parallel for
      for (j = 0; j < num_feats; j++)
	{	  
	  dot += w[j]*x[num_samp*j + i];
	}
      int guess = dot > 0 ? 1 : -1;
      num_wrong += guess != y[i];
    }
  return (double) num_wrong / num_samp; 
}

double choose_lambda (int fold_size, double* x, int* y, int num_samp, int num_feat, int batch, int it, double eta, int p_batch, int p_w_samp)
{
  int i,j,k,l,m;
  double lambdas[] = {0, .001, .01, .1, 1, 10, 100};
  double weights[num_feat];
  double min_lambda = -1;
  double min_err = DBL_MAX;
  double test_err;
  for (i = 0; i < sizeof(lambdas) / sizeof(double); i++)
    {
      test_err = 0;
      double *w = malloc(num_feat * sizeof(double));
      int fold = num_samp / fold_size;
      double* trainX = malloc((num_samp - fold_size) * num_feat * sizeof(double));
      int* trainY = malloc(fold * sizeof(int));
      for (j = 0; j < fold; j++)
	{
	  int start = j*fold_size;
	  int end = start + fold_size - 1;
	  l = 0;
	  for (k = 0; k < start; k++)
	    {
	      trainY[l++] = y[k];
	    }
	  for (k = end; k < num_samp; k++)
	    {
	      trainY[l++] = y[k];
	    }
	  l = 0;
	  for (m = 0; m < num_feat; m++)
	    {
	      for (k = 0; k < start; k++)
		{
		  trainX[l++] = x[k + num_samp*m];
		}
	      for (k = end; k < num_samp; k++)
		{
		  trainX[l++] = x[k + num_samp*m];
		}
	    }
	  runSCD(batch, w, trainX, trainY, lambdas[i], num_samp, num_feat, p_batch, p_w_samp, it, eta);
	  test_err += test_w(w, &x[start], &y[start], num_feat, fold_size);
	}
      test_err = (double) (test_err / fold);
      if (test_err < min_err)
	{
	  min_err = test_err;
	  min_lambda = lambdas[i];
	}
      free(trainX);
      free(trainY);
      return min_lambda;
    }
}

int main(void)
{

  clock_t start;
  clock_t end;

  double times[4];
  char *xFile = "whatever.csv";
  char *yFile = "whateverelse.csv";
  
  int num_samp = 3;
  int num_feat = 5;

  int fold_size = 10;
  double *x = malloc(num_samp*num_feat*sizeof(double));
  readX(x, xFile);
  int *y = malloc(num_samp*sizeof(int));
  readY(y, yFile);

  double weights[num_feat];
  int batch = 2;
  int it = 1000;
  int eta = 0.001;
  int p_batch = 1;
  int p_w_samp = 1;
  int lamb = choose_lambda (fold_size, x, y, num_samp, num_feat, batch, it, eta, p_batch, p_w_samp);

  start = clock();
  runSCD(batch, weights, x, y, lamb, num_samp, num_feat, 0, 0, it, eta);
  end = clock();
  times[0] = ((double)(end - start))/CLOCKS_PER_SEC;

  start = clock();
  runSCD(batch, weights, x, y, lamb, num_samp, num_feat, 0, 1, it, eta);
  end = clock();
  times[1] = ((double)(end - start))/CLOCKS_PER_SEC;

  start = clock();
  runSCD(batch, weights, x, y, lamb, num_samp, num_feat, 1, 0, it, eta);
  end = clock();
  times[2] = ((double)(end - start))/CLOCKS_PER_SEC;

  start = clock();
  runSCD(batch, weights, x, y, lamb, num_samp, num_feat, 1, 1, it, eta);
  end = clock();
  times[3] = ((double)(end - start))/CLOCKS_PER_SEC;

  //Debugging code + checking results


  // free stuff
  free(x);
  free(y);
  return 0;
}
