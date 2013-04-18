#include "scd.h"
#include "data_proc.h"
#include <time.h>
#include <stdlib.h>
#include <float.h>

int check_nan(double w[])
{
  int i;
  int ret = 0;
  int length = sizeof(w)/sizeof(double);
# pragma omp parallel for
  for (i = 0; i < length; i++)
    {
      if (isnan(w[i]))
	ret = 1;
    }
  return ret;
}

double test_w(double* w, double* x, int* y, int num_feat, int num_samp)
{
  int i,j;
  int guess;
  int num_wrong = 0;
# pragma omp parallel for
  for (i = 0; i < num_samp; i++)
    {
      double dot = 0;
# pragma omp parallel for
      for (j = 0; j < num_feat; j++)
	{	  
	  dot += w[j]*x[num_samp*j + i];
	}
      guess = dot > 0 ? 1 : -1;
      num_wrong += (guess != y[i]);
    }
  return (double) num_wrong / num_samp; 
}

double choose_lambda (int fold, double* x, int* y, int num_samp, int num_feat, int batch, int s_batch,  int it, double eta)
{
  int i,j,k,m;
  double lambdas[] = {0, .001, .01, .1, 1, 10, 100, 1000};//, 10000, 100000};
  int len = sizeof(lambdas)/sizeof(double);
  double weights[num_feat];
  int fold_size = num_samp / fold;
  double errs[len];
  int inan[len];

# pragma omp parallel for
  for (i = 0; i < len; i++)
    {
      double test_err = 0;
      double *w = malloc(num_feat * sizeof(double));
      double* trainX = malloc((num_samp - fold_size + 1) * num_feat * sizeof(double));
      int* trainY = malloc((num_samp - fold_size + 1) * sizeof(int));
# pragma omp parallel for
      for (j = 0; j < fold; j++)
	{
	  int start = j*fold_size;
	  int end = start + fold_size;
	  int l = 0;
# pragma omp parallel for if (p_fold)
	  for (k = 0; k < start; k++)
	    {
	      trainY[l++] = y[k];
	    }
# pragma omp parallel for if (p_fold)
	  for (k = end; k < num_samp; k++)
	    {
	      trainY[l++] = y[k];
	    }
	  l = 0;
# pragma omp parallel for if (p_fold)
	  for (m = 0; m < num_feat; m++)
	    {
# pragma omp parallel for if (p_fold)
	      for (k = 0; k < start; k++)
		{
		  trainX[l++] = x[k + num_samp*m];
		}
# pragma omp parallel for if (p_fold)
	      for (k = end; k < num_samp; k++)
		{
		  trainX[l++] = x[k + num_samp*m];
		}
	    }

	  //clock_t cst = clock();
	  runSCD(batch, w, trainX, trainY, lambdas[i], num_samp - fold_size, num_feat, s_batch, 1, 1, it, eta);
	  //clock_t cend = clock();
	  //printf("runSCD took %.2lf sec for lambda = %lf\n", ((double) (cend-cst)/CLOCKS_PER_SEC), lambdas[i]);
	  //NOTE: Probably should make atomic, but this doesn't need to be precise so I guess it doesn't matter?
	  test_err += test_w(w, &x[start], &y[start], num_feat, fold_size);
	}
      errs[i] = (double) (test_err / fold);
      inan[i] = check_nan(w);
      free(trainY);
      free(trainX);
      free(w);
    }
  
  double min_lambda = -1;
  double min_err = DBL_MAX;
  for (i = 0; i < len; i++)
    {
      if ((errs[i] <= min_err) && !inan[i])
	{
	  min_err = errs[i];
	  min_lambda = lambdas[i];
	}
    }

  return min_lambda;
}

int main(void)
{

  int i, j, k;

  //Training Set
//  char *xFile = "/usr2/home/kearly/rcv_tr.csv"; 
  char *xFile = "/usr0/home/nrafidi/Data/ij_tr.csv";
//  char *yFile = "/usr2/home/kearly/rcv_labels_tr.csv";
  char* yFile = "/usr0/home/nrafidi/Data/ij_labels_tr.csv";
//  char *xTestFile = "/usr2/home/kearly/rcv_ts_short.csv"; 
  char* xTestFile = "/usr0/home/nrafidi/Data/ij_ts.csv"; //"/usr2/home/kearly/arc_ts.csv";
//  char *yTestFile = "/usr2/home/kearly/rcv_labels_ts_short.csv"; 
  char *yTestFile = "/usr0/home/nrafidi/Data/ij_labels_ts.csv"; //"/usr2/home/kearly/arc_labels_ts.csv";

  int num_samp = 24995; // for ij dataset
  int num_samp_ts = num_samp; // for ij dataset
  int num_feat = 22; // for ij dataset

//  int num_samp = 20242; // for rcv dataset 
//  int num_samp_ts = 1000; // for rcv dataset
//  int num_feat = 47236; // for rcv dataset

  double *x = malloc(num_samp*num_feat*sizeof(double));
  double *testX = malloc(num_samp_ts*num_feat*sizeof(double));
  readX(x, xFile);
  readX(testX, xTestFile);
  int *y = malloc(num_samp*sizeof(int));
  readY(y, yFile);
  int *testY = malloc(num_samp_ts*sizeof(int));
  readY(testY, yTestFile);
// Print data after reading it in
  /*
    for (i = 0; i < num_samp; i++)
    printf("Y[%d] = %d\n",i,y[i]);
    for (i = 0; i < num_feat; i++)
    {
    for (j = 0; j < num_samp; j++)
    printf("%f ", x[num_samp*i + j]);
    printf("\n");
    }
  */

  //Feature batch values
//  int batch_max = num_feat;
  int batch_max = num_feat; // set to num_feat
  int batch_min = 0; // set to 0
  int batch_step = 2;
  //Sample batch values
  int s_batch_max = num_samp;
  int s_batch_min = 0; // set to 0
  int s_batch_step = 2400;
  //Iterations, step size and fold size for cross-validation
  int it = 100;
  double eta = 0.00001;
  int num_folds = 10;
  //  int fold_size = ;
  //Number of experiments
  int numexp = batch_max/batch_step + s_batch_max/s_batch_step;
  

  double timel[numexp], lambs[numexp], times[numexp*4], errs[numexp*4];
  int b, s_b;

# pragma omp parallel for
  for (b = batch_min; b <=batch_max; b +=batch_step)
    {
      int exp =  b/batch_step;
      int batch = b != 0 ? b : 1;
//      printf("Feature batch size batch = %d\n", batch);
# pragma omp parallel for
      for (s_b = s_batch_min; s_b <= s_batch_max; s_b+=s_batch_step)
	{
	  int s_exp = exp + s_b/s_batch_step;
	  int s_batch = s_b != 0 ? s_b : 1;
//          printf("Sample batch size s_batch = %d\n", s_batch);
	 
	  //choose lambda
	  clock_t start = clock();
	  lambs[s_exp] = choose_lambda (num_folds, x, y, num_samp, num_feat, batch, s_batch, it, eta);
	  clock_t end = clock();
	  timel[s_exp] = ((double) (end - start))/CLOCKS_PER_SEC;
	  //printf("%lf, ", ((double) (end - start))/CLOCKS_PER_SEC);
	  // printf("Chosen lambda = %f\n",lambs[exp]); //if this prints -1 we're having problems
# pragma omp parallel for
	  for (i = 0; i < 4; i++)
	    {
	      int first = i-2 >= 0 ? 1: 0;
	      int second = i%2;
	      double weights[num_feat];
	      clock_t startt = clock();
	      runSCD(batch, weights, x, y, lambs[s_exp], num_samp, num_feat, s_batch, first, second, it, eta);
	      if (check_nan(weights))
		  printf("Weights are nan - discard trial %d,%d,%d,\n", batch, s_batch, i);
	      clock_t endd = clock();
	      times[s_exp + i] = ((double)(endd - startt))/CLOCKS_PER_SEC;
	      errs[s_exp + i] = test_w(weights, testX, testY, num_feat, num_samp_ts);
//	      if (errs[s_exp + i] <= 0)
//		{
	      /* printf("printing weights\n");
		  int ww;
		  for (ww = 0; ww < num_feat; ww++)
                    {
                      if (weights[ww] != 0)
		        printf("Weight %d = %lf\n", ww, weights[ww]);
			}*/
//		}
	    }
	}
    }

  //print formatted output

  //first line
  printf("Batch, S_batch, TimeL, L, T1, E1, T2, E2, T3, E3, T4, E4,\n");
  //Rest of data
  for (b = batch_min; b <= batch_max; b+= batch_step)
    {
      int exp =  b/batch_step;
      int batch = b != 0 ? b : 1;
      for (s_b = s_batch_min; s_b <= s_batch_max; s_b+=s_batch_step)
	{
	  int s_exp = exp + s_b/s_batch_step;
	  int s_batch = s_b != 0 ? s_b : 1;
	  printf("%d, %d, ", batch, s_batch);
	  printf("%lf, ", timel[s_exp]);
	  printf("%lf, ", lambs[s_exp]);
      for (i = 0; i < 4; i++)
	{
	  printf("%lf, ", times[s_exp + i]);
	  printf("%lf, ", errs[s_exp + i]);
	}
      printf("\n");
	}
    }


  // free stuff
  free(x);
  free(y);
  free(testX);
  free(testY);
  return 0;
}
