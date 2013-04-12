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

double test_w(double* w, double* x, int* y, int num_feats, int num_samp)
{
  int i,j;
  int guess;
  int num_wrong = 0;
  # pragma omp parallel for
  for (i = 0; i < num_samp; i++)
    {
      double dot = 0;
      # pragma omp parallel for
      for (j = 0; j < num_feats; j++)
	{	  
	  dot += w[j]*x[num_samp*j + i];
	}
      guess = dot > 0 ? 1 : -1;
      //printf("guess at %d = %d\n", i, guess);
      //printf("y[%d] = %d\n", i, y[i]);
      //printf("guess != y[%d] = %d\n",i, (guess != y[i]));
      num_wrong += (guess != y[i]);
      //printf("num_wrong at %d = %d\n", i, num_wrong);
    }
  //printf("num_wrong = %d\n", num_wrong);
  //printf("num_samp = %d\n", num_samp);
  return (double) num_wrong / num_samp; 
}

double choose_lambda (int fold_size, double* x, int* y, int num_samp, int num_feat, int batch, int it, double eta)
{
  //printf("Just entered choose_lambda!!\n");
  int i,j,k,l,m, q;
  double lambdas[] = {0, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000};
  int len = sizeof(lambdas)/sizeof(double);
  double weights[num_feat];
  double min_lambda = -1;
  double min_err = DBL_MAX;
  int fold = num_samp / fold_size;
  double test_err;
  double errs[len];
  int inan[len];
  # pragma omp parallel for
  for (i = 0; i < len; i++)
    {
      test_err = 0;
      double *w = malloc(num_feat * sizeof(double));
      double* trainX = malloc((num_samp - fold_size + 1) * num_feat * sizeof(double));
      int* trainY = malloc((num_samp - fold_size + 1) * sizeof(int));
      # pragma omp parallel for
      for (j = 0; j < fold; j++)
	{
	  int start = j*fold_size;
	  int end = start + fold_size;
	  l = 0;
	  # pragma omp parallel for
	  for (k = 0; k < start; k++)
	    {
	      trainY[l++] = y[k];
	    }
	  # pragma omp parallel for
	  for (k = end; k < num_samp; k++)
	    {
	      trainY[l++] = y[k];
	    }
	  l = 0;
	  # pragma omp parallel for
	  for (m = 0; m < num_feat; m++)
	    {
              # pragma omp parallel for
	      for (k = 0; k < start; k++)
		{
		  trainX[l++] = x[k + num_samp*m];
		}
	      # pragma omp parallel for
	      for (k = end; k < num_samp; k++)
		{
		  trainX[l++] = x[k + num_samp*m];
		}
	    }

	  //printf("About to call runSCD on trainX and trainY\n");
	  
	  /* // Print data
	  for (i = 0; i < num_samp - fold_size; i++)
	    // printf("trainY[%d] = %d\n",i,trainY[i]);
	  for (i = 0; i < num_feat; i++)
	    {
	      for (j = 0; j < num_samp - fold_size; j++)
		//printf("%f ", trainX[(num_samp - fold_size)*i + j]);
		// printf("\n");
		}*/

	  runSCD(batch, w, trainX, trainY, lambdas[i], num_samp - fold_size, num_feat, 1, 1, it, eta);
	  //for (q = 0; q < num_feat; q++)
	  //printf("Weight in choose_lambda %d, feature  %d: %lf\n", i, q, w[q]);
	  test_err += test_w(w, &x[start], &y[start], num_feat, fold_size);
	  //printf("test_err at %d = %lf\n", j, test_err);
	}
      errs[i] = (double) (test_err / fold);
      inan[i] = check_nan(w);
      //printf("About to free stuff in choose_lambda\n");
      free(trainY);
      //printf("Freed trainY\n");
      free(trainX);
      //printf("Freed trainX\n");
      free(w);
      // printf("Freed w\n");
    }
  
  for (i = 0; i < len; i++)
    {
      //printf("Test error at %d is %f\n",i, test_err);
      if ((errs[i] <= min_err) && !inan[i])
	{
	  //printf("Updated mins\n");
	  min_err = errs[i];
	  min_lambda = lambdas[i];
	}
    }

  return min_lambda;
}

int main(void)
{

  clock_t start;
  clock_t end;
  int i, j, k, guess, b, batch;
  double dot;
  double times[4];
  char *xFile = "/usr0/home/nrafidi/Data/arc_tr.csv";
  char *yFile = "/usr0/home/nrafidi/Data/arc_labels_tr.csv";
  
  //FILE *fid = fopen(xFile);
  //perror("fopen");
  //fclose(fid);
  int num_samp = 100;
  int num_feat = 10000;

  int fold_size = 10;
  double *x = malloc(num_samp*num_feat*sizeof(double));
  readX(x, xFile);
  int *y = malloc(num_samp*sizeof(int));
  readY(y, yFile);

  //printf("DBL_MIN = %.20lf\n",DBL_MIN);
  // Print data
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
  double weights[num_feat];
  int batch_max = 10000;
  int it = 100000;
  double eta = 0.00001;

  for (b = 0; b <=batch_max; b +=1000)
    {
      batch = b != 0 ? b : 1;
      printf("%d, ", batch);
      //printf("About to call choose_lambda\n");
      start = clock();
      double lamb = choose_lambda (fold_size, x, y, num_samp, num_feat, batch, it, eta);
      end = clock();
      printf("%lf, ", ((double) (end - start))/CLOCKS_PER_SEC);
      printf("%lf, ", lamb);
      //printf("Chosen lambda = %f\n",lamb); //if this prints -1 we're having problems
      for (i = 0; i < 4; i++)
	{
	  int first = i-2 >= 0 ? 1: 0;
	  int second = i%2;
	  start = clock();
	  runSCD(batch, weights, x, y, lamb, num_samp, num_feat, first, second, it, eta);
	  end = clock();
	  times[i] = ((double)(end - start))/CLOCKS_PER_SEC;
	  
	  //Debugging code
	  
	  printf("%lf, ", times[i]);
	  printf("%lf, ", test_w(weights, x, y, num_feat, num_samp));
	  
	  //      for (j = 0; j < num_feat; j++)
	  //	printf("Weight %d: %lf\n", j, weights[j]);
	  
	  /*for (j = 0; j < num_samp; j++)
	    {
	    dot = 0;
	    for (k = 0; k < num_feat; k++)
	    dot += x[num_samp*k + j]*weights[k];
	    guess = dot > 0 ? 1: -1;
	    printf("Guess %d: %d\n", j, guess); 
	    }*/
	}
      printf("\n");
    }
  // free stuff
  free(x);
  free(y);
  return 0;
}
