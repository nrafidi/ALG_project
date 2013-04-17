#include <math.h>
#include "scd.h"

double compare_and_swap(double* reg, double oldval, double newval)
{
  //printf("In CAS: oldval is %lf\n",oldval);
  
  double old_reg_val = *reg;
  if (old_reg_val == oldval)
    *reg = newval;
  return old_reg_val;
}


//Updates a weight w given a set of features x, labels y and a lambda for regularization
double update_w(double w, double x[], int y[], double lamb, int num_samples, int p_samp, double eta)
{

  double new_w;
  double sum = 0;
  int s;
  # pragma omp parallel for if(p_samp)
  for(s = 0; s < num_samples; s++)
    {
      double val = exp(y[s]*w*x[s]);
      double oldsum = sum;
      // printf("About to compare_and_swap\n");
      // printf("oldsum is %lf\n",oldsum);
      if (isnan(sum))
	{
	  // printf("NaN!\n");
	  break;	  
	}
      double newsum = sum + (y[s]*x[s]*val)/(1 + val);
      //      printf("val = %lf\ny[s] = %d\n x[s] = %lf\n s = %d\n",val, y[s], x[s], s);
      while (compare_and_swap(&sum, oldsum, newsum) != oldsum) {}
      //printf("sum is %lf\n", sum);
      //  printf("Swap over\n");
    }
  new_w = w + eta*sum + eta*lamb*w;
  //  printf("Weight update = %lf\nsum = %lf\nsamples = %d\n", new_w, sum, num_samples);
  return new_w;
}


void runSCD(int batch, double weights[], double* x, int y[], int lamb, int num_samples, int num_feats, int s_batch, int p_batch, int p_w_samp, int it, double eta)
{
  int i, j, k;
  for (i = 0; i < num_feats; i++)
    {
      weights[i] = 0;
    }
  double* batchX = malloc(s_batch * num_feats * sizeof(double));
  int* batchY = malloc(s_batch * sizeof(double));
  for (i = 0; i < it; i++)
    {
      // Picking samples in s_batch
      # pragma omp parallel for
      for (j = 0; j < s_batch; j++)
	{
	  int r = rand() % num_samples;
	  # pragma omp parallel for
	  for (k = 0; k < num_feats; k++)
	    {
	      batchX[k*s_batch + j] = x[k*num_samples + r];
	    }
	  batchY[j] = y[r];
	}
      
      //if (lamb >= 10000)
	//printf("runSCD iteration %d\n", i);
      # pragma omp parallel for if(p_batch)
      for (j = 0; j < batch; j++)
	{
	  int r = rand()%num_feats;
	  //printf("r = %d\n", r);
	  // printf("About to update_w\n");
	  weights[r] = update_w(weights[r], &batchX[r*s_batch], batchY, lamb, s_batch, p_w_samp, eta);
	}
    }  
  //if (lamb >=10000)
    // printf("Leaving runSCD\n");
  free(batchX);
  free(batchY);
}

/*
// MAIN for testing:
//For each combination of lambda and step size, runSCD for a fixed number of iterations
//Choose the best lambda-eta pair, and then runSCD for 10x as many iterations
//Test resulting weights on test data
int main(void)
{
  double lambdas[] = {.001, .01, .1, 1, 10, 100, 1000};

  // Test Code - feature array
  double w[2];
  int num_samples = 6;
  int num_feats = 2;
  double* x = malloc(num_samples*num_feats*sizeof(double));
  double vals[] = {-3, -2, -1, 1, 2, 3};
  int i,j;
  for (i = 0; i < num_feats; i++)
    {
      for (j = 0; j < num_samples; j++)
	{
	  x[num_samples*i + j] = i%2 == 0 ? vals[j] : -vals[j];
	}
    }

  //Code for printing x
  printf("About to print x: feats x samples\n");
  for (i = 0; i < num_feats; i++)
    {
      for (j = 0; j < num_samples; j++)
	{
	  printf("%f ",x[num_samples*i + j]);
	}
      printf("\n");
      }

  
  // Test code - labels and testing
  int y[] = {-1, -1, -1, 1, 1, 1};

  clock_t start = clock();
  runSCD(1, w, x, y, lambdas[2], 6, 2, 1, 1, 10, .01); 
  clock_t end = clock();
  double time_elapsed = (end - start) / CLOCKS_PER_SEC;
  printf("Time elapsed for runSCD: %f\n", time_elapsed);

  for (i = 0; i < num_samples; i++)
    {
      double dot = 0;
      for (j = 0; j < num_feats; j++)
	{	  
	  dot += w[j]*x[num_samples*j + i];
	}
      printf("Truth: %d\n", y[i]);
      int guess = dot > 0 ? 1 : -1;
      printf("Guess: %d\n", guess);
    }

  //Test code - print weights
  printf("Result: %f\n", w[0]);
  printf("Result: %f\n", w[1]);

  free(x);
  return 0;
}
*/
	       
