#include <math.h>
#include "scd.h"

//Updates a weight w given a set of features x, labels y and a lambda for regularization
double update_w(double w, double x[], int y[], double lamb, int num_samples, int p_samp, double eta)
{

  double sum = 0;
  int s;
  # pragma omp parallel for if(p_samp)
  for(s = 0; s < num_samples; s++)
    {
      double val = exp(y[s]*w*x[s]);
      sum += (y[s]*x[s]*val)/(1 + val);
    }

  return w + eta*sum + eta*lamb*w;
}


void runSCD(int batch, double weights[], double* x, int y[], int lamb, int num_samples, int num_feats, int p_batch, int p_w_samp, int it, double eta)
{
  int i, j;
  for (i = 0; i < num_feats; i++)
    {
      weights[i] = 0;
    }
  
  for (i = 0; i < it; i++)
    {
      # pragma omp parallel for if(p_batch)
      for (j = 0; j < batch; j++)
	{
	  int r = rand()%num_feats;	  
	  weights[r] = update_w(weights[r], &x[r*num_samples], y, lamb, num_samples, p_w_samp, eta);
	}
    }
  
}


/* MAIN for testing:
//For each combination of lambda and step size, runSCD for a fixed number of iterations
//Choose the best lambda-eta pair, and then runSCD for 10x as many iterations
//Test resulting weights on test data
int main(void)
{
  double test = exp(3);
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
  runSCD(1, w, x, y, lambdas[2], 6, 2, 1, 1); 
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
	       
