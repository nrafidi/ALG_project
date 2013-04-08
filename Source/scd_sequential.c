#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double ETA = .00001;
double  IT = 1000;

//Updates a weight w given a set of features x, labels y and a lambda for regularization
//We could update w directly if we passed a pointer instead
double update_w(double w, double x[], int y[], double lamb, int num_samples)
{

  double sum = 0;
  int s;
  for(s = 0; s < num_samples; s++)
    {
      double val = exp(y[s]*w*x[s]);
      sum += (y[s]*x[s]*val)/(1 + val);
    }

  return w + ETA*sum + ETA*lamb*w;
}


void runSCD(int batch, double weights[], double* x, int y[], int lamb, int num_samples, int num_feats)
{
  int i, j;
  for (i = 0; i < num_feats; i++)
    {
      weights[i] = 0;
    }

  for (i = 0; i < IT; i++)
    {
      for (j = 0; j < batch; j++)
	{
	  int r = rand()%num_feats;	  
	  weights[r] = update_w(weights[r], &x[r*num_samples], y, lamb, num_samples);
	}
    }
  
}

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
  /* printf("About to print x: feats x samples\n");
  for (i = 0; i < num_feats; i++)
    {
      for (j = 0; j < num_samples; j++)
	{
	  printf("%f ",x[num_samples*i + j]);
	}
      printf("\n");
      }*/

  
  // Test code - labels and testing
  int y[] = {-1, -1, -1, 1, 1, 1};
  runSCD(1, w, x, y, lambdas[2], 6, 2); 

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


	       
