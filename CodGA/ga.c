#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "nk.h"
#include "random.h"

/*===============================================================*/
/* HOW TO USE NK-RELATED FUNCTIONS

1. Before running the GA, load the NK instance using load_nk
2. Evaluate solutions in either of the following 2 ways
   - evaluate_nk              (without any hill climbing)
   - evaluate_nk_local_search (with deterministic hill climber)
3. Check any solution for optimality with is_optimal_nk
4. After finishing up with the GA, free up the memory using 
   free_nk

For example use of these functions, look at the functions
in main.c and ga.c
                                                                 */
/*===============================================================*/


/*===============================================================*/
/**
   Allocates memory for a population of strings (int arrays).
*/
/*===============================================================*/

int **allocatePopulation(int N, int n)
{
  int i;
  int **p = (int**) malloc(N*sizeof(int*));
  
  for (i=0; i<N; i++)
    p[i] = (int*) malloc(n*sizeof(int));

  return p;
}

/*===============================================================*/
/**
   Frees the population.
*/
/*===============================================================*/

void freePopulation(int **p, int N)
{
  int i;

  for (i=0; i<N; i++)
    free(p[i]);
  free(p);
}

/*===============================================================*/
/**
   Generates the population.
*/
/*===============================================================*/

void generatePopulation(int **p, int N, int n)
{
  int i, j;

  for (i=0; i<N; i++)
    for (j=0; j<n; j++)
      p[i][j] = (drand()<0.5)? 0:1;
}

/*===============================================================*/
/**
   Returns onemax fitness function value (used for debugging).
*/
/*===============================================================*/

double onemax(int *x, int n)
{
  int i;
  double f=0;

  for (i=0; i<n; i++)
    f+=x[i];

  return f;
}

/*===============================================================*/
/**
   Evaluates the population.
*/
/*===============================================================*/

void evaluatePopulation(int **p, 
			double *f, 
			int N, 
			int n, 
			double *best_f,
			NK_instance *nk_instance)
{
  int i;

  *best_f = f[0] = evaluate_nk_local_search(p[0],nk_instance);
  /*best_f = f[0] = evaluate_nk(p[0],nk_instance);*/

  for (i=1; i<N; i++)
    {
      /*f[i] = evaluate_nk_local_search(p[i],nk_instance);*/
      f[i] = evaluate_nk(p[i],nk_instance);
      if (f[i]>*best_f)
	*best_f=f[i];
    };
}

/*===============================================================*/
/**
   Is the GA done?
*/
/*===============================================================*/

int done(double best_f, 
	 int t, 
	 int t_max, 
	 NK_instance *nk_instance)
{
  if (is_optimal_nk(nk_instance,best_f))
    return 1;

  if (t>=t_max)
    return -1;

  return 0;
}

/*===============================================================*/
/**
   Executes selection (binary tournament selection).
*/
/*===============================================================*/

void selection(int **s, int **p, double *f, int N, int n)
{
  int k;

  for (k=0; k<N; k++)
    {
      int i=(int)((double)drand()*N);
      int j=(int)((double)drand()*(N-1));
      if (j>=i)
	j++;

      if (f[i]>f[j])
	memcpy(s[k],p[i],n*sizeof(int));
      else
	memcpy(s[k],p[j],n*sizeof(int));
    }
}

/*===============================================================*/
/**
   Performs onepoint crossover on two individuals.
*/
/*===============================================================*/

void onepoint_crossover(int *x, int *y, int n)
{
  /*printf("onepoint_crossover\n");*/
  int i;
  int k = (int) ((double)drand()*(n-1));

  for (i=k+1; i<n; i++)
    {
      int tmp=x[i];
      x[i]=y[i];
      y[i]=tmp;
    }
}

/*===============================================================*/
/**
   Performs uniform crossover on two individuals.
*/
/*===============================================================*/

void uniform_crossover(int *x, int *y, int n)
{
  /*printf("uniform_crossover\n");*/
  int i;

  for (i=0; i<n; i++)
    if (drand()<0.5)
      {
	int tmp=x[i];
	x[i]=y[i];
	y[i]=tmp;
      }
}

/*===============================================================*/
/**
   Performs bit-flip mutation on an individual.
*/
/*===============================================================*/

void mutation(int *x, int n, double p_m)
{
  int i;

  for (i=0; i<n; i++)
    if (drand()<p_m)
      x[i]=1-x[i];
}

/*===============================================================*/
/**
   Performs variation operators on the population.
*/
/*===============================================================*/

void variation(int **o, int **s, int N, int n, double p_c, double p_m, int c_o)
{
  int i;
  /*printf("variation\n");*/

  for (i=0; i<N; i+=2)
    {
      memcpy(o[i],s[i],n*sizeof(int));
      memcpy(o[i+1],s[i+1],n*sizeof(int));

      if (drand()<p_c)
      {
	if(c_o == 1)
	  uniform_crossover(o[i],o[i+1],n);
	else if(c_o == 2)
	  onepoint_crossover(o[i],o[i+1],n);
	else
	 printf("ERROR en seleccion operador de cruzamiento\n");
      }
      
      mutation(o[i],n,p_m);
      mutation(o[i+1],n,p_m);	
    }
}

/*===============================================================*/
/**
   Performs full replacemnt.
*/
/*===============================================================*/

void replacement(int **p, int **o, int N, int n)
{
  int i;

  for (i=0; i<N; i++)
    memcpy(p[i],o[i],n*sizeof(int));
}

/*===============================================================*/
/**
   Executes a GA.
*/
/*===============================================================*/

void print_status(int t, double best_f)
{
  printf("   t = %3u   best_f = %lf\n", t, best_f);
}

/*===============================================================*/
/**
   Executes a GA.
*/
/*===============================================================*/

int ga(int N, 
       int t_max, 
       double p_c, 
       double p_m,
       NK_instance *nk_instance,
       char * output_file,
       int c_o)
{
  int   n = nk_instance->n;
  int **p = allocatePopulation(N,n);
  int **s = allocatePopulation(N,n);
  int **o = allocatePopulation(N,n);

  double *f = (double*) calloc(N,sizeof(double));

  double best_f;
  int    t;
  int    num_evals;

  generatePopulation(p,N,n);
  evaluatePopulation(p,f,N,n,&best_f,nk_instance);
  num_evals=N;

  t=0;
  /*print_status(t,best_f);*/
  while (!done(best_f,t,t_max,nk_instance))
    {  
      /*printf("evolucionanodo?\n");*/
      selection(s,p,f,N,n);
      
      variation(o,s,N,n,p_c,p_m, c_o);

      replacement(p,o,N,n);

      evaluatePopulation(p,f,N,n,&best_f,nk_instance);

      num_evals+=N;

      t++;
      /*print_status(t,best_f);*/
    }

  freePopulation(p,N);
  freePopulation(s,N);
  freePopulation(o,N);
  
  FILE *output;
  output = fopen(output_file,"a");
  int tuning=0;
  int testing=1-tuning;
  if(tuning)
    fprintf(output, "%d\n", num_evals);

  if (is_optimal_nk(nk_instance,best_f))
    {
      /*printf("Success: n=%d, N=%d, num_gens=%d, num_evals=%d\n",n,N,t,num_evals);*/
      if(testing)
	fprintf(output, "1 %d %d %d %d\n",n,N,t,num_evals);
      return 1;
    }
  else
    if (nk_instance->optimum_defined)
      {
	/*printf("Failure: n=%d, N=%d, num_gens=%d, best_fitness=%f\n",n,N,t,(double)best_f);*/
	if(testing)
	  fprintf(output, "0 %d %d %d %d\n",n,N,t,num_evals);
	return 0;
      }
    else
      {
	/*printf("Final result: n=%d, N=%d, num_gens=%d, best_fitness=%f\n",n,N,t,(double)best_f);*/
	if(testing)
	  fprintf(output, "-1 %d %d %d %d\n",n,N,t,num_evals);
	return 0;
      }
   
   fclose(output);
}
