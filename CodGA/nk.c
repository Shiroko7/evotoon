#include <stdlib.h>
#include <stdio.h>
   
#include "random.h"
#include "nk.h"

/*===============================================================*/
/* Should we use multiple restarts for the local searcher?       */
/* (better to use than not)                                      */

#define MULTIPLE_LOCAL_SEARCH_RESTARTS

double opt_f; 

/*===============================================================*/
/**
   Generates a random NK instance. Neighbors are chosen randomly,
   function values are uniformly generated from [0,1).
*/
/*===============================================================*/

void generate_nk(int n, int k, NK_instance *nk)
{
  int i,j, two2k1;

  int *generated=(int*) calloc(n,sizeof(int));

  /* --------------------------------------------- */

  if (k<=0)
    {
      printf("ERROR: Expecting k>0, got k=%d. Exiting.\n",k);
      exit(-1);
    }

  nk->n = n;
  nk->k = k;
  two2k1 = (1<<(k+1));

  /* --------------------------------------------- */

  nk->neighbors   = (int**) calloc(n,sizeof(int*));
  nk->friends     = (int**) calloc(n,sizeof(int*));
  nk->num_friends = (int*)  calloc(n,sizeof(int));
  nk->f           = (double**) calloc(n,sizeof(double*));
  nk->is_neighbor = (int**) calloc(n,sizeof(int*));

  nk->num_subproblems  = (int*) calloc(n,sizeof(int));
  nk->subproblems      = (int**) calloc(n,sizeof(int*));

  nk->num_subproblems2 = (int*) calloc(n,sizeof(int));
  nk->subproblems2     = (int**) calloc(n,sizeof(int*));

  /* --------------------------------------------- */

  for (i=0; i<n; i++)
    {
      nk->neighbors[i] = (int*) calloc(k+1,sizeof(int));
      nk->f[i] = (double*) calloc(two2k1,sizeof(double));
      nk->num_friends[i] = 0;
      nk->is_neighbor[i] = (int*) calloc(n,sizeof(int));

      for (j=0; j<n; j++)
	nk->is_neighbor[i][j]=-1;

      nk->num_subproblems[i]=0;
      nk->num_subproblems2[i]=0;
    }

  /* --------------------------------------------- */

  for (i=0; i<n; i++)
    {
      for (j=0; j<n; j++)
	generated[j]=0;
      generated[i]=1;

      nk->num_subproblems[i]++;

      for (j=0; j<k; j++)
	{
	  int neighbor;

	  do {
	    neighbor=intRand(n);
	  } while (generated[neighbor]);

	  generated[neighbor]=1;
	  nk->neighbors[i][j]=neighbor;
	  nk->num_friends[neighbor]++;
	  
	  nk->num_subproblems[neighbor]++;
	}

      nk->neighbors[i][k]=i;

      sort_int_array(nk->neighbors[i],k+1);

      for (j=0; j<k; j++)
	nk->num_subproblems2[nk->neighbors[i][j]]++;

      for (j=0; j<two2k1; j++)
	nk->f[i][j]=drand();
    }

  /* --------------------------------------------- */

  for (i=0; i<n; i++)
    {
      if (nk->num_friends[i]>0)
	{
	  nk->friends[i] = (int*) calloc(nk->num_friends[i],sizeof(int));
	  nk->num_friends[i]=0;
	}
      else
	nk->friends[i]=NULL;

      nk->subproblems[i] = (int*) calloc(nk->num_subproblems[i],sizeof(int));
      nk->num_subproblems[i] = 0;

      if (nk->num_subproblems2[i]>0)
	{
	  nk->subproblems2[i] = (int*) calloc(nk->num_subproblems2[i],sizeof(int));
	  nk->num_subproblems2[i] = 0;
	}
      else
	nk->subproblems2[i]=NULL;
    }

  /* --------------------------------------------- */

  for (i=0; i<n; i++)
    {
      for (j=0; j<=k; j++)
	{
	  int neighbor=nk->neighbors[i][j];
	  
	  if (neighbor!=i)
	    nk->friends[neighbor][nk->num_friends[neighbor]++]=i;
	  
	  nk->is_neighbor[i][neighbor]=j;
	  nk->subproblems[neighbor][nk->num_subproblems[neighbor]++]=i;
	  
	  if (j<k)
	    nk->subproblems2[neighbor][nk->num_subproblems2[neighbor]++]=i;
	}
    }

  /* --------------------------------------------- */

  nk->optimum_defined = 0;
  nk->optimum         = 0;

  /* --------------------------------------------- */
  
  free(generated);
}

/*===============================================================*/
/**
   Frees memory occupied by NK instance.
*/
/*===============================================================*/

void free_nk(NK_instance *nk)
{
  int i;

  for (i=0; i<nk->n; i++)
    {
      free(nk->neighbors[i]);
      if (nk->friends[i])
	free(nk->friends[i]);
      free(nk->is_neighbor[i]);
      if (nk->subproblems[i])
	free(nk->subproblems[i]);
      if (nk->subproblems2[i])
	free(nk->subproblems2[i]);
      free(nk->f[i]);
    }

  free(nk->neighbors);
  free(nk->num_friends);
  free(nk->subproblems);
  free(nk->num_subproblems);
  free(nk->subproblems2);
  free(nk->num_subproblems2);
  free(nk->is_neighbor);
  free(nk->f);
  free(nk->friends);
}

/*===============================================================*/
/**
   Prepares data structures for the BB solver for NK instance
   (called automatically from solve_sk).
*/
/*===============================================================*/

void prepare_for_solve_nk(NK_instance *nk, 
			  int **index, 
			  int *index_size, 
			  double *max_contrib, 
			  double ***best)
{
  int i,j;
  int two2k1 = (1<<(nk->k+1));

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    index_size[i]=0;

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    {
      int max_idx=i;
      for (j=0; j<=nk->k; j++)
	{
	  int neighbor=nk->neighbors[i][j];

	  if (neighbor>max_idx)
	    max_idx=neighbor;
	}

      index_size[max_idx]++;

      best[i]=(double**)calloc(nk->k,sizeof(double*));

      for (j=0; j<nk->k; j++)
	{
	  int ii,jj,num=1<<(j+1);
	  best[i][j]=(double*)calloc(num,sizeof(double));

	  for (ii=0; ii<num; ii++)
	    {
	      int base=(ii<<(nk->k-j));
	      int num2=((int)1)<<(nk->k-j);

	      double max_f=-1E30;

	      for (jj=0; jj<num2; jj++)
		if (nk->f[i][base+jj]>max_f)
		  max_f=nk->f[i][base+jj];
	      
	      best[i][j][ii]=max_f;
	    }
	}
    }

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    {
      if (index_size[i]>0)
	index[i]=(int*) calloc(index_size[i],sizeof(int));
      else
	index[i]=NULL;

      index_size[i]=0;
    }

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    {
      int max_idx=i;
      for (j=0; j<=nk->k; j++)
	{
	  int neighbor=nk->neighbors[i][j];
	  
	  if (neighbor>max_idx)
	    max_idx=neighbor;
	}
      
      index[max_idx][index_size[max_idx]++]=i;
    }

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    {
      double max=-1E10;
      max_contrib[i]=0;

      for (j=0; j<two2k1; j++)
	if (nk->f[i][j]>max)
	  max=nk->f[i][j];

      max_contrib[i]=max;
    }
}

/*===============================================================*/
/**
   Solves NK instance using branch and bound.
*/
/*===============================================================*/

double solve_nk(NK_instance *nk)
{
  int i,j;
  int *x = (int*) calloc(nk->n,sizeof(int));
  double *max_contrib = (double*) calloc(nk->n,sizeof(double));
  double ***best;
  double max_remain=0;

  /* index[i] stores indices of subproblems that contain variable i and others <i */
  int **index;    

  /* index_size[i] stores the number of entries in index[i] */
  int *index_size;

  /* --------------------------------------------- */

  if (nk->optimum_defined)
    return nk->optimum;

  /* --------------------------------------------- */

  index=(int**) calloc(nk->n,sizeof(int*));
  index_size=(int*) calloc(nk->n,sizeof(int));
  best=(double***) calloc(nk->n,sizeof(double**));

  /* --------------------------------------------- */

  prepare_for_solve_nk(nk,index,index_size,max_contrib,best);

  /* --------------------------------------------- */
  
  for (i=0; i<nk->n; i++)
    max_remain+=max_contrib[i];

  opt_f=local_search_nk(nk);
  
/*   printf("      - running BB\n"); */

  /* --------------------------------------------- */

  bb_nk(x,0,0,index,index_size,max_contrib,max_remain,nk,best);

  /* --------------------------------------------- */
  
  free(x);
  free(max_contrib);

  for (i=0; i<nk->n; i++)
    {
      if (index[i]!=NULL)
	free(index[i]);
      for (j=0; j<nk->k; j++)
	free(best[i][j]);
      free(best[i]);
    }

  free(index);
  free(index_size);
  free(best);

  return opt_f;
}

/*===============================================================*/
/**
   Evaluates NK instance (without local search).
*/
/*===============================================================*/

double evaluate_nk(int *x, NK_instance *nk)
{
  int i,j;
  double f=0;

  for (i=0; i<nk->n; i++)
    {
      int idx=0;
  
      for (j=0; j<=nk->k; j++)
	idx=((idx<<1)+x[nk->neighbors[i][j]]);
      
      f+=nk->f[i][idx];
    }

  return f;
}

/*===============================================================*/
/**
   Evaluates NK instance (with deterministic local search based
   on single-bit flips).
*/
/*===============================================================*/

double evaluate_nk_local_search(int *x, NK_instance *nk)
{
  int i;
  int n=nk->n;
  double current_f;

  /* ------------------------------------------------------- */

  double max_improvement;
  current_f=evaluate_nk(x,nk);
  
  /* ------------------------------------------------------- */
  /* can still optimize the following (no need to recompute  */
  /* all), but it's still pretty fast so who cares...        */
  int cont=0;
  do {
    max_improvement=-1;
    int change=-1;
    for (i=0; i<n; i++)
      {
	double improvement = evaluate_flip(x,i,nk);
	
	if (improvement>max_improvement)
	  {
	    max_improvement=improvement;
	    change=i;
	  }
      }
   
    if (max_improvement>0)
      {
	x[change]=1-x[change];
	current_f+=max_improvement;
      }
    /*printf("max_improvement: %f\n", max_improvement);*/
    cont=cont+1;
  }while ((max_improvement>0) && (cont<n*n));
 
  /*printf("Salí!\n"); */
  /* ------------------------------------------------------- */
  
  return current_f;
}

/*===============================================================*/
/**
   Recursive branch and bound for NK (called automatically
   from solve_sk).
*/
/*===============================================================*/

void bb_nk(int *x, 
	   int current, 
	   double current_f, 
	   int **index, 
	   int *index_size, 
	   double *max_contrib, 
	   double max_remain, 
	   NK_instance *nk,
	   double ***best)
{
  int i,j,ii,n=nk->n,k=nk->k,num;
  double d;

  double inc0=0, inc1=0;
  
  for (i=0,num=index_size[current]; i<num; i++)
    {
      int idx=0;
      int which=index[current][i];
      int *neighbors=nk->neighbors[which];
      
      for (j=0; j<k; j++, neighbors++)
	idx=((idx<<1)+x[*neighbors]);
      
      idx<<=1;
      
      inc0+=nk->f[which][idx];
      inc1+=nk->f[which][idx+1];

      max_remain-=max_contrib[which];
    }
  
  if (current==n-1)
    {
      if (inc0>inc1)
	{
	  x[current]=0;
	  current_f+=inc0;
	}
      else
	{
	  x[current]=1;
	  current_f+=inc1;
	}

      current_f=evaluate_nk(x,nk);

      if (current_f>opt_f)
	{
	  opt_f=current_f;
/* 	  printf("      - BB found better optimum: %lf\n",opt_f); */
	}

      return;
    }
  
  /* ------------------------------------------------------- */

  double max_remain0=0;
  for (i=0,num=nk->num_subproblems2[current]; i<num; i++)
    {
      int which=nk->subproblems2[current][i];

      int idx=0;
      int neighbor;

      for (ii=0; ((neighbor=nk->neighbors[which][ii])<current); ii++)
	idx=((idx<<1)+x[neighbor]);

      max_remain-=max_contrib[which];
      max_remain0+=(max_contrib[which]=best[which][ii][idx<<1]);
    }

  max_remain+=max_remain0;
  
  if (current_f+inc0+max_remain>opt_f)
    {
      x[current]=0;
      bb_nk(x,current+1,current_f+inc0,index,index_size,max_contrib,max_remain,nk,best);
    }

  max_remain-=max_remain0;

  /* ------------------------------------------------------- */

  double max_remain1=0;
  for (i=0,num=nk->num_subproblems2[current]; i<num; i++)
    {
      int which=nk->subproblems2[current][i];
      
      int idx=0;
      int neighbor;
      
      for (ii=0; ((neighbor=nk->neighbors[which][ii])<current); ii++)
	idx=((idx<<1)+x[neighbor]);
      
      max_remain1+=(max_contrib[which]=best[which][ii][(idx<<1)+1]);
    }

  max_remain+=max_remain1;

  if (current_f+inc1+max_remain>opt_f)
    {
      x[current]=1;
      bb_nk(x,current+1,current_f+inc1,index,index_size,max_contrib,max_remain,nk,best);
    }

  /* ------------------------------------------------------- */

  for (i=0,num=nk->num_subproblems[current]; i<num; i++)
    {
      int which=nk->subproblems[current][i];
      
      int idx=0;
      int neighbor;

      for (ii=0; ((neighbor=nk->neighbors[which][ii])<current); ii++)
	idx=((idx<<1)+x[neighbor]);
      
      idx<<=(k-ii+1);
      
      int max=(1<<(k-ii+1));
      double max_f=-10E10;
      
      for (ii=0; ii<max; ii++)
	if ((d=nk->f[which][idx+ii])>max_f)
	  max_f=d;
      
      max_contrib[which]=max_f;
    }
}

/*===============================================================*/
/**
   Sorts integer array (using insertion sort). Speed is secondary
   because this is used only to sort the neighbor indices when 
   generating a new instance.
*/
/*===============================================================*/

void sort_int_array(int *x, int n)
{
  int i;

  for (i=1; i<n; i++)
    {
      int tmp=x[i];
      int j=i;

      while ((j>0)&&(x[j-1]>tmp))
	{
	  x[j]=x[j-1];
	  j--;
	}
      
      x[j]=tmp;
    }
}

/*===============================================================*/
/**
   Performs deterministic local search based on sigle-bit flips,
   also called deterministic hill climber, on one or more random
   instances.
*/
/*===============================================================*/

double local_search_nk(NK_instance *nk)
{
  int i;
  int n=nk->n;
  double current_f;
  int    *x = (int*) calloc(n,sizeof(int));
  double best_f=-10E30;
  int run;
  int num_runs=n*n;

  /* ------------------------------------------------------- */

#ifdef MULTIPLE_LOCAL_SEARCH_RESTARTS
  for (run=0; run<num_runs; run++)
#endif
    {
      for (i=0; i<n; i++)
	x[i]=(drand()<0.5)? 0:1;
      
      double max_improvement;
      current_f=evaluate_nk(x,nk);

      /* ------------------------------------------------------- */
      /* can still optimize the following (no need to recompute  */
      /* all), but it's still pretty fast so who cares...        */
            
      do {
	max_improvement=-1;
	int change=-1;
	for (i=0; i<n; i++)
	  {
	    double improvement = evaluate_flip(x,i,nk);
	    
	    if (improvement>max_improvement)
	      {
		max_improvement=improvement;
		change=i;
	      }
	  }
	
	if (max_improvement>0)
	  {
	    x[change]=1-x[change];
	    current_f+=max_improvement;
	  }
      } while (max_improvement>0);

      /* ------------------------------------------------------- */
      
      if (current_f>best_f)
	best_f=current_f;
    }

/*   printf("      - local search with %u restarts: %lf\n",num_runs,best_f); */


  /* ------------------------------------------------------- */
  
  free(x);

  /* ------------------------------------------------------- */

  return best_f;
}

/*===============================================================*/
/**
   Saves an NK problem instance.
*/
/*===============================================================*/

void save_nk(char *fname, NK_instance *nk)
{
  int i,j;
  FILE *f=fopen(fname,"w");

  /* ------------------------------------------------------- */

  fprintf(f,"%u %u\n",nk->n,nk->k);

  for (i=0; i<nk->n; i++)
    for (j=0; j<=nk->k; j++)
      fprintf(f,"%u\n",nk->neighbors[i][j]);

  for (i=0; i<nk->n; i++)
    for (j=0; j<(1<<(nk->k+1)); j++)
      fprintf(f,"%lf\n",nk->f[i][j]);

  if (nk->optimum_defined)
    fprintf(f,"Optimum: %.9lf",nk->optimum);
  else
    fprintf(f,"Optimum: ?");

  /* ------------------------------------------------------- */

  fclose(f);
}

/*===============================================================*/
/**
   Loads an NK instance.
*/
/*===============================================================*/

void load_nk(char *fname, NK_instance *nk)
{
  int i,j;
  int n,k;
  FILE *f=fopen(fname,"r");

  fscanf(f,"%u %u\n",&nk->n,&nk->k);

  n=nk->n;
  k=nk->k;

  nk->neighbors   = (int**) calloc(n,sizeof(int*));
  nk->friends     = (int**) calloc(n,sizeof(int*));
  nk->num_friends = (int*)  calloc(n,sizeof(int));
  nk->f           = (double**) calloc(n,sizeof(double*));
  nk->is_neighbor = (int**) calloc(n,sizeof(int*));

  nk->num_subproblems = (int*) calloc(n,sizeof(int));
  nk->subproblems = (int**) calloc(n,sizeof(int*));

  nk->num_subproblems2 = (int*) calloc(n,sizeof(int));
  nk->subproblems2 = (int**) calloc(n,sizeof(int*));

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    {
      nk->neighbors[i] = (int*) calloc(nk->k+1,sizeof(int));

      for (j=0; j<=nk->k; j++)
	{
	  int neighbor;
	  fscanf(f,"%u\n",&neighbor);
	  nk->neighbors[i][j]=neighbor;
	  nk->num_subproblems[neighbor]++;
	  if (j<nk->k)
	    nk->num_subproblems2[nk->neighbors[i][j]]++;
	  nk->num_friends[neighbor]++;
	}

      nk->is_neighbor[i] = (int*) calloc(nk->n,sizeof(int));
      for (j=0; j<nk->n; j++)
	nk->is_neighbor[i][j]=0;
    }

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    {
      int max=(1<<(nk->k+1));

      nk->f[i] = (double*) calloc(max,sizeof(double));

      for (j=0; j<max; j++)
	fscanf(f,"%lf\n",&nk->f[i][j]);
    }

  /* --------------------------------------------- */

  for (i=0; i<nk->n; i++)
    {
      if (nk->num_friends[i]>0)
	{
	  nk->friends[i] = (int*) calloc(nk->num_friends[i],sizeof(int));
	  nk->num_friends[i]=0;
	}
      else
	nk->friends[i]=NULL;
      
      nk->subproblems[i] = (int*) calloc(nk->num_subproblems[i],sizeof(int));
      nk->num_subproblems[i] = 0;
      
      if (nk->num_subproblems2[i]>0)
	{
	  nk->subproblems2[i] = (int*) calloc(nk->num_subproblems2[i],sizeof(int));
	  nk->num_subproblems2[i] = 0;
	}
      else
	nk->subproblems2[i]=NULL;
    }

  /* --------------------------------------------- */

  for (i=0; i<n; i++)
    {
      for (j=0; j<=k; j++)
	{
	  int neighbor=nk->neighbors[i][j];
	  
	  if (neighbor!=i)
	    nk->friends[neighbor][nk->num_friends[neighbor]++]=i;
	  
	  nk->is_neighbor[i][neighbor]=j;
	  nk->subproblems[neighbor][nk->num_subproblems[neighbor]++]=i;
	  
	  if (j<k)
	    nk->subproblems2[neighbor][nk->num_subproblems2[neighbor]++]=i;
	}
    }

  /* --------------------------------------------- */

  if (fscanf(f,"Optimum: %lf\n",&nk->optimum)!=1)
    {
      nk->optimum         = 0;
      nk->optimum_defined = 0;
    }
  else
    nk->optimum_defined = 1;
  
  /* --------------------------------------------- */

  fclose(f);
}

/*===============================================================*/
/**
   Evaluates a flip (for deterministic local searcher).
*/
/*===============================================================*/

double evaluate_flip(int *x, int i, NK_instance *nk)
{
  int ii,j;
  double f=0;

  for (ii=0; ii<nk->num_subproblems[i]; ii++)
    {
      int which=nk->subproblems[i][ii];
      int idx=0;
      int pos_i=-1000;

      for (j=0; j<=nk->k; j++)
	{
	  int pos=nk->neighbors[which][j];
	 
	  idx=((idx<<1)+x[pos]);

	  if (pos==i)
	    pos_i=j;
	}

      f-=nk->f[which][idx];

      idx^=(int) (((int)1)<<(nk->k-pos_i));

      f+=nk->f[which][idx];
    }
  if (f<0.000)
	return 0;  
  return f;
}

/*===============================================================*/
/**
   Is the given fitness the same as the global optimum?
*/
/*===============================================================*/

int is_optimal_nk(NK_instance *nk, double best_f)
{
  if ((nk->optimum_defined)&&(best_f>=nk->optimum-1E-9))
    return 1;
  else
    return 0;
}
