#include <stdio.h>
#include <stdlib.h>

#include "nk.h"
#include "ga.h"
#include "random.h"

/*===============================================================*/
/**
   Prints a short header.
*/
/*===============================================================*/

void print_header()
{
  printf("---------------------------------------------------------------\n");
  printf(" Simple genetic algorithm for NK landscapes\n");
  printf(" Author: Martin Pelikan (2008)\n");
  printf("\n See README for further information.\n");
  printf("---------------------------------------------------------------\n");
}

/*===============================================================*/
/**
   Prints a short help.
*/
/*===============================================================*/

void print_help()
{
  printf("Requires file name of NK instances to process as arguments\n");

  exit(-1);
}

/*===============================================================*/
/**
   Main function (processes arguments, calls genetic algorithm).
*/
/*===============================================================*/

int main(int argc, char **argv)
{
  int N;
  int t_max;
  double p_c;
  double p_m;
  int c_o;
  NK_instance nk_instance;

  setSeed(atoi(argv[8])); /*123*/

  /* -------------------------------------- */

  /*print_header();*/

  /* -------------------------------------- */

  if (argc!=9)
    {
      print_help();
    }

  /* -------------------------------------- */
  /* load the NK instance first from the    */
  /* given file name                        */
  
  load_nk(argv[1], &nk_instance);
  
  
  /* -------------------------------------- */
  /* set up and run the genetic algorithm   */

  /*N     = nk_instance.n;
  t_max = nk_instance.n;
  p_c   = 0.6;
  p_m   = 1.0/nk_instance.n;
*/
  char * output_file = argv[2];
  p_c = atof(argv[3]);
  if(p_c==0)
      p_c=0.001;
  p_m = atof(argv[4]);
  if(p_m==0)
      p_m=0.001;
  N = atoi(argv[5]);
  if(N%2 != 0)
      N=N+1;
  t_max = atoi(argv[7]);/*(int)(atof(argv[6])/N);*/

  c_o = atoi(argv[6]);
  if(c_o < 1 || c_o > 2)
    printf("ERROR en seleccion operador de cruzamiento\n");
  /*printf("c_o: %d\n", c_o);*/

  /*
  printf("arg0: %s\n", argv[0]);
  printf("nk_instance: %s\n", argv[1]);
  printf("output: %s\n", argv[2]);
  printf("p_c: %s\n", argv[3]);
  printf("p_m: %s\n", argv[4]);
  printf("N: %s\n", argv[5]);
  printf("arg6: %s\n", argv[6]);
  printf("t_max: %d\n", t_max);
  printf("seed: %s\n", argv[7]);
  printf("c_o: %s\n", argv[8]);

    p_c: %f,
     p_m: %f,
      N: %d,
       6: %s, 
       t_max: %d, 
       c_0: %d"
       , atof(argv[0]), argv[1], output_file, p_c, p_m, N, argv[6], t_max, c_o);
*/
  ga(N,t_max,p_c,p_m,&nk_instance, output_file, c_o);  

  /* -------------------------------------- */
  return 0;
}
