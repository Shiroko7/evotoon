int **allocatePopulation(int N, int n);

void freePopulation(int **p, int N);

void freePopulation(int **p, int N);

void generatePopulation(int **p, int N, int n);

void evaluatePopulation(int **p, 
			double *f, 
			int N, 
			int n, 
			double *best_f,
			NK_instance *nk_instance);

int done(double best_f, double f_opt, int t, int t_max);

void selection(int **s, int **p, double *f, int N, int n);

void onepoint_crossover(int *x, int *y, int n);

void uniform_crossover(int *x, int *y, int n);

void mutation(int *x, int n, double p_m);

void variation(int **o, int **s, int N, int n, double p_c, double p_m);

void replacement(int **p, int **o, int N, int n);

int ga(int N, 
       int t_max, 
       double p_c, 
       double p_m,
       NK_instance *nk_instance,
       char * output_file,
       int c_o);
