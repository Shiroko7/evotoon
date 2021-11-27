typedef struct {
  /*! Number of bits, n. */
  int n;

  /*! Neighborhood size, k. */
  int k;

  /*! List of neighbors for each bit. */
  int **neighbors;

  /*! List of connected (interacting) bits with each bit. */
  int **friends;

  /*! Number of connected bits with each bit. */
  int *num_friends;

  /*! Coincidence matrix w.r.t. neighborhoods. */
  int **is_neighbor;

  /*! Look-up tables for the subproblems. */
  double **f;

  /*! Value of the global optimum. */
  double optimum;

  /*! Is the global optimum defined? */
  int    optimum_defined;

  int *num_subproblems;
  int **subproblems;
  int *num_subproblems2;
  int **subproblems2;  
} NK_instance;

void generate_nk(int n, int k, NK_instance *nk);

double solve_nk(NK_instance *nk);

void bb_nk(int *x, 
	   int current, 
	   double current_f, 
	   int **index, 
	   int *index_size, 
	   double *max_contrib, 
	   double max_remain, 
	   NK_instance *nk,
	   double ***best);

void sort_int_array(int *x, int n);

double local_search_nk(NK_instance *nk);

void save_nk(char *fname, NK_instance *nk);

void load_nk(char *fname, NK_instance *nk);

void free_nk(NK_instance *nk);

double evaluate_flip(int *x, int i, NK_instance *nk);

void prepare_for_solve_nk(NK_instance *nk, 
			  int **index, 
			  int *index_size, 
			  double *max_contrib, 
			  double ***best);

int is_optimal_nk(NK_instance *nk, double best_f);

double evaluate_nk(int *x, NK_instance *nk);

double evaluate_nk_local_search(int *x, NK_instance *nk);
