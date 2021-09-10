#include "Initialize.h"
#include "Ant.h"
#include "ACO.h"

char* InstanciaFile;
float Opt;
unsigned int N,M,Seed,TotalAnts;
vector < float > Profits;
vector < float > Aux;
vector <vector < float > > Weights;
vector <vector < float > > Pheromone;
float alpha,beta,ph_max,ph_min,rho;
vector < int > BestSolutionFound;
double BestFitnessFound;
vector < float > b_i;
int TotalEvaluations;
int eval;


using namespace std;

int main(int argc, char** argv) {
	Capture_Params(argc,argv);
	Read_Instance();
	srand48(Seed);
	DefineInitial();
	//PrintInstancia();
	ACO();
	cout << 100*(Opt - BestFitnessFound)/Opt << endl;

	return 0;
}
