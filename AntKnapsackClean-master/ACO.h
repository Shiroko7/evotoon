/*
 * ACO.h
 *
 *  Created on: 22-12-2014
 *      Author: nicolas
 */

#ifndef ACO_H_
#define ACO_H_
#include <math.h>

using namespace std;

extern vector < int > BestSolutionFound;
void InitVector();
void CompareSolutions();

int decide (vector <float> CapRem, bool Step);
void CheckAllowedNodes(vector <float> CRem, vector <int> Vis);
void UpdatePheromone(vector <int> Sol, float quality);
float InsertObjectToCandSolution(int obj,vector <float> &CapRem);
bool LocalOptimaCheck(float Fit);
void ACO();
void RhoPheromone();
bool ValidateSolution();
extern vector < float > Profits;
extern vector < float > Aux;
extern vector <vector < float > > Weights;
extern vector <vector < float > > Pheromone;
extern vector < int > BestSolutionFound;
void CleanVector();
#endif /* ACO_H_ */
