/*
 * Ant.cpp
 *
 *  Created on: 22-12-2014
 *      Author: nicolas
 */

#include "Ant.h"
#include "Initialize.h"
using namespace std;


Ant::Ant():id(0){}

Ant::~Ant(){}

int Ant::getId(){
	return id;
}

void Ant::setId(int id_ext){
	id = id_ext;
}

void Ant::insertInMem(int object){
	Memoria.push_back(object);
	evaluacion += Profits[object];
}

int Ant::getSomeObject(int index){
	return Memoria[index];
}

int Ant::getSizeMem(){
	return Memoria.size();
}

vector <int> Ant::getMem(){
	return Memoria;
}

void Ant::initAnt(){
	for(unsigned int i = 0; i < N; i++){
		Memoria.push_back(0);
	}
	evaluacion = 0;
}

void Ant::setEvaluacion(float eval){
	evaluacion = eval;
}

void Ant::subEval(float penal){
	evaluacion -= penal;
}

float Ant::getEval(){
	return evaluacion;
}

void Ant::setMemoria(vector<int> aux){
	Memoria.clear();
	Memoria = aux;
}
