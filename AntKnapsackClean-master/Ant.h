/*
 * Ant.h
 *
 *  Created on: 22-12-2014
 *      Author: nicolas
 */

#ifndef ANT_H_
#define ANT_H_

#include <vector>

using namespace std;

class Ant {

	private:
		int id;
		vector <int> Memoria;
		float evaluacion;

	public:
		int getId();
		void initAnt();
		void setId(int id_ext);
		void insertInMem(int object); //debería sumar a eval, cuando agrega
		int getSizeMem();
		int getSomeObject(int index);
		void subEval(float penal);
		void setMemoria(vector<int> aux);
		void setEvaluacion(float eval);
		vector <int> getMem();
		float getEval();
		Ant();
		virtual ~Ant();
};

#endif /* ANT_H_ */
