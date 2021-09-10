#include "Initialize.h"
#include "Ant.h"
#include "ACO.h"
using namespace std;
vector<int> auxiliar;
vector<int> allowed;
vector<int> visited;
vector<float> CapRem;
vector<int> included;
bool factible;

void ACO() {
    bool print = false;
    vector<Ant> Hormigas;
    float mejor, lowerq;
    float QualityNew = 0;

    unsigned int i, depositant, iteration,selected;
    int curr_eval = 0;
    bool stop = true;
    bool end = true;
    InitVector();

    selected = 0;

    depositant = lowerq = mejor = 0;
    CapRem = b_i;
    iteration = 0;
    Ant ant;
    ant.initAnt();

    //vector de hormigas
    for (i = 0; i < TotalAnts; i++) {
        ant.setId(i);
        Hormigas.push_back(ant);
    }

    //Init values
    BestFitnessFound = 0;

    while (curr_eval < TotalEvaluations && end) {
        for (i = 0; (i < TotalAnts) && end; i++) {
            stop = true;

            selected = int_rand(0, (N - 1));
            included.push_back(selected);
            QualityNew += InsertObjectToCandSolution(selected, CapRem);
            CheckAllowedNodes(CapRem,visited);

            //CONSTRUCCION DE LAS SOLUCIONES
            while (stop) {
                selected = decide(CapRem, false);
                QualityNew += InsertObjectToCandSolution(selected, CapRem);
                included.push_back(selected);
                CheckAllowedNodes(CapRem,visited);
                if (allowed.size() == 0)stop = false;
            }

            if (print) {
                cout << "curr_eval ->" << curr_eval << " - Ant#" << i << " - Q:" << QualityNew << endl;
            }

            if (i == 0) {
                mejor = QualityNew;
                depositant = i;
                //lowerq = QualityNew;
            } else {
                if (mejor < QualityNew) {
                    mejor = QualityNew;
                    depositant = i;
                }
            }

            if (QualityNew != Opt) {
                    if (QualityNew > BestFitnessFound) {
                        BestSolutionFound = auxiliar;
                        BestFitnessFound = QualityNew;
                    }
            } else {
                BestSolutionFound = auxiliar;
                end = false;
                BestFitnessFound = QualityNew;
            }

            Hormigas[i].setEvaluacion(QualityNew);
            Hormigas[i].setMemoria(auxiliar);


            QualityNew = 0;
            CleanVector(); //auxiliar y visited no lleva ningun objeto
            included.clear();
            CapRem = b_i; // empezamos con toda la capacidad en las M dimensiones
            stop = true;
            curr_eval++;
        }

        RhoPheromone();
        UpdatePheromone(Hormigas[depositant].getMem(), mejor);
        iteration++;

    }
}


float InsertObjectToCandSolution(int obj, vector<float> &CapRem) {
    unsigned int j;
    visited[obj] = 0;
    auxiliar[obj] = 1;

    //UPDATE OF CAP REM
    for (j = 0; j < Weights.size(); j++) {
        CapRem[j] -= Weights[j][obj];
    }

    return Profits[obj];

}

void RhoPheromone() {
    unsigned int i, j;
    for (i = 0; i < (N - 1); i++) {
        for (j = (i + 1); j < N; j++) {
            if (i != j) {
                Pheromone[i][j] = Pheromone[i][j] * (1 - rho);
                Pheromone[j][i] = Pheromone[i][j];
                if (Pheromone[i][j] < ph_min) {
                    Pheromone[i][j] = ph_min;
                    Pheromone[j][i] = ph_min;
                }
            }
        }
    }
}

int decide(vector<float> CapRem, bool Step) {
    unsigned int i, j;
    float rnd;
    bool print = false;
    vector<float> Matriz;
    float num1, num2, mult;
    float div = 0;
    double suma_total = 0;
    vector<float> remanenteobj;
    float rempromedio = 0;

    for (i = 0; i < allowed.size(); i++) {
        //Por cada objeto permitido, suma en DIV de peso/cap_remanente en todas las dimensiones
        for (j = 0; j < CapRem.size(); j++) {
            div = (float) (Weights[j][allowed[i]] / CapRem[j]);
            rempromedio += div;
        }

        mult = (float) (Profits[allowed[i]] / rempromedio);
        remanenteobj.push_back(mult);
        rempromedio = 0;
    }

    //Roulette
    suma_total = 0;
    float suma_obj_allowed = 0;
    for (i = 0; i < allowed.size(); i++) {
        for (j = 0; j < visited.size(); j++) {
            if (visited[j] == 0)suma_obj_allowed += Pheromone[j][allowed[i]];
        }

        num1 = powf(suma_obj_allowed, alpha);
        num2 = powf(remanenteobj[i], beta);
        mult = num1 * num2;
        Matriz.push_back(mult);
        suma_total += mult;
        suma_obj_allowed = 0;
    }
    rnd = float_rand(0.00, 1.00);
    bool flag = true;
    int index_mellevo = 0;
    //a_ij de cada objeto
    float acumulado = 0;
    for (i = 0; (i < Matriz.size()) && (flag); i++) {
        mult = Matriz[i] / suma_total;
        acumulado += mult;
        if (rnd < acumulado) {
            index_mellevo = i;
            flag = false;
        }
        Matriz[i] = acumulado;
    }
    if (print) {
        cout << "Se queda con -->" << allowed[index_mellevo];
        getchar();
    }
    return allowed[index_mellevo];
}


void UpdatePheromone(vector<int> Sol, float quality) {
    unsigned int i, j;
    float howmuch = 0;
    float upperph = 0;
    howmuch = (float) (1 / (1 + BestFitnessFound - quality));
    upperph = ph_max;

    for (i = 0; i < (N - 1); i++) {
        if (Sol[i] == 1) {
            for (j = (i + 1); j < N; j++) {
                if ((Sol[j] == 1) && (i != j)) {
                    Pheromone[i][j] += howmuch;
                    Pheromone[j][i] = Pheromone[i][j];
                    if (Pheromone[i][j] > upperph) {
                        Pheromone[i][j] = upperph;
                        Pheromone[j][i] = upperph;
                    }

                }
            }
        }
    }
}


void CleanVector() {
    unsigned int i;
    for (i = 0; i < N; i++) {
        auxiliar[i] = 0;
        visited[i] = 1;
    }
}

void InitVector() {
    unsigned int i;
    for (i = 0; i < N; i++) {
        auxiliar.push_back(0);
        visited.push_back(1);
    }
}

bool ValidateSolution() {

//		float profit = 0;
//		float howmuch = 0;
    float auxmuch = 0;
    unsigned int i, j;
//		for(i = 0; i < N; i++){
//			profit += Profits[i]*BestSolutionFound[i];
//		}
    for (i = 0; i < Weights.size(); i++) {
        for (j = 0; j < Weights[i].size(); j++) {
            //auxmuch me dice el peso de los objetos que llevo en auxiliar
            auxmuch += Weights[i][j] * BestSolutionFound[j];
        }
        //si es que auxmuch es más que b_i: ojo con los indices de b_i, puede estar raro
        if (auxmuch > b_i[i]) {
            return false;
        }
        auxmuch = 0;
    }

    return true;
}

//Retorna en Allowed los objetos posibles a evaluar
void CheckAllowedNodes(vector<float> CRem, vector<int> Vis) {
    unsigned int i, j;
    allowed.clear();
    bool flag = true;
    for (i = 0; i < N; i++) {
        //El objeto i en visited es 1 cuando no lo he visitado - es disponible
        if (Vis[i]) {
            //cout << "objeto i -->" << i << endl;
            //Ve que la capacidad remanente, en cada dimension, sea mayor que el peso del objeto i
            for (j = 0; (j < M) && flag; j++) {
                //cout << CRem[j] << " - " << Weights[j][i] << endl;
                if (CRem[j] < Weights[j][i]) {
                    flag = false;
                }
            }
            // si cabe, es allowed
            if (flag) {
                allowed.push_back(i);
            }
            flag = true;
        }
    }
}

void CheckAllowedNodesNew(){


}


//Swap object
bool LocalOptimaCheck(float Fit){
//en auxiliar esta la solucion
//en CapRem la cap remanente de cada dimension
//Weights[i][j] el peso de cada cosa
//visited = objeto es 1 cuando no lo han visitado

    unsigned int i, indi,indiall,cicles,newallow;
    vector <int> auxiauxi;
    vector <int> auxiVisited;
    vector <float> capAux;
    vector <int> LOActual;

    vector <int> auxiauxiAllowed;
    vector <int> auxiVisitedAllowed;
    vector <float> capAuxAllowed;
    float FitAuxAllowed;

    LOActual = auxiliar;
    bool flagLO = true;
    bool verbose = false;
    float FitAux, LOQ;


    //sol candidata inicial
    auxiauxi = auxiliar;
    capAux = CapRem;
    auxiVisited = visited;
    FitAux = Fit;
    LOQ = Fit;


    if(verbose){
        cout << "ANTES DE EMPEZAR" << endl;
        for(i = 0; i < auxiliar.size(); i++) if(auxiliar[i] == 1) cout << auxiliar[i] << " <--" << i << endl;
        for(i = 0; i < included.size(); i++) cout << included[i] << " " << endl;
        cout << endl;
    }


    for(cicles = 0; ((cicles < included.size()) && flagLO) ; cicles++) {

            //quito el objeto indi de la sol candidata inicial
            indi = included[cicles];
            auxiauxi[indi] = 0;
            auxiVisited[indi] = 1;
            FitAux = FitAux - Profits[indi];

            //devuelvo los recursos que estaba ocupando
            if(verbose) cout << "voy a sacar a indi:-->" << indi << " con Profit:" << Profits[indi] << " - FitAux sin indi:" << FitAux << endl;

            if(verbose) cout << "devuelvo recursos" << endl;
            for (i = 0; i < capAux.size(); i++) {
                capAux[i] += Weights[i][indi];
                if(verbose) cout << capAux[i] << " = " << Weights[i][indi] << " + " << CapRem[i] << endl;

            }

            //Veo cuales son los nuevos objetos allowed al sacar "indi"
            CheckAllowedNodes(capAux,auxiVisited);



            if(verbose){
                for(i = 0; i < allowed.size(); i++) cout << allowed[i] << " ";
                cout << endl;
            }
            if(verbose) cout << "size of allowed-->" << allowed.size() << endl;

            //Almaceno la sol candidata sin indi
            auxiauxiAllowed = auxiauxi;
            auxiVisitedAllowed = auxiVisited;
            capAuxAllowed = capAux;
            FitAuxAllowed = FitAux;

            //recorro los allowed, voy probando uno a uno.
            for(newallow = 0; (newallow < allowed.size()) && flagLO ; newallow++){

                //intento agregar indiall
                indiall = allowed[newallow];
                if(verbose) cout << "allowed#: "<< newallow <<  " intento agregar a --> "<< indiall << endl;

                if(indi != indiall){

                    auxiauxiAllowed[indiall] = 1;
                    FitAuxAllowed += Profits[indiall];
                    for (i = 0; i < capAuxAllowed.size(); i++) {
                        capAuxAllowed[i] -= Weights[i][indiall];
                        if(verbose) cout << capAuxAllowed[i] << " = " << Weights[i][indiall] << " - " << capAux[i] << endl;
                    }

                    //check if it is better. Si hay un vecino mejor, se acaba.
                    if(verbose) cout << "FitAux:" << FitAux <<"ProfObj:" << Profits[indiall] << " - LOQ:" << LOQ << " - FITAUXALLOWED:" << FitAuxAllowed << endl;
                    if (LOQ < FitAuxAllowed) {
                        if(verbose) cout << "ES MEJOR" << endl;
                        if(verbose) getchar();
                        flagLO = false;
                    }
                    else{
                        //Si no es mejor, vuelvo a la sol candidata sin indi
                        auxiauxiAllowed = auxiauxi;
                        auxiVisitedAllowed = auxiVisited;
                        capAuxAllowed = capAux;
                        FitAuxAllowed = FitAux;
                    }
                }
                else{
                    //Es el mismo que saqué (indi)
                }

            }


            FitAux = Fit;
            auxiauxi = auxiliar;
            capAux = CapRem;
            auxiVisited = visited;

    }


    return flagLO;
}