/*
 * Initialize.cpp
 *
 *  Created on: 22-12-2014
 *      Author: nicolas
 */
#include "Initialize.h"
#include "ACO.h"

vector<float> WeightsProm;

using namespace std;
using std::ifstream;


void Capture_Params(int argc, char **argv) {
    InstanciaFile = argv[1];
    Seed = atoi(argv[2]);
    TotalAnts = atoi(argv[3]);
    TotalEvaluations = atoi(argv[4]);
    alpha = atof(argv[5]);
    beta = atof(argv[6]);
    ph_max = atof(argv[7]);
    ph_min = atof(argv[8]);
    rho = atof(argv[9]);
}


bool Read_Instance() {
    ifstream fin;
    //bool new_instance = false;
    bool profits = false;
    bool wei = false;
    //OJO CON ESTOS PARAMETROS
    const int MAX_CHARS_PER_LINE = 10280;
    const int MAX_TOKENS_PER_LINE = 10280;
    const char *const DELIMITER = " ";
    unsigned int bloque_uno, bloque_dos, bloque_tres;
    unsigned int i;
    int contador = 0;
    fin.open(InstanciaFile); // open a file
    if (!fin.good())
        return true; // exit if file not found

    bloque_uno = 0;
    bloque_dos = 0;
    bloque_tres = 0;
    BestFitnessFound = 0;

    while (!fin.eof()) {
        // read an entire line into memory
        char buf[MAX_CHARS_PER_LINE];
        fin.getline(buf, MAX_CHARS_PER_LINE);
        // parse the line into blank-delimited tokens
        unsigned int n = 0; // a for-loop index
        // array to store memory addresses of the tokens in buf
        const char *token[MAX_TOKENS_PER_LINE] = {}; // initialze to 0
        // parse the line
        token[0] = strtok(buf, DELIMITER); // first token
        if (token[0]) // zero if line is blank
        {
            for (n = 1; n < MAX_TOKENS_PER_LINE; n++) {
                token[n] = strtok(0, DELIMITER); // subsequent tokens
                if (!token[n]) break; // no more tokens
            }
        }
        if (contador == 0) {
            N = atoi(token[0]);
            M = atoi(token[1]);
            Opt = atof(token[2]);
            //new_instance = false;
            profits = true;
            bloque_uno = 0;
        } else {
            if (profits) {
                for (i = 0; i < n; i++) {
                    Profits.push_back(atof(token[i]));
                    BestFitnessFound += atof(token[i]);
                    bloque_uno++;
                }
                if (N == bloque_uno) {
                    profits = false;
                    wei = true;
                    bloque_dos = 0;
                    bloque_tres = 0;
                }
            } else {
                if (wei) {
                    for (i = 0; i < n; i++) {
                        Aux.push_back(atof(token[i]));
                        bloque_dos++;
                    }
                    if (bloque_dos == N) {
                        bloque_tres++;
                        Weights.push_back(Aux);
                        Aux.clear();
                        if (bloque_tres == M) {
                            wei = false;
                        }
                        bloque_dos = 0;
                    }
                } else {
                    for (i = 0; i < n; i++) {
                        b_i.push_back(atof(token[i]));
                        bloque_dos++;
                    }
                }
            }
        }
        contador++;
    }
    return false;
}

//Inicializa la feromona
void DefineInitial() {
    unsigned int i;
    float initialph;
    vector<float> au;
    vector<float> ph;
    initialph = ph_max;

    for (i = 0; i < N; i++) {
        ph.push_back(initialph);
    }
    for (i = 0; i < N; i++)Pheromone.push_back(ph);
    for (i = 0; i < N; i++)Pheromone[i][i] = initialph;

    //Mat_A = Pheromone;
}

void PrintInstancia() {

    unsigned int i, j;
    cout << "N --> " << N << endl;
    cout << "M --> " << M << endl;
    cout << "OPT -->" << Opt << endl;
    cout << "PROFITS" << endl;
    for (i = 0; i < Profits.size(); i++) {
        cout << Profits[i] << " ";
    }
    cout << endl;
    cout << "b_i" << endl;
    for (i = 0; i < b_i.size(); i++) {
        cout << b_i[i] << " ";
    }
    cout << endl;
    cout << "Weights" << endl;
    for (i = 0; i < Weights.size(); i++) {
        for (j = 0; j < Weights[i].size(); j++) {
            cout << Weights[i][j] << " ";
        }
        cout << endl;
    }

}

int int_rand(int a, int b) {
    int retorno = 0;

    if (a < b) {
        retorno = (int) ((b - a) * drand48());
        retorno = retorno + a;
    } else {
        retorno = (int) ((a - b) * drand48());
        retorno = retorno + b;
    }

    return retorno;
}

float float_rand(float a, float b) {
    float retorno = 0;

    if (a < b) {
        retorno = (float) ((b - a) * drand48());
        retorno = retorno + a;
    } else {
        retorno = (float) ((a - b) * drand48());
        retorno = retorno + b;
    }

    return retorno;
}

