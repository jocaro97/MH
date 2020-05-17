
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <cmath>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <random>
#include <stdio.h>
#include <time.h>
#include <set>

using namespace std;

const int K = 5; //< Numero de particiones
const double alpha = 0.5;

// Seed for randomness
int seed = 62894; //anterior semilla: 23456789

// Random engine generator
default_random_engine gen;

const int MAX_ITER = 5000;

const int MAX_VECINOS = 20;

clock_t start_time;
double elapsed;

void start_timers()

{
    start_time = clock();
}



double elapsed_time()

{
    elapsed = clock()- start_time;
    return elapsed / CLOCKS_PER_SEC * 1000.0;
}

inline std::string trim(const std::string &s)
{
   auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
   auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
   return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}

struct Ejemplo{
  vector<double> caract;
  string clase;
  int n;
};

struct Cromosoma{
  vector<double> genes;
  double obj;
};

bool comparador_cromosoma(const Cromosoma& c1, const Cromosoma& c2){
  return c1.obj < c2.obj;
}

void leer_csv(string nombre_archivo, vector<Ejemplo> &resultado){

  ifstream archivo(nombre_archivo);
  string linea;


  while(getline(archivo, linea)){
    Ejemplo ej;
    ej.n = 0;
    istringstream s(linea);
    string palabra;

    while( getline(s, palabra, ';')){

      if(s.peek() != '\n' && s.peek() != EOF){
        ej.caract.push_back(stod(palabra));
        ej.n ++;
      }
    }

    ej.clase = trim(palabra);

    resultado.push_back(ej);
  }

}

vector<vector<Ejemplo>> hacerParticiones(const vector<Ejemplo>& datos) {
  unordered_map<string, int> categorias;
  vector<vector<Ejemplo>> particiones;

  for (int i = 0; i < K; i++)
    particiones.push_back(vector<Ejemplo>());

  for (auto& ej : datos) {
    int cat = categorias[ej.clase];
    particiones[cat].push_back(ej);

    categorias[ej.clase] = (categorias[ej.clase] + 1) % K;
  }

  return particiones;
}


double distanciaEuclidia(const Ejemplo& e1, const Ejemplo& e2) {
  double distancia = 0.0;
  for (int i = 0; i < e1.n; i++)
    distancia += (e2.caract[i] - e1.caract[i]) * (e2.caract[i] - e1.caract[i]);
  return distancia;
}

double distanciaConPesos(const Ejemplo& e1, const Ejemplo& e2, const vector<double>& w) {
  double distancia = 0.0;
  for (int i = 0; i < e1.n; i++){
    if (w[i] >= 0.2)
      distancia += w[i] * (e2.caract[i] - e1.caract[i]) * (e2.caract[i] - e1.caract[i]);

  }

  return distancia;
}

void buscarAmigoEnemigo(unsigned posicion, vector<Ejemplo> ejemplos, Ejemplo& amigo, Ejemplo& enemigo){
  double distancia;
  double distanciaAmigoMin = numeric_limits<double>::max();
  double distanciaEnemigoMin = numeric_limits<double>::max();
  unsigned posAmigoMin = 0;
  unsigned posEnemigoMin = 0;
  Ejemplo ej;
  for(unsigned i = 0; i < ejemplos.size(); ++i){
    if( i != posicion){

          distancia = distanciaEuclidia(ejemplos[i], ejemplos[posicion]);

          if(ejemplos[posicion].clase == ejemplos[i].clase){
            if(distancia < distanciaAmigoMin){
              distanciaAmigoMin = distancia;
              posAmigoMin = i;
            }
          }else{
            if(distancia < distanciaEnemigoMin){
              distanciaEnemigoMin = distancia;
              posEnemigoMin = i;
            }
          }
    }
  }

  amigo = ejemplos[posAmigoMin];
  enemigo = ejemplos[posEnemigoMin];

}

//------------------------------------------------------------
//-------------------------1-NN-------------------------------
//------------------------------------------------------------


string clasificar1NN(Ejemplo e, vector<Ejemplo> ejemplos, vector<double> pesos, int j){
  double distancia;
  double distanciaMin = numeric_limits<double>::max();
  int posMin = 0;

  for(int i = 0; i < ejemplos.size(); ++i){

    if( i != j){
      distancia = distanciaConPesos(ejemplos[i], e, pesos);

      if(distancia < distanciaMin){
        distanciaMin = distancia;
        posMin = i;
      }
    }

  }

  return ejemplos[posMin].clase;

}

//------------------------------------------------------------
//-------------------ANALISIS DE DATOS -----------------------
//------------------------------------------------------------


float tasaClas(const vector<string>& clasificados, const vector<Ejemplo>& test){
    int aciertos = 0;

    for(unsigned i = 0; i < clasificados.size(); ++i)
      if(clasificados[i] == test[i].clase)
        aciertos ++;

    return 100.0 * aciertos / clasificados.size();
}

float tasaRed(const vector<double>& p) {
  int descartados = 0;

  for (auto peso : p)
    if (peso < 0.2)
      descartados++;

  return 100.0 * descartados / p.size();
}

float objetivo(float tasa_clas, float tasa_red) {
  return alpha * tasa_clas + (1.0 - alpha) * tasa_red;
}

void limpiar(vector<double>& pesos){
  for(unsigned i = 0; i < pesos.size(); ++i)
    pesos[i] = 0.0;
}

void recalcularObjetivo(Cromosoma& cromosoma, const vector<Ejemplo>& entrenamiento){
  vector<string> clasificados;

  for(unsigned k = 0; k < entrenamiento.size(); ++k){
    clasificados.push_back( clasificar1NN(entrenamiento[k], entrenamiento, cromosoma.genes, k));
  }

  cromosoma.obj = objetivo(tasaClas(clasificados, entrenamiento), tasaRed(cromosoma.genes));
}

//------------------------------------------------------------
//----------------------------BL------------------------------
//------------------------------------------------------------

void busquedaLocal(const vector<Ejemplo>& entrenamiento, Cromosoma& c){
  normal_distribution<double> normal(0.0, 0.3);
  uniform_real_distribution<double> unirforme(0.0,1.0);

  double mejor_objetivo;
  int n = c.genes.size();
  int iteracion = 0;

  std::vector<int> indices;
  for(int i = 0; i < n; ++i){
    indices.push_back(i);
  }


  mejor_objetivo = c.obj;

  shuffle(indices.begin(), indices.end(), gen);

  //Busqueda

  while(iteracion < 2*n){
    int j = indices[iteracion % n];

    Cromosoma c_mut = c;

    c_mut.genes[j] += normal(gen);

    if(c_mut.genes[j] < 0) c_mut.genes[j] = 0;
    if(c_mut.genes[j] > 1) c_mut.genes[j] = 1;

    recalcularObjetivo(c_mut, entrenamiento);

    if(c_mut.obj > mejor_objetivo){
      mejor_objetivo = c_mut.obj;
      c = c_mut;
    }

    iteracion ++;

    if(iteracion % n == 0){
      shuffle(indices.begin(), indices.end(), gen);
    }

  }

}

//------------------------------------------------------------
//-------------------------OPERADORES-------------------------
//------------------------------------------------------------
void op_seleccion(vector< Cromosoma > poblacion, vector< Cromosoma > & seleccionados){

  int n = poblacion.size();
  Cromosoma c1, c2;

  seleccionados.clear();

  for(int i=0; i<2; ++i){
    int i1 = rand() % n;
    int i2 = rand() % n;

    c1 = poblacion[i1];
    c2 = poblacion[i2];

    if(c1.obj > c2.obj){
      seleccionados.push_back(c1);
    }else{
      seleccionados.push_back(c2);
    }

  }
}

void op_mutacion(vector< Cromosoma >& poblacion, int num_mutaciones){
  normal_distribution<double> normal(0.0, 0.3);
  int i = 0;
  set<pair<int, int>> indices;
  int n = poblacion.size();
  int m = poblacion[0].genes.size();
  int j,k;

  while( i < num_mutaciones ){
    j = rand() % n;
    k = rand() % m;

    if(indices.count(make_pair(j,k)) == 0){
      //mutamos
      poblacion[j].genes[k] += normal(gen);

      if(poblacion[j].genes[k] > 1) poblacion[j].genes[k] = 1.0;
      if(poblacion[j].genes[k] < 0) poblacion[j].genes[k] = 0.0;

      i++;
      indices.insert(make_pair(j,k));
      poblacion[j].obj = -1;
    }
  }

}

//------------------------------------------------------------
//-------------------------CRUCES----------------------------
//------------------------------------------------------------

void cruce_blx(vector< Cromosoma > padres, vector< Cromosoma >& hijos) {
  vector<double> cmin, cmax, ind;

  cmin.resize(padres[0].genes.size());
  cmax.resize(padres[0].genes.size());
  ind.resize(padres[0].genes.size());

  hijos.clear();
  hijos.resize(2);

  for(int i=0; i<padres[0].genes.size(); ++i){
    cmin[i] = min(padres[0].genes[i], padres[1].genes[i]);
    cmax[i] = max(padres[0].genes[i], padres[1].genes[i]);

    ind[i] = cmax[i] - cmin[i];

    uniform_real_distribution<float>
      random_real(cmin[i] - ind[i] * 0.3, cmax[i] + ind[i] * 0.3);

    hijos[0].genes.push_back(random_real(gen));
    hijos[1].genes.push_back(random_real(gen));

  }

  for(int i=0; i<2; ++i){
    for(int j=0; j<hijos[0].genes.size(); ++j){
      if(hijos[i].genes[j] < 0)
        hijos[i].genes[j] = 0.0;

      if(hijos[i].genes[j] > 1)
        hijos[i].genes[j] = 1.0;

    }
    hijos[i].obj = -1;
  }

}

void cruce_aritmetico(vector< Cromosoma > padres, Cromosoma& hijo){

  hijo.genes.clear();

  for(int i = 0; i< padres[0].genes.size(); ++i){
    hijo.genes.push_back( (padres[0].genes[i] + padres[1].genes[i]) / 2 );
  }
  hijo.obj = -1;
}

//------------------------------------------------------------
//-------------------------AGG-BLX----------------------------
//------------------------------------------------------------

void agg_blx(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  uniform_real_distribution<double> unirforme(0.0,1.0);
  vector < Cromosoma > poblacion, pob_intermedia;
  vector < Cromosoma > padres, hijos;
  Cromosoma mejor_sol_padre, mejor_sol_hijo, peor_sol;
  int n = pesos.size();
  int pc = 0.7 * 30;
  int pm = 0.001 * n * 30;
  int evaluaciones = 0;

  poblacion.resize(30);

  for(int i = 0; i < 30; ++i){
    for(int j = 0; j < pesos.size(); ++j){
      poblacion[i].genes.push_back(unirforme(gen));
    }
    recalcularObjetivo(poblacion[i], entrenamiento);
    evaluaciones ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

  while(evaluaciones < 15000){
    //Guardamos la mejor mejor solucion
    mejor_sol_padre = poblacion[poblacion.size()-1];

    for(int i = 0; i < pc; i+=2){
      op_seleccion(poblacion, padres);

      cruce_blx(padres, hijos);

      pob_intermedia.push_back(hijos[0]);
      pob_intermedia.push_back(hijos[1]);

    }

    for(int i = 0; i < (30 - pc); i+=2){
      op_seleccion(poblacion, padres);

      pob_intermedia.push_back(padres[0]);
      pob_intermedia.push_back(padres[1]);

    }

    op_mutacion(pob_intermedia, pm);

    poblacion = pob_intermedia;
    pob_intermedia.clear();

    for(int i=0; i<poblacion.size(); ++i){
      if(poblacion[i].obj == -1){
        recalcularObjetivo(poblacion[i], entrenamiento);
        evaluaciones++;
      }
    }

    sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    mejor_sol_hijo = poblacion[poblacion.size() -1];


    if(mejor_sol_hijo.obj < mejor_sol_padre.obj){

      poblacion[0] = mejor_sol_padre;
      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);
    }
  }

  pesos = poblacion[poblacion.size() -1].genes;

}

//------------------------------------------------------------
//-------------------------AGG-CA-----------------------------
//------------------------------------------------------------

void agg_ca(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  uniform_real_distribution<double> unirforme(0.0,1.0);
  vector < Cromosoma > poblacion, pob_intermedia, padres;

  Cromosoma mejor_sol_padre, mejor_sol_hijo, peor_sol, hijos;
  int n = pesos.size();
  int pc = 0.7 * 30;
  int pm = 0.001 * n * 30;
  int evaluaciones = 0;

  poblacion.resize(30);

  for(int i = 0; i < 30; ++i){
    for(int j = 0; j < pesos.size(); ++j){
      poblacion[i].genes.push_back(unirforme(gen));
    }
    recalcularObjetivo(poblacion[i], entrenamiento);
    evaluaciones ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

  while(evaluaciones < 15000){
    //Guardamos la mejor mejor solucion
    mejor_sol_padre = poblacion[poblacion.size() - 1];

    for(int i = 0; i < pc; i++){
      op_seleccion(poblacion, padres);

      cruce_aritmetico(padres, hijos);
      pob_intermedia.push_back(hijos);

    }

    for(int i = 0; i < (30 - pc); i+=2){
      op_seleccion(poblacion, padres);

      pob_intermedia.push_back(padres[0]);
      pob_intermedia.push_back(padres[1]);

    }

    op_mutacion(pob_intermedia, pm);

    poblacion = pob_intermedia;
    pob_intermedia.clear();

    for(int i=0; i<poblacion.size(); ++i){
      if(poblacion[i].obj == -1){
        recalcularObjetivo(poblacion[i], entrenamiento);
        evaluaciones++;
      }
    }

    sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    mejor_sol_hijo = poblacion[poblacion.size() -1];


    if(mejor_sol_hijo.obj < mejor_sol_padre.obj){

      poblacion[0] = mejor_sol_padre;
      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);
    }
  }

    pesos = poblacion[poblacion.size() -1].genes;

}

//------------------------------------------------------------
//-------------------------AGE-BLX----------------------------
//------------------------------------------------------------

void age_blx(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  uniform_real_distribution<double> unirforme(0.0,1.0);
  normal_distribution<double> normal(0.0, 0.3);

  vector < Cromosoma > poblacion, padres, hijos;
  Cromosoma mejor_sol;


  int n = pesos.size();
  int pc = 0.7 * 30;
  float pm = 0.001 * n * 2;
  uniform_int_distribution<int> random_int(0, n - 1);

  int evaluaciones = 0;

  poblacion.resize(30);

  for(int i = 0; i < 30; ++i){
    for(int j = 0; j < pesos.size(); ++j){
      poblacion[i].genes.push_back(unirforme(gen));
    }
    recalcularObjetivo(poblacion[i], entrenamiento);
    evaluaciones ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

  while(evaluaciones < 15000){
    //Guardamos la mejor mejor solucion
    op_seleccion(poblacion, padres);
    cruce_blx(padres, hijos);

    //PREGUNTAR COMO MUTAR
    uniform_real_distribution<double> random_real(0.0, 1.0);
    for (int i = 0; i < 2; i++) {
      if (random_real(gen) <= pm) {
        int k = random_int(gen);
        hijos[i].genes[k] += normal(gen);

        if(hijos[i].genes[k] > 1) hijos[i].genes[k] = 1.0;
        if(hijos[i].genes[k] < 0) hijos[i].genes[k] = 0.0;

      }
      hijos[i].obj = -1;
    }

    for(int i=0; i<hijos.size(); ++i){
      if(hijos[i].obj == -1){
        recalcularObjetivo(hijos[i], entrenamiento);
        evaluaciones++;
      }
    }

    //ARREGLAR

    for(int i = 0; i<2; ++i){
      hijos.push_back(poblacion[0]);
      poblacion.erase(poblacion.begin());
    }

    sort(hijos.begin(), hijos.end(), comparador_cromosoma);

    for(int i = 0; i<2; ++i){
      poblacion.push_back(hijos[hijos.size() -i-1]);

    }

    sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);
    hijos.clear();

  }
    pesos = poblacion[poblacion.size() -1].genes;

}

//------------------------------------------------------------
//-------------------------AGE-CA-----------------------------
//------------------------------------------------------------

void age_ca(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  uniform_real_distribution<double> unirforme(0.0,1.0);
  normal_distribution<double> normal(0.0, 0.3);

  vector < Cromosoma > poblacion, padres, hijos, peores;
  Cromosoma mejor_sol;
  vector< vector<double>> ::iterator it;

  int n = pesos.size();
  int pc = 0.7 * 30;
  float pm = 0.001 * n * 2;
  uniform_int_distribution<int> random_int(0, n - 1);

  int evaluaciones = 0;

  poblacion.resize(30);

  for(int i = 0; i < 30; ++i){
    for(int j = 0; j < pesos.size(); ++j){
      poblacion[i].genes.push_back(unirforme(gen));
    }
    recalcularObjetivo(poblacion[i], entrenamiento);
    evaluaciones ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

  while(evaluaciones < 15000){
    //Guardamos la mejor mejor solucion
    hijos.resize(2);

    for(int i=0; i<2; ++i){
      op_seleccion(poblacion, padres);
      cruce_aritmetico(padres, hijos[i]);
    }

    //PREGUNTAR COMO MUTAR
    uniform_real_distribution<double> random_real(0.0, 1.0);
    for (int i = 0; i < 2; i++) {
      if (random_real(gen) <= pm) {
        int k = random_int(gen);
        hijos[i].genes[k] += normal(gen);

        if(hijos[i].genes[k] > 1) hijos[i].genes[k] = 1.0;
        if(hijos[i].genes[k] < 0) hijos[i].genes[k] = 0.0;

      }
      hijos[i].obj = -1;
    }

    for(int i=0; i<hijos.size(); ++i){
      if(hijos[i].obj == -1){
        recalcularObjetivo(hijos[i], entrenamiento);
        evaluaciones++;
      }
    }

    //ARREGLAR

    for(int i = 0; i<2; ++i){
      hijos.push_back(poblacion[0]);
      poblacion.erase(poblacion.begin());
    }

    sort(hijos.begin(), hijos.end(), comparador_cromosoma);

    for(int i = 0; i<2; ++i){
      poblacion.push_back(hijos[hijos.size() -i-1]);

    }

    sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);
    hijos.clear();

  }
    pesos = poblacion[poblacion.size() -1].genes;
}

//------------------------------------------------------------
//-------------------------AM-10,1.0--------------------------
//------------------------------------------------------------
void am_1010(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  uniform_real_distribution<double> unirforme(0.0,1.0);
  vector < Cromosoma > poblacion, pob_intermedia, padres, hijos;

  Cromosoma mejor_sol_padre, mejor_sol_hijo, peor_sol;
  int n = pesos.size();
  int pc = 0.7 * 10;
  int pm = 0.001 * n * 10;
  int evaluaciones = 0;
  int generaciones = 1;

  poblacion.resize(10);

  for(int i = 0; i < 10; ++i){
    for(int j = 0; j < pesos.size(); ++j){
      poblacion[i].genes.push_back(unirforme(gen));
    }
    recalcularObjetivo(poblacion[i], entrenamiento);
    evaluaciones ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

  while(evaluaciones < 15000){
    //Guardamos la mejor mejor solucion
    mejor_sol_padre = poblacion[poblacion.size() - 1];

    for(int i = 0; i < pc; i+=2){
      op_seleccion(poblacion, padres);

      cruce_blx(padres, hijos);
      pob_intermedia.push_back(hijos[0]);
      pob_intermedia.push_back(hijos[1]);

    }

    for(int i = 0; i < (10 - pc); i+=2){
      op_seleccion(poblacion, padres);

      pob_intermedia.push_back(padres[0]);
      pob_intermedia.push_back(padres[1]);

    }

    op_mutacion(pob_intermedia, pm);

    poblacion = pob_intermedia;
    pob_intermedia.clear();

    for(int i=0; i<poblacion.size(); ++i){
      if(poblacion[i].obj == -1){
        recalcularObjetivo(poblacion[i], entrenamiento);
        evaluaciones++;
      }
    }

    sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    mejor_sol_hijo = poblacion[poblacion.size() -1];


    if(mejor_sol_hijo.obj < mejor_sol_padre.obj){

      poblacion[0] = mejor_sol_padre;
      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);
    }
    if(generaciones % 10 == 0){
      for(int i=0; i<poblacion.size(); ++i){
        busquedaLocal(entrenamiento, poblacion[i]);
        evaluaciones += 2*poblacion[i].genes.size();
      }

      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    }

    generaciones ++;
  }

    pesos = poblacion[poblacion.size() -1].genes;

}

//------------------------------------------------------------
//-------------------------AM-10,0.1--------------------------
//------------------------------------------------------------
void am_1001(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  uniform_real_distribution<double> unirforme(0.0,1.0);
  vector < Cromosoma > poblacion, pob_intermedia, padres, hijos;

  Cromosoma mejor_sol_padre, mejor_sol_hijo, peor_sol;
  int n = pesos.size();
  int pc = 0.7 * 10;
  int pm = 0.001 * n * 10;
  int evaluaciones = 0;
  int generaciones = 1;

  poblacion.resize(10);

  for(int i = 0; i < 10; ++i){
    for(int j = 0; j < pesos.size(); ++j){
      poblacion[i].genes.push_back(unirforme(gen));
    }
    recalcularObjetivo(poblacion[i], entrenamiento);
    evaluaciones ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

  while(evaluaciones < 15000){
    //Guardamos la mejor mejor solucion
    mejor_sol_padre = poblacion[poblacion.size() - 1];

    for(int i = 0; i < pc; i+=2){
      op_seleccion(poblacion, padres);

      cruce_blx(padres, hijos);
      pob_intermedia.push_back(hijos[0]);
      pob_intermedia.push_back(hijos[1]);


    }

    for(int i = 0; i < (10 - pc); i+=2){
      op_seleccion(poblacion, padres);

      pob_intermedia.push_back(padres[0]);
      pob_intermedia.push_back(padres[1]);

    }

    op_mutacion(pob_intermedia, pm);

    poblacion = pob_intermedia;
    pob_intermedia.clear();

    for(int i=0; i<poblacion.size(); ++i){
      if(poblacion[i].obj == -1){
        recalcularObjetivo(poblacion[i], entrenamiento);
        evaluaciones++;
      }
    }

    sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    mejor_sol_hijo = poblacion[poblacion.size() -1];


    if(mejor_sol_hijo.obj < mejor_sol_padre.obj){

      poblacion[0] = mejor_sol_padre;
      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);
    }
    if(generaciones % 10 == 0){
        int indice = rand() % 10;
        busquedaLocal(entrenamiento, poblacion[indice]);
        evaluaciones += 2*poblacion[indice].genes.size();


      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    }

    generaciones ++;
  }

    pesos = poblacion[poblacion.size() -1].genes;

}

//------------------------------------------------------------
//-------------------------AM-10,0.1 MEJ----------------------
//------------------------------------------------------------
void am_1001_mej(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  uniform_real_distribution<double> unirforme(0.0,1.0);
  vector < Cromosoma > poblacion, pob_intermedia, padres, hijos;

  Cromosoma mejor_sol_padre, mejor_sol_hijo, peor_sol;
  int n = pesos.size();
  int pc = 0.7 * 10;
  int pm = 0.001 * n * 10;
  int evaluaciones = 0;
  int generaciones = 1;

  poblacion.resize(10);

  for(int i = 0; i < 10; ++i){
    for(int j = 0; j < pesos.size(); ++j){
      poblacion[i].genes.push_back(unirforme(gen));
    }
    recalcularObjetivo(poblacion[i], entrenamiento);
    evaluaciones ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

  while(evaluaciones < 15000){
    //Guardamos la mejor mejor solucion
    mejor_sol_padre = poblacion[poblacion.size() - 1];

    for(int i = 0; i < pc; i+=2){
      op_seleccion(poblacion, padres);

      cruce_blx(padres, hijos);
      pob_intermedia.push_back(hijos[0]);
      pob_intermedia.push_back(hijos[1]);


    }

    for(int i = 0; i < (10 - pc); i+=2){
      op_seleccion(poblacion, padres);

      pob_intermedia.push_back(padres[0]);
      pob_intermedia.push_back(padres[1]);

    }

    op_mutacion(pob_intermedia, pm);

    poblacion = pob_intermedia;
    pob_intermedia.clear();

    for(int i=0; i<poblacion.size(); ++i){
      if(poblacion[i].obj == -1){
        recalcularObjetivo(poblacion[i], entrenamiento);
        evaluaciones++;
      }
    }

    sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    mejor_sol_hijo = poblacion[poblacion.size() -1];


    if(mejor_sol_hijo.obj < mejor_sol_padre.obj){

      poblacion[0] = mejor_sol_padre;
      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);
    }
    if(generaciones % 10 == 0){
        busquedaLocal(entrenamiento, poblacion[9]);
        evaluaciones += 2*poblacion[9].genes.size();


      sort(poblacion.begin(), poblacion.end(), comparador_cromosoma);

    }

    generaciones ++;
  }

    pesos = poblacion[poblacion.size() -1].genes;

}

void practica1(string nombre_archivo){
  vector<Ejemplo> ejemplos;
  vector<Ejemplo> entrenamiento;
  vector<string> clasificados;

  double tasa_clasificacion[7];
  double tasa_reduccion[7];
  double agregado[7];
  double tiempo[7];

  leer_csv(nombre_archivo, ejemplos);

  cout << "----------------------------------------------------------" << endl;
  cout << "CONJUNTO DE DATOS: " << nombre_archivo << endl;
  cout << "----------------------------------------------------------" << endl << endl;

  vector<double> pesos(ejemplos[0].caract.size(), 0);

  double tasa_clasificacion_global[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double tasa_reduccion_global[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double agregado_global[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double tiempo_global[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for(unsigned i = 0; i < K; ++i){
    auto particiones = hacerParticiones(ejemplos);

    auto test = particiones[i];

    for(unsigned j = 0; (j < i) ; ++j)
      entrenamiento.insert(entrenamiento.end(), particiones[j].begin(), particiones[j].end());


    for(unsigned j = i+1; (j < K) ; ++j)
      entrenamiento.insert(entrenamiento.end(), particiones[j].begin(), particiones[j].end());

    //AGG - BLX
    clasificados.clear();
    start_timers();

    agg_blx(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[0] = elapsed_time();

    tasa_clasificacion[0] = tasaClas(clasificados, test);
    tasa_reduccion[0] = tasaRed(pesos);
    agregado[0] = objetivo(tasa_clasificacion[0], tasa_reduccion[0]);

    limpiar(pesos);

    tasa_clasificacion_global[0] += tasa_clasificacion[0];
    tasa_reduccion_global[0] += tasa_reduccion[0];
    agregado_global[0] += agregado[0];
    tiempo_global[0] += tiempo[0];

    //AGG - CA
    clasificados.clear();
    start_timers();

    agg_ca(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[1] = elapsed_time();

    tasa_clasificacion[1] = tasaClas(clasificados, test);
    tasa_reduccion[1] = tasaRed(pesos);
    agregado[1] = objetivo(tasa_clasificacion[1], tasa_reduccion[1]);

    limpiar(pesos);

    tasa_clasificacion_global[1] += tasa_clasificacion[1];
    tasa_reduccion_global[1] += tasa_reduccion[1];
    agregado_global[1] += agregado[1];
    tiempo_global[1] += tiempo[1];

    //AGE - BLX
    clasificados.clear();
    start_timers();

    age_blx(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[2] = elapsed_time();

    tasa_clasificacion[2] = tasaClas(clasificados, test);
    tasa_reduccion[2] = tasaRed(pesos);
    agregado[2] = objetivo(tasa_clasificacion[2], tasa_reduccion[2]);

    limpiar(pesos);

    tasa_clasificacion_global[2] += tasa_clasificacion[2];
    tasa_reduccion_global[2] += tasa_reduccion[2];
    agregado_global[2] += agregado[2];
    tiempo_global[2] += tiempo[2];

    //AGE - CA
    clasificados.clear();
    start_timers();

    age_ca(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[3] = elapsed_time();

    tasa_clasificacion[3] = tasaClas(clasificados, test);
    tasa_reduccion[3] = tasaRed(pesos);
    agregado[3] = objetivo(tasa_clasificacion[3], tasa_reduccion[3]);

    limpiar(pesos);

    tasa_clasificacion_global[3] += tasa_clasificacion[3];
    tasa_reduccion_global[3] += tasa_reduccion[3];
    agregado_global[3] += agregado[3];
    tiempo_global[3] += tiempo[3];

    //AM - 10,1.0
    clasificados.clear();
    start_timers();

    am_1010(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[4] = elapsed_time();

    tasa_clasificacion[4] = tasaClas(clasificados, test);
    tasa_reduccion[4] = tasaRed(pesos);
    agregado[4] = objetivo(tasa_clasificacion[4], tasa_reduccion[4]);

    limpiar(pesos);

    tasa_clasificacion_global[4] += tasa_clasificacion[4];
    tasa_reduccion_global[4] += tasa_reduccion[4];
    agregado_global[4] += agregado[4];
    tiempo_global[4] += tiempo[4];

    //AM - 10,0.1
    clasificados.clear();
    start_timers();

    am_1001(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[5] = elapsed_time();

    tasa_clasificacion[5] = tasaClas(clasificados, test);
    tasa_reduccion[5] = tasaRed(pesos);
    agregado[5] = objetivo(tasa_clasificacion[5], tasa_reduccion[5]);

    limpiar(pesos);

    tasa_clasificacion_global[5] += tasa_clasificacion[5];
    tasa_reduccion_global[5] += tasa_reduccion[5];
    agregado_global[5] += agregado[5];
    tiempo_global[5] += tiempo[5];

    //AM - 10,0.1 mej
    clasificados.clear();
    start_timers();

    am_1001_mej(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[6] = elapsed_time();

    tasa_clasificacion[6] = tasaClas(clasificados, test);
    tasa_reduccion[6] = tasaRed(pesos);
    agregado[6] = objetivo(tasa_clasificacion[6], tasa_reduccion[6]);

    limpiar(pesos);

    tasa_clasificacion_global[6] += tasa_clasificacion[6];
    tasa_reduccion_global[6] += tasa_reduccion[6];
    agregado_global[6] += agregado[6];
    tiempo_global[6] += tiempo[6];



    std::cout << "----------------- Ejecucion "<< i+1 <<" -----------------------"<< endl;
    std::cout << "AGG-BLX" << endl;
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[0] <<" %"<< endl;
    std::cout << "Tasa reduccion: " <<tasa_reduccion[0] <<" %"<<endl;
    std::cout << "Agregado: "<< agregado[0] << endl;
    std::cout << "Tiempo: "<< tiempo[0] << endl;
    std::cout << "-----------------------------------------------------" << endl;
    std::cout << "AGG-CA" << endl;
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[1] <<" %"<< endl;
    std::cout << "Tasa reduccion: " <<tasa_reduccion[1] <<" %"<<endl;
    std::cout << "Agregado: "<< agregado[1] << endl;
    std::cout << "Tiempo: "<< tiempo[1] << endl;
    std::cout << "-----------------------------------------------------" << endl;
    std::cout << "AGE-BLX" << endl;
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[2] <<" %"<< endl;
    std::cout << "Tasa reduccion: " <<tasa_reduccion[2] <<" %"<<endl;
    std::cout << "Agregado: "<< agregado[2] << endl;
    std::cout << "Tiempo: "<< tiempo[2] << endl;
    std::cout << "-----------------------------------------------------" << endl;
    std::cout << "AGE-CA" << endl;
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[3] <<" %"<< endl;
    std::cout << "Tasa reduccion: " <<tasa_reduccion[3] <<" %"<<endl;
    std::cout << "Agregado: "<< agregado[3] << endl;
    std::cout << "Tiempo: "<< tiempo[3] << endl;
    std::cout << "-----------------------------------------------------" << endl;
    std::cout << "AM-10, 1.0" << endl;
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[4] <<" %"<< endl;
    std::cout << "Tasa reduccion: " <<tasa_reduccion[4] <<" %"<<endl;
    std::cout << "Agregado: "<< agregado[4] << endl;
    std::cout << "Tiempo: "<< tiempo[4] << endl;
    std::cout << "-----------------------------------------------------" << endl;
    std::cout << "AM-10, 0.1" << endl;
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[5] <<" %"<< endl;
    std::cout << "Tasa reduccion: " <<tasa_reduccion[5] <<" %"<<endl;
    std::cout << "Agregado: "<< agregado[5] << endl;
    std::cout << "Tiempo: "<< tiempo[5] << endl;
    std::cout << "-----------------------------------------------------" << endl;
    std::cout << "AM-10, 0.1 mej" << endl;
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[6] <<" %"<< endl;
    std::cout << "Tasa reduccion: " <<tasa_reduccion[6] <<" %"<<endl;
    std::cout << "Agregado: "<< agregado[6] << endl;
    std::cout << "Tiempo: "<< tiempo[6] << endl;
    std::cout << "-----------------------------------------------------" << endl;

    entrenamiento.clear();
    clasificados.clear();
    limpiar(pesos);

  }

  std::cout << "----------------- RESUTADOS GLOBALES -----------------------"<< endl;
  std::cout << "AGG-BLX" << endl;
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[0] / K <<" %"<< endl;
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[0] / K <<" %"<<endl;
  std::cout << "Agregado: "<< agregado_global[0] / K << endl;
  std::cout << "Tiempo: "<< tiempo_global[0] / K << endl;
  std::cout << "-----------------------------------------------------" << endl;
  std::cout << "AGG-CA" << endl;
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[1] / K <<" %"<< endl;
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[1] / K <<" %"<<endl;
  std::cout << "Agregado: "<< agregado_global[1] / K << endl;
  std::cout << "Tiempo: "<< tiempo_global[1] / K << endl;
  std::cout << "-----------------------------------------------------" << endl;
  std::cout << "AGE-BLX" << endl;
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[2] / K <<" %"<< endl;
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[2] / K <<" %"<<endl;
  std::cout << "Agregado: "<< agregado_global[2] / K << endl;
  std::cout << "Tiempo: "<< tiempo_global[2] / K << endl;
  std::cout << "-----------------------------------------------------" << endl;
  std::cout << "AGE-CA" << endl;
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[3] / K <<" %"<< endl;
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[3] / K <<" %"<<endl;
  std::cout << "Agregado: "<< agregado_global[3] / K << endl;
  std::cout << "Tiempo: "<< tiempo_global[3] / K << endl;
  std::cout << "-----------------------------------------------------" << endl;
  std::cout << "AM-10,1.0" << endl;
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[4] / K <<" %"<< endl;
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[4] / K <<" %"<<endl;
  std::cout << "Agregado: "<< agregado_global[4] / K << endl;
  std::cout << "Tiempo: "<< tiempo_global[4] / K << endl;
  std::cout << "-----------------------------------------------------" << endl;
  std::cout << "AM-10,0.1" << endl;
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[5] / K <<" %"<< endl;
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[5] / K <<" %"<<endl;
  std::cout << "Agregado: "<< agregado_global[5] / K << endl;
  std::cout << "Tiempo: "<< tiempo_global[5] / K << endl;
  std::cout << "-----------------------------------------------------" << endl;
  std::cout << "AM-10,0.1 mej" << endl;
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[6] / K <<" %"<< endl;
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[6] / K <<" %"<<endl;
  std::cout << "Agregado: "<< agregado_global[6] / K << endl;
  std::cout << "Tiempo: "<< tiempo_global[6] / K << endl;
  std::cout << "-----------------------------------------------------" << endl;

}

int main(int argc, char const *argv[]){

  if (argc > 1)
    seed = stoi(argv[1]);

  gen = default_random_engine(seed);
  srand(seed);

  practica1("DATA/ionosphere_normalizados.csv");

  practica1("DATA/colposcopy_normalizados.csv");

  practica1("DATA/texture_normalizados.csv");

}
