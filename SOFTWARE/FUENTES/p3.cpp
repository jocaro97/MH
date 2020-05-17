
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
const float mu = 0.3;
const float phi = 0.3;
const float sigma = 0.3;
const float sigma_ils = 0.4;

float temperatura_final = 1e-3;

// Seed for randomness
int seed =  327; //int seed = 62894; // otra semilla 23456789;

// Random engine generator
default_random_engine gen;

const int MAX_ITER = 15000;

const int MAX_VECINOS_POR_CARACTERISTICA = 10;
const float MAX_EXITOS_POR_VECINOS = 0.1;

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

struct Solucion{
  vector<double> w;
  float objetivo;
};

bool comparador_solucion(const Solucion& sol1, const Solucion& sol2){
  return sol1.objetivo < sol2.objetivo;
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

void relief(vector<Ejemplo> ejemplos, vector<double> & pesos){
  double wmax = 0;
  Ejemplo amigo, enemigo;
  for(unsigned i = 0; i < ejemplos.size(); ++i){
    buscarAmigoEnemigo(i, ejemplos, amigo, enemigo);

    for(unsigned j = 0; j < ejemplos[i].caract.size(); ++j){
      pesos[j] += abs(ejemplos[i].caract[j] - enemigo.caract[j]) - abs(ejemplos[i].caract[j] - amigo.caract[j]);
    }
  }

  for(auto p: pesos){
    if(p > wmax)
      wmax = p;
  }

  for(unsigned j = 0; j < pesos.size(); ++j){

    if( pesos[j] < 0 )
      pesos[j] = 0;
    else
      pesos[j] = pesos[j] / wmax;
  }
}

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

//-------------------------------------------------------------------------
//---------------------------OPERADORES COMUNES----------------------------
//-------------------------------------------------------------------------

void recalcularObjetivo(Solucion& sol, const vector<Ejemplo>& entrenamiento){
  vector<string> clasificados;

  for(unsigned k = 0; k < entrenamiento.size(); ++k){
    clasificados.push_back( clasificar1NN(entrenamiento[k], entrenamiento, sol.w, k));
  }

  sol.objetivo = objetivo(tasaClas(clasificados, entrenamiento), tasaRed(sol.w));
}


Solucion inicializar_solucion(const vector<Ejemplo> entrenamiento, int n){
  Solucion sol;
  uniform_real_distribution<double> random_real(0.0, 1.0);

  sol.w.resize(n);
  for(int i = 0; i < n; i++)
    sol.w[i] = random_real(gen);

  recalcularObjetivo(sol, entrenamiento);

  return sol;
}

void mutar(vector<double>& w, int i, float s){
  normal_distribution<double> normal(0.0, s);
  w[i] += normal(gen);

  if (w[i] < 0.0) w[i] = 0.0;
  if (w[i] > 1.0) w[i] = 1.0;
}


//------------------------------------------------------------------------
//------------------------ENFRIAMIENTO SIMULADO---------------------------
//------------------------------------------------------------------------

void enfriamiento_simulado(const vector<Ejemplo> entrenamiento, vector<double>& pesos){
  const int MAX_VECINOS = MAX_VECINOS_POR_CARACTERISTICA * pesos.size();
  const int MAX_EXITOS = MAX_EXITOS_POR_VECINOS * MAX_VECINOS;
  const int M = MAX_ITER / MAX_VECINOS;
  Solucion mejor_sol;
  int it, num_vecinos;
  float exito;
  float temperatura_inicial, temperatura;
  uniform_int_distribution<int> random_int(0, pesos.size() - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  it = 0;
  Solucion sol = inicializar_solucion(entrenamiento, pesos.size());
  mejor_sol = sol;
  it ++;
  temperatura_inicial = (mu * mejor_sol.objetivo) / (- 1.0 * log(phi));
  temperatura = temperatura_inicial;

  while(temperatura_final >= temperatura)
    temperatura_final = temperatura / 100.0;

  const float beta = (float) (temperatura_inicial - temperatura_final) / (M * temperatura_inicial * temperatura_final);

  exito = MAX_EXITOS;

  while(it < MAX_ITER && exito != 0){
    num_vecinos = 0;
    exito = 0;

    while(it < MAX_ITER && num_vecinos < MAX_VECINOS && exito < MAX_EXITOS){
      int i = random_int(gen);
      Solucion sol_mutada = sol;
      mutar(sol_mutada.w, i, sigma);
      recalcularObjetivo(sol_mutada, entrenamiento);
      it ++;
      num_vecinos ++;

      float dif = sol.objetivo - sol_mutada.objetivo;

      if(dif == 0)
        dif = 0.001;

      if(dif < 0 || random_real(gen) <= exp(-1.0 * dif / temperatura) ){
        exito ++;
        sol = sol_mutada;
        if(sol.objetivo > mejor_sol.objetivo)
          mejor_sol = sol;
      }
    }
    temperatura = temperatura / (1.0 + beta * temperatura);
  }

  pesos = mejor_sol.w;

}

//------------------------------------------------------------------------
//-------------------------------ILS--------------------------------------
//------------------------------------------------------------------------

void busquedaLocal(const vector<Ejemplo>& entrenamiento, Solucion& sol){
  const int MAX_ITER_BL = 1000;
  const int MAX_VECINOS_BL = 20;
  double mejor_obejtivo = sol.objetivo;
  int n = sol.w.size();
  int iteracion = 0;
  int num_vecinos = 0;
  bool mejora = false;

  std::vector<int> indices;
  for(int i = 0; i < n; ++i){
    indices.push_back(i);
  }

  shuffle(indices.begin(), indices.end(), gen);

  //Busqueda

  while( (iteracion < MAX_ITER_BL) && (num_vecinos < n * MAX_VECINOS_BL) ){
    int j = indices[iteracion % n];

    Solucion sol_mutada = sol;

    mutar(sol_mutada.w, j, sigma);

    recalcularObjetivo(sol_mutada, entrenamiento);

    if(sol_mutada.objetivo > mejor_obejtivo){
      mejor_obejtivo = sol_mutada.objetivo;
      sol = sol_mutada;
      num_vecinos = 0;
      mejora = true;

    }else
      num_vecinos ++;

    iteracion ++;

    if(mejora || (iteracion % n == 0)){
      mejora = false;
      shuffle(indices.begin(), indices.end(), gen);
    }

  }

}

void ils(const vector<Ejemplo> entrenamiento, vector<double>& pesos){
  const int MAX_ITER_ILS = 15;
  const float FACTOR_MUTACION_ILS = 0.1;
  uniform_int_distribution<int> random_int(0, pesos.size() -1);
  Solucion sol = inicializar_solucion(entrenamiento, pesos.size());
  int n = pesos.size();
  int indice;

  busquedaLocal(entrenamiento, sol);

  for(int i = 1; i < MAX_ITER_ILS; ++i){
    Solucion sol_mutada = sol;

    set<int> indices_mutados;
    for(int j = 0; j < (int) FACTOR_MUTACION_ILS * n; ++j){
      while(indices_mutados.size() == j){
        indice = random_int(gen);
        indices_mutados.insert(indice);
      }
      mutar(sol_mutada.w, indice, sigma_ils);
    }
    recalcularObjetivo(sol_mutada, entrenamiento);
    busquedaLocal(entrenamiento, sol_mutada);

    if(sol_mutada.objetivo > sol.objetivo)
      sol = sol_mutada;
  }
  pesos = sol.w;
}

//-------------------------------------------------------------------------
//-----------------------EVOLUCION DIFERENCIAL-----------------------------
//-------------------------------------------------------------------------

void seleccion_padres(const vector<Solucion> poblacion, vector<Solucion>& seleccionados, int num_padres, int padre){
  uniform_int_distribution<int> random_int(0, poblacion.size() -1);
  set<int> candidatos;
  int indice;

  for(int i = 0; i < num_padres; ++i){
    while(candidatos.size() == i){
      indice = random_int(gen);
      if(indice != padre)
        candidatos.insert(indice);
    }
    seleccionados.push_back(poblacion[indice]);
  }
}

void evolucion_direfencial_aleatoria(const vector<Ejemplo> entrenamiento, vector<double>& pesos){
  const int TAM_POB = 50;
  const int MAX_ITER_DE = 15000;
  const int NUM_PADRES_DE_ALEATORIO = 3;
  const float PROB_CRUCE = 0.5;
  const float FREC = 0.5;
  int n = pesos.size();
  uniform_int_distribution<int> random_int(0, n -1);
  uniform_real_distribution<double> random_real(0.0, 1.0);
  vector<Solucion> poblacion;
  int it = 0;

  for(int i = 0; i < TAM_POB; ++i){
    poblacion.push_back(inicializar_solucion(entrenamiento, pesos.size()));
    it ++;
  }

  while(it < MAX_ITER_DE){
    for(int i = 0; i < TAM_POB; ++i){
      vector<Solucion> padres;
      Solucion hija;
      hija.w.resize(n);

      seleccion_padres(poblacion, padres, NUM_PADRES_DE_ALEATORIO, i);
      int elegido = random_int(gen);
      for(int k = 0; k < n; ++k){
        if(k == elegido || random_real(gen) <= PROB_CRUCE){
          hija.w[k] = padres[0].w[k] + FREC * (padres[1].w[k] - padres[2].w[k]);

          if(hija.w[k] < 0.0) hija.w[k] = 0.0;
          if(hija.w[k] > 1.0) hija.w[k] = 1.0;
        }else{
          hija.w[k] = poblacion[i].w[k];
        }
      }

      recalcularObjetivo(hija, entrenamiento);
      it ++;

      if(hija.objetivo > poblacion[i].objetivo)
        poblacion[i] = hija;
    }
  }

  sort(poblacion.begin(), poblacion.end(), comparador_solucion);

  pesos = poblacion[TAM_POB - 1].w;
}

void evolucion_direfencial_ctb(const vector<Ejemplo> entrenamiento, vector<double>& pesos){
  const int TAM_POB = 50;
  const int MAX_ITER_DE = 15000;
  const int NUM_PADRES_DE_CTB = 2;
  const float PROB_CRUCE = 0.5;
  const float FREC = 0.5;
  int n = pesos.size();
  uniform_int_distribution<int> random_int(0, n -1);
  uniform_real_distribution<double> random_real(0.0, 1.0);
  vector<Solucion> poblacion;
  Solucion mejor_sol;
  int it = 0;

  for(int i = 0; i < TAM_POB; ++i){
    poblacion.push_back(inicializar_solucion(entrenamiento, pesos.size()));
    it ++;
  }

  sort(poblacion.begin(), poblacion.end(), comparador_solucion);
  mejor_sol = poblacion[TAM_POB - 1];

  while(it < MAX_ITER_DE){
    for(int i = 0; i < TAM_POB; ++i){
      vector<Solucion> padres;
      Solucion hija;
      hija.w.resize(n);

      seleccion_padres(poblacion, padres, NUM_PADRES_DE_CTB, i);
      int elegido = random_int(gen);
      for(int k = 0; k < n; ++k){
        if(k == elegido || random_real(gen) <= PROB_CRUCE){
          hija.w[k] = poblacion[i].w[k] + FREC * (mejor_sol.w[k] - poblacion[i].w[k]) + FREC * (padres[0].w[k] - padres[1].w[k]);

          if(hija.w[k] < 0.0) hija.w[k] = 0.0;
          if(hija.w[k] > 1.0) hija.w[k] = 1.0;
        }else{
          hija.w[k] = poblacion[i].w[k];
        }
      }

      recalcularObjetivo(hija, entrenamiento);
      it ++;

      if(hija.objetivo > poblacion[i].objetivo)
        poblacion[i] = hija;
    }

    sort(poblacion.begin(), poblacion.end(), comparador_solucion);
    mejor_sol = poblacion[TAM_POB - 1];
  }

  pesos = poblacion[TAM_POB - 1].w;
}



void practica1(string nombre_archivo){
  vector<Ejemplo> ejemplos;
  vector<Ejemplo> entrenamiento;
  vector<string> clasificados;

  double tasa_clasificacion[4];
  double tasa_reduccion[4];
  double agregado[4];
  double tiempo[4];

  leer_csv(nombre_archivo, ejemplos);

  cout << "----------------------------------------------------------" << endl;
  cout << "CONJUNTO DE DATOS: " << nombre_archivo << endl;
  cout << "----------------------------------------------------------" << endl << endl;

  vector<double> pesos(ejemplos[0].caract.size(), 0);
  vector<double> pesos1(ejemplos[0].caract.size(), 1);

  double tasa_clasificacion_global[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double tasa_reduccion_global[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double agregado_global[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double tiempo_global[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for(unsigned i = 0; i < K; ++i){
    auto particiones = hacerParticiones(ejemplos);

    auto test = particiones[i];

    for(unsigned j = 0; (j < i) ; ++j)
      entrenamiento.insert(entrenamiento.end(), particiones[j].begin(), particiones[j].end());


    for(unsigned j = i+1; (j < K) ; ++j)
      entrenamiento.insert(entrenamiento.end(), particiones[j].begin(), particiones[j].end());

    clasificados.clear();
    std::cout << "1-nn" << '\n';


    // 1-NN
    start_timers();
    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos1, -1));
    }

    tiempo[0] = elapsed_time();

    tasa_clasificacion[0] = tasaClas(clasificados, test);
    agregado[0] = objetivo(tasa_clasificacion[0], tasa_reduccion[0]);

    tasa_clasificacion_global[0] += tasa_clasificacion[0];
    tasa_reduccion_global[0] += tasa_reduccion[0];
    agregado_global[0] += agregado[0];
    tiempo_global[0] += tiempo[0];

    // RELIEF
    clasificados.clear();
    std::cout << "RELIEF" << '\n';

    start_timers();
    relief(entrenamiento, pesos);

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


    //Enfriamiento simulado
    std::cout << "Enfriamiento simulado" << '\n';
    clasificados.clear();

    start_timers();
    enfriamiento_simulado(entrenamiento, pesos);

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

    //ILS
    clasificados.clear();
    std::cout << "ILS" << '\n';

    start_timers();
    ils(entrenamiento, pesos);

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

    //Evolucion difrencial rand
    std::cout << "DE -rand" << '\n';
    clasificados.clear();

    start_timers();
    evolucion_direfencial_aleatoria(entrenamiento, pesos);

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

    //Evolucion difrencial current to best
    std::cout << "DE-ctb" << '\n';
    clasificados.clear();

    start_timers();
    evolucion_direfencial_ctb(entrenamiento, pesos);

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


    std::cout << "----------------- Ejecucion "<< i+1 <<" -----------------------"<< '\n';
    std::cout << "1NN" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[0] <<" %"<< '\n';
    std::cout << "Tasa reduccion: "<<tasa_reduccion[0] <<" %"<< '\n';
    std::cout << "Agregado: "<< agregado[0] << '\n';
    std::cout << "Tiempo: "<< tiempo[0] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';
    std::cout << "RELIEF" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[1]<<" %" << '\n';
    std::cout << "Tasa reduccion: " <<tasa_reduccion[1] <<" %"<<'\n';
    std::cout << "Agregado: "<< agregado[1] << '\n';
    std::cout << "Tiempo: "<< tiempo[1] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';
    std::cout << "ES" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[2] <<" %"<< '\n';
    std::cout << "Tasa reduccion: " <<tasa_reduccion[2] <<" %"<<'\n';
    std::cout << "Agregado: "<< agregado[2] << '\n';
    std::cout << "Tiempo: "<< tiempo[2] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';
    std::cout << "ILS" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[3] <<" %"<< '\n';
    std::cout << "Tasa reduccion: " <<tasa_reduccion[3] <<" %"<<'\n';
    std::cout << "Agregado: "<< agregado[3] << '\n';
    std::cout << "Tiempo: "<< tiempo[3] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';
    std::cout << "DE-rand" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[4] <<" %"<< '\n';
    std::cout << "Tasa reduccion: " <<tasa_reduccion[4] <<" %"<<'\n';
    std::cout << "Agregado: "<< agregado[4] << '\n';
    std::cout << "Tiempo: "<< tiempo[4] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';
    std::cout << "DE-ctb" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[5] <<" %"<< '\n';
    std::cout << "Tasa reduccion: " <<tasa_reduccion[5] <<" %"<<'\n';
    std::cout << "Agregado: "<< agregado[5] << '\n';
    std::cout << "Tiempo: "<< tiempo[5] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';



    entrenamiento.clear();
    clasificados.clear();
    limpiar(pesos);

  }

  std::cout << "----------------- RESUTADOS GLOBALES -----------------------"<< '\n';
  std::cout << "1NN" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[0] / K <<" %"<< '\n';
  std::cout << "Tasa reduccion: "<<tasa_reduccion_global[0] / K <<" %"<< '\n';
  std::cout << "Agregado: "<< agregado_global[0] / K << '\n';
  std::cout << "Tiempo: "<< tiempo_global[0] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';
  std::cout << "RELIEF" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[1] / K<<" %" << '\n';
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[1] / K <<" %"<<'\n';
  std::cout << "Agregado: "<< agregado_global[1] / K<< '\n';
  std::cout << "Tiempo: "<< tiempo_global[1] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';
  std::cout << "ES" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[2] / K <<" %"<< '\n';
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[2] / K <<" %"<<'\n';
  std::cout << "Agregado: "<< agregado_global[2] / K << '\n';
  std::cout << "Tiempo: "<< tiempo_global[2] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';
  std::cout << "ILS" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[3] / K <<" %"<< '\n';
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[3] / K <<" %"<<'\n';
  std::cout << "Agregado: "<< agregado_global[3] / K << '\n';
  std::cout << "Tiempo: "<< tiempo_global[3] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';
  std::cout << "DE-rand" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[4] / K <<" %"<< '\n';
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[4] / K <<" %"<<'\n';
  std::cout << "Agregado: "<< agregado_global[4] / K << '\n';
  std::cout << "Tiempo: "<< tiempo_global[4] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';
  std::cout << "DE-ctb" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[5] / K <<" %"<< '\n';
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[5] / K <<" %"<<'\n';
  std::cout << "Agregado: "<< agregado_global[5] / K << '\n';
  std::cout << "Tiempo: "<< tiempo_global[5] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';


}

int main(int argc, char const *argv[]){

  if (argc > 1)
    seed = stoi(argv[1]);

  gen = default_random_engine(seed);

  //practica1("DATA/ionosphere_normalizados.csv");

  //practica1("DATA/colposcopy_normalizados.csv");

  practica1("DATA/texture_normalizados.csv");

}
