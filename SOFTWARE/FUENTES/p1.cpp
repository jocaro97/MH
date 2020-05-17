
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

using namespace std;

const int K = 5; //< Numero de particiones
const double alpha = 0.5;

// Seed for randomness
int seed = 62894; // otra semilla 23456789;

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

void busquedaLocal(const vector<Ejemplo>& entrenamiento, vector<double>& pesos){
  normal_distribution<double> normal(0.0, 0.3);
  uniform_real_distribution<double> unirforme(0.0,1.0);

  double mejor_obejtivo;
  int n = pesos.size();
  int iteracion = 0;
  int num_vecinos = 0;

  std::vector<string> clasificados;
  bool mejora = false;

  std::vector<int> indices;
  for(int i = 0; i < n; ++i){
    pesos[i] = unirforme(gen);
    indices.push_back(i);
  }

  for(unsigned k = 0; k < entrenamiento.size(); ++k){
    clasificados.push_back( clasificar1NN(entrenamiento[k], entrenamiento, pesos, k));
  }

  mejor_obejtivo = objetivo(tasaClas(clasificados, entrenamiento), tasaRed(pesos));

  clasificados.clear();

  shuffle(indices.begin(), indices.end(), gen);

  //Busqueda

  while( (iteracion < MAX_ITER) && (num_vecinos < n * MAX_VECINOS) ){
    int j = indices[iteracion % n];

    std::vector<double> pesos_mutados = pesos;

    pesos_mutados[j] += normal(gen);

    if(pesos_mutados[j] < 0) pesos_mutados[j] = 0;
    if(pesos_mutados[j] > 1) pesos_mutados[j] = 1;

    for(unsigned k = 0; k < entrenamiento.size(); ++k){
      clasificados.push_back( clasificar1NN(entrenamiento[k], entrenamiento, pesos_mutados, k));
    }

    double obj_act = objetivo(tasaClas(clasificados, entrenamiento), tasaRed(pesos_mutados));

    clasificados.clear();

    if(obj_act > mejor_obejtivo){
      mejor_obejtivo = obj_act;
      pesos = pesos_mutados;
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

string aleatorio(vector<Ejemplo> entrenamiento, Ejemplo elemento){
  int n = entrenamiento.size();

  return entrenamiento[rand() % n].clase;
}

void practica1(string nombre_archivo){
  vector<Ejemplo> ejemplos;
  vector<Ejemplo> entrenamiento;
  vector<string> clasificados;

  double tasa_clasificacion[4];
  double tasa_reduccion[4];
  double agregado[4];
  double tiempo[4];

//  tasa_clasificacion_global = tasa_reduccion_global = agregado_global = tiempo_global = {0.0,0.0,0.0};

  leer_csv(nombre_archivo, ejemplos);

  cout << "----------------------------------------------------------" << endl;
  cout << "CONJUNTO DE DATOS: " << nombre_archivo << endl;
  cout << "----------------------------------------------------------" << endl << endl;

  vector<double> pesos(ejemplos[0].caract.size(), 0);
  vector<double> pesos1(ejemplos[0].caract.size(), 1);

  double tasa_clasificacion_global[4] = {0.0, 0.0, 0.0, 0.0};
  double tasa_reduccion_global[4] = {0.0, 0.0, 0.0, 0.0};
  double agregado_global[4] = {0.0, 0.0, 0.0, 0.0};
  double tiempo_global[4] = {0.0, 0.0, 0.0, 0.0};

  for(unsigned i = 0; i < K; ++i){
    auto particiones = hacerParticiones(ejemplos);

    auto test = particiones[i];

    for(unsigned j = 0; (j < i) ; ++j)
      entrenamiento.insert(entrenamiento.end(), particiones[j].begin(), particiones[j].end());


    for(unsigned j = i+1; (j < K) ; ++j)
      entrenamiento.insert(entrenamiento.end(), particiones[j].begin(), particiones[j].end());

    //aleatorio
    start_timers();
    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( aleatorio( entrenamiento, test[k]));
    }
    tiempo[3] = elapsed_time();

    tasa_clasificacion[3] = tasaClas(clasificados, test);
    tasa_reduccion[3] = 100;
    agregado[3] = objetivo(tasa_clasificacion[3], tasa_reduccion[3]);

    tasa_clasificacion_global[3] += tasa_clasificacion[3];
    tasa_reduccion_global[3] += tasa_reduccion[3];
    agregado_global[3] += agregado[3];
    tiempo_global[3] += tiempo[3];

    clasificados.clear();


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


    //Busqueda local
    clasificados.clear();

    start_timers();
    busquedaLocal(entrenamiento, pesos);

    for(unsigned k = 0; k < test.size(); ++k){
      clasificados.push_back( clasificar1NN(test[k], entrenamiento, pesos, -1));
    }
    tiempo[2] = elapsed_time();

    tasa_clasificacion[2] = tasaClas(clasificados, test);
    tasa_reduccion[2] = tasaRed(pesos);
    agregado[2] = objetivo(tasa_clasificacion[2], tasa_reduccion[2]);

    tasa_clasificacion_global[2] += tasa_clasificacion[2];
    tasa_reduccion_global[2] += tasa_reduccion[2];
    agregado_global[2] += agregado[2];
    tiempo_global[2] += tiempo[2];


    std::cout << "----------------- Ejecucion "<< i+1 <<" -----------------------"<< '\n';
    std::cout << "Aleatorio" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[3] <<" %"<< '\n';
    std::cout << "Tasa reduccion: "<<tasa_reduccion[3] <<" %"<< '\n';
    std::cout << "Agregado: "<< agregado[3] << '\n';
    std::cout << "Tiempo: "<< tiempo[3] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';
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
    std::cout << "BL" << '\n';
    std::cout << "Tasa clasificacion: "<<tasa_clasificacion[2] <<" %"<< '\n';
    std::cout << "Tasa reduccion: " <<tasa_reduccion[2] <<" %"<<'\n';
    std::cout << "Agregado: "<< agregado[2] << '\n';
    std::cout << "Tiempo: "<< tiempo[2] << '\n';
    std::cout << "-----------------------------------------------------" << '\n';


    entrenamiento.clear();
    clasificados.clear();
    limpiar(pesos);

  }

  std::cout << "----------------- RESUTADOS GLOBALES -----------------------"<< '\n';
  std::cout << "Aleatorio" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[3] / K <<" %"<< '\n';
  std::cout << "Tasa reduccion: "<<tasa_reduccion_global[3] / K <<" %"<< '\n';
  std::cout << "Agregado: "<< agregado_global[3] / K << '\n';
  std::cout << "Tiempo: "<< tiempo_global[3] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';
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
  std::cout << "BL" << '\n';
  std::cout << "Tasa clasificacion: "<<tasa_clasificacion_global[2] / K <<" %"<< '\n';
  std::cout << "Tasa reduccion: " <<tasa_reduccion_global[2] / K <<" %"<<'\n';
  std::cout << "Agregado: "<< agregado_global[2] / K << '\n';
  std::cout << "Tiempo: "<< tiempo_global[2] / K << '\n';
  std::cout << "-----------------------------------------------------" << '\n';


}

int main(int argc, char const *argv[]){

  if (argc > 1)
    seed = stoi(argv[1]);

  gen = default_random_engine(seed);

  practica1("DATA/ionosphere_normalizados.csv");

  practica1("DATA/colposcopy_normalizados.csv");

  practica1("DATA/texture_normalizados.csv");

}
