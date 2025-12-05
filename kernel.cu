#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include "preprocesamiento.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace std;

__global__ void MultiMat(const double* A, const double* pesos, const double* sesgo, double* salida, int filas, int dentrada, int dsalida) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila >= filas) {
        return;
    }

    for (int i = 0; i < dsalida; i++) {
        double acumulador = 0.0;
        const double* fila_pesos = &pesos[i * dentrada];
        const double* fila_A = &A[fila * dentrada];
        for (int j = 0; j < dentrada; j++) {
            acumulador = acumulador + fila_A[j] * fila_pesos[j];
        }

        salida[fila * dsalida + i] = acumulador + (sesgo ? sesgo[i] : 0.0);
    }
}

__global__ void PositionalEncoding(const double* X, const double* PE, double* Xpe, int filas, int modelo) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila >= filas) {
        return;
    }
    const double* fila_X = &X[fila * modelo];
    const double* fila_PE = &PE[fila * modelo];
    double* fila_salida = &Xpe[fila * modelo];

    for (int j = 0; j < modelo; j++) {
        fila_salida[j] = fila_X[j] + fila_PE[j];
    }
}

__global__ void Atencion(const double* Q, const double* K, const double* V, double* O, int longitudsecuencia, int k, int mascara_causal) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila >= longitudsecuencia) {
        return;
    }
    double maximo = -1e300;
    const double escala = 1.0 / sqrt((double)k);
    double denominador = 0.0;


    for (int i = 0; i < longitudsecuencia; i++) {

        if (mascara_causal && i > fila) {
            continue;
        }
        double puntaje = 0.0;

        for (int j = 0; j < k; j++) {
            puntaje = puntaje + Q[fila * k + j] * K[i * k + j];
        }
        puntaje = puntaje * escala;
        if (puntaje > maximo) {
            maximo = puntaje;
        }
    }
    if (maximo == -1e300) {
        maximo = 0.0;
    }

    for (int i = 0; i < longitudsecuencia; i++) {
        double puntaje = -1e9;
        if (!(mascara_causal && i > fila)) {
            puntaje = 0.0;
            for (int j = 0; j < k; j++) {
                puntaje = puntaje + Q[fila * k + j] * K[i * k + j];
            }
            puntaje = (puntaje * escala) - maximo;
        }
        denominador = denominador + exp(puntaje);
    }
    denominador = fmax(denominador, 1e-12);

    for (int j = 0; j < k; j++) {
        O[fila * k + j] = 0.0;
    }
    for (int i = 0; i < longitudsecuencia; i++) {
        double puntaje = -1e9;
        if (!(mascara_causal && i > fila)) {
            puntaje = 0.0;
            for (int j = 0; j < k; j++) {
                puntaje = puntaje + Q[fila * k + j] * K[i * k + j];
            }
            puntaje = (puntaje * escala) - maximo;
        }
        double peso = exp(puntaje) / denominador;
        for (int j = 0; j < k; j++) {
            O[fila * k + j] = O[fila * k + j] + peso * V[i * k + j];
        }
    }
}

__global__ void AR(const double* A, const double* B, double* O, int filas, int modelo) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila >= filas) {
        return;
    }
    for (int j = 0; j < modelo; j++) {
        int w = fila * modelo + j;
        O[w] = A[fila * modelo + j] + B[fila * modelo + j];
    }
}

__global__ void linearHeadKernel(const double* X, const double* cabeza, double sesgo, double* prediccion, int filas, int modelo) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila >= filas) {
        return;
    }
    const double* fila_X = &X[fila * modelo];
    double suma = 0.0;

    for (int j = 0; j < modelo; j++) {
        suma = suma + fila_X[j] * cabeza[j];
    }

    prediccion[fila] = suma + sesgo;
}

__global__ void squaredErrorKernel(const double* prediccion, const double* objetivo, double* cuadrados, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double d = prediccion[i] - objetivo[i];
        cuadrados[i] = d * d;
    }
}

double RMSE(const double* d_prediccion, const double* d_objetivo, float n) {
    double* d_cuadrados = 0;
    cudaMalloc(&d_cuadrados, n * sizeof(double));

    int bloque = 256;
    int malla = (n + bloque - 1) / bloque;

    squaredErrorKernel << <malla, bloque >> > (d_prediccion, d_objetivo, d_cuadrados, n);
    cudaDeviceSynchronize();

    thrust::device_ptr<double> ptr(d_cuadrados);
    double suma = thrust::reduce(ptr, ptr + n);

    cudaFree(d_cuadrados);

    //cout << sqrt(suma / n) << endl;

    return sqrt(suma / n);
}

__global__ void headGradKernel(const double* X, const double* prediccion, const double* objetivo, double* gradw, int filas, int modelo) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= modelo) {
        return;
    }

    double g = 0.0;

    for (int fila = 0; fila < filas; fila++) {
        double error = prediccion[fila] - objetivo[fila];
        g = g + error * X[fila * modelo + j];
    }

    gradw[j] = (2.0 / (double)filas) * g;
}

__global__ void headUpdateKernel(double* head, const double* gradw, double* sesgo, double gradb, double tasa, int modelo) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < modelo) {
        head[j] = head[j] - tasa * gradw[j];
    }

    if (j == 0) {
        *sesgo = *sesgo - tasa * gradb;
    }
}

static vector<double> crearCodificacionPosicional(int longitud_secuencia, int modelo) {
    vector<double> pe(longitud_secuencia * modelo, 0.0);

    for (int pos = 0; pos < longitud_secuencia; pos++) {

        for (int i = 0; i < modelo; i++) {
            double divisor = pow(10000.0, (2.0 * (i / 2)) / modelo);
            if (i % 2 == 0) {
                pe[pos * modelo + i] = sin(pos / divisor);
            }

            else {
                pe[pos * modelo + i] = cos(pos / divisor);
            }
        }
    }
    return pe;
}

void guardarVector(const double* d_vec, int tamano, const string& ruta) {
    vector<double> h(tamano);
    cudaMemcpy(h.data(), d_vec, tamano * sizeof(double), cudaMemcpyDeviceToHost);
    ofstream f(ruta, ios::binary);
    f.write(reinterpret_cast<const char*>(h.data()), tamano * sizeof(double));
    f.close();
}

void cargarVector(double* d_vec, int tamano, const string& ruta) {
    vector<double> h(tamano);
    ifstream f(ruta, ios::binary);

    f.read(reinterpret_cast<char*>(h.data()), tamano * sizeof(double));

    f.close();
    cudaMemcpy(d_vec, h.data(), tamano * sizeof(double), cudaMemcpyHostToDevice);
}

double forwardEvaluate(const double* entradas, int longsec, int modelo, const double* pe, const double* WQ, const double* WK, const double* WV, const double* head, double bias, const double* objetivo, int causal, double* Xpe, double* Q, double* K, double* V, double* contexto, double* caracteristicas, double* prediccion) {

    int bloque = 256;
    int malla = (longsec + bloque - 1) / bloque;

    PositionalEncoding << <malla, bloque >> > (entradas, pe, Xpe, longsec, modelo);
    cudaDeviceSynchronize();

    MultiMat << <malla, bloque >> > (Xpe, WQ, nullptr, Q, longsec, modelo, modelo);
    MultiMat << <malla, bloque >> > (Xpe, WK, nullptr, K, longsec, modelo, modelo);
    MultiMat << <malla, bloque >> > (Xpe, WV, nullptr, V, longsec, modelo, modelo);
    cudaDeviceSynchronize();

    Atencion << <malla, bloque >> > (Q, K, V, contexto, longsec, modelo, causal);
    cudaDeviceSynchronize();

    AR << <malla, bloque >> > (contexto, Xpe, caracteristicas, longsec, modelo);
    cudaDeviceSynchronize();

    linearHeadKernel << <malla, bloque >> > (caracteristicas, head, bias, prediccion, longsec, modelo);
    cudaDeviceSynchronize();

    if (objetivo) {
        return RMSE(prediccion, objetivo, longsec);
    }

    return 0.0;
}

void guardarCSVPredicciones(const vector<Data>& datos_validacion, const vector<pair<vector<double>, double>>& ventanas_validacion, const double* d_prediccion_val, const double* d_objetivo_val, int longitud_validacion) {

    vector<double> pred(longitud_validacion);
    vector<double> eti(longitud_validacion);

    cudaMemcpy(pred.data(), d_prediccion_val, longitud_validacion * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(eti.data(), d_objetivo_val, longitud_validacion * sizeof(double), cudaMemcpyDeviceToHost);

    int tam = ventanas_validacion.front().first.size();

    ofstream csv("prediccion.csv");

    csv << "fecha,precio,predicho\n";
    for (int i = 0; i < longitud_validacion; i++) {
        int j = i + tam;
        string fecha = "";

        if (j >= 0 && j < (int)datos_validacion.size()) {

            fecha = datos_validacion[j].Fecha;
        }
        //cout << fecha <<  " " << eti[i] << endl;

        csv << fecha << "," << eti[i] << "," << pred[i] << "\n";
    }
    csv.close();
}

void trainTransformerHead(const double* entradas, const double* objetivo, int longsec, int modelo, const double* entradasval, const double* objetivoval, int longval, double* WQ, double* WK, double* WV, double* head, double* bias, const double* entrenamiento, const double* validacion, double* Xpe, double* Q, double* K, double* V, double* contexto, double* caracteristicas, double* prediccion, double* Xpeval, double* Qval, double* Kval, double* Vval, double* contextoval, double* caracteristicasval, double* prediccionval, int epocas, double tasa, int causal) {

    int bloque = 256;
    int malla = (modelo + bloque - 1) / bloque;

    vector<double> biashost(1);
    cudaMemcpy(biashost.data(), bias, sizeof(double), cudaMemcpyDeviceToHost);
    double sesgo = biashost[0];

    double* grad = nullptr;
    cudaMalloc(&grad, modelo * sizeof(double));

    for (int epoca = 0; epoca < epocas; epoca++) {
        double rmseentrenamiento = forwardEvaluate(entradas, longsec, modelo, entrenamiento, WQ, WK, WV, head, sesgo, objetivo, causal, Xpe, Q, K, V, contexto, caracteristicas, prediccion);

        headGradKernel << <malla, bloque >> > (caracteristicas, prediccion, objetivo, grad, longsec, modelo);
        cudaDeviceSynchronize();

        vector<double> predhost(longsec);
        vector<double> objhost(longsec);
        cudaMemcpy(predhost.data(), prediccion, longsec * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(objhost.data(), objetivo, longsec * sizeof(double), cudaMemcpyDeviceToHost);
        double suma = 0.0;
        for (int i = 0; i < longsec; i++) {
            suma = suma + (predhost[i] - objhost[i]);
        }
        double gradbias = (2.0 / longsec) * suma;

        headUpdateKernel << <malla, bloque >> > (head, grad, bias, gradbias, tasa, modelo);
        cudaDeviceSynchronize();

        cudaMemcpy(&sesgo, bias, sizeof(double), cudaMemcpyDeviceToHost);

        double rmsevalidacion = forwardEvaluate(entradasval, longval, modelo, validacion, WQ, WK, WV, head, sesgo, objetivoval, causal, Xpeval, Qval, Kval, Vval, contextoval, caracteristicasval, prediccionval);

        cout << "Epoca " << epoca << endl;
        cout << "RMSE: " << endl;
        cout << " Entrenamiento: " << rmseentrenamiento << endl;
        cout << " Validacion: " << rmsevalidacion << endl;
    }

    cudaFree(grad);
}

int main() {
    cudaSetDevice(0);

    string entrenamiento = "SP50020152020.csv";
    string validacion = "SP500.csv";

    vector<Data> ENTRENAMIENTO = LeerCSV(entrenamiento);
    vector<Data> VALIDACION = LeerCSV(validacion);
    Normalizar(ENTRENAMIENTO);
    Normalizar(VALIDACION);

    double* entradas = 0;
    double* etiquetas = 0;
    double* entradasV = 0;
    double* etiquetasV = 0;
    double* PEEntrenamieto = 0;
    double* PEValidacion = 0;

    int size = 5;
    double modelo = size;

    double* wq = 0;
    double* wk = 0;
    double* wv = 0;

    vector<pair<vector<double>, double>> VE = Ventana(ENTRENAMIENTO, size);
    vector<pair<vector<double>, double>> VV = Ventana(VALIDACION, size);

    int SizeEntrenamiento = (int)VE.size();
    int SizeValidacion = (int)VV.size();

    vector<double> entradasEntrenamiento(SizeEntrenamiento * modelo);
    vector<double> ETIQUETAS(SizeEntrenamiento);

    for (int i = 0; i < SizeEntrenamiento; i++) {

        for (int j = 0; j < modelo; j++) {

            int w = i * modelo + j;
            entradasEntrenamiento[w] = VE[i].first[j];
        }
        ETIQUETAS[i] = VE[i].second;
    }

    vector<double> entradasValidacion(SizeValidacion * modelo);
    vector<double> ETIQUETASV(SizeValidacion);
    for (int i = 0; i < SizeValidacion; i++) {
        for (int j = 0; j < modelo; j++) {
            int w = i * modelo + j;
            entradasValidacion[w] = VV[i].first[j];
        }
        ETIQUETASV[i] = VV[i].second;
    }


    cudaMalloc(&entradas, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&etiquetas, SizeEntrenamiento * sizeof(double));
    cudaMalloc(&entradasV, SizeValidacion * modelo * sizeof(double));
    cudaMalloc(&etiquetasV, SizeValidacion * sizeof(double));
    cudaMemcpy(entradas, entradasEntrenamiento.data(), SizeEntrenamiento * modelo * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(etiquetas, ETIQUETAS.data(), SizeEntrenamiento * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(entradasV, entradasValidacion.data(), SizeValidacion * modelo * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(etiquetasV, ETIQUETASV.data(), SizeValidacion * sizeof(double), cudaMemcpyHostToDevice);

    vector<double> PET = crearCodificacionPosicional(SizeEntrenamiento, modelo);
    vector<double> PEV = crearCodificacionPosicional(SizeValidacion, modelo);

    cudaMalloc(&PEEntrenamieto, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&PEValidacion, SizeValidacion * modelo * sizeof(double));
    cudaMemcpy(PEEntrenamieto, PET.data(), SizeEntrenamiento * modelo * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(PEValidacion, PEV.data(), SizeValidacion * modelo * sizeof(double), cudaMemcpyHostToDevice);

    auto M = [](vector<double>& v, double escala) {std::mt19937 rng(42); std::uniform_real_distribution<double> dist(-escala, escala);
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = dist(rng);
    }
        };

    vector<double> WQ(modelo * modelo);
    vector<double> WK(modelo * modelo);
    vector<double> WV(modelo * modelo);
    M(WQ, 1.0 / sqrt(modelo));
    M(WK, 1.0 / sqrt(modelo));
    M(WV, 1.0 / sqrt(modelo));

    cudaMalloc(&wq, modelo * modelo * sizeof(double));
    cudaMalloc(&wk, modelo * modelo * sizeof(double));
    cudaMalloc(&wv, modelo * modelo * sizeof(double));
    cudaMemcpy(wq, WQ.data(), modelo * modelo * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(wk, WK.data(), modelo * modelo * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(wv, WV.data(), modelo * modelo * sizeof(double), cudaMemcpyHostToDevice);

    vector<double> head(modelo, 0.0);
    double bias = 0.0;
    double* head2 = 0;
    double* bias2 = 0;
    cudaMalloc(&head2, modelo * sizeof(double));
    cudaMalloc(&bias2, sizeof(double));
    cudaMemcpy(head2, head.data(), modelo * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bias2, &bias, sizeof(double), cudaMemcpyHostToDevice);

    double* X = 0;
    double* Q = 0;
    double* K = 0;
    double* V = 0;
    double* contexto = 0;
    double* caracteristicas = 0;
    double* d_prediccion = 0;

    double* XV = 0;
    double* QV = 0;
    double* KV = 0;
    double* Vval = 0;
    double* contextoV = 0;
    double* caracteristicasv = 0;
    double* prediccionV = 0;


    cudaMalloc(&X, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&Q, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&K, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&V, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&contexto, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&caracteristicas, SizeEntrenamiento * modelo * sizeof(double));
    cudaMalloc(&d_prediccion, SizeEntrenamiento * sizeof(double));
    cudaMalloc(&XV, SizeValidacion * modelo * sizeof(double));
    cudaMalloc(&QV, SizeValidacion * modelo * sizeof(double));
    cudaMalloc(&KV, SizeValidacion * modelo * sizeof(double));
    cudaMalloc(&Vval, SizeValidacion * modelo * sizeof(double));
    cudaMalloc(&contextoV, SizeValidacion * modelo * sizeof(double));
    cudaMalloc(&caracteristicasv, SizeValidacion * modelo * sizeof(double));
    cudaMalloc(&prediccionV, SizeValidacion * sizeof(double));

    std::ifstream f1("cabeza.bin", std::ios::binary);
    if (f1.good()) {
        cargarVector(head2, modelo, "cabeza.bin");
    }

    std::ifstream f2("sesgo_cabeza.bin", std::ios::binary);
    if (f2.good()) {
        cargarVector(bias2, 1, "sesgo_cabeza.bin");
    }

    int epocas = 20;
    double tasa = 0.01;

    trainTransformerHead(entradas, etiquetas, SizeEntrenamiento, modelo, entradasV, etiquetasV, SizeValidacion, wq, wk, wv, head2, bias2, PEEntrenamieto, PEValidacion, X, Q, K, V, contexto, caracteristicas, d_prediccion, XV, QV, KV, Vval, contextoV, caracteristicasv, prediccionV, epocas, tasa, 1);

    double bias3 = 0.0;
    cudaMemcpy(&bias3, bias2, sizeof(double), cudaMemcpyDeviceToHost);


    forwardEvaluate(entradasV, SizeValidacion, modelo, PEValidacion, wq, wk, wv, head2, bias3, etiquetasV, 1, XV, QV, KV, Vval, contextoV, caracteristicasv, prediccionV);
    guardarCSVPredicciones(VALIDACION, VV, prediccionV, etiquetasV, SizeValidacion);

    guardarVector(head2, modelo, "cabeza.bin");
    guardarVector(bias2, 1, "sesgo_cabeza.bin");

    int rc = system("python \"Resultado.py\"");

    return 0;
}