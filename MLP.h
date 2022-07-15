#pragma once
#include<algorithm>
#include<vector>
#include<iostream>
#include<random>
#include<numeric>
#include<cmath>
#include<time.h>

using namespace std;

class Perceptron {
public:
	vector<double>weights;
	double bias;
	Perceptron(int inputs, double bias = 1.0);
	double run(vector<double>x);
	void set_weights(vector<double>w_init);
	double sigmoid(double x);
};

class MultiLayerPerceptron {
public:
	MultiLayerPerceptron(vector<int>layers, double bias = 1.0, double eta = 0.5);
	void set_weights(vector < vector<vector<double>>>w_init); //vector1 - liczba warstw, vector2 - liczba neuronow w warstwie, vector3 - liczba wejsc na neuron
	void print_weights();
	vector<double>run(vector<double>x);
	double bp(vector<double>x, vector<double>y);

	//layers {10,5,3,1} == 10 wejsc w warstwie 0, 5 neuronow w warswie 1, 3 neurony w warstwie 2 i 1 w ostaniej
	vector<int>layers; //reprezentuje liczbe neuronow na kazda warstwe, włącznie z warstwą wejściową (która nie ma neuronów ale liczba ta oznacza liczbę wejść)
	double bias;
	double eta; // współczynnik uczenia
	vector<vector<Perceptron>>network; //wektor wektorów perceptronów
	vector<vector<double>>values; // wartosci wyjść neuronów, ilość wyjść == ilość perceptronów
	vector<vector<double>>d; //wartości błędu neuronów

};