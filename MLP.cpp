#include"MLP.h"

double frand() {
	return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}
//konstruktor, ustawienie biasu, rozszerzenie wag o bias, wygenerowanie wag
Perceptron::Perceptron(int inputs, double bias) {
	this->bias = bias;
	weights.resize(inputs + 1);
	generate(weights.begin(), weights.end(), frand);
}

double Perceptron::run(vector<double>x) {
	x.push_back(bias);
	double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);
	return sigmoid(sum);
}

void Perceptron::set_weights(vector<double>w_init) {
	weights = w_init;
}

double Perceptron::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

MultiLayerPerceptron::MultiLayerPerceptron(vector<int>layers, double bias, double eta) {
	this->layers = layers;
	this->bias = bias;
	this->eta = eta;

	//np. layers = {3,2,1};
	for (int i = 0; i < layers.size(); i++)
	{
		//do kazdej warstwy dodajemy wektor wartosci (np layers[2]==2, czyli dwa wejscia rowne 0,0)
		values.push_back(vector<double>(layers[i], 0.0));
		d.push_back(vector<double>(layers[i], 0.0));
		//do kazdej warstwy dodajemy wektor neuronow
		network.push_back(vector<Perceptron>());
		if (i > 0)//
			for (int j = 0; j < layers[i]; j++)
			{
				network[i].push_back(Perceptron(layers[i - 1], bias));
			}
	}
}
//			warstwa0			warstwa1	warstwa2
// ({{ {2,1},{1,1},{3,2} }, {{2,1},{1,1}}, {{1,1}} })
void MultiLayerPerceptron::set_weights(vector<vector<vector<double>>>w_init) {
	for (int i = 0; i < w_init.size(); i++)
	{
		for (int j = 0; j < w_init[i].size(); j++)
		{
			network[i+1][j].set_weights(w_init[i][j]);
		}
	}
}
void MultiLayerPerceptron::print_weights() {
	cout << endl;
	for (int i = 1; i < layers.size(); i++) {
		for (int j = 0; j < layers[i]; j++) {
			cout << "Layer " << i + 1 << " Neuron " << j << ": ";
			for (auto& it : network[i][j].weights)
				cout << it << " ";
			cout << endl;
		}
	}
	cout << endl;
}

vector<double>MultiLayerPerceptron::run(vector<double>x) {

	values[0] = x;
	for (int i = 1; i < network.size(); i++)
	{
		for (int j = 0; j < layers[i]; j++)
		{
			values[i][j] = network[i][j].run(values[i - 1]);
		}
	}
	return values.back();
}


double MultiLayerPerceptron::bp(vector<double>x, vector<double>y) {
	//1 feed a sample to the network
	vector<double> output = run(x);
	
	//2 calculate mse
	double MSE = 0.0;
	vector<double>error;
	for (int i = 0; i < y.size();i++) {
		error.push_back(y[i] - output[i]);
		MSE += error[i] * error[i];
	}
	MSE /= layers.back();

	//3 calculate output error terms
	for (int i = 0; i < output.size(); i++)
	{
		d.back()[i]= output[i] * (1 - output[i]) * (error[i]);
	}
	//4 calculate the error term of each unit on each layer
	for (int i = network.size() - 2; i > 0; i--)
		for (int h = 0; h < network[i].size(); h++) {
			double fwd_error = 0.0;
			for (int k = 0; k < layers[i + 1]; k++)
				//fill in the blank
				//i- ilosc warstw, h-ilosc neuronow na warstwe, k- ilosc wejsc na neuron
				fwd_error += network[i + 1][k].weights[h] * d[i + 1][k];
			d[i][h] = values[i][h] * (1 - values[i][h]) * fwd_error;
		}

	//5,6 calculate deltas and update the weights
	for (int i = 1; i < network.size(); i++) {
		for (int j = 0; j < layers[i]; j++) {
			for (int k = 0; k < layers[i - 1] + 1; k++) {
				double delta_weight;
				if (k == layers[i - 1])
					delta_weight = eta * d[i][j] * bias;
				else
					delta_weight = eta * d[i][j] * values[i-1][k];
				network[i][j].weights[k] += delta_weight;
			}
		}
	}
	return MSE;
}
