#include<iostream>
#include"MLP.h"

int main() {
	srand(time(NULL));
	rand();

	Perceptron* p = new Perceptron(2);
	p->set_weights({ 10,10,-15 });//and gate

	cout << "Gate: " << endl;
	cout << p->run({ 0,0 }) << endl;
	cout << p->run({ 1,0 }) << endl;
	cout << p->run({ 0,1 }) << endl;
	cout << p->run({ 1,1 }) << endl;


	MultiLayerPerceptron mlp = MultiLayerPerceptron({ 2,2,1 });
	mlp.set_weights({ {{-10,-10,15},{15,15,-10}},{{10,10,-15} }});
	mlp.print_weights();

	cout << "Gate: " << endl;
	cout << mlp.run({ 0,0 })[0] << endl;
	cout << mlp.run({ 1,0 })[0] << endl;
	cout << mlp.run({ 0,1 })[0] << endl;
	cout << mlp.run({ 1,1 })[0] << endl;

	cout << "LEARNING!!!" << endl;
	mlp = MultiLayerPerceptron({ 2,2,1 });
	double MSE;
	for (size_t i = 0; i < 3000; i++)
	{
		MSE = 0.0;
		MSE += mlp.bp({ 0,0 }, { 0 });
		MSE += mlp.bp({ 0,1 }, { 1 });
		MSE += mlp.bp({ 1,0 }, { 1 });
		MSE += mlp.bp({ 1,1 }, { 0 });
		MSE /= 4.0;
		if (i % 100 == 0)
			cout << "MSE = " << MSE << endl;
	}

	cout << "TRAINED WEIGHTS: " << endl;
	mlp.print_weights();

	cout << "TEST: " << endl;
	cout << mlp.run({ 0,0 })[0] << endl;
	cout << mlp.run({ 1,0 })[0] << endl;
	cout << mlp.run({ 0,1 })[0] << endl;
	cout << mlp.run({ 1,1 })[0] << endl;
}