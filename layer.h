#pragma once
#include <vector>
#include <map>
#include <cmath>

using std::vector;
using std::pair;
using std::map;
using std::string;


class layer {

public:
	

	vector<vector<double>> w1_;
	vector<vector<double>> w2_;
	vector<double> b1_;
	vector<double> b2_;

	double loss_;

	layer(int input_size, int hidden_size, int output_size);
	vector<vector<double>> sigmoid(vector<vector<double>> &x);
	vector < vector < double >> sigmoid_grad(vector<vector<double>> &x);
	vector<vector<double>> predict(vector<vector<double>> &x);
	vector<vector <double>> mse_loss(vector<vector<double>> &out, vector<vector<double>> &y);
	pair<
		map <string, vector<vector<double>>>,
		map <string, vector <double>>
	> process(vector<vector<double>> &x, vector<vector<double>> &y);



	vector<vector<double>> relu( vector<vector<double>> &x);
	vector < vector < double >> relu_grad(vector<vector<double>> &x);
	void show_weight_bias();
	vector<vector<double>> tanh(vector<vector<double>> &x);
	vector<vector<double>> tanh_grad(vector<vector<double>> &x);
	void bce_loss(vector<vector<double>> &out, vector<vector<double>> &y);
};