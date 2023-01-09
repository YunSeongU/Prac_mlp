#pragma once
#include <vector>
#include <cmath>

using std::vector;
using std::pair;

class layer {
private:
	vector<vector<double>> x_;

	vector<vector<double>> w_;
	vector<double> b_;

	vector<vector<double>> dw;
	vector<double> db;

public:

	layer(vector<vector<double>> w, vector<double> b);

};