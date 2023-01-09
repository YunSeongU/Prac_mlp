#pragma once
#include<vector>
#include<iostream>

using std::vector;
//class matrix {
//public:
//	//±¸Çö : matmul, transpose, reshape
//	vector <vector <double>> matmul(vector<vector<double>>A, vector<vector<double>>B);
//	void matPrint(vector<vector<double>>&A);
//	vector <vector <double>> T(vector<vector<double>>A);
//
//};

vector <vector <double>> matmul(vector<vector<double>>A, vector<vector<double>>B);
void matPrint(vector<vector<double>>&A);
vector <vector <double>> T(vector<vector<double>>A);
vector<vector<double>> addb(vector<vector<double>> arr1, vector<double> arr2);
vector <double> sum_axis(vector<vector<double>> A,int axis);

vector<vector<double>> backPorp_element_mul(vector<vector<double>> A, vector<vector<double>> B);

vector<vector<double>> mse_matsub(vector<vector<double>> A, vector<vector<double>> B);
