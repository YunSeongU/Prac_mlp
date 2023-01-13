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

vector <vector <double>> matmul(const vector<vector<double>>&A, const vector<vector<double>>&B);

void matPrint(vector<vector<double>> A);
vector <vector <double>> T(vector<vector<double>>&A);

vector <double> sum_axis(const vector<vector<double>>& A, int axis);

vector<vector<double>> mat_element_mul(const vector<vector<double>>& A, const vector<vector<double>>& B);

vector<vector<double>> mat_element_sub(const vector<vector<double>>& A, const vector<vector<double>>& B);


vector<double> mat_square(vector<vector<double>> &A);
vector<vector<double>> mse_grad(vector<vector<double>> A);

vector<vector<double>> mat_scalr_mul(double v,vector<vector<double>> &A);
vector<double> vec_elem_sub(vector<double> A, vector<double> &B);
vector<double> vec_scalr_mul(double v, vector<double> &A);
vector<vector<double>> weight_init(int r_size, int c_size);
void vecPrint(vector<double> A);
vector<vector<double>> add_bias(vector<vector<double>>& arr1, vector<double>& arr2);
