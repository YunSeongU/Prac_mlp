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
void matPrint(vector<vector<double>> A);
vector <vector <double>> T(vector<vector<double>>A);
vector<vector<double>> addb(vector<vector<double>> arr1, vector<vector<double>> arr2);
vector <double> sum_axis(vector<vector<double>> A, int axis);

vector<vector<double>> mat_element_mul(vector<vector<double>> A, vector<vector<double>> B);

vector<vector<double>> mat_element_sub(vector<vector<double>> A, vector<vector<double>> B);
vector<vector<double>> mat_sigmoid(vector<vector<double>> A);

vector<double> mat_square(vector<vector<double>> A);
vector<vector<double>> mse_grad(vector<vector<double>> A);
vector<vector<double>> make_bias_mat(int row_size, vector<double> A);
vector<vector<double>> mat_scalr_mul(double v,vector<vector<double>> A);
vector<double> vec_elem_sub(vector<double> A, vector<double> B);
vector<double> vec_scalr_mul(double v, vector<double> A);
