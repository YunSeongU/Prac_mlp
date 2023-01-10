#include "matrix.h"
#include "layer.h"


layer::layer(int input_size,int hidden_size ,int output_size) {
	w1_ = vector<vector<double>>(input_size, vector <double>(hidden_size, 0.5));
	b1_ = vector<double>(hidden_size, 0);
	w2_ = vector<vector<double>>(hidden_size, vector <double>(output_size, 0.5));
	b2_ = vector<double>(output_size, 0);
}

vector<vector<double>> layer::sigmoid(vector<vector<double>> x) {
	return mat_sigmoid(x);
}


vector<vector<double>> layer::predict(vector<vector<double>> x) {
	vector<vector<double>> affine1, active1, affine2, out,b1,b2;

	affine1 = matmul(x, w1_);
	b1 = make_bias_mat(affine1.size(), b1_);
	affine1 = addb(affine1, b1);
	active1 = sigmoid(affine1);



	affine2 = matmul(active1, w2_);
	b2 = make_bias_mat(affine2.size(), b2_);
	affine2 = addb(affine2, b2);
	out = sigmoid(affine2);

	return out;
}

vector<vector <double>> layer::loss(vector<vector<double>> out, vector<vector<double>> y) {

	vector<vector<double>> diff = mat_element_sub(out, y);
	vector<double> diff_sq = mat_square(diff);
	int size = diff_sq.size();
	double sum = 0;
	for (int i = 0; i < size; i++) {
		sum += diff_sq[i];
	}
	loss_ = sum / size;
	return diff;
}

pair<
	map <string, vector<vector<double>>>,
	map <string, vector <double>>
> layer::process(vector<vector<double>> x, vector<vector<double>> y) {

	vector<vector<double>> affine1, z1, affine2, out,diff,b1,b2;
	vector<vector<double>> dout, daffine1, dz1;
	

	pair<
		map <string, vector<vector<double>>>, 
		map <string, vector <double>> 
	> ret_p_wNb;

	map <string, vector<vector <double>> > grad_w;
	map <string, vector <double>> grad_b;

	//forward
	affine1 = matmul(x, w1_); // 4x2 2x10 = 4x10
	b1 = make_bias_mat(affine1.size(), b1_);//1,10-->4x10
	affine1 = addb(affine1, b1);//4x10
	z1 = relu(affine1); //4x10
	affine2 = matmul(z1, w2_);// 4x10 10x1 = 4x1
	b2 = make_bias_mat(affine2.size(), b2_);//1x1 --> 4x1
	affine2 = addb(affine2, b2);// 4x1
	out = relu(affine2);// 4x1

	//backward
	diff = loss(out, y);
	dout = mse_grad(diff);
	grad_w["w2"] = matmul(T(z1), dout);
	grad_b["b2"] = sum_axis(dout, 0);

	daffine1 = matmul(dout, T(w2_));
	
	dz1 = mat_element_mul(relu_grad(affine1), daffine1);
	grad_w["w1"] = matmul(T(x), dz1);
	grad_b["b1"] = sum_axis(dz1, 0);
	ret_p_wNb.first = grad_w;
	ret_p_wNb.second = grad_b;

	return ret_p_wNb;;
}


vector <vector < double >> layer::sigmoid_grad(vector<vector<double>> x) {
	vector<vector<double>> vec_sig = mat_sigmoid(x);
	vector<vector<double>> minus_sig;
	vector<double> tmp;
	int row = x.size(), col = x[0].size();
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			tmp.push_back(1.0 - vec_sig[r][c]);
		}
		minus_sig.push_back(tmp);
	}
	return mat_element_mul(vec_sig, minus_sig);

}

vector<vector<double>> layer::relu(vector<vector<double>> x) {
	vector<vector<double>> retm;
	vector<double> tmp;
	int row = x.size(); int col = x[0].size();
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			if (x[r][c] > 0) {
				tmp.push_back(x[r][c]);
			}
			else {
				tmp.push_back(0.0);
			}
		}
		retm.push_back(tmp);
	}
	return retm;
}

vector < vector < double >> layer::relu_grad(vector<vector<double>> x) {
	vector<vector<double>> retm;
	vector<double> tmp;
	int row = x.size(); int col = x[0].size();
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			if (x[r][c] > 0) {
				tmp.push_back(1.0);
			}
			else {
				tmp.push_back(0.0);
			}
		}
		retm.push_back(tmp);
	}
	return retm;
}
