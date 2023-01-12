#include "matrix.h"
#include "layer.h"

//생성자 정의
layer::layer(int input_size,int hidden_size ,int output_size) {
	w1_ = weight_init(input_size, hidden_size);
	b1_ = vector<double>(hidden_size, 0);
	w2_ = weight_init(hidden_size, output_size);
	b2_ = vector<double>(output_size, 0);
}
//시그모이드 함수 정의
vector<vector<double>> layer::sigmoid(vector<vector<double>> x) {
	return mat_sigmoid(x);
}

//최종 w_,b_를 사용한 가설
vector<vector<double>> layer::predict(vector<vector<double>> x) {
	vector<vector<double>> affine1, z1, affine2, out,b1,b2;

	affine1 = matmul(x, w1_); // 4x2 2x10 = 4x10
	b1 = make_bias_mat(affine1.size(), b1_);//1,10-->4x10
	affine1 = addb(affine1, b1);//4x10 4x10 요소 합
	z1 = relu(affine1); //4x10

	affine2 = matmul(z1, w2_);// 4x10 10x1 = 4x1
	b2 = make_bias_mat(affine2.size(), b2_);//1x1 --> 4x1
	affine2 = addb(affine2, b2);// 4x1 4x1 요소 합
	out = sigmoid(affine2);// 4x1

	return out;
}

vector<vector <double>> layer::mse_loss(vector<vector<double>> out, vector<vector<double>> y) {

	vector<vector<double>> diff = mat_element_sub(out, y);

	vector<double> diff_sq = mat_square(diff);
	int size = diff_sq.size();
	double sum = 0;
	for (int i = 0; i < size; i++) {
		sum += diff_sq[i];
	}
	loss_ = sum / size; //loss값은 클래스 멤버변수에 저장
	return diff;
}


void layer::bce_loss(vector<vector<double>> out, vector<vector<double>> y) {
	int row1 = out.size(), col1 = out[0].size();
	int row2 = y.size(), col2 = y[0].size();
	if ((row1 == row2) and (col1 == col2)) {
		double val = 0;
		for (int r = 0; r < row1; r++) {
			
			for (int c = 0; c < col1; c++) {
		        val += ((-y[r][c]) * std::log(out[r][c])) - ((1 - y[r][c]) * std::log(1 - out[r][c]));

			}

		}
		loss_ = val / row1*col1;

	}
}

//forward와 backward를 진행하는 함수
//w,b미분 정보를 포함한 map을 반환한다
pair<
	map <string, vector<vector<double>>>,
	map <string, vector <double>>
> layer::process(vector<vector<double>> x, vector<vector<double>> y) {

	vector<vector<double>> affine1, z1, affine2, out,diff,b1,b2;
	vector<vector<double>> dout, daffine1, dloss, dz1;
	

	pair<
		map <string, vector<vector<double>>>, 
		map <string, vector <double>> 
	> ret_p_wNb; //가중치와 편향을 받는 페어object

	map <string, vector<vector <double>> > grad_w; //가중치 정보를 받아옴
	map <string, vector <double>> grad_b; //편향 정보를 받아옴

	//forward
	affine1 = matmul(x, w1_); // 4x2 2x10 = 4x10
	b1 = make_bias_mat(affine1.size(), b1_);//1,10-->4x10
	affine1 = addb(affine1, b1);//4x10 4x10 요소 합
	z1 = relu(affine1); //4x10

	affine2 = matmul(z1, w2_);// 4x10 10x1 = 4x1
	b2 = make_bias_mat(affine2.size(), b2_);//1x1 --> 4x1
	affine2 = addb(affine2, b2);// 4x1 4x1 요소 합
	out = sigmoid(affine2);// 4x1

	//backward
	//MSE사용할때 -> relu+relu
	/*
	diff = mse_loss(out, y); // 4x1
	dloss = mse_grad(diff); //4x1
	dout = mat_element_mul(sigmoid_grad(affine2), dloss); // 4x1 4x1 행렬 요소 곱 --> 4x1
	grad_w["w2"] = matmul(T(z1), dout); //10x4 4x1 --> 10x1
	grad_b["b2"] = sum_axis(dout, 0); // 4x1 --> 1x1

	daffine1 = matmul(dout, T(w2_)); // 4x1 1x10 --> 4x10
	dz1 = mat_element_mul(relu_grad(affine1), daffine1); //4x10 4x10 행렬 요소곱 --> 4x10
	grad_w["w1"] = matmul(T(x), dz1); // 2x4 4x10 --> 2x10
	grad_b["b1"] = sum_axis(dz1, 0); // 4x10 --> 1x10
	*/

	//BCE사용할때  -> relu+sigmoid
	/*
	bce_loss(out, y);
	dloss = mat_element_sub(out, y);
	grad_w["w2"] = matmul(T(z1), dloss); //10x4 4x1 --> 10x1 
	grad_b["b2"] = sum_axis(dloss, 0); // 4x1 --> 1x1

	daffine1 = matmul(dloss, T(w2_)); // 4x1 1x10 --> 4x10
	dz1 = mat_element_mul(relu_grad(affine1), daffine1); //4x10 4x10 행렬 요소곱 --> 4x10
	grad_w["w1"] = matmul(T(x), dz1); // 2x4 4x10 --> 2x10
	grad_b["b1"] = sum_axis(dz1, 0); // 4x10 --> 1x10
	*/
	ret_p_wNb.first = grad_w; //pair의 첫번째에 w정보를 담고있는 <문자열,행렬> map을 저장
	ret_p_wNb.second = grad_b; //pair의 두번째에 b정보를 담고있는  <문자열,벡터>map을 저장

	return ret_p_wNb;
}

//시그모이드 미분 => (1.0 - sigmoid(x))sigmoid(x)
vector <vector < double >> layer::sigmoid_grad(vector<vector<double>> x) {
	vector<vector<double>> sig = mat_sigmoid(x); //sigmoid(x)
	vector<vector<double>> retm;
	vector<double> tmp;
	int row = x.size(), col = x[0].size();
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			tmp.push_back((1.0 - sig[r][c])*sig[r][c]); //(1.0 - sigmoid(x))sigmoid(x)
		}
		retm.push_back(tmp);
	}
	return retm;

}
// 행열 요소마다 x>0 ? x : 0
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
// 행열 요소마다 x>0 ? 1 : 0
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
//가중치와 편향값 출력
void layer::show_weight_bias() {
	std::cout << " weight1 : ";
	matPrint(w1_);
	std::cout << " weight2 : ";
	matPrint(w2_);
	std::cout << " bias1 : ";
	vecPrint(b1_);
	std::cout << " bias2 : ";
	vecPrint(b2_);
}


vector<vector<double>> layer::tanh(vector<vector<double>> x) {
	int row = x.size(); int col = x[0].size();
	vector<double> tmp;
	vector<vector<double>> retm;
	double exp_plus, exp_minus, val;
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			exp_plus = std::exp(x[r][c]);
			exp_minus = std::exp(-x[r][c]);
			val = (exp_plus - exp_minus) / (exp_plus + exp_minus);
			tmp.push_back(val);
		}
		retm.push_back(tmp);
	}
	return retm;
}

vector<vector<double>> layer::tanh_grad(vector<vector<double>> x) {
	int row = x.size(); int col = x[0].size();
	vector<double> tmp;
	vector<vector<double>> retm;
	double exp_plus, exp_minus, val;
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			val = 4 / ( (std::exp(2*x[r][c])) + 2 + (std::exp(-2 * x[r][c])) );
			tmp.push_back(val);
		}
		retm.push_back(tmp);
	}
	return retm;
}
