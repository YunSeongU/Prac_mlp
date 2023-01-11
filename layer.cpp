#include "matrix.h"
#include "layer.h"

//������ ����
layer::layer(int input_size,int hidden_size ,int output_size) {
	w1_ = weight_init(input_size, hidden_size);
	b1_ = vector<double>(hidden_size, 0);
	w2_ = weight_init(hidden_size, output_size);
	b2_ = vector<double>(output_size, 0);
}
//�ñ׸��̵� �Լ� ����
vector<vector<double>> layer::sigmoid(vector<vector<double>> x) {
	return mat_sigmoid(x);
}

//���� w_,b_�� ����� ����
vector<vector<double>> layer::predict(vector<vector<double>> x) {
	vector<vector<double>> affine1, active1, affine2, out,b1,b2;

	affine1 = matmul(x, w1_);
	b1 = make_bias_mat(affine1.size(), b1_);
	affine1 = addb(affine1, b1);
	active1 = relu(affine1);



	affine2 = matmul(active1, w2_);
	b2 = make_bias_mat(affine2.size(), b2_);
	affine2 = addb(affine2, b2);
	out = relu(affine2);

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
	loss_ = sum / size; //loss���� Ŭ���� ��������� ����
	return diff;
}


//forward�� backward�� �����ϴ� �Լ�
//w,b�̺� ������ ������ map�� ��ȯ�Ѵ�
pair<
	map <string, vector<vector<double>>>,
	map <string, vector <double>>
> layer::process(vector<vector<double>> x, vector<vector<double>> y) {

	vector<vector<double>> affine1, z1, affine2, out,diff,b1,b2;
	vector<vector<double>> dout, daffine1, dloss, dz1;
	

	pair<
		map <string, vector<vector<double>>>, 
		map <string, vector <double>> 
	> ret_p_wNb; //����ġ�� ������ �޴� ���object

	map <string, vector<vector <double>> > grad_w; //����ġ ������ �޾ƿ�
	map <string, vector <double>> grad_b; //���� ������ �޾ƿ�

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
	diff = mse_loss(out, y); // 4x1

	dloss = mse_grad(diff); //4x1
	dout = mat_element_mul(relu_grad(affine2), dloss); // 4x1 4x1 ��� ��� �� --> 4x1
	grad_w["w2"] = matmul(T(z1), dout); //10x4 4x1 --> 10x1 
	grad_b["b2"] = sum_axis(dout, 0); // 4x1 --> 1x1

	daffine1 = matmul(dout, T(w2_)); // 4x1 1x10 --> 4x10
	dz1 = mat_element_mul(relu_grad(affine1), daffine1); //4x10 4x10 ��� ��Ұ� --> 4x10
	grad_w["w1"] = matmul(T(x), dz1); // 2x4 4x10 --> 2x10
	grad_b["b1"] = sum_axis(dz1, 0); // 4x10 --> 1x10

	ret_p_wNb.first = grad_w; //pair�� ù��°�� w������ ����ִ� <���ڿ�,���> map�� ����
	ret_p_wNb.second = grad_b; //pair�� �ι�°�� b������ ����ִ�  <���ڿ�,����>map�� ����

	return ret_p_wNb;
}

//�ñ׸��̵� �̺� => (1.0 - sigmoid(x))sigmoid(x)
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
// �࿭ ��Ҹ��� x>0 ? x : 0
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
// �࿭ ��Ҹ��� x>0 ? 1 : 0
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
//����ġ�� ���Ⱚ ���
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
