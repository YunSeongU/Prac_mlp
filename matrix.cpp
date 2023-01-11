#include "matrix.h"
#include <stdio.h>
#include <random>
#include <cmath>

using std::cout;
using std::endl;


//행렬 곱연산
vector <vector <double>> matmul(vector<vector<double>> A, vector<vector<double>> B) { 
	int row1 = A.size(), col1 = A[0].size();
	int row2 = B.size(), col2 = B[0].size();
	vector<vector<double>> matrix(row1, vector<double>(col2));

	if (col1==row2) { 
		for (int i = 0; i < row1; i++) {
			for (int j = 0; j < col2; j++) {
				matrix[i][j] = 0;
				for (int k = 0; k < col1; k++)
					matrix[i][j] += A[i][k] * B[k][j]; //A의 행과 B의 열의 곱의 합을 저장한다
			}
		}
		return matrix;
	}
	else { //행렬 곱을 위한 조건(A행렬의 열과 B행렬의 행의수가 같아야함)을 만족하지 않을때
		cout << "ValueError : (n?,k),(k,m?)->(n?,m?)" << "(size " << col1 << " is different from " << row2 << ")" << endl;
	}

	

	
}
//2차원 행렬 출력
void matPrint(vector<vector<double>> A) {
	int row1 = A.size(), col1 = A[0].size();
	for (int r = 0; r < row1; ++r) {
		for (int c = 0; c < col1; ++c) {
			cout << A[r][c] << " ";
		}
		cout << endl;
	}
	cout << "ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ" << endl;
}
// 행렬 전치
vector <vector <double>> T(vector<vector<double>>A) {
	int row = A.size(), col = A[0].size();
	int i = 0; int j = 0;
	vector<vector<double>> retm(col, vector<double>(row));
	for (int c = 0; c < col; c++) {
		for (int r = 0; r < row; r++) { 
			retm[i][j++] = A[r][c];  //행방향(axis =0)으로 먼저 읽어서 열방향으로 넣음
		}
		i += 1;
		j = 0;
	}
	return retm;

}

//편향벡터 더하기 
vector<vector<double>> addb(vector<vector<double>> arr1, vector<vector<double>> arr2) {
	vector<vector<double>> answer;
	vector<double> temp;
	int r1 = arr1.size(); int c1 = arr1[0].size();
	int r2 = arr2.size(); int c2 = arr2[0].size();

	if ((r1 == r2) and (c1 == c2)) {
		for (int i = 0; i < arr1.size(); i++)
		{
			temp.clear();
			for (int j = 0; j < arr1[i].size(); j++)
			{
				temp.push_back(arr1[i][j] + arr2[i][j]); //요소 합 반환
			}
			answer.push_back(temp);
		}
		return answer;
	}
	else {
		cout << "addb :: size mismatch " << endl;
	}
}
//축에따라 값더하기, bias역전파에 사용(gradient 구할때 행방향 합)
vector <double> sum_axis(vector<vector<double>> A, int axis) {
	vector <double> retv;
	if (axis==0){
		int row = A.size(); int col = A[0].size();
		double sum = 0.0;
		for (int c = 0; c < col; c++) {
			for (int r = 0; r < row; r++) {
				sum += A[r][c]; //col이 고정된 상태로 row를 이동하며 더한다
			}
			retv.push_back(sum);
			sum = 0;
		}
		return retv;
	}else if (axis == 1) {
		int row = A.size(); int col = A[0].size();
		double sum = 0.0;
		for (int r = 0; r < row; r++) {
			for (int c = 0; c < col; c++) {
				sum += A[r][c]; // row를 고정한 상태로 col을 이동하며 더한다
			}
			retv.push_back(sum);
			sum = 0;
		}
		return retv;
	}
}
	

//행렬 요소 곱 --> 활성화함수 gradient에 사용
vector<vector<double>> mat_element_mul(vector<vector<double>> A, vector<vector<double>> B) {
	
	int row1 = A.size(), col1 = A[0].size();
	int row2 = B.size(), col2 = B[0].size();

	if ((row1 == row2) and (col1 == col2)) {
		vector<vector<double>> retm;
		vector<double> tmp;
		for (int i = 0; i < row1; i++) {
			tmp.clear();
			for (int j = 0; j < col1; j++) {
				tmp.push_back((A[i][j] * B[i][j]));
			}
			retm.push_back(tmp);
		}
		return retm;
	}


}
//행렬 요소 차 --> mse에 사용
vector<vector<double>> mat_element_sub(vector<vector<double>> A, vector<vector<double>> B) {
	int row1 = A.size(), col1 = A[0].size();
	int row2 = B.size(), col2 = B[0].size();
	vector<vector<double>> retMat;
	vector<double> tmp;
	if ((row1 == row2) and (col1 == col2)) {
		for (int i = 0; i < row1; i++) {
			tmp.clear();
			for (int j = 0; j < col1; j++) {
				tmp.push_back(A[i][j] - B[i][j]);
			}
			retMat.push_back(tmp);
		}
		return retMat;
	}
}
//행렬 요소별 시그모이드 함수 적용
vector < vector < double >> mat_sigmoid(vector<vector<double>> A) {
	int row = A.size(), col = A[0].size();
	vector<vector<double>> sigm;
	vector<double> tmp;
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			tmp.push_back(1 / (1 + std::exp(-A[r][c])));
		}
		sigm.push_back(tmp);
	}
	return sigm;
}


//행렬 요소 제곱 --> mse
vector<double> mat_square(vector<vector<double>> A) {
	int row = A.size(), col = A[0].size();
	vector<double> tmp;
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			tmp.push_back(A[r][c]*A[r][c]);
		}
	}
	return tmp;
}


//mse미분 -> 파이썬에서 2*diff/np.prod(diff.shape) c++로 표현
vector<vector<double>> mse_grad(vector<vector<double>> A) {
	int row = A.size(), col = A[0].size();
	int div = row * col;
	vector<vector<double>> retm;
	vector<double> tmp;
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			tmp.push_back((2.0*A[r][c])/div);
		}
		retm.push_back(tmp);
	}
	return retm;
}
//vector인 bias를 연산을위해 복제하여 행렬로 만듬
vector<vector<double>> make_bias_mat(int row_size, vector<double> A) {
	vector<vector<double>> retm;
	vector<double> tmp;
	int col = A.size();
	for (int r = 0; r < row_size; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			tmp.push_back(A[c]);
		}
		retm.push_back(tmp);
	}
	return retm;
}
//행렬과 스칼라의 곱 --> learning_rate적용에 사용
vector<vector<double>> mat_scalr_mul(double v,vector<vector<double>> A) {
	int row = A.size(), col = A[0].size();
	vector<vector<double>> retm;
	vector<double> tmp;
	for (int r = 0; r < row; r++) {
		tmp.clear();
		for (int c = 0; c < col; c++) {
			tmp.push_back(v*A[r][c]);
		}
		retm.push_back(tmp);
	}
	return retm;
}
//벡터 요소 차 --> bias 업데이트에 사용
vector<double> vec_elem_sub(vector<double> A, vector<double> B) {
	vector<double> retvec;
	int size_a = A.size(); int size_b = B.size();
	if (size_a == size_b) {
		for (int s = 0; s < size_a; s++) {
			retvec.push_back(A[s] - B[s]);
		}
		return retvec;
	}
	else {
		cout << "vec_elem_sub :: size error" << endl;
	}
}
//벡터와 스칼라의 곱 --> learning_rate적용에 사용
vector<double> vec_scalr_mul(double v, vector<double> A) {
	int row = A.size();
	vector<double> tmp;
	for (int s = 0; s < row; s++) {
		tmp.push_back(v * A[s]);
	}
	return tmp;
	
}
//random_device를 사용한 가중치 초기화 
vector<vector<double>> weight_init(int r_size, int c_size) {
	vector<vector<double>> retm;
	vector<double> tmp;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double>dis(0, 1); //0~1사이 균일분포에서 실수 랜덤한 실수 반환
	for (int r = 0; r < r_size; r++) {
		tmp.clear();
		for (int c = 0; c < c_size; c++) {
			tmp.push_back(dis(gen));
		}
		retm.push_back(tmp);
	}
	return retm;
}
//vector 출력
void vecPrint(vector<double> A) {
	int s = A.size();
	for (int i = 0; i < s; i++) {
		cout << A[i] << " ";
	}
	cout << endl;
}