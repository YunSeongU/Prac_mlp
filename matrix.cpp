#include "matrix.h"
#include <stdio.h>


#include <cmath>

using std::cout;
using std::endl;



//vector <vector <double>> matrix::matmul(vector<vector<double>>A, vector<vector<double>>B) {
//	int row1 = A.size(), col1 = A[0].size();
//	int row2 = B.size(), col2 = B[0].size();
//	vector<vector<double>> matrix(row1, vector<double>(col2));
//
//	for (int i = 0; i < row1; i++) {
//		for (int j = 0; j < col2; j++) {
//			matrix[i][j] = 0;
//			for (int k = 0; k < col1; k++)
//				matrix[i][j] += A[i][k] * B[k][j];
//		}
//	}
//	return matrix;
//}
//
//void matrix::matPrint(vector<vector<double>>&A) {
//	int row1 = A.size(), col1 = A[0].size();
//	for (int r = 0; r < row1; ++r) {
//		for (int c = 0; c < col1; ++c) {
//			cout << A[r][c] << " ";
//		}
//		cout << endl;
//	}
//	cout << "�ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�" << endl;
//}
//
//vector <vector <double>> matrix::T(vector<vector<double>>A) {
//	int row = A.size(), col = A[0].size();
//	int i = 0; int j = 0;
//	vector<vector<double>> retm(col, vector<double>(row));
//	for (int c = 0; c < col; c++) {
//		for (int r = 0; r < row; r++) {
//			retm[i][j++] = A[r][c];
//		}
//		i += 1;
//		j = 0;
//	}
//	return retm;
//
//}

vector <vector <double>> matmul(vector<vector<double>> A, vector<vector<double>> B) {
	int row1 = A.size(), col1 = A[0].size();
	int row2 = B.size(), col2 = B[0].size();

	if (col1!=row2) {
		cout<< "ValueError : (n?,k),(k,m?)->(n?,m?)" << "(size " << col1 <<" is different from "<< row2 <<")"<<endl;
		return A;
	}
	vector<vector<double>> matrix(row1, vector<double>(col2));

	for (int i = 0; i < row1; i++) {
		for (int j = 0; j < col2; j++) {
			matrix[i][j] = 0;
			for (int k = 0; k < col1; k++)
				matrix[i][j] += A[i][k] * B[k][j];
		}
	}
	return matrix;
}

void matPrint(vector<vector<double>> A) {
	int row1 = A.size(), col1 = A[0].size();
	for (int r = 0; r < row1; ++r) {
		for (int c = 0; c < col1; ++c) {
			cout << A[r][c] << " ";
		}
		cout << endl;
	}
	cout << "�ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�" << endl;
}

vector <vector <double>> T(vector<vector<double>>A) {
	int row = A.size(), col = A[0].size();
	int i = 0; int j = 0;
	vector<vector<double>> retm(col, vector<double>(row));
	for (int c = 0; c < col; c++) {
		for (int r = 0; r < row; r++) {
			retm[i][j++] = A[r][c];
		}
		i += 1;
		j = 0;
	}
	return retm;

}


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
				temp.push_back(arr1[i][j] + arr2[i][j]);
			}
			answer.push_back(temp);
		}
		return answer;
	}
	else {
		cout << "addb :: size mismatch " << endl;
	}
}

vector <double> sum_axis(vector<vector<double>> A, int axis) {
	vector <double> retv;
	if (axis==0){
		int row = A.size(); int col = A[0].size();
		double sum = 0.0;
		for (int c = 0; c < col; c++) {
			for (int r = 0; r < row; r++) {
				sum += A[r][c];
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
				sum += A[r][c];
			}
			retv.push_back(sum);
			sum = 0;
		}
		return retv;
	}
}
	


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

vector<double> vec_scalr_mul(double v, vector<double> A) {
	int row = A.size();
	vector<double> tmp;
	for (int s = 0; s < row; s++) {
		tmp.push_back(v * A[s]);
	}
	return tmp;
	
}