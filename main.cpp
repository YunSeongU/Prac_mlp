#include <iostream>
#include <vector>
#include "layer.h"
#include "matrix.h"

using std::vector;
using std::cout;
using std::endl;

void vecPrint(vector<double> A);

int main() {

    pair<
        map <string, vector<vector<double>>>,
        map <string, vector <double>>
    > grad_map;

    map <string, vector<vector<double>>> grad_w;
    map <string, vector <double>> grad_b;

    vector < vector<double>> tmp_2dim;
    vector<double> tmp_dim;

    double lr = 0.001;

    vector<vector<double>> inx = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };
    vector<vector<double>> y = {
        {0},
        {1},
        {1},
        {0}
    };

    layer xor_mlp(2, 10, 1);
 
    for (int i = 0; i < 10001; i++) {
        grad_map=xor_mlp.process(inx, y);
        grad_w = grad_map.first;
        grad_b = grad_map.second;


        tmp_2dim = grad_w.find("w1")->second;
        tmp_2dim = mat_scalr_mul(lr, tmp_2dim);
        xor_mlp.w1_ = mat_element_sub(xor_mlp.w1_, tmp_2dim); // w -= lr*gradient
        
        tmp_dim = grad_b.find("b1")->second;
        tmp_dim = vec_scalr_mul(lr, tmp_dim);
        xor_mlp.b1_ = vec_elem_sub(xor_mlp.b1_, tmp_dim); // b -= lr*gradient

        tmp_2dim = grad_w.find("w2")->second;
        tmp_2dim = mat_scalr_mul(lr, tmp_2dim);
        xor_mlp.w2_ = mat_element_sub(xor_mlp.w2_, tmp_2dim);

        tmp_dim = grad_b.find("b2")->second;
        tmp_dim = vec_scalr_mul(lr, tmp_dim);
        xor_mlp.b2_ = vec_elem_sub(xor_mlp.b2_, tmp_dim);

        if (i % 1000 == 0) {
            cout << "epoch : " << i << " , " << "loss val : " << xor_mlp.loss_ << endl;
        }

    }

    tmp_2dim = xor_mlp.predict(inx);
    matPrint(tmp_2dim);
}

void vecPrint(vector<double> A) {
    int s = A.size();
    for (int i = 0; i < s; i++) {
        cout << A[i] << " ";
    }
    cout << endl;
}
