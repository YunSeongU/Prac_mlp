#include <iostream>
#include <vector>
#include "layer.h"
#include "matrix.h"
#include <chrono>

using std::vector;
using std::cout;
using std::endl;

void vecPrint(vector<double> A);

int main() {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    pair<
        map <string, vector<vector<double>>>,
        map <string, vector <double>>
    > grad_map;

    map <string, vector<vector<double>>> grad_w;
    map <string, vector <double>> grad_b;

    vector < vector<double>> tmp_2dim;
    vector<double> tmp_dim;

    

    vector<vector<double>> inx = { //xor input
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };
    vector<vector<double>> outy = { //xor target
        {0},
        {1},
        {1},
        {0}
    };
    
    
    double lr = 0.001;

    layer xor_mlp(2, 10, 1);// �����ڿ� ���� ��ü ����
 
    for (int i = 0; i < 15001; i++) {
        /*pair<map,map>*/grad_map = xor_mlp.process(inx, outy); //process -> forward ~ backward

        /*map<���ڿ�,���>*/grad_w = grad_map.first; //����ġ �̺�����
        /*map<���ڿ�,����>*/grad_b = grad_map.second; // ���� �̺�����


        tmp_2dim = grad_w.find("w1")->second; //map���� key "w1"������ value�� ã�Ƽ� ����
        tmp_2dim = mat_scalr_mul(lr, tmp_2dim); //lr*gradient
        xor_mlp.w1_ = mat_element_sub(xor_mlp.w1_, tmp_2dim); // w1 -= lr*gradient
        
        tmp_dim = grad_b.find("b1")->second; //map���� key "b1"������ value�� ã�Ƽ� ����
        tmp_dim = vec_scalr_mul(lr, tmp_dim); //learning_rate*gradient
        xor_mlp.b1_ = vec_elem_sub(xor_mlp.b1_, tmp_dim); // b1 -= lr*gradient

        tmp_2dim = grad_w.find("w2")->second; //map���� key "w2"������ value�� ã�Ƽ� ����
        tmp_2dim = mat_scalr_mul(lr, tmp_2dim);
        xor_mlp.w2_ = mat_element_sub(xor_mlp.w2_, tmp_2dim);

        tmp_dim = grad_b.find("b2")->second; //map���� key "b2"������ value�� ã�Ƽ� ����
        tmp_dim = vec_scalr_mul(lr, tmp_dim);
        xor_mlp.b2_ = vec_elem_sub(xor_mlp.b2_, tmp_dim);

        if (i % 1000 == 0) {
            cout << "epoch : " << i << " , " << "loss val : " << xor_mlp.loss_ << endl;


        }

    }
    xor_mlp.show_weight_bias(); //���� ����ġ,���̾ ���
    tmp_2dim = xor_mlp.predict(inx);//���� ����ġ,���̾�� ������� ������ xor�Է��� �ִ´�
    matPrint(tmp_2dim); //��� ���

    for (int i = 0; i < tmp_2dim.size(); i++) {
        if (tmp_2dim[i][0] > 0.5) {
            cout << 1 << endl;
        }
        else {
            cout << 0 << endl;
        }
    }
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    std::cout << "layer_use_refer�� �����µ� �ɸ��� �ð�(��) : " << sec.count() << "seconds" << std::endl;
}


