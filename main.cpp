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

    layer xor_mlp(2, 10, 1);// 생성자에 맞춰 객체 생성
 
    for (int i = 0; i < 15001; i++) {
        /*pair<map,map>*/grad_map = xor_mlp.process(inx, outy); //process -> forward ~ backward

        /*map<문자열,행렬>*/grad_w = grad_map.first; //가중치 미분정보
        /*map<문자열,벡터>*/grad_b = grad_map.second; // 편향 미분정보


        tmp_2dim = grad_w.find("w1")->second; //map에서 key "w1"에대한 value를 찾아서 저장
        tmp_2dim = mat_scalr_mul(lr, tmp_2dim); //lr*gradient
        xor_mlp.w1_ = mat_element_sub(xor_mlp.w1_, tmp_2dim); // w1 -= lr*gradient
        
        tmp_dim = grad_b.find("b1")->second; //map에서 key "b1"에대한 value를 찾아서 저장
        tmp_dim = vec_scalr_mul(lr, tmp_dim); //learning_rate*gradient
        xor_mlp.b1_ = vec_elem_sub(xor_mlp.b1_, tmp_dim); // b1 -= lr*gradient

        tmp_2dim = grad_w.find("w2")->second; //map에서 key "w2"에대한 value를 찾아서 저장
        tmp_2dim = mat_scalr_mul(lr, tmp_2dim);
        xor_mlp.w2_ = mat_element_sub(xor_mlp.w2_, tmp_2dim);

        tmp_dim = grad_b.find("b2")->second; //map에서 key "b2"에대한 value를 찾아서 저장
        tmp_dim = vec_scalr_mul(lr, tmp_dim);
        xor_mlp.b2_ = vec_elem_sub(xor_mlp.b2_, tmp_dim);

        if (i % 1000 == 0) {
            cout << "epoch : " << i << " , " << "loss val : " << xor_mlp.loss_ << endl;


        }

    }
    xor_mlp.show_weight_bias(); //최종 가중치,바이어스 출력
    tmp_2dim = xor_mlp.predict(inx);//최종 가중치,바이어스로 만들어진 가설에 xor입력을 넣는다
    matPrint(tmp_2dim); //행렬 출력

    for (int i = 0; i < tmp_2dim.size(); i++) {
        if (tmp_2dim[i][0] > 0.5) {
            cout << 1 << endl;
        }
        else {
            cout << 0 << endl;
        }
    }
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    std::cout << "layer_use_refer를 돌리는데 걸리는 시간(초) : " << sec.count() << "seconds" << std::endl;
}


