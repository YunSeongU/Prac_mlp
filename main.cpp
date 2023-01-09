#include <iostream>
#include <vector>

using std::vector;


void init_Weight_Bias(vector<vector<double>>& W, vector<double>& B, int input_size, int output_size);

int main() {



}

void init_Weight_Bias(vector<vector<double>>& W, vector<double>& B, int input_size, int output_size) {
    W = vector<vector<double>>(input_size, vector <double>(output_size, 0.5));
    B = vector <double>(output_size, 0);
}
