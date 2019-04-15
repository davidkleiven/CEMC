#include "multidirectional_khachaturyan.hpp"
#include <cmath>

using namespace std;

void MultidirectionalKhachaturyan::add_model(const Khachaturyan &model, unsigned int shape_field){
    strain_models[shape_field] = model;
}

MultidirectionalKhachaturyan::~MultidirectionalKhachaturyan(){
    delete fft; fft = nullptr;
}

double MultidirectionalKhachaturyan::B_tensor_element(MMSP::vector<double> &dir, const mat3x3 &green, \
                                                      const mat3x3 &eff_stress1, const mat3x3 &eff_stress2) const
{
    MMSP::vector<double> temp_vec(3);
    dot(eff_stress2, dir, temp_vec);

    MMSP::vector<double> temp_vec2(3);
    dot(green, temp_vec, temp_vec2);
    dot(eff_stress1, temp_vec2, temp_vec);
    return dot(dir, temp_vec);
}



double MultidirectionalKhachaturyan::contract_tensors(const mat3x3 &mat1, const mat3x3 &mat2) const{
    double value = 0.0;
    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        value += mat1[i][j]*mat2[i][j];
    }
    return value;
}

void MultidirectionalKhachaturyan::get_effective_stresses(vector<mat3x3> &eff_stress) const{
    for (auto iter=strain_models.begin(); iter != strain_models.end(); ++iter){
        mat3x3 stress;
        iter->second.effective_stress(stress);
        eff_stress.push_back(stress);
    }
}

void MultidirectionalKhachaturyan::index_map(const std::vector<int> &shape_fields, std::map<unsigned int, unsigned int> &mapping) const{
    for (unsigned int i=0;i<shape_fields.size();i++){
        mapping[shape_fields[i]] = i;
    }
}