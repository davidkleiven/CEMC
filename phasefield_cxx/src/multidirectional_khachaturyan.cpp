#include "multidirectional_khachaturyan.hpp"
#include <cmath>

using namespace std;

void MultidirectionalKhachaturyan::add_model(const Khachaturyan &model, unsigned int shape_field){
    strain_models[shape_field] = model;
}

MultidirectionalKhachaturyan::~MultidirectionalKhachaturyan(){
    delete fft; fft = nullptr;
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