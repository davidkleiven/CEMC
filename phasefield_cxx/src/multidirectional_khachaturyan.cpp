#include "multidirectional_khachaturyan.hpp"

void MultidirectionalKhachaturyan::add_model(const Khachaturyan &model){
    strain_models.push_back(model);
}