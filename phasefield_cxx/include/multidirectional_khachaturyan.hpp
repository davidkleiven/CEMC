#ifndef MULTIDIRECTIONAL_KHACKATURYAN_H
#define MULTIDIRECTIONAL_KHACKATURYAN_H
#include <vector>
#include "khachaturyan.hpp"

class MultidirectionalKhachaturyan{
    public:
        MultidirectionalKhachaturyan(){};

        /** Add a new model */
        void add_model(const Khachaturyan &model);        
    private:
        std::vector<Khachaturyan> strain_models;
};
#endif