#include <fstream>
#include <omp.h>
#include <limits>

#ifdef HAS_FFTW

template<int dim>
void fft_mmsp_grid(const MMSP::grid<dim, MMSP::vector<fftw_complex> > & grid_in, MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_out, fftw_direction direction,
                    const int *dims, const std::vector<int> &ft_fields){

    // Initialize the dimensionality array
    fftwnd_plan plan = fftwnd_create_plan(dim, dims, direction, FFTW_ESTIMATE | FFTW_IN_PLACE);

    int num_elements = MMSP::nodes(grid_in);
    double normalization = 1.0;

    if (direction == FFTW_BACKWARD){
        normalization = num_elements;
    }

    // Ensure consistent nodes
    if (MMSP::nodes(grid_out) != num_elements){
        std::stringstream ss;
        ss << "Output and input lattice has different number of nodes! ";
        ss << "Input: " << num_elements << " Output: " << MMSP::nodes(grid_out);
        throw std::invalid_argument(ss.str());
    }

    // Ensure consistent number of fields
    if (MMSP::fields(grid_in) != MMSP::fields(grid_out)){
        std::stringstream ss;
        ss << "Number of fields in input and output lattice are different! ";
        ss << "Input: " << MMSP::fields(grid_in) << " Output: " << MMSP::fields(grid_out);
        throw std::invalid_argument(ss.str());
    }

    // Check consistent dimensions
    int num_elems_from_dims = 1;
    for (unsigned int i=0;i<dim;i++){
        num_elems_from_dims *= dims[i];
    }

    if (num_elems_from_dims != num_elements){
        std::stringstream ss;
        ss << "Dimension passed is inconsistent. Number of elements ";
        ss << "in the lattices " << num_elements << " Number of elements ";
        ss << "from dimension array: " << num_elems_from_dims;
        throw std::invalid_argument(ss.str());
    }

    // Check that the ft_fields argument is find
    for (auto field : ft_fields){
        if (field >= MMSP::fields(grid_in)){
            std::stringstream ss;
            ss << "Argument ft_fields is inconsistent ";
            ss << "FFT of field no. " << field << " requested, ";
            ss << "there are only " << MMSP::fields(grid_in) << " fields";
            throw std::invalid_argument(ss.str());
        }
    }

    // Construct array that FFTW can use
    fftw_complex *A = new fftw_complex[num_elements];

    // Loop over all fields that should be fourier transformed
    for (auto field : ft_fields){
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<MMSP::nodes(grid_in);i++){
            A[i] = grid_in(i)[field];
        }
        // Perform the FFT
        // TODO: See if FFTW can utilize multithreading
        fftwnd_one(plan, A, NULL);

        // Insert FT in the out field variable
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<MMSP::nodes(grid_out);i++){
            A[i].re /= normalization;
            grid_out(i)[field] = A[i];
        }
    }

    delete [] A;
    fftwnd_destroy_plan(plan);
};
#endif


template<int dim>
void get_dims(const MMSP::grid<dim, MMSP::vector<fftw_complex> >&grid_in, int dims[3]){
    // dims[0] = grid_in.xlength();
    // dims[1] = grid_in.ylength();
    // dims[2] = grid_in.zlength();

    dims[0] = MMSP::xlength(grid_in);
    dims[1] = MMSP::ylength(grid_in);
    dims[2] = MMSP::zlength(grid_in);
};

template<class T>
void divide(MMSP::vector<T> &vec, double val){
    for (unsigned int i=0;i<vec.length();i++){
        vec[i] /= val;
    }
};


template<int dim>
void max_value(const MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid, MMSP::vector<double> &max_val){
    for (unsigned int field=0;field<MMSP::fields(grid);field++){
        max_val[field] = 0.0;
        for (unsigned int node=0;node<MMSP::nodes(grid);node++){
            double abs_val = sqrt(pow(grid(node)[field].re, 2) + pow(grid(node)[field].im, 2));

            if (abs_val > max_val[field]){
                max_val[field] = abs_val;
            }
        }
    }
}

template<class T>
std::ostream& operator<<(std::ostream &out, MMSP::vector<T> &vec){
    for (unsigned int i=0;i<vec.length();i++){
        out << vec[i] << " ";
    }
    return out;
}

template<int dim>
void save_complex_field(const std::string &fname, MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid, unsigned int field){

    std::ofstream ofs(fname);

    if (!ofs.good()){
        throw std::runtime_error("Cannot open file!");
    }

    for (unsigned int i=0;i<MMSP::nodes(grid);i++){
        ofs << sqrt(pow(grid(i)[field].re, 2) + pow(grid(i)[field].im, 2));

        if (i < MMSP::nodes(grid) - 1){
            ofs << ",";
        }
    }
    ofs.close();
}

template<int dim>
double inf_norm_diff(const MMSP::grid<dim, MMSP::vector<double> > &grid1, const MMSP::grid<dim, MMSP::vector<double> > &grid2){
    double max_value = 0.0;

    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for reduction(max : max_value)
    #endif
    for (unsigned int node=0;node<MMSP::nodes(grid1);node++){
        for (unsigned int field=0;field<MMSP::fields(grid1);field++){
            double diff = grid1(node)[field] - grid2(node)[field];
            if (isnan(diff)){
                max_value = std::numeric_limits<double>::max();
            }
            else if (abs(diff) > max_value){
                max_value = abs(diff);
            }
        }
    }
    return max_value;
}

template<int dim>
fftw_complex average_nearest_neighbours(const MMSP::grid<dim, MMSP::vector<fftw_complex> > &gr, unsigned int field, unsigned int center_node){
    MMSP::vector<int> pos = gr.position(center_node);

    fftw_complex value;
    value.re = 0.0;
    value.im = 0.0;
   
    int num = pow(2, dim);
    for (int dir=0;dir<dim;dir++){
        int old_val = pos[dir];
        for (int i=-1;i<2;i+=2){
            pos[dir] += i;
            fftw_complex new_val = gr(pos)[field];
            value.re += new_val.re/num;
            value.im += new_val.im/num;
            pos[dir] = old_val;
        }
    }
    return value;
}