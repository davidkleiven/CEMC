#ifdef HAS_FFTW

template<int dim>
void fft_mmsp_grid(const MMSP::grid<dim, MMSP::vector<fftw_complex> > & grid_in, MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_out, int direction,
                    const int *dims, const std::vector<int> &ft_fields){

    // Initialize the dimensionality array
    fftwnd_plan plan = fftwnd_create_plan(dim, dims, direction, FFTW_ESTIMATE | FFTW_IN_PLACE);

    int num_elements = MMSP::nodes(grid_in);
    // Construct array that FFTW can use
    fftw_complex *A = new fftw_complex[num_elements];

    // Loop over all fields that should be fourier transformed
    for (auto field : ft_fields){
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<MMSP::nodes(grid_in);i++){
            A[i] = grid_in(field)[i];
        }

        // Perform the FFT
        // TODO: See if FFTW can utilize multithreading
        fftwnd_one(plan, A, NULL);

        // Insert FT in the out field variable
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<grid_out.nodes();i++){
            grid_out(field)[i] = A[i];
        }
    }

    delete [] A;
    fftwnd_destroy_plan(plan);
}
#endif

template<int dim>
void get_dims(const MMSP::grid<dim, MMSP::vector<fftw_complex> >&grid_in, int dims[3]){
    // dims[0] = grid_in.xlength();
    // dims[1] = grid_in.ylength();
    // dims[2] = grid_in.zlength();

    dims[0] = MMSP::xlength(grid_in);
    dims[1] = MMSP::ylength(grid_in);
    dims[2] = MMSP::zlength(grid_in);
}

template<class T>
void divide(MMSP::vector<T> &vec, double val){
    for (unsigned int i=0;i<vec.length();i++){
        vec[i] /= val;
    }
}