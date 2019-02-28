#ifdef HAS_FFTW

    template<int dim>
    void fft_mmsp_grid(const MMSP::grid<dim, MMSP::vector<fftw_complex> > & grid_in, MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_out, int direction,
                       const int *dims, const std::vector<int> &ft_fields){
    
        // Initialize the dimensionality array
        fftwnd_plan plan = fftwnd_create_plan(dim, dims, direction, FFTW_ESTIMATE | FFTW_IN_PLACE);

        int num_elements = grid_in.nodes();
        // Construct array that FFTW can use
        fftw_complex *A = new fftw_complex[num_elements];

        // Loop over all fields that should be fourier transformed
        for (auto field : ft_fields){
            #ifndef NO_PHASEFIELD_PARALLEL
            #pragma omp parallel for
            #endif
            for (unsigned int i=0;i<grid_in.nodes();i++){
                A[i] = grid_in(field)[i];
            }

            // Perform the FFT
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