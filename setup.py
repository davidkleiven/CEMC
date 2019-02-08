import sys
from setuptools import setup, Extension, find_packages
import numpy as np
from Cython.Build import cythonize

"""Install instructions

1. To parallelize the update of correlation functions
    usng openMP run the installation with
    pip install --install-option="--PARALLEL_CF_UPDATE"
2. To compile with -g flag use
   --install-option="--DEBUG"
3. To not parallize the Khacaturyan integrals run 
    pip install --install-option="--NO_PARALLEL_KHACHATURYAN_INTEGRAL"
"""

src_folder = "cpp/src"
inc_folder = "cpp/include"
base_line_sources = []

ce_updater_sources = ["ce_updater.cpp", "cf_history_tracker.cpp",
                      "additional_tools.cpp", "histogram.cpp",
                      "wang_landau_sampler.cpp", "adaptive_windows.cpp",
                      "mc_observers.cpp", "linear_vib_correction.cpp",
                      "cluster.cpp", "cluster_tracker.cpp",
                      "named_array.cpp",
                      "row_sparse_struct_matrix.cpp", "pair_constraint.cpp",
                      "eshelby_tensor.cpp", "eshelby_sphere.cpp",
                      "eshelby_cylinder.cpp", "init_numpy_api.cpp",
                      "symbols_with_numbers.cpp", "basis_function.cpp",
                      "mat4D.cpp", "khacaturyan.cpp"]

ce_updater_sources = [src_folder+"/"+srcfile for srcfile in ce_updater_sources]
ce_updater_sources.append("cemc/cpp_ext/cemc_cpp_code.pyx")

define_macros = [("PARALLEL_KHACHATURYAN_INTEGRAL", None)]
extra_comp_args = ["-std=c++11", "-fopenmp"]
extracted_args = []
for arg in sys.argv:
    if arg == "--PARALLEL_CF_UPDATE":
        define_macros.append(("PARALLEL_CF_UPDATE", None))
        extracted_args.append(arg)
    elif arg == "--DEBUG":
        extracted_args.append(arg)
        extra_comp_args.append("-g")
    elif arg == "--NO_PARALLEL_KHACHATURYAN_INTEGRAL":
        define_macros.remove((("PARALLEL_KHACHATURYAN_INTEGRAL", None)))
        extracted_args.append(arg)

# Filter out of sys.argv
for arg in extracted_args:
    sys.argv.remove(arg)

cemc_cpp_code = Extension("cemc_cpp_code", sources=ce_updater_sources,
                          include_dirs=[inc_folder, np.get_include()],
                          extra_compile_args=extra_comp_args,
                          language="c++", libraries=["gomp", "pthread"],
                          define_macros=define_macros)

setup(
    name="cemc",
    ext_modules=cythonize(cemc_cpp_code),
    version=1.0,
    author="David Kleiven",
    author_email="davidkleiven446@gmail.com",
    description="Monte Carlo routines for Cluster Expansion",
    packages=find_packages(),
    include_package_data=True
)
