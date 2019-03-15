import sys
from setuptools import setup, Extension, find_packages
import numpy as np
from Cython.Build import cythonize
import os

"""Install instructions

1. To parallelize the update of correlation functions
    usng openMP run the installation with
    pip install --install-option="--PARALLEL_CF_UPDATE"
2. To compile with -g flag use
   --install-option="--DEBUG"
3. To not parallize the Khacaturyan integrals run 
    pip install --install-option="--NO_PARALLEL_KHACHATURYAN_INTEGRAL"
4. To build the phase field extension module (which depend on the 
    mesoscale/MMSP project)
5. Many of the phase field models parallizes internally using
    hyperthreading. To disable run the installation with
    --install-option="--NO_PHASEFIELD_PARALLEL"

To build the phase field module the following environement variable
may be set
MMSP_HOME: Path to the root directory of the MMSP package.
    (Header files are assumed to be located at $MMSP_HOME/include)

VTKLIB: Path to the VTK libraries needed by MMSP
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
                      "symbols_with_numbers.cpp", "basis_function.cpp"]

ce_updater_sources = [src_folder+"/"+srcfile for srcfile in ce_updater_sources]
ce_updater_sources.append("cemc/cpp_ext/cemc_cpp_code.pyx")

src_phase = "phasefield_cxx/src"
phasefield_sources = ["two_phase_landau.cpp", "mat4D.cpp", "khacaturyan.cpp",
                      "linalg.cpp", "cahn_hilliard.cpp", "polynomial_term.cpp",
                      "polynomial.cpp", "regression_kernels.cpp"]

phasefield_sources = [src_phase + "/" + x for x in phasefield_sources]
phasefield_sources.append("cemc/phasefield/cython/phasefield_cxx.pyx")

define_macros = [("PARALLEL_KHACHATURYAN_INTEGRAL", None)]
extra_comp_args = ["-std=c++11", "-fopenmp"]
extracted_args = []
optional_lib_phasefield = []
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
    elif arg == "--NO_PHASEFIELD_PARALLEL":
        define_macros.append(("NO_PHASEFIELD_PARALLEL", None))
        extracted_args.append(arg)
    elif arg == "--HAS_FFTW":
        define_macros.append(("HAS_FFTW", None))
        optional_lib_phasefield.append("fftw")
        extracted_args.append(arg)

# Filter out of sys.argv
for arg in extracted_args:
    sys.argv.remove(arg)

cemc_cpp_code = Extension("cemc_cpp_code", sources=ce_updater_sources,
                          include_dirs=[inc_folder, np.get_include()],
                          extra_compile_args=extra_comp_args,
                          language="c++", libraries=["gomp", "pthread"],
                          define_macros=define_macros)

phase_field_mod = Extension("phasefield_cxx", sources=phasefield_sources,
                            include_dirs=[np.get_include(),
                                          "phasefield_cxx/include",
                                          "phasefield_cxx/src",
                                          os.environ.get("MMSP_HOME", "./")+"/include"],
                            extra_compile_args=extra_comp_args,
                            language="c++", define_macros=define_macros,
                            libraries=["gomp", "pthread", "z", "png",
                                       "vtkCommonCore", "vtkCommonDataModel",
                                       "vtkIOXML"] + optional_lib_phasefield,
                            library_dirs=[os.environ.get("VTKLIB", "./")])

ext_mods = [cemc_cpp_code]
if "--with-phasefield" in sys.argv:
    ext_mods.append(phase_field_mod)
    sys.argv.remove("--with-phasefield")
    print("Including phase field module in the build...")

setup(
    name="cemc",
    ext_modules=cythonize(ext_mods),
    version=1.0,
    author="David Kleiven",
    author_email="davidkleiven446@gmail.com",
    description="Monte Carlo routines for Cluster Expansion",
    packages=find_packages(),
    include_package_data=True
)
