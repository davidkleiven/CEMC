from setuptools import setup, Extension, find_packages
import numpy as np
from Cython.Build import cythonize

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
                      "eshelby_cylinder.cpp", "init_numpy_api.cpp"]

ce_updater_sources = [src_folder+"/"+srcfile for srcfile in ce_updater_sources]
ce_updater_sources.append("cemc/cpp_ext/cemc_cpp_code.pyx")

cemc_cpp_code = Extension("cemc_cpp_code", sources=ce_updater_sources,
                          include_dirs=[inc_folder, np.get_include()],
                          extra_compile_args=["-std=c++11", "-fopenmp"],
                          language="c++", libraries=["gomp", "pthread"])
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
