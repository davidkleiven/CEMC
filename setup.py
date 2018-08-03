from setuptools import setup, Extension, find_packages
import subprocess as sub
import sys
import numpy as np

# Print the version of swig
sub.call(["swig","-version"])
print ("Numpy development files:")
print (np.get_include())

swig_opts=["-modern","-Icpp/include","-c++"]
if ( sys.version_info >= (3,0) ):
    pass
    #swig_opts.append("-python3")
    #sys.stdout.write( "Currently SWIG only works with python2" )
    #sys.exit(1)

src_folder = "cpp/src"
inc_folder = "cpp/include"
ce_updater_sources = ["ce_updater.cpp", "cf_history_tracker.cpp",
                      "additional_tools.cpp", "histogram.cpp",
                      "wang_landau_sampler.cpp", "adaptive_windows.cpp",
                      "mc_observers.cpp", "linear_vib_correction.cpp",
                      "cluster.cpp", "cluster_tracker.cpp",
                      "named_array.cpp",
                      "row_sparse_struct_matrix.cpp", "pair_constraint.cpp",
                      "eshelby_tensor.cpp", "eshelby_sphere.cpp",
                      "eshelby_cylinder.cpp"]
                      
ce_updater_sources = [src_folder+"/"+srcfile for srcfile in ce_updater_sources]
ce_updater_sources.append( "cemc/ce_updater/ce_updater.i" )
ce_updater = Extension( "_ce_updater", sources=ce_updater_sources,include_dirs=[inc_folder,np.get_include()],
extra_compile_args=["-std=c++11","-fopenmp"], language="c++", swig_opts=swig_opts,
libraries=["gomp","pthread"] )

setup(
    name="cemc",
    ext_modules = [ce_updater],
    version=1.0,
    author="David Kleiven",
    author_email="davidkleiven446@gmail.com",
    description="Monte Carlo routines for Cluster Expansion",
    packages=find_packages(),
    package_data={"ce_updater":["_ce_updater.so"]},
    include_package_data=True
)
