from setuptools import setup, Extension

src_folder = "cpp/src"
inc_folder = "cpp/include"
ce_updater_sources = ["ce_updater.cpp","cf_history_tracker.cpp"]
ce_updater_sources = [src_folder+"/"+srcfile for srcfile in ce_updater_sources]
ce_updater_sources.append( "cpp/swig/ce_updater.i" )
ce_updater = Extension( "_ce_updater", sources=ce_updater_sources,include_dirs=[inc_folder],
extra_compile_args=["-std=c++11"], language="c++", swig_opts=["-modern","-Icpp/include","-c++"] )
setup(
    name="CEMonteCarlo",
    ext_modules = [ce_updater],
    py_modules=["ce_updater"],
    version=1.0,
    author="David Kleiven",
    author_email="davidkleiven446@gmail.com",
    description="Monte Carlo routines for Cluster Expansion",
    packages=["wanglandau","mcmc"]
)
