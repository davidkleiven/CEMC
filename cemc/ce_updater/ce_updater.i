%module ce_updater
%include "exception.i"
%include <std_string.i>
%{
#define SWIG_FILE_WITH_INIT
#include "ce_updater.hpp"
#include "wang_landau_sampler.hpp"
#include "adaptive_windows.hpp"
%}
%include "numpy.i"

%init %{
  import_array();
%}

%include "ce_updater.hpp"
%include "matrix.hpp"
%include "matrix.tpp"
%include "cf_history_tracker.hpp"
%include "wang_landau_sampler.hpp"
%include "histogram.hpp"
%include "adaptive_windows.hpp"
%include "additional_tools.hpp"
%include "additional_tools.tpp"
%include "mc_observers.hpp"
