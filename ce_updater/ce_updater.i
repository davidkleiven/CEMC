%module ce_updater
%include "exception.i"
%include <std_string.i>
%{
#include "ce_updater.hpp"
#include "wang_landau_sampler.hpp"
%}
%include "ce_updater.hpp"
%include "matrix.hpp"
%include "cf_history_tracker.hpp"
%include "wang_landau_sampler.hpp"
%include "histogram.hpp"
