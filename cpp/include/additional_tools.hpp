#ifndef ADDITIONAL_TOOLS_H
#define ADDITIONAL_TOOLS_H

namespace additional_tools
{
  template<class T>
  void product( std::vector<T> &in, std::vector< std::vector<T> > &out, unsigned int repeat );
}

#include "additional_tools.tpp"
#endif
