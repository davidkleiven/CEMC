# distutils: language = c++

cdef extern from "ce_updater.hpp":
  cdef cppclass CEUpdater:
      CEUpdater() except +

      # Initialize the object
      void init(object BC, object corrFunc, object ecis) except +

      # Clear update history
      void clear_history()

      # Undo all chages since last call to clear_history
      void undo_changes()

      # Update the correlation functions
      void update_cf(object system_changes) except +

      # Add linear vibration correction
      void add_linear_vib_correction(object dict_str_double)

      # Get vibration energy
      double vib_energy(double T)

      double get_energy()

      double calculate(object system_changes) except +

      object get_cf()

      void set_ecis(object ecis)

      object get_singlets()
      void get_singlets(object array)
