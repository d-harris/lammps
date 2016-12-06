/*----------------------------------------------------------------------
 *   dump_hdf5 - parallel HDF5 routine
 *
 *     Crown Copyright 2015 AWE.
 *
 *     This program is free software: you can redistribute it and/or
 *     modify it under the terms of the GNU General Public License as
 *     published by the Free Software Foundation, either version 3 of
 *     the License, or (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 *     See the GNU General Public License for more details:
 *       <http://www.gnu.org/licenses/>.
 *----------------------------------------------------------------------*/

#ifdef DUMP_CLASS

DumpStyle(hdf5,DumpHDF5)

#else

#ifndef DUMP_HDF5_H
#define DUMP_HDF5_H

#include "dump.h"
#include "typhonio.h"

namespace LAMMPS_NS {

class DumpHDF5 : public Dump {
 public:
  DumpHDF5(class LAMMPS *, int, char **);
  virtual ~DumpHDF5();
  virtual void init_style();
  bigint memory_usage();

 private:
// VERSION INFO
#define VERSION "lammps"

  int nevery;                // dump frequency to check Fix against

  int nmine;                 // # of lines I am dumping
  int *vtype;                // type of each vector (INT, DOUBLE)
  char **vformat;            // format string for each vector element

  char *columns;             // column labels

  int maxlocal;              // size of atom selection and variable arrays
  int *choose;               // 1 if output this atom, 0 if no
  double *dchoose;           // value for each atom to threshhold against

  int nfield;                // # of keywords listed by user

  int *field2index;          // which compute,fix,variable calcs this field
  int *argindex;             // index into compute,fix scalar_atom,vector_atom
                             // 0 for scalar_atom, 1-N for vector_atom values

  int ncompute;              // # of Compute objects used by dump
  char **id_compute;         // their IDs
  class Compute **compute;   // list of ptrs to the Compute objects

  int nvariable;             // # of Variables used by dump
  char **id_variable;        // their names
  int *variable;             // list of indices for the Variables
  double **vbuf;             // local storage for variable evaluation

  int   tio_exists;          // 1 if HDF5 file exists
  int   state_exists;        // 0 if new state, else 1
  int   no_pack;             // 1 if mesh_only arg given, else 0
  int   lenbody;             // length of first part of filename
  int   state_per_file;      // one state per file (true if multifile true)

// Type to hold Quant arrays in linked list
  struct Quant
  {
     char           *name;   // Quant name
     double         *ptr;    // used to point at existing vector array
     double         *data;   // used to store copied data (need alloc/dealloc)
     struct Quant   *next;
     struct Quant   *prev;
  };

  Quant         *first;    // first item in linked list.
  Quant         *last;     // last item in linked list.
  unsigned int   nquant;   // number of quants in list

  // HDF5 variables 

  MPI_Info       mpiinfo;
  TIO_t          err;          // HDF5 error flag
  TIO_File_t     fid;          // HDF5 file ID
  TIO_Object_t   stid, msid;   // HDF5 state and mesh IDs
  TIO_Object_t   matid, qid;   // HDF5 material and quantity IDs

  // coordinates

  double   *x;                  // local x atom coordinates
  double   *y;                  // local y atom coordinates
  double   *z;                  // local z atom coordinates

  // private methods

  void openfile();                        // create or open HDF5 file
  void write_header(bigint);
  int  count();
  void pack(int *);
  void write_data(int, double *);
  void parse_fields(int, char **);
  int add_compute(char *);
  int add_variable(char *);
  int modify_param(int, char **);
  void apply_PBC();                // Applies Periodic Boundary Conditions to x,y,z

  // customize by adding a method prototype

  typedef void (DumpHDF5::*FnPtrPack)(int);
  FnPtrPack *pack_choice;              // ptrs to pack functions

  void pack_mass(int);

  void pack_vx(int);
  void pack_vy(int);
  void pack_vz(int);
  void pack_fx(int);
  void pack_fy(int);
  void pack_fz(int);
  void pack_q(int);

  void pack_compute(int);
  void pack_variable(int);
  DumpHDF5::Quant * create_quant(char *name);
  void write_quant(Quant * qdat);
  void clear_quants();
};

}

#endif
#endif
