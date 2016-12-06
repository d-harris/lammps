/*----------------------------------------------------------------------
 *   write_hdf5_restart - parallel HDF5 restarts
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

#ifdef COMMAND_CLASS

CommandStyle(write_hdf5_restart,WriteHDF5Restart)

#else

#ifndef WRITE_HDF5_RESTART_H
#define WRITE_HDF5_RESTART_H

#include "pointers.h"
#include "typhonio.h"

namespace LAMMPS_NS {

class WriteHDF5Restart : protected Pointers {
 public:
  WriteHDF5Restart(class LAMMPS *);
  void command(int, char **);
//void multiproc_options(int, int, int, char **);
  void write(char *);

 private:
  int me,nprocs;
  FILE *fp;
  bigint natoms;         // natoms (sum of nlocal) to write into file

  int multiproc;             // 0 = proc 0 writes for all
                             // else # of procs writing files
  int nclusterprocs;         // # of procs in my cluster that write to one file
  int filewriter;            // 1 if this proc writes a file, else 0
  int fileproc;              // ID of proc in my cluster who writes to file
  int icluster;              // which cluster I am in
  int mpiioflag;             // 1 for MPIIO output, else 0

  void header();
  void type_arrays();
  void force_fields();
  void file_layout(int);

  void magic_string();
  void endian();
  void version_numeric();

  void openfile(char *file);
  void closefile();

  void write_int(int, int, TIO_Object_t);
  void write_bigint(int, bigint, TIO_Object_t);
  void write_double(int, double, TIO_Object_t);
  void write_string(int, const char *, TIO_Object_t);
  void write_int_vec(int, int, int *, TIO_Object_t);
  void write_double_vec(int, int, double *, TIO_Object_t);

  // HDF5 variables 

  MPI_Info       mpiinfo;
  TIO_t          err;          // HDF5 error flag
  TIO_File_t     fid;          // HDF5 file ID
  TIO_Object_t   stid, msid;   // HDF5 state and mesh IDs
  TIO_Object_t   vgid, vid;    // HDF5 vargroup and variable IDs

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Write_restart command before simulation box is defined

The write_restart command cannot be used before a read_data,
read_restart, or create_box command.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Restart file MPI-IO output not allowed with % in filename

This is because a % signifies one file per processor and MPI-IO
creates one large file for all processors.

E: Writing to MPI-IO filename when MPIIO package is not installed

Self-explanatory.

E: Cannot use write_restart fileper without % in restart file name

Self-explanatory.

E: Cannot use write_restart nfile without % in restart file name

Self-explanatory.

E: Atom count is inconsistent, cannot write restart file

Sum of atoms across processors does not equal initial total count.
This is probably because you have lost some atoms.

E: Cannot open restart file %s

Self-explanatory.

*/
