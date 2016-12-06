/*----------------------------------------------------------------------
 *   dump_hdf5 - parallel HDF5 routine
 *
 *     British Crown Owned Copyright 2015/AWE.
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


#include "stdlib.h"
#include "string.h"
#include "dump_hdf5.h"
#include "atom.h"
#include "force.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "input.h"
#include "variable.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "time.h"

using namespace LAMMPS_NS;

// customize by adding keyword
// same list as in compute_property.cpp, also customize that command
// have changed the first enum list so may need changing back to full list.
enum{MASS,VX,VY,VZ,FX,FY,FZ,Q,COMPUTE};
enum{LT,LE,GT,GE,EQ,NEQ};
enum{INT,DOUBLE};

#define INVOKED_PERATOM 8

/* ---------------------------------------------------------------------- */
// HDF5 dump class for Lammps. Based on dump_custom.

// Requires TyphonIO HDF5 format libary, available via git from:

//   https://github.com/UK-MAC/typhonio

// State    == Each dump is a new State within a file or in a separate file.
// Mesh     == Atomic coordinates (as a point mesh)
// Material == Atomic Type
// Quants   == vector compute data. Must be mesh-wide so masks not really an option

// parse routine parses the input arguments, determines which quants are required
// and sets up pointer function to appropriate pack routine

// pack routines create a linked list. Each item in the list is set either as a pointer
// to the compute's vector array, or else it creates a 1D array that holds the
// selected column's data from a 2D array. Any memory allocated here is freed
// after the write is complete

// All floating point data is converted to type float on writing

// Writes a single file across all procs so modify multiproc and multifile below
// if multifile dump_hdf5 writes one state per file
// multiproc must not be defined by user. dump_hdf5 always sets it to true
// Must use group all

// All Create/Open/Close TyphonIO routines must be called collectively
// All Write (and Read if used) routines can be called independently

DumpHDF5::DumpHDF5(LAMMPS *lmp, int narg, char **arg) :
  Dump(lmp, narg, arg)
{
  no_pack = 0;
  if (5 == narg) no_pack = 1;  // no args to parse. Dumping coordinatess only.

  if (igroup != group->find("all")) error->all(FLERR,"DumpHDF5 must use group all");
  if (binary || compressed || multiproc)
    error->all(FLERR,"Invalid dumpHDF5 filename");


  // if multifile set state_per_file to true. reset multifile to 0 to
  // ensure correct behaviour later on

  state_per_file = 0;
  if (multifile) {
     state_per_file = 1;
     multifile = 0;
  }

  // set multiproc to true. Ensures Dump::write() behaves as we want it to
  // need to copy a section of dump_write.cpp out to here though as it
  // normally triggers on the filename (looking for a '%')
  multiproc = 1;
  nclusterprocs = 1;
  filewriter = 1;
  fileproc = me;
  MPI_Comm_split(world,me,0,&clustercomm);


  // set clearstep flag - will ensure computes are cleared for next dump step

  clearstep = 1;

  nevery = atoi(arg[3]);

  // nullify quant linked list

  first = NULL;
  last = NULL;

// MPI Hints to improve performance on Lustre filesystems
   MPI_Info_create(&mpiinfo);
   MPI_Info_set(mpiinfo, "romio_ds_write", "DISABLE");
   MPI_Info_set(mpiinfo, "romio_ds_read", "DISABLE");
   MPI_Info_set(mpiinfo, "romio_cb_write", "enable");
   MPI_Info_set(mpiinfo, "romio_cb_read", "disable");

  // Nullify pointers that are not used by dump_hdf5 but are
  // deleted in ~Dump()

  fp = NULL; 
  format_default = NULL;


  // number of fields (= nquants for dumping)

  size_one = nfield = narg-5;
  pack_choice = new FnPtrPack[nfield];
  vtype = new int[nfield];

  // computes, fixes, variables which the dump accesses

  memory->create(field2index,nfield,"dump:field2index");
  memory->create(argindex,nfield,"dump:argindex");

  ncompute = 0;
  id_compute = NULL;
  compute = NULL;

  nvariable = 0;
  id_variable = NULL;
  variable = NULL;
  vbuf = NULL;

  // process attributes

  parse_fields(narg,arg);


  // setup column string

  int n = 0;
  for (int iarg = 5; iarg < narg; iarg++) n += strlen(arg[iarg]) + 2;
  columns = new char[n];
  columns[0] = '\0';
  for (int iarg = 5; iarg < narg; iarg++) {
    strcat(columns,arg[iarg]);
    strcat(columns," ");
  }
}

/* ---------------------------------------------------------------------- */

DumpHDF5::~DumpHDF5()
{
  delete [] pack_choice;
  delete [] vtype;
  memory->destroy(field2index);
  memory->destroy(argindex);

  for (int i = 0; i < ncompute; i++) delete [] id_compute[i];
  memory->sfree(id_compute);
  delete [] compute;

  for (int i = 0; i < nvariable; i++) delete [] id_variable[i];
  memory->sfree(id_variable);
  delete [] variable;
  for (int i = 0; i < nvariable; i++) memory->destroy(vbuf[i]);
  delete [] vbuf;

  delete [] columns;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::init_style()
{
  // find current ptr for each compute,fix,variable
  // check that fix frequency is acceptable

  int icompute;
  for (int i = 0; i < ncompute; i++) {
    icompute = modify->find_compute(id_compute[i]);
    if (icompute < 0) error->all(FLERR,"Could not find dump hdf5 compute ID");
    compute[i] = modify->compute[icompute];
  }

  int ivariable;
  for (int i = 0; i < nvariable; i++) {
    ivariable = input->variable->find(id_variable[i]);
    if (ivariable < 0)
      error->all(FLERR,"Could not find dump custom variable name");
    variable[i] = ivariable;
  }
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::write_header(bigint ndump)
{
  char        st_name[TIO_STRLEN];
  TIO_Time_t  st_time;

//   if (me==0) fprintf(screen,"Writing header\n");

  // Create or open the HDF5 file

  openfile();


  // HDF5 State - named after timestep

  sprintf(st_name,BIGINT_FORMAT,update->ntimestep);
  st_time = (TIO_Time_t) (update->ntimestep * update->dt);

  // Create state. If this fails then state already exists so close file and
  // set flag to skip later writes within routine
  // (ideally would want to skip out of dump entirely)

  state_exists = 0;
  err = TIO_Create_State(fid, st_name, &stid, update->ntimestep, st_time, "-");
  if (TIO_SUCCESS != err) {
     state_exists = 1;
     err = TIO_Close(fid);
     if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not close HDF5 file");
     if (0 == me) {
        fprintf(screen,"HDF5 State %s exists. Skipping...\n", st_name);
        if (logfile) fprintf(logfile,"HDF5 State %s exists. Skipping...\n", st_name);
     }
  }

}

/* ---------------------------------------------------------------------- */

int DumpHDF5::count()
{
  // invoke Computes for per-atom quantities

  if (ncompute) {
    for (int i = 0; i < ncompute; i++)
      if (!(compute[i]->invoked_flag & INVOKED_PERATOM)) {
        compute[i]->compute_peratom();
        compute[i]->invoked_flag |= INVOKED_PERATOM;
      }
  }

  // evaluate atom-style Variables for per-atom quantities

  if (nvariable)
    for (int i = 0; i < nvariable; i++)
      input->variable->compute_atom(variable[i],igroup,vbuf[i],1,0);


  return 1;   // Dummy value recognised as non-error by main code.
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack(int *ids)  // ids not used in this routine but needed for compatibility
{
  // This routine actually sets up the quants that need writing in the HDF5 file

  // skip pack if state already exists in file or no quantities need packing

  if (state_exists || no_pack) return;

  // These 2D LAMMPS arrays need copying as the data is stored in the wrong
  // order for us to simply point to them as a 1D array.
  // The pointers to the 2D arrays' data are set up here. The copies themselves
  // are done in write_data() on a per quant basis. This is slower but
  // uses less memory.

//   if (me==0) fprintf(screen,"Packing\n");
  int icompute;
  for (int i = 0; i < ncompute; i++) {
    icompute = modify->find_compute(id_compute[i]);
    if (icompute < 0) error->all(FLERR,"Could not find dump hdf5 compute ID");
    compute[i] = modify->compute[icompute];
  }

  // loop through quants that need setting up for write

  for (int n = 0; n < size_one; n++) (this->*pack_choice[n])(n);

}

/* ---------------------------------------------------------------------- */

void DumpHDF5::write_data(int n, double *mybuf)
{
  int        me     = static_cast<int> (comm->me);
  int        nprocs = static_cast<int> (comm->nprocs);
  bigint     nlocal = atom->nlocal;
  bigint     natoms = atom->natoms;

  // skip write if state already exists in file

  if (state_exists) return;

  // Determine chunk extents on each processor

  bigint *nl = (long int *) 
       memory->smalloc(nprocs * sizeof(bigint), "dump_hdf5:nl");
  bigint *nh = (long int *) 
       memory->smalloc(nprocs * sizeof(bigint), "dump_hdf5:nh");
  bigint *chunksize = (long int *) 
       memory->smalloc(nprocs * sizeof(bigint), "dump_hdf5:chunksize");

  MPI_Allgather(&nlocal, 1, MPI_LMP_BIGINT, chunksize, 1, MPI_LMP_BIGINT, world);

  nl[0] = 0;
  nh[0] = chunksize[0]-1;

  for (int i=1; i < nprocs; i++) {
     nl[i] = nh[i-1] + 1;
     nh[i] = chunksize[i] + nh[i-1];
  }


  // ************************************************
  // HDF5 Mesh

  err = TIO_Create_Mesh(fid, stid, "Atom Coordinates", &msid, TIO_MESH_POINT,
        TIO_COORD_CARTESIAN, static_cast<TIO_Bool_t> (0), "atoms", 1,
        TIO_DATATYPE_NULL, TIO_FLOAT, TIO_3D, natoms, 0, 0, 0, nprocs,
        "Angstroms", "Angstroms", "Angstroms", "x", "y", "z");
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 mesh");

  // Set the mesh range

  err = TIO_Set_Mesh_Range(fid, msid, TIO_DOUBLE, TIO_3D, &boxxlo, &boxxhi, &boxylo,
                           &boxyhi, &boxzlo, &boxzhi);
  if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not set HDF5 mesh range");

  // Set the mesh chunk

  for (int i = 0; i < nprocs; i++) {
      err = TIO_Set_Point_Chunk(fid,msid,i,TIO_3D,nl[i],nh[i],0);
      if (TIO_SUCCESS != err)
         error->all(FLERR,"Could not set HDF5 mesh chunks");
  }

  memory->destroy(nl);
  memory->destroy(nh);
  memory->destroy(chunksize);

  // Write the mesh - need to copy coordinates into 1D arrays

  double  **px = atom->x;      // pointer to coordinate 2D array

  memory->create(x, nlocal, "dump_hdf5:x");
  memory->create(y, nlocal, "dump_hdf5:y");
  memory->create(z, nlocal, "dump_hdf5:z");

  for (int i = 0; i < nlocal; i++) {

     x[i]   = px[i][0];
     y[i]   = px[i][1];
     z[i]   = px[i][2];
  }

  // apply periodic boundary conditions to the coordinates
  apply_PBC();

  err = TIO_Write_PointMesh_Chunk(fid,msid,me,TIO_XFER_COLLECTIVE,TIO_DOUBLE,x,y,z);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not write HDF5 mesh chunks");

  memory->destroy(x);
  memory->destroy(y);
  memory->destroy(z);

  // ************************************************
  // HDF5 Material (Atom Types)

  int  *type = atom->type;
  int  ntypes = static_cast<int>(atom->ntypes);

  err = TIO_Create_Material(fid, msid, "Atom Types",&matid, TIO_INT, ntypes,
                            TIO_GHOSTS_NONE, TIO_FALSE, TIO_DATATYPE_NULL,
                            TIO_DATATYPE_NULL, TIO_DATATYPE_NULL);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 material");

  err = TIO_Write_PointMaterial_Chunk(fid, matid, me, TIO_XFER_COLLECTIVE,
                                      TIO_INT, type);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not write HDF5 material chunk");

  err = TIO_Close_Material(fid, matid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 material");

  // ************************************************
  // Quantities

  if (NULL != first)
     write_quant(first);

  // delete quant linked list

  clear_quants();

  // Close Mesh/State/File

  err = TIO_Close_Mesh(fid, msid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 mesh");
  err = TIO_Close_State(fid, stid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 state");
  err = TIO_Close(fid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 file");
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::parse_fields(int narg, char **arg)
{
  // customize by adding to if statement

  if (no_pack) {
     size_one = 1;  // dummy value to prevent crash in dump.cpp
     return;  // no arguments to parse
  }

  int i;
  for (int iarg = 5; iarg < narg; iarg++) {
    i = iarg-5;

    if (0 == strcmp(arg[iarg],"mass")) {
      pack_choice[i] = &DumpHDF5::pack_mass;
      vtype[i] = DOUBLE;
    } else if (0 == strcmp(arg[iarg],"vx")) {
      pack_choice[i] = &DumpHDF5::pack_vx;
      vtype[i] = DOUBLE;
    } else if (0 == strcmp(arg[iarg],"vy")) {
      pack_choice[i] = &DumpHDF5::pack_vy;
      vtype[i] = DOUBLE;
    } else if (0 == strcmp(arg[iarg],"vz")) {
      pack_choice[i] = &DumpHDF5::pack_vz;
      vtype[i] = DOUBLE;
    } else if (0 == strcmp(arg[iarg],"fx")) {
      pack_choice[i] = &DumpHDF5::pack_fx;
      vtype[i] = DOUBLE;
    } else if (0 == strcmp(arg[iarg],"fy")) {
      pack_choice[i] = &DumpHDF5::pack_fy;
      vtype[i] = DOUBLE;
    } else if (0 == strcmp(arg[iarg],"fz")) {
      pack_choice[i] = &DumpHDF5::pack_fz;
      vtype[i] = DOUBLE;

    } else if (0 == strcmp(arg[iarg],"q")) {
      if (!atom->q_flag)
      error->all(FLERR,"Dumping an atom property that isn't allocated");
      pack_choice[i] = &DumpHDF5::pack_q;
      vtype[i] = DOUBLE;

    // compute value = c_ID
    // if no trailing [], then arg is set to 0, else arg is int between []

    } else if (0 == strncmp(arg[iarg],"c_",2)) {
      pack_choice[i] = &DumpHDF5::pack_compute;
      vtype[i] = DOUBLE;

      int n = strlen(arg[iarg]);
      char *suffix = new char[n];
      strcpy(suffix,&arg[iarg][2]);

      char *ptr = strchr(suffix,'[');
      if (ptr) {
         if (suffix[strlen(suffix)-1] != ']')
         error->all(FLERR,"Invalid attribute in dump hdf5 command");
         argindex[i] = atoi(ptr+1);
         *ptr = '\0';
      } else argindex[i] = 0;

      n = modify->find_compute(suffix);
      if (n < 0) error->all(FLERR,"Could not find dump hdf5 compute ID");
      if (modify->compute[n]->peratom_flag == 0)
         error->all(FLERR,"Dump hdf5 compute does not compute per-atom info");
      if (argindex[i] == 0 && modify->compute[n]->size_peratom_cols > 0)
         error->all(FLERR,"Dump hdf5 compute does not calculate per-atom vector");
      if (argindex[i] > 0 && modify->compute[n]->size_peratom_cols == 0)
         error->all(FLERR,"Dump hdf5 compute does not calculate per-atom array");
      if (argindex[i] > 0 && 
         argindex[i] > modify->compute[n]->size_peratom_cols)
      error->all(FLERR,"Dump hdf5 compute vector is accessed out-of-range");

      field2index[i] = add_compute(suffix);
      delete [] suffix;

    // variable value = v_name

    } else if (strncmp(arg[iarg],"v_",2) == 0) {
      pack_choice[i] = &DumpHDF5::pack_variable;
      vtype[i] = DOUBLE;

      int n = strlen(arg[iarg]);
      char *suffix = new char[n];
      strcpy(suffix,&arg[iarg][2]);

      argindex[i] = 0;

      n = input->variable->find(suffix);
      if (n < 0) error->all(FLERR,"Could not find dump custom variable name");
      if (input->variable->atomstyle(n) == 0)
        error->all(FLERR,"Dump custom variable is not atom-style variable");

      field2index[i] = add_variable(suffix);
      delete [] suffix;

    } else error->all(FLERR,"Invalid attribute in dump hdf5 command");
  }
}

/* ----------------------------------------------------------------------
   add Compute to list of Compute objects used by dump
   return index of where this Compute is in list
   if already in list, do not add, just return index, else add to list
------------------------------------------------------------------------- */

int DumpHDF5::add_compute(char *id)
{
  int icompute;
  for (icompute = 0; icompute < ncompute; icompute++)
    if (0 == strcmp(id,id_compute[icompute])) break;
  if (icompute < ncompute) return icompute;

  id_compute = (char **)
    memory->srealloc(id_compute,(ncompute+1)*sizeof(char *),"dump:id_compute");
  delete [] compute;
  compute = new Compute*[ncompute+1];

  int n = strlen(id) + 1;
  id_compute[ncompute] = new char[n];
  strcpy(id_compute[ncompute],id);
  ncompute++;
  return ncompute-1;
}

/* ----------------------------------------------------------------------
   add Variable to list of Variables used by dump
   return index of where this Variable is in list
   if already in list, do not add, just return index, else add to list
------------------------------------------------------------------------- */

int DumpHDF5::add_variable(char *id)
{
  int ivariable;
  for (ivariable = 0; ivariable < nvariable; ivariable++)
    if (strcmp(id,id_variable[ivariable]) == 0) break;
  if (ivariable < nvariable) return ivariable;

  id_variable = (char **)
    memory->srealloc(id_variable,(nvariable+1)*sizeof(char *),
                     "dump:id_variable");
  delete [] variable;
  variable = new int[nvariable+1];
  delete [] vbuf;
  vbuf = new double*[nvariable+1];
  for (int i = 0; i <= nvariable; i++) vbuf[i] = NULL;

  int n = strlen(id) + 1;
  id_variable[nvariable] = new char[n];
  strcpy(id_variable[nvariable],id);
  nvariable++;
  return nvariable-1;
}

/* ---------------------------------------------------------------------- */

int DumpHDF5::modify_param(int narg, char **arg)
{
  // DumpHDF5 has no modification parameters so return 0

  return 0;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory in buf, choose, variable arrays
------------------------------------------------------------------------- */

bigint DumpHDF5::memory_usage()
{
  bigint   nlocal = static_cast<bigint> (atom->nlocal);

  bigint bytes  = Dump::memory_usage();
  bytes += 3 * nlocal * sizeof(double);         // mesh
  bytes +=   nquant * nlocal * sizeof(double);  // quants
  bytes += memory->usage(vbuf,nvariable,maxlocal);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::apply_PBC()
{
  // Apply periodic boundary conditions to the local coordinates.
  // Main code coordinates unaffected.

  double     boxxlen, boxylen, boxzlen;
  long int   nlocal = static_cast<long int> (atom->nlocal);

  boxxlen = abs(boxxlo) + abs(boxxhi);
  boxylen = abs(boxylo) + abs(boxyhi);
  boxzlen = abs(boxzlo) + abs(boxzhi);


  for (int i=0; i < nlocal; i++) {

    if (x[i] < boxxlo) x[i] += boxxlen;
    if (x[i] > boxxhi) x[i] -= boxxlen;
    if (y[i] < boxylo) y[i] += boxylen;
    if (y[i] > boxyhi) y[i] -= boxylen;
    if (z[i] < boxzlo) z[i] += boxzlen;
    if (z[i] > boxzhi) z[i] -= boxzlen;

  }
}

/* ----------------------------------------------------------------------
   extraction of Compute results
------------------------------------------------------------------------- */

void DumpHDF5::pack_compute(int n)
{
  double *vector = compute[field2index[n]]->vector_atom;
  double **array = compute[field2index[n]]->array_atom;
  int index = argindex[n];
  int nlocal = atom->nlocal;
  char name[TIO_STRLEN];
  char str[20];

  if (0 == index) {
     Quant *newquant = create_quant(compute[field2index[n]]->id);
     newquant->ptr = vector;
  } else {
    strcpy(name,compute[field2index[n]]->id);
    strcat(name,"_");
    sprintf(str,"%d",index);
    strcat(name,str);
    Quant *newquant = create_quant(name);
    newquant->data = new double[nlocal];
    index--;
    for (int i = 0; i < nlocal; i++) {
       newquant->data[i] = array[i][index];
    }
    nquant++;
  }
}

void DumpHDF5::pack_variable(int n)
{
  double *vector = vbuf[field2index[n]];

  Quant *newquant = create_quant(id_variable[n]);
  newquant->ptr = vector;
}

/* ----------------------------------------------------------------------
   one method for every attribute dump hdf5 can output
   the atom property is packed into buf starting at n with stride size_one
   customize a new attribute by adding a method
------------------------------------------------------------------------- */

void DumpHDF5::pack_mass(int n)
{
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;
  double *qmass;

  DumpHDF5::Quant *newquant = create_quant("Mass");

  if (rmass) {
    newquant->ptr = rmass;
  } else {
    newquant->data = new double[nlocal];
    for (int i = 0; i < nlocal; i++) {
      newquant->data[i] = mass[type[i]];
    }
    nquant++;
  }
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack_vx(int n)
{
  double **v = atom->v;
  int nlocal = atom->nlocal;

  Quant *newquant = create_quant("Velocity_X");

  newquant->data = new double[nlocal];
  for (int i = 0; i < nlocal; i++) {
     newquant->data[i] = v[i][0];
  }
  nquant++;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack_vy(int n)
{
  double **v = atom->v;
  int nlocal = atom->nlocal;

  Quant *newquant = create_quant("Velocity_Y");

  newquant->data = new double[nlocal];
  for (int i = 0; i < nlocal; i++) {
     newquant->data[i] = v[i][1];
  }
  nquant++;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack_vz(int n)
{
  double **v = atom->v;
  int nlocal = atom->nlocal;

  Quant *newquant = create_quant("Velocity_Z");

  newquant->data = new double[nlocal];
  for (int i = 0; i < nlocal; i++) {
     newquant->data[i] = v[i][2];
  }
  nquant++;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack_fx(int n)
{
  double **f = atom->f;
  int nlocal = atom->nlocal;

  Quant *newquant = create_quant("Force_X");

  newquant->data = new double[nlocal];
  for (int i = 0; i < nlocal; i++) {
     newquant->data[i] = f[i][0];
  }
  nquant++;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack_fy(int n)
{
  double **f = atom->f;
  int nlocal = atom->nlocal;

  Quant *newquant = create_quant("Force_Y");

  newquant->data = new double[nlocal];
  for (int i = 0; i < nlocal; i++) {
     newquant->data[i] = f[i][1];
  }
  nquant++;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack_fz(int n)
{
  double **f = atom->f;
  int nlocal = atom->nlocal;

  Quant *newquant = create_quant("Force_X");

  newquant->data = new double[nlocal];
  for (int i = 0; i < nlocal; i++) {
     newquant->data[i] = f[i][2];
  }
  nquant++;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::pack_q(int n)
{
  double *q = atom->q;
  int nlocal = atom->nlocal;

  Quant *newquant = create_quant("Atom Charge");
  newquant->ptr = q;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::openfile()
{
   char       count[5];
   time_t     now;
   char       date[TIO_STRLEN];
   char       filestore[TIO_STRLEN];
   struct tm  *ts;

   // setup filename using parsed args

   // first remove the .h5 if it's been included
   char *ptr = strcasestr(filename,".h5");    // "myfile*.h5\0"
   if (ptr) {
      *ptr = '\0';                        // "myfile*\0"
   }

   char *filecurrent = filename;          // filename used to open dumps
   if (state_per_file) {
     char *filestar = filecurrent;
     filecurrent = new char[strlen(filename) + 16];
     char *ptr = strchr(filestar,'*');
     *ptr = '\0';                         // "myfile\0"
     sprintf(filecurrent,"%s" BIGINT_FORMAT "%s",
             filestar,update->ntimestep,ptr+1);  // "myfile0001\0"
     *ptr = '*';
   }

   // now append ".h5"
   if (state_per_file) {
     filecurrent = (char *)
         memory->srealloc(filecurrent,(strlen(filecurrent)+3)*sizeof(char *),
         "dump_typhonio:filecurrent");
   }
   else
   {
     filecurrent = new char[strlen(filename) + 4];
     sprintf(filecurrent,"%s",filename);
   }
   strcat(filecurrent,".h5");

   tio_exists = 0;
   if (0 == me) {
     err = TIO_CheckFile(filecurrent);
     if (TIO_SUCCESS == err)  tio_exists = 1;
   }
   MPI_Bcast(&tio_exists, 1, MPI_INT, 0, world);

   // Get time/date

   time(&now);
   ts = localtime(&now);
   strftime(date, sizeof(date), "%a %d-%m-%Y %H:%M", ts);

   if (0 == tio_exists) {     // New file required
     err = TIO_Create(filecurrent, &fid, TIO_ACC_REPLACE, "LAMMPS", VERSION,
                            date, "title", world, mpiinfo, comm->me);
     if (TIO_SUCCESS != err)
        error->all(FLERR,"Could not create HDF5 dump file");
   } else {                   // Open existing file
     err = TIO_Open(filecurrent, &fid, TIO_ACC_READWRITE, "LAMMPS", VERSION,
                            date, "title",world, mpiinfo, comm->me);
     if (TIO_SUCCESS != err)
        error->all(FLERR,"Could not open HDF5 dump file");
   }
   delete [] filecurrent;
}

/* ---------------------------------------------------------------------- */

DumpHDF5::Quant * DumpHDF5::create_quant(char *name)
{
  Quant *qptr;
  if (NULL != first) {
    qptr=first;
    // check if list item already exists and return address if so
    while(1) {
      if (0 == strcmp(qptr->name,name)) {
        return qptr;
      }
      if (NULL == qptr->next) break;
      qptr = qptr->next;
    }
  }
  Quant *newquant = new Quant;
  if (NULL == newquant)
     error->all(FLERR,"Error creating new Dump HDF5 Quant");
  newquant->ptr = NULL;
  newquant->data = NULL;
  newquant->next = NULL;
  newquant->name = new char[TIO_STRLEN];
  if (NULL == newquant->name)
     error->all(FLERR,"Error creating new Dump HDF5 Quant name");
  strcpy(newquant->name,name);

  if (NULL == first) {
    first = newquant;
  } else {
    last->next = newquant;
  }
  last = newquant;
//   if (0 == me)
//     printf("Added Quant: %s\n", newquant->name);

  return last;
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::clear_quants()
{
  // nullify/free quant data pointers
  if (NULL == first) {
    return;
  } else {
    Quant * qptr = first;
    while (1) {
//       if (0 == me) printf("Clearing Quant: %s\n", qptr->name);
      if (NULL == qptr->data) {
        qptr->ptr = NULL;
      } else {
        memory->sfree(qptr->data);
        qptr->data = NULL;
      }
      if (NULL == qptr->next) {
        break;
      }
      qptr = qptr->next;
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpHDF5::write_quant(Quant *qdat)
{
  int   me = static_cast<int> (comm->me);
  Quant * qptr = qdat;

  while (1) {
//     if (0 == me) printf("Writing Quant: %s\n", qptr->name);
    err = TIO_Create_Quant(fid, msid, qptr->name, &qid, TIO_FLOAT, TIO_CENTRE_NODE,
                           TIO_GHOSTS_NONE, TIO_FALSE, "-");
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not create HDF5 quantity");

    if (NULL == qptr->data) {
       err = TIO_Write_PointQuant_Chunk(fid, qid, me, TIO_XFER_COLLECTIVE,
                                        TIO_DOUBLE, qptr->ptr);
    } else {
       err = TIO_Write_PointQuant_Chunk(fid, qid, me, TIO_XFER_COLLECTIVE,
                                        TIO_DOUBLE, qptr->data);
    }
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not write HDF5 quantity chunk");

    err = TIO_Close_Quant(fid, qid);
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not close HDF5 quantity");

    if (NULL == qptr->next) {
       break;
    } else {
       qptr = qptr->next;
    }
  }
}

// EOF
