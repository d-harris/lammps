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

#include <mpi.h>
#include <string.h>
#include "write_hdf5_restart.h"
#include "atom.h"
#include "atom_vec.h"
#include "atom_vec_hybrid.h"
#include "group.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "update.h"
#include "neighbor.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "universe.h"
#include "comm.h"
#include "output.h"
#include "thermo.h"
#include "mpiio.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

// same as read_restart.cpp

#define MAGIC_STRING "LammpS RestartT"
#define ENDIAN 0x0001
#define ENDIANSWAP 0x1000
#define VERSION_NUMERIC 0

enum{VERSION,SMALLINT,TAGINT,BIGINT,
     UNITS,NTIMESTEP,DIMENSION,NPROCS,PROCGRID,
     NEWTON_PAIR,NEWTON_BOND,
     XPERIODIC,YPERIODIC,ZPERIODIC,BOUNDARY,
     ATOM_STYLE,NATOMS,NTYPES,
     NBONDS,NBONDTYPES,BOND_PER_ATOM,
     NANGLES,NANGLETYPES,ANGLE_PER_ATOM,
     NDIHEDRALS,NDIHEDRALTYPES,DIHEDRAL_PER_ATOM,
     NIMPROPERS,NIMPROPERTYPES,IMPROPER_PER_ATOM,
     TRICLINIC,BOXLO,BOXHI,XY,XZ,YZ,
     SPECIAL_LJ,SPECIAL_COUL,
     MASS,PAIR,BOND,ANGLE,DIHEDRAL,IMPROPER,
     MULTIPROC,MPIIO,PROCSPERFILE,PERPROC,
     IMAGEINT,BOUNDMIN,TIMESTEP,
     ATOM_ID,ATOM_MAP_STYLE,ATOM_MAP_USER,ATOM_SORTFREQ,ATOM_SORTBIN,
     COMM_MODE,COMM_CUTOFF,COMM_VEL};

enum{IGNORE,WARN,ERROR};                    // same as thermo.cpp

/* ---------------------------------------------------------------------- */

WriteHDF5Restart::WriteHDF5Restart(LAMMPS *lmp) : Pointers(lmp)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  multiproc = 0;
  fp = NULL;
}

/* ----------------------------------------------------------------------
   called as write_hdf5_restart command in input script
------------------------------------------------------------------------- */

void WriteHDF5Restart::command(int narg, char **arg)
{
  if (domain->box_exist == 0)
    error->all(FLERR,"write_hdf5_restart command before simulation box is defined");
  if (narg < 1) error->all(FLERR,"Illegal write_hdf5_restart command");

  // if filename contains a "*", replace with current timestep

  char *ptr;
  int n = strlen(arg[0]) + 16;
  char *file = new char[n];

  if ((ptr = strchr(arg[0],'*'))) {
    *ptr = '\0';
    sprintf(file,"%s" BIGINT_FORMAT "%s",arg[0],update->ntimestep,ptr+1);
  } else strcpy(file,arg[0]);

  // check for multiproc output and an MPI-IO filename - invalid with hdf5

  if (strchr(arg[0],'%')) multiproc = nprocs;
  else multiproc = 0;
  if (strstr(arg[0],".mpiio")) mpiioflag = 1;
  else mpiioflag = 0;

  if (multiproc || mpiioflag)
  error->all(FLERR,"Invalid write_hdf5_restart filename; mpiio and % not allowed");

  // init entire system since comm->exchange is done
  // comm::init needs neighbor::init needs pair::init needs kspace::init, etc

  if (comm->me == 0 && screen)
    fprintf(screen,"System init for write_hdf5_restart ...\n");
  lmp->init();

  // move atoms to new processors before writing file
  // enforce PBC in case atoms are outside box
  // call borders() to rebuild atom map since exchange() destroys map
  // NOTE: removed call to setup_pre_exchange
  //   used to be needed by fixShearHistory for granular
  //   to move history info from neigh list to atoms between runs
  //   but now that is done via FIx::post_run()
  //   don't think any other fix needs this or should do it
  //   e.g. fix evaporate should not delete more atoms

  // modify->setup_pre_exchange();
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);

  // write single restart file

  write(file);
  delete [] file;
}

/* ----------------------------------------------------------------------
   called from command() and directly from output within run/minimize loop
   file = final file name to write, except may contain a "%"
------------------------------------------------------------------------- */

void WriteHDF5Restart::write(char *file)
{
  // special case where reneighboring is not done in integrator
  //   on timestep restart file is written (due to build_once being set)
  // if box is changing, must be reset, else restart file will have
  //   wrong box size and atoms will be lost when restart file is read
  // other calls to pbc and domain and comm are not made,
  //   b/c they only make sense if reneighboring is actually performed

  if (neighbor->build_once) domain->reset_box();

  // natoms = sum of nlocal = value to write into restart file
  // if unequal and thermo lostflag is "error", don't write restart file

  bigint nblocal = atom->nlocal;
  MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (natoms != atom->natoms && output->thermo->lostflag == ERROR)
    error->all(FLERR,"Atom count is inconsistent, cannot write restart file");

  // open single restart file
  
  openfile(file);

  // proc 0 writes magic string, endian flag, numeric version

  if (me == 0) {
    magic_string();
    endian();
    version_numeric();
  }

  // write header, groups, pertype info, force field info
  // all procs involved for create/close, but only proc 0 writes
  
  header();
/*
  group->write_hdf5_restart(fp);
  type_arrays();
  force_fields();

  // all procs write fix info

  modify->write_hdf5_restart(fp);

  // communication buffer for my atom info
  // max_size = largest buffer needed by any proc

  int max_size;
  int send_size = atom->avec->size_restart();
  MPI_Allreduce(&send_size,&max_size,1,MPI_INT,MPI_MAX,world);

  double *buf;
  memory->create(buf,max_size,"write_hdf5_restart:buf");

  // all procs write file layout info which may include per-proc sizes

  file_layout(send_size);

  // header info is complete
  // if multiproc output:
  //   close header file, open multiname file on each writing proc,
  //   write PROCSPERFILE into new file

  if (multiproc) {
    if (me == 0 && fp) {
      fclose(fp);
      fp = NULL;
    }

    char *multiname = new char[strlen(file) + 16];
    char *ptr = strchr(file,'%');
    *ptr = '\0';
    sprintf(multiname,"%s%d%s",file,icluster,ptr+1);
    *ptr = '%';

    if (filewriter) {
      fp = fopen(multiname,"wb");
      if (fp == NULL) {
        char str[128];
        sprintf(str,"Cannot open restart file %s",multiname);
        error->one(FLERR,str);
      }
      write_int(PROCSPERFILE,nclusterprocs);
    }

    delete [] multiname;
  }

  // pack my atom data into buf

  AtomVec *avec = atom->avec;
  int n = 0;
  for (int i = 0; i < atom->nlocal; i++) n += avec->pack_restart(i,&buf[n]);

  // if any fix requires it, remap each atom's coords via PBC
  // is because fix changes atom coords (excepting an integrate fix)
  // just remap in buffer, not actual atoms

  if (modify->restart_pbc_any) {
    int triclinic = domain->triclinic;
    double *lo,*hi,*period;

    if (triclinic == 0) {
      lo = domain->boxlo;
      hi = domain->boxhi;
      period = domain->prd;
    } else {
      lo = domain->boxlo_lamda;
      hi = domain->boxhi_lamda;
      period = domain->prd_lamda;
    }

    int xperiodic = domain->xperiodic;
    int yperiodic = domain->yperiodic;
    int zperiodic = domain->zperiodic;

    double *x;
    int m = 0;
    for (int i = 0; i < atom->nlocal; i++) {
      x = &buf[m+1];
      if (triclinic) domain->x2lamda(x,x);

      if (xperiodic) {
        if (x[0] < lo[0]) x[0] += period[0];
        if (x[0] >= hi[0]) x[0] -= period[0];
        x[0] = MAX(x[0],lo[0]);
      }
      if (yperiodic) {
        if (x[1] < lo[1]) x[1] += period[1];
        if (x[1] >= hi[1]) x[1] -= period[1];
        x[1] = MAX(x[1],lo[1]);
      }
      if (zperiodic) {
        if (x[2] < lo[2]) x[2] += period[2];
        if (x[2] >= hi[2]) x[2] -= period[2];
        x[2] = MAX(x[2],lo[2]);
      }

      if (triclinic) domain->lamda2x(x,x);
      m += static_cast<int> (buf[m]);
    }
  }

  // MPI-IO output to single file

  if (mpiioflag) {
    if (me == 0 && fp) {
      fclose(fp);
      fp = NULL;
    }
    mpiio->openForWrite(file);
    mpiio->write(headerOffset,send_size,buf);
    mpiio->close();
  }

  // output of one or more native files
  // filewriter = 1 = this proc writes to file
  // ping each proc in my cluster, receive its data, write data to file
  // else wait for ping from fileproc, send my data to fileproc

  else {
    int tmp,recv_size;

    if (filewriter) {
      MPI_Status status;
      MPI_Request request;
      for (int iproc = 0; iproc < nclusterprocs; iproc++) {
        if (iproc) {
          MPI_Irecv(buf,max_size,MPI_DOUBLE,me+iproc,0,world,&request);
          MPI_Send(&tmp,0,MPI_INT,me+iproc,0,world);
          MPI_Wait(&request,&status);
          MPI_Get_count(&status,MPI_DOUBLE,&recv_size);
        } else recv_size = send_size;

        write_double_vec(PERPROC,recv_size,buf);
      }
      fclose(fp);
      fp = NULL;

    } else {
      MPI_Recv(&tmp,0,MPI_INT,fileproc,0,world,MPI_STATUS_IGNORE);
      MPI_Rsend(buf,send_size,MPI_DOUBLE,fileproc,0,world);
    }
  }

  // clean up

  memory->destroy(buf);

  // invoke any fixes that write their own restart file

  for (int ifix = 0; ifix < modify->nfix; ifix++)
    if (modify->fix[ifix]->restart_file)
      modify->fix[ifix]->write_hdf5_restart_file(file);
*/
  closefile();
  
}

/* ----------------------------------------------------------------------
   proc 0 writes out problem description
------------------------------------------------------------------------- */

void WriteHDF5Restart::header()
{
  
  err = TIO_Create_Vargroup(fid,stid,"Header",&vgid);

  write_string(VERSION,universe->version,vgid);
  write_int(SMALLINT,sizeof(smallint),vgid);
  write_int(IMAGEINT,sizeof(imageint),vgid);
  write_int(TAGINT,sizeof(tagint),vgid);
  write_int(BIGINT,sizeof(bigint),vgid);
  write_string(UNITS,update->unit_style,vgid);
  write_bigint(NTIMESTEP,update->ntimestep,vgid);
  write_int(DIMENSION,domain->dimension,vgid);
  write_int(NPROCS,nprocs,vgid);
  write_int_vec(PROCGRID,3,comm->procgrid,vgid);
  write_int(NEWTON_PAIR,force->newton_pair,vgid);
  write_int(NEWTON_BOND,force->newton_bond,vgid);
  write_int(XPERIODIC,domain->xperiodic,vgid);
  write_int(YPERIODIC,domain->yperiodic,vgid);
  write_int(ZPERIODIC,domain->zperiodic,vgid);
  write_int_vec(BOUNDARY,6,&domain->boundary[0][0],vgid);

  // added field for shrink-wrap boundaries with minimum - 2 Jul 2015

  double minbound[6];
  minbound[0] = domain->minxlo; minbound[1] = domain->minxhi;
  minbound[2] = domain->minylo; minbound[3] = domain->minyhi;
  minbound[4] = domain->minzlo; minbound[5] = domain->minzhi;
  write_double_vec(BOUNDMIN,6,minbound,vgid);

  // write atom_style and its args

  write_string(ATOM_STYLE,atom->atom_style,vgid);
/*  fwrite(&atom->avec->nargcopy,sizeof(int),1,fp);
  for (int i = 0; i < atom->avec->nargcopy; i++) {
    int n = strlen(atom->avec->argcopy[i]) + 1;
    fwrite(&n,sizeof(int),1,fp);
    fwrite(atom->avec->argcopy[i],sizeof(char),n,fp);
  }
*/
  write_bigint(NATOMS,natoms,vgid);
  write_int(NTYPES,atom->ntypes,vgid);
  write_bigint(NBONDS,atom->nbonds,vgid);
  write_int(NBONDTYPES,atom->nbondtypes,vgid);
  write_int(BOND_PER_ATOM,atom->bond_per_atom,vgid);
  write_bigint(NANGLES,atom->nangles,vgid);
  write_int(NANGLETYPES,atom->nangletypes,vgid);
  write_int(ANGLE_PER_ATOM,atom->angle_per_atom,vgid);
  write_bigint(NDIHEDRALS,atom->ndihedrals,vgid);
  write_int(NDIHEDRALTYPES,atom->ndihedraltypes,vgid);
  write_int(DIHEDRAL_PER_ATOM,atom->dihedral_per_atom,vgid);
  write_bigint(NIMPROPERS,atom->nimpropers,vgid);
  write_int(NIMPROPERTYPES,atom->nimpropertypes,vgid);
  write_int(IMPROPER_PER_ATOM,atom->improper_per_atom,vgid);

  write_int(TRICLINIC,domain->triclinic,vgid);
  write_double_vec(BOXLO,3,domain->boxlo,vgid);
  write_double_vec(BOXHI,3,domain->boxhi,vgid);
  write_double(XY,domain->xy,vgid);
  write_double(XZ,domain->xz,vgid);
  write_double(YZ,domain->yz,vgid);

  write_double_vec(SPECIAL_LJ,3,&force->special_lj[1],vgid);
  write_double_vec(SPECIAL_COUL,3,&force->special_coul[1],vgid);

  write_double(TIMESTEP,update->dt,vgid);

  write_int(ATOM_ID,atom->tag_enable,vgid);
  write_int(ATOM_MAP_STYLE,atom->map_style,vgid);
  write_int(ATOM_MAP_USER,atom->map_user,vgid);
  write_int(ATOM_SORTFREQ,atom->sortfreq,vgid);
  write_double(ATOM_SORTBIN,atom->userbinsize,vgid);

  write_int(COMM_MODE,comm->mode,vgid);
  write_double(COMM_CUTOFF,comm->cutghostuser,vgid);
  write_int(COMM_VEL,comm->ghost_velocity,vgid);

  // -1 flag signals end of header

//  int flag = -1;
//  fwrite(&flag,sizeof(int),1,fp);

  err = TIO_Close_Vargroup(fid,vgid);
  
}

/* ----------------------------------------------------------------------
   proc 0 writes out any type-based arrays that are defined
------------------------------------------------------------------------- */

void WriteHDF5Restart::type_arrays()
{
  if (atom->mass) write_double_vec(MASS,atom->ntypes,&atom->mass[1],vgid);

  // -1 flag signals end of type arrays

  int flag = -1;
  fwrite(&flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 writes out and force field styles and data that are defined
------------------------------------------------------------------------- */

void WriteHDF5Restart::force_fields()
{
  if (force->pair && force->pair->restartinfo) {
    write_string(PAIR,force->pair_style,vgid);
    force->pair->write_hdf5_restart(fp);
  }
  if (atom->avec->bonds_allow && force->bond) {
    write_string(BOND,force->bond_style,vgid);
    force->bond->write_hdf5_restart(fp);
  }
  if (atom->avec->angles_allow && force->angle) {
    write_string(ANGLE,force->angle_style,vgid);
    force->angle->write_hdf5_restart(fp);
  }
  if (atom->avec->dihedrals_allow && force->dihedral) {
    write_string(DIHEDRAL,force->dihedral_style,vgid);
    force->dihedral->write_hdf5_restart(fp);
  }
  if (atom->avec->impropers_allow && force->improper) {
    write_string(IMPROPER,force->improper_style,vgid);
    force->improper->write_hdf5_restart(fp);
  }

  // -1 flag signals end of force field info

  int flag = -1;
  fwrite(&flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 writes out file layout info
   all procs call this method, only proc 0 writes to file
------------------------------------------------------------------------- */

void WriteHDF5Restart::file_layout(int send_size)
{
  if (me == 0) {
    write_int(MULTIPROC,multiproc,vgid);
    write_int(MPIIO,mpiioflag,vgid);
  }

  if (mpiioflag) {
    int *all_send_sizes;
    memory->create(all_send_sizes,nprocs,"write_hdf5_restart:all_send_sizes");
    MPI_Gather(&send_size, 1, MPI_INT, all_send_sizes, 1, MPI_INT, 0,world);
    if (me == 0) fwrite(all_send_sizes,sizeof(int),nprocs,fp);
    memory->destroy(all_send_sizes);
  }

  // -1 flag signals end of file layout info

  if (me == 0) {
    int flag = -1;
    fwrite(&flag,sizeof(int),1,fp);
  }

  // if MPI-IO file, broadcast the end of the header offste
  // this allows all ranks to compute offset to their data

/*  if (mpiioflag) {
    if (me == 0) headerOffset = ftell(fp);
    MPI_Bcast(&headerOffset,1,MPI_LMP_BIGINT,0,world);
  } */
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// low-level fwrite methods
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------


void WriteHDF5Restart::openfile(char *file)
{
  err = TIO_Create(file, &fid, TIO_ACC_REPLACE, "LAMMPS", VERSION_NUMERIC,
                         "-", "title", world, mpiinfo, comm->me);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 restart file");

  TIO_Time_t st_time = (TIO_Time_t) (update->ntimestep * update->dt);

  err = TIO_Create_State(fid, "Restart", &stid, update->ntimestep, st_time, "-");
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 restart State");
}

/* ---------------------------------------------------------------------- */

void WriteHDF5Restart::closefile()
{
  err = TIO_Close_State(fid, stid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 restart state");

  err = TIO_Close(fid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 restart file");
}

/* ---------------------------------------------------------------------- */

void WriteHDF5Restart::magic_string()
{
  int n = strlen(MAGIC_STRING) + 1;
  char *str = new char[n];
  strcpy(str,MAGIC_STRING);
  fwrite(str,sizeof(char),n,fp);
  delete [] str;
}

/* ---------------------------------------------------------------------- */

void WriteHDF5Restart::endian()
{
  int endian = ENDIAN;
  fwrite(&endian,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   write a scalar int into restart file
------------------------------------------------------------------------- */

void WriteHDF5Restart::write_int(const char * name, const int value, 
                                 TIO_Object_t objid)
{
  TIO_Size_t dims0D[0] = 1;

  err = TIO_Create_Variable(fid, objid, name, &vid, TIO_INT, TIO_0D,
                            dims0D, NULL);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 int variable");

  if (me==0) {
    err = TIO_Write_Variable(fid, vid, TIO_INT, value);
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not write HDF5 int variable");
  }

  err = TIO_Close_Variable(fid, vid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 int variable");
}

/* ----------------------------------------------------------------------
   write a flag and a bigint into restart file
------------------------------------------------------------------------- */

void WriteHDF5Restart::write_bigint(const char * name, const int value, 
                                    TIO_Object_t objid)
{
  TIO_Size_t dims0D[0] = 1;

  err = TIO_Create_Variable(fid, objid, name, &vid, TIO_ULLONG, TIO_0D,
                            dims0D, NULL);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 bigint variable");

  if (me==0) {
    err = TIO_Write_Variable(fid, vid, TIO_ULLONG, value);
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not write HDF5 bigint variable");
  }

  err = TIO_Close_Variable(fid, vid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 bigint variable");
}

/* ----------------------------------------------------------------------
   write a flag and a double into restart file
------------------------------------------------------------------------- */

void WriteHDF5Restart::write_double(int flag, double value)
                                    TIO_Object_t objid)
{
  TIO_Size_t dims0D[0] = 1;

  err = TIO_Create_Variable(fid, objid, name, &vid, TIO_DOUBLE, TIO_0D,
                            dims0D, NULL);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 double variable");

  if (me==0) {
    err = TIO_Write_Variable(fid, vid, TIO_DOUBLE, value);
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not write HDF5 double variable");
  }

  err = TIO_Close_Variable(fid, vid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 double variable");
}

/* ----------------------------------------------------------------------
   write a flag and a char string (including NULL) into restart file
------------------------------------------------------------------------- */

void WriteHDF5Restart::write_string(int flag, const char *value)
                                    TIO_Object_t objid)
{
  TIO_Size_t dims1D[1] = 1;

  err = TIO_Create_Variable(fid, objid, name, &vid, TIO_STRING, TIO_1D,
                            dims1D, NULL);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 string variable");

  if (me==0) {
    err = TIO_Write_Variable(fid, vid, TIO_STRING, value);
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not write HDF5 string variable");
  }

  err = TIO_Close_Variable(fid, vid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 string variable");
}

/* ----------------------------------------------------------------------
   write a flag and vector of N ints into restart file
------------------------------------------------------------------------- */

void WriteHDF5Restart::write_int_vec(int flag, int n, int *vec)
                                     TIO_Object_t objid)
{
  TIO_Size_t dims1D[0] = n;

  err = TIO_Create_Variable(fid, objid, name, &vid, TIO_INT, TIO_1D,
                            dims1D, NULL);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 int vec variable");

  if (me==0) {
    err = TIO_Write_Variable(fid, vid, TIO_INT, value);
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not write HDF5 int vec variable");
  }

  err = TIO_Close_Variable(fid, vid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 int vec variable");
}

/* ----------------------------------------------------------------------
   write a flag and vector of N doubles into restart file
------------------------------------------------------------------------- */

void WriteHDF5Restart::write_double_vec(int flag, int n, double *vec)
                                        TIO_Object_t objid)
{
  TIO_Size_t dims1D[0] = n;

  err = TIO_Create_Variable(fid, objid, name, &vid, TIO_DOUBLE, TIO_1D,
                            dims1D, NULL);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not create HDF5 double vec variable");

  if (me==0) {
    err = TIO_Write_Variable(fid, vid, TIO_DOUBLE, value);
    if (TIO_SUCCESS != err)
       error->all(FLERR,"Could not write HDF5 double vec variable");
  }

  err = TIO_Close_Variable(fid, vid);
  if (TIO_SUCCESS != err)
     error->all(FLERR,"Could not close HDF5 double vec variable");
}
