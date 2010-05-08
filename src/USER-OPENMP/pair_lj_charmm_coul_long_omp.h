/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(lj/charmm/coul/long/omp,PairLJCharmmCoulLongOMP)

#else

#ifndef LMP_PAIR_LJ_CHARMM_COUL_LONG_OMP_H
#define LMP_PAIR_LJ_CHARMM_COUL_LONG_OMP_H

#include "pair_omp.h"

namespace LAMMPS_NS {

class PairLJCharmmCoulLongOMP : public PairOMP {
 public:
  PairLJCharmmCoulLongOMP(class LAMMPS *);
  ~PairLJCharmmCoulLongOMP();

  virtual void compute(int, int);
  virtual void compute_inner();
  virtual void compute_middle();
  virtual void compute_outer(int, int);

  virtual void settings(int, char **);
  virtual void coeff(int, char **);

  virtual void init_style();
  virtual void init_list(int, class NeighList *);
  virtual double init_one(int, int);

  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);

  virtual double single(int, int, int, int, double, double, double, double &);

  void *extract(char *);

  virtual double memory_usage();

 protected:
  template <int EVFLAG, int EFLAG, int NEWTON_PAIR> void eval();
  template <int NEWTON_PAIR> void eval_inner();
  template <int NEWTON_PAIR> void eval_middle();
  template <int EVFLAG, int EFLAG, int VFLAG, int NEWTON_PAIR> 
  void eval_outer();

 protected:
  int implicit;
  double cut_lj_inner,cut_lj;
  double cut_lj_innersq,cut_ljsq;
  double cut_coul,cut_coulsq;
  double cut_bothsq;
  double denom_lj;
  double **epsilon,**sigma,**eps14,**sigma14;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double **lj14_1,**lj14_2,**lj14_3,**lj14_4;
  double *cut_respa;
  double g_ewald;

  double tabinnersq;
  double *rtable,*drtable,*ftable,*dftable,*ctable,*dctable;
  double *etable,*detable,*ptable,*dptable,*vtable,*dvtable;
  int ncoulshiftbits,ncoulmask;

  void allocate();
  void init_tables();
  void free_tables();
};

}

#endif
#endif
