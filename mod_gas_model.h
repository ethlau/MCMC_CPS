// class for the gas model
// code originally by Suman Bhattacharya, modified by Samuel Flender
#ifndef GAS_MODEL_HEADER_INCLUDED
#define GAS_MODEL_HEADER_INCLUDED

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_dht.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include "xray.h"

using namespace std;

struct parameters {
  double alpha0; // fiducial : 0.18
  double n_nt;   // fiducial : 0.80
  double beta;   // fiducial : 0.50
  double eps_f;  // fiducial : 3.97e-6
  double eps_DM; // fiducial : 0.00
  double f_star; // fiducial : 0.026
  double S_star; // fiducial : 0.12
  double A_C;    // fiducial : 1.00
  double gamma_mod0; // fiducial : 0.10
  double gamma_mod_zslope; // fiducial : 1.72
  double x_break; // fiducial : 0.195
  double x_smooth; // fiducial : 0.01
  double n_nt_mod; // fiducial : 0.80
  double clump0;
  double clump_zslope;
  double x_clump;
  double alpha_clump1;
  double alpha_clump2;
};

void set_fiducial_parameters (struct parameters *params);

void set_parameters (double pzero, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16, double p17, struct parameters *params);

/*
struct model_param{

  float conc_norm = 1.0;
  float conc_mass_norm = 1.0;
  float delta_rel, delta_rel_n, delta_rel_zslope;
  float ad_index;
  float eps_fb, eps_dm;
  float fs_0, fs_alpha;
  int pturbrad = 2;
  bool verbose = false;
  float overden_id = -1.0;
  int relation = 3;
  float rcutoff = 2.0;
};
*/

struct my_func_params { float a; float b; float c; double d; int e;float f;};

class gas_model {

    friend double gasmod_apply_bc(const gsl_vector * x, void *p);

    protected:

    double delta_rel, delta_rel_n, n, eps, eps_dm, fs_0, fs_alpha, f_s, Mpiv, chi_turb, delta_rel_zslope;
    double clump0, alpha_clump1, alpha_clump2, x_clump;
    float C, ri, rhoi, mass, radius, vcmax, mgas, Ytot, pressurebound, R500toRvir;
    double xs;
    double final_beta, final_Cf, p0, rho0, T0; // need to define these
    double PI, m_sun, G, mpc, mu_e, mmw, m_p, clight, eV, sigma_T, me_csq, m_e, q;
    //double Aprime, Bprime;
    public:

    gas_model (float inp1, float inp2, float inp3, float inp4, float inp5) {
        Mpiv = 3.0e14; //in Msol
        set_constants();
        delta_rel = inp1;
        n = inp2;
        C = inp3;
        f_s = inp4;
        delta_rel_n = inp5; //0.8;
        chi_turb = 0.0;
        pressurebound = 1.0;
    }

};

// -- end of class
// functions for integrals go here


#endif
