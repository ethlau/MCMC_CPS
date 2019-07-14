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
#include "gas_model.h"

using namespace std;

double kappa_integrant(double x, void* p){ // see Ma et al 2015 Eq 3.2
    double *params = (double *) p;
    double ell = params[0];
    double chi = params[1];
    double Rs = params[2]; //NFW scale radius in Mpc
    double rhoS = params[3]; // NFW density at the scale radius
    double Rvir = params[4];
    double rho_NFW = rhoS/( (x*Rvir/Rs) * pow(1+x*Rvir/Rs,2.0) );
    double integrant;
    if(ell*x>0){
        integrant = 4.0*M_PI*pow(Rvir,3)*pow(x,2) * sin(ell*Rvir*x/chi) / (ell*Rvir*x/chi) * rho_NFW;
    }
    else if(ell*x==0) {
        integrant = 4.0*M_PI*pow(Rvir,3)*pow(x,2) * rho_NFW;
    }
    return integrant;
}

double ttx_func(double x, void * p) {

  double *params = (double *)p;
  gas_model gmod;
  gmod.set_A_nt(params[0]);
  gmod.set_n(params[1]);
  gmod.set_C(params[2]);
  double beta = params[3];
  double n = params[1];

  double ff = pow(gmod.theta(x, beta), n)*pow(x,2);
  return ff;
}

double tx_func(double x, void * p) {

  double *params = (double *)p;
  gas_model gmod;
  gmod.set_A_nt(params[0]);
  gmod.set_n(params[1]);
  gmod.set_C(params[2]);
  double beta = params[3];
  double n = params[1];

  double ff = pow(gmod.theta(x, beta),n+1.0)*pow(x,2);
  return ff;
}

double tx_func_p(double x, void * p) {

  double *params = (double *)p;
  gas_model gmod;
  gmod.set_A_nt(params[0]);
  gmod.set_n(params[1]);
  gmod.set_C(params[2]);
  double beta = params[3];
  double A_nt = params[0];
  double n = params[1];

  double ff = A_nt*pow(gmod.theta(x,beta),n-1.0)*pow(x,2);
  return ff;
}


double ftx_func(double x, void * p) {

  double *params = (double *)p;
  gas_model gmod;
  gmod.set_A_nt(params[0]);
  gmod.set_n(params[1]);
  gmod.set_C(params[2]);
  double beta = params[3];
  double A_nt = params[0];
  double n = params[1];

  double ff = gmod.f(x)*pow(gmod.theta(x, beta),n)*pow(x,2);
  return ff;
}

double sx_func (double x, void * params) {
    double f = (x - (1.0+x)*log(1.0+x)) / (pow(x,3)*pow(1.0+x,3));
    return f;
}


double ss_func(double x, void * params) {
  double C = *(double *) params;
  gas_model gmod;
  gmod.set_C(C);
  //cout << "In ss_func: C=" << C << endl;
  double f = gmod.S_cx(x)*pow(x,2);
  return f;
}

double fx_func(double x, void * params) {
  double C = *(double *) params;
  gas_model gmod;
  gmod.set_C(C);
  //cout << "In fx_func: C=" << C << endl;
  double g = gmod.f(x)*x/pow(1.0+x,2);
  return g;
}


// function for solving model goes here

double gasmod_apply_bc(const gsl_vector * v, void *p) {

  double *params_all = (double *)p;
  double params[8];
  for (int i = 0; i < 8; i ++) params[i] = params_all[i];
  gas_model gmod(params);
  gmod.set_C(params_all[8]);
  gmod.set_mass(params_all[9]);
  gmod.set_vcmax(params_all[10]);
  gmod.set_ri(params_all[11]);
  gmod.set_mgas(params_all[12]);
  gmod.set_xs(params_all[13]);
  gmod.set_f_s(params_all[14]);
 
  double x0 = gsl_vector_get (v, 0); // beta
  double x1 = gsl_vector_get (v, 1); // Cf
  if (x0<0.01) x0 = 0.01;
  //if (x0>14) x0 = 14;
  if (x1<0.01) x1 = 0.01;
  //if (x1>14) x1 = 14;
  return sqrt(pow(gmod.energy_constraint(x0, x1),2)+pow(gmod.pressure_constraint(x0, x1),2));
}

int gasmod_constraints(const gsl_vector *x, void *p, gsl_vector *f) {

  double *params_all = (double *)p;
  double params[8];
  for (int i = 0; i < 8; i ++) params[i] = params_all[i];
    
  //double p[15] = {n, eps, eps_dm, fs_0, fs_alpha, A_nt, B_nt, gamma_nt, C, mass, vcmax, ri, mgas, xs, f_s};
  gas_model gmod(params);

  gmod.set_C(params_all[8]);
  gmod.set_mass(params_all[9]);
  gmod.set_vcmax(params_all[10]);
  gmod.set_ri(params_all[11]);
  gmod.set_mgas(params_all[12]);
  gmod.set_xs(params_all[13]);
  gmod.set_f_s(params_all[14]);
  //cout << "in gasmod_constraints: C=" << gmod.get_C() << endl;

  double x0 = gsl_vector_get (x, 0); // beta
  double x1 = gsl_vector_get (x, 1); // Cf
  if (x0 < 1e-7) x0 = 100.0;
  //if (x0>14) x0 = 14;
  if (x1 < 1e-7) x1 = 100.0;
  //if (x1>14) x1 = 14;
  
  gsl_vector_set (f, 0, gmod.energy_constraint(x0, x1));
  gsl_vector_set (f, 1, gmod.pressure_constraint(x0, x1));

  return GSL_SUCCESS;
}


int gasmod_constraints_fdf(const gsl_vector *x, void *p, gsl_vector *f, gsl_matrix *J) {

  gasmod_constraints (x, p, f);

  return GSL_SUCCESS;
}

double gxs (double x, void *p) {
  double *params = (double *)p;
  double C = params[0];
  double f_s = params[1];
  double y;

  //cout << "In gxs" << endl;
  y = (log(1.0+x) - x/(1.0+x)) - (log(1.0+C) - C/(1.0+C))*f_s/(1.0+f_s);
  if (y != y  || f_s <= 0) {
    cout << C << " " << f_s << " " << y << endl;
  }
  return y;

}

double dgxs (double x, void *p) {
  double *params = (double *)p;
  double dy;

  dy =  x/((1.+x)*(1.+x));

  return dy;
}

void gxs_fdf (double x, void *p, double *y, double *dy){

  double *params = (double *)p;
  double C = params[0];
  double f_s = params[1];

  *y = log(1.0+x) - x/(1.0+x) - (log(1.0+C) - C/(1.0+C))*f_s/(1+f_s);
  *dy = x/((1.+x)*(1.+x));

  if (*y != *y || *dy != *dy ) {
    cout << C << " " << f_s << endl;
    cout << *y << " " << *dy << endl;
  }

}
