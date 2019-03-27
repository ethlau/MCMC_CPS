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

double ttx_func(double x, void * p) {
  gas_model gmod(p);
  double n = gmod.n;
  double ff = pow(gmod.theta(x, beta), n)*pow(x,2);
  return ff;
}

double tx_func(double x, void * p) {
  gas_model gmod(p);
  double n = gmod.n;
  return  pow(gmod.theta(x, beta),n+1.0)*pow(x,2);
}

double tx_func_p(double x, void * p) {
  gas_model gmod(p);
  double n = gmod.n;
  return delta_rel*pow(gmod.theta(x,beta),n-1.0)*pow(x,2);
}


double ftx_func(double x, void * p) {
  gas_model gmod(p);
  double n = gmod.n;
  return gmod.f(x)*pow(gmod.theta(x, beta),gmod.n)*pow(x,2);
}

double sx_func (double x, void * p) {
  return (x - (1.0+x)*log(1.0+x)) / (pow(x,3)*pow(1.0+x,3));
}

double ss_func(double x, void *p) {
  gas_model gmod(p);
  return gmod.S_cx(x)*pow(x,2);
}

double fx_func(double x, void *p) {
  gas_model gmod(p);
  return gmod.f(x)*x/pow(1.0+x,2);
}


// function for solving model goes here

double gasmod_apply_bc(const gsl_vector * v, void *p) {

  gas_model gmod(p);
  double x0 = gsl_vector_get (v, 0); // beta
  double x1 = gsl_vector_get (v, 1); // Cf
  if (x0<0.01) x0 = 0.01;
  //if (x0>14) x0 = 14;
  if (x1<0.01) x1 = 0.01;
  //if (x1>14) x1 = 14;
  return sqrt(pow(gmod.energy_constraint(x0, x1),2)+pow(gmod.pressure_constraint(x0, x1),2));
}

int gasmod_constraints(const gsl_vector *x, void *p, gsl_vector *f) {

  gas_model gmod(p);
  const double x0 = gsl_vector_get (x, 0); // beta
  const double x1 = gsl_vector_get (x, 1); // Cf
  //if (x0<0.01) x0 = 0.01;
  ////if (x0>14) x0 = 14;
  //if (x1<0.01) x1 = 0.01;
  ////if (x1>14) x1 = 14;
  
  gsl_vector_set (f, 0, gmod.energy_constraint(x0, x1));
  gsl_vector_set (f, 1, gmod.pressure_constraint(x0, x1));

  return GSL_SUCCESS;
}

double gxs (double x, void *p) {
  double *params = (double *)p;
  double C = params[0];
  double f_s = params[1];
  double y;

  y = (log(1.0+x) - x/(1.0+x)) - (log(1.0+C) - C/(1.0+C))*f_s/(1.0+f_s);
  //cout  << "y = " << y << endl;
  if (y != y  || f_s <= 0) {
    cout << C << " " << f_s << " " << y << endl;
  }
  return y;

}

double dgxs (double x, void *p) {
  double *params = (double *)p;
  double dy;

  dy =  x/((1.+x)*(1.+x));
  //cout  << "dy = " << dy << endl;

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
