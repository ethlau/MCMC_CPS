#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <gsl/gsl_integration.h>

#include "cosmo.h"
#include "cluster.h"
#include "gas_model.h"
#include "xray.h"

#include "allvars.h"
#include "nrutil.h"
#include "proto.h"

#include "cfortran.h"

#define BOOST_PYTHON_MAX_ARITY 20

#define MAXBINS 500

double tarray[ntmax]; //keV
double zarray[nmmax]; //Solar unit
double rarray[nrmax]; 
double lambda_table[ntmax][nrmax];
double tres, zres, eres;

const double megapc = 3.0857e24; // in cm/s
const double mmw = 0.58824; // X=0.76 assumed
const double m_p  = 1.6726e-24;// mass proton, g
const double Msun = 1.99e33; //g 
const double sigma_T = 0.665245854e-24; //(cm)^2 Thomson cross-section
const double m_elect = 5.10998902e2; //keV/c^2
const double X = 0.76;// primordial hydrogen fraction
const double factor = (2.0*X+2.0)/(5.0*X+3.0);//conversion factor from gas pressure to electron pressure

using namespace std;

#define Nx 100
static double xp[Nx], yp[Nx], yp2[Nx];
static int Nspline;

std::vector<double> calc_Shaw_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x);

double calc_Flender_xray_luminosity (cosmo cosm_model, float z, float Mvir, std::vector<float> x);

double calc_Flender_mgas (cosmo cosm_model, float z, float Mvir, std::vector<float> x);

double calc_Flender_xray_temperature (cosmo cosm_model, float z, float Mvir, std::vector<float> x);

std::vector<double> calc_Flender_pressure_profile (cosmo cosm_model, float z, float Mvir, std::vector<float> x, double *Rs);

std::vector<double> calc_Flender_density_profile (cosmo cosm_model, float z, float Mvir, std::vector<float> x);

std::vector<double> calc_Flender_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x, double *Rs);

std::vector<double> calc_beta_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x, double *Rs);

void free_FFTdata();

void FFT_density_profile(double *output, double *bin, int nbin);

double sinc (double x);

struct Shaw_param{
  double alpha0; // fiducial : 0.18
  double n_nt;   // fiducial : 0.80
  double beta;   // fiducial : 0.50
  double eps_f;  // fiducial : 3.97e-6
  double eps_DM; // fiducial : 0.00
  double f_star; // fiducial : 0.026
  double S_star; // fiducial : 0.12
  double A_C;    // fiducial : 1.00
};

static struct Shaw_param S;

struct Flender_param{
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

static struct Flender_param F;

extern "C"
{
  void fhti_(int * n, double * mu, double * q, double * dlnr,
	     double * kr, int * kropt, double * wsave, int * ok);
  void fht_(int * n, double * a , int * dir, double * wsave);
  void fhtq_(int * n, double * a , int * dir, double * wsave);
}

//FFTlog paramaters
static double logrmin;
static double logrmax;
static int N;

static double q;
static double kr;
static int kropt;
//kropt = 0 to use input kr as is;                                                                                                     
//        1 to change kr to nearest low-ringing kr, quietly;                                                                           
//        2 to change kr to nearest low-ringing kr, verbosely;                                                                         
//        3 for option to change kr interactively.
static int dir;
static double *r, *a, *k, *wsave;

void set_FFTlog_param(){
	
  logrmin=log10(1e-8);
  logrmax=log10(1e+8);
  N=512;
  //N=64;
	
  kr=1.;
  kropt=1;
	
  dir=1;
	
  r= new double [N];
  a= new double [N];
  k= new double [N];
  wsave = new double [5*N];
	
}

//integral with input tables
double tab_spline_and_integral(int Nbin, double *xlist, double *ylist, double *zlist);

#include "boost/python/numpy.hpp"
#include "boost/python.hpp"
#include <stdexcept>
#include <algorithm>

namespace py = boost::python;
namespace npy = boost::python::numpy;

struct cosmo_params{
  double H0;
  double Omega_M;
  double Omega_b;
  double wt;
  double Omega_k;
  double ns;
  double hubble;
};

static struct cosmo_params CP;

void init_cosmology(double H0, double Omega_M, double Omega_b, double wt, double Omega_k, double ns, double nH, char *inputPk, int opt){

  set_cosmology_halo_info(inputPk, Omega_M, Omega_b, wt, H0/100.0, ns);
  CP.H0 = H0;
  CP.Omega_M = Omega_M;
  CP.Omega_b = Omega_b;
  CP.wt = wt;
  CP.Omega_k = Omega_k;
  CP.ns = CP.ns;
  CP.hubble = H0/100.0;

  set_emission_table(opt, nH,"apec_table.dat"); 

}

void free_cosmology(){
  free_halo_info();
}

void set_Flender_params(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16, double p17) {
  F.alpha0 =p0;
  F.n_nt   =p1;
  F.beta   =p2;
  F.eps_f  =p3;
  F.eps_DM =p4;
  F.f_star =p5;
  F.S_star =p6;
  F.A_C    =p7;
  F.gamma_mod0 =p8;
  F.gamma_mod_zslope =p9;
  F.x_break =p10;
  F.x_smooth =p11;
  F.n_nt_mod =p12;
  F.clump0 =p13;
  F.clump_zslope =p14;
  F.x_clump =p15;
  F.alpha_clump1 =p16;
  F.alpha_clump2 =p17;
}

npy::ndarray return_xx_power(npy::ndarray x_input){
  int nzbin = 21;
  float zmin = 1e-3;
  float zmax = 3.0;
	
  int nmbin = 21;
  float logMvir_min= 13.0;
  float logMvir_max= 16.0;

  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  
  float z, Mvir, x;
  std::vector<float> z_fft, M_fft, xlist;
  /* set up for redshift, virial mass, radius bin */
  //redshift
  for(int i=0;i<nzbin;i++){
    z = (log(zmax)-log(zmin))/(nzbin-1)*(float)(1.*i) + log(zmin);
    z = exp(z);
    z_fft.push_back(z);
  }
	
  //virial mass (Bryan & Norman) in [Msun], not in [Msun/h]
  for(int i=0;i<nmbin;i++){
    Mvir = (logMvir_max-logMvir_min)/(nmbin-1)*(float)(1.*i) + logMvir_min;
    Mvir = pow(10.0, Mvir);
    M_fft.push_back(Mvir);
  }
	
  //radius in unit of R500, x = r/R500
  float xmin = 1e-4;
  float xmax = 100.0;
  for(int i=0;i<Nx;i++){
    x = (log10(xmax)-log10(xmin))/Nx*(float)(1.*i+0.5) + log10(xmin);
    x = pow(10.0, x);
    xlist.push_back(x);
  }
	  
  set_FFTlog_param();
		
  double dlog_ell = (3.-(-5.))/(Nx-1);
  double tab_l_ls[Nx];
  double bin[Nx];
  double tab_yl_int[Nx];
  //double flux[nmbin][nzbin];
  
  double *tab_Fourier = new double[Nx * nzbin * nmbin];
  double *tab_r500 = new double[nzbin * nmbin];
  
  for(int i=0;i<Nx;i++){
    tab_l_ls[i] = -5.0+(double)(i)*dlog_ell;
    bin[i] = pow(10., tab_l_ls[i]);
  }
	  
  for(int i=0;i<nzbin;i++){
    //fprintf(stdout, ".");fflush(stdout);
    for(int j=0;j<nmbin;j++){
			
      //cout << zlist[i] << " " << Mlist[j] << endl;
      //flux[i][j] = calc_Flender_xray_flux (cosm_model, z_fft[i], M_fft[j], xlist); //ergs/s/cm^2
      double Rs;		
      std::vector<double> emission;
      emission = calc_Flender_xray_emissivity_profile(cosm_model, z_fft[i], M_fft[j], xlist, &Rs); // ergs/s/cm^3/str
      tab_r500[j+nmbin*i] = Rs;
      
      double yp1 = 1.e31;
      double ypn = 1.e31;
			
      Nspline=0;
      for(int k=0;k<Nx;k++){
	if(emission[k] > 0){
	  Nspline += 1;
	}
      }
			
      if(Nspline > 2){
	int id =0;
	for(int k=0;k<Nx;k++){
	  if(emission[k] > 0){
	    xp[id] = log10(xlist[k]);
	    yp[id] = log10(emission[k]);
	    id += 1;
	  }
	}
	double xout = pow(10., xp[Nspline-1]);
	spline(xp-1, yp-1, Nspline, yp1, ypn, yp2-1);

	FFT_density_profile(tab_yl_int, bin, Nx);	
				
      }else{
	for(int k=0;k<Nx;k++) tab_yl_int[k] = 0.0;
      }

      for(int k=0;k<Nx;k++) {
        tab_Fourier[k + Nx * ( j + nmbin * i)] = tab_yl_int[k];
      }
    }
  }
  
  /* Start computing a halo model */
  int nd = x_input.get_nd();
  if (nd != 1)
    throw std::runtime_error("a must be 1-dimensional");
  size_t nsh = x_input.shape(0);
  if (x_input.get_dtype() != npy::dtype::get_builtin<double>())
    throw std::runtime_error("a must be float64 array");

  auto xshape = x_input.get_shape();
  auto xstrides = x_input.get_strides();
  
  int Nbin=xshape[0];
  double lbin[Nbin];
  double signal[Nbin];
  for (int i = 0; i < xshape[0]; ++i) {
    lbin[i] = *reinterpret_cast<double *>(x_input.get_data() + i * xstrides[0]);
  }
  double yp1 = 1.e31;
  double ypn = 1.e31;

  int Nell = Nx;
  double tab_fc_int[Nell], tab_fc_int2[Nell];
				
  float dlnz = (log(zmax)-log(zmin))/(nzbin-1);
  float dlogm = (logMvir_max-logMvir_min)/(nmbin-1);
    
  double zlist[nzbin];
		
  double cl_xx_1_int_z[nzbin], cl_xx_1_int_z2[nzbin];
  double cl_xx_2_int_z[nzbin], cl_xx_2_int_z2[nzbin];
				
  double mlist[nmbin];
		
  double cl_xx_1_int_m[nmbin], cl_xx_1_int_m2[nmbin];
  double cl_xx_2_int_m[nmbin], cl_xx_2_int_m2[nmbin];
  
  for(int iz=0;iz<nzbin;iz++){
    cl_xx_1_int_z[iz] = 0.0;
    cl_xx_2_int_z[iz] = 0.0;
    cl_xx_1_int_z2[iz] = 0.0;
    cl_xx_2_int_z2[iz] = 0.0;

  }

  for(int jm=0;jm<nmbin;jm++){
    cl_xx_1_int_m[jm] = 0.0;
    cl_xx_2_int_m[jm] = 0.0;
    cl_xx_1_int_m2[jm] = 0.0;
    cl_xx_2_int_m2[jm] = 0.0;

  }

  for(int i=0;i<Nbin;i++){
    double ell_bin = lbin[i];						
    double cl_xx_1=0.0, cl_xx_2=0.0;
				
    for(int iz=0;iz<nzbin;iz++){
			
      double zhere = exp(dlnz*(double)(1.*iz) + log(zmin));
      zlist[iz] = log(zhere);
			
      // comoving distance in units of Mpc/h;
      double covd = chi_fast(zhere);
      double dVdz = covd*covd*C*CP.hubble/H_z(zhere); //dVdz in (Mpc/h)^3
      double calc_k = ell_bin/covd; // k in h Mpc^-1
      double gfac = (growth(1./(1.+zhere))/growth(1.0));
      // PowerSpec takes k in h/Mpc, outputs h^3 Mpc^-3
      double Pk = gfac*gfac*PowerSpec(calc_k);
			
      for(int jm=0;jm<nmbin;jm++){
         //if ( flux[iz][jm] >= 0.0 ) {	
	    double logMvir = dlogm*(double)(1.*jm) + logMvir_min;
	    mlist[jm] = logMvir;
	    double Mvir = pow(10., logMvir) * CP.hubble ;// in the unit of Msun/h
	    double rs = tab_r500[jm+nmbin*iz]; // in physical Mpc
	    double ells = covd/(1+zhere)/(rs *CP.hubble); // = ell_500 = D_A / Rs; D_a in Mpc/h;
	    double l_ls = ell_bin/ells;
	    for(int il=0;il<Nell;il++){
	        tab_fc_int[il] = tab_Fourier[il + Nell * (jm + nmbin * iz)]; // ergs/s/cm^3/sr
	    }
				
	    double xl;
	    int index;
				
	    index = -1;
	    for(int il=0;il<Nell-1;il++){
	        if(tab_l_ls[il] < log10(l_ls) && log10(l_ls) < tab_l_ls[il+1]){
	            index = il;
	        }
	    }
				
	    if(index < 0){
	        xl = 0;
	    } else {
	        xl = (tab_fc_int[index+1]-tab_fc_int[index])/(tab_l_ls[index+1]-tab_l_ls[index])*(log10(l_ls)-tab_l_ls[index]) + tab_fc_int[index];
	        xl = xl * 4*M_PI*(rs*megapc)/ells/ells; // ergs/s/cm^3/sr
	    }

	    //double m200m = M_vir_to_M_delta(zhere, Mvir, 200.0);		
	    double mf = dndlogm_fast(log10(Mvir), zhere); //input Mvir in Msun/h; mf in (h^3 Mpc^-3)
	    double b = halo_bias_fast(log10(Mvir), zhere); // b is dimensionless
								
	    cl_xx_1_int_m[jm] = mf * xl * xl; // erg^2/cm^4/sr^2 h^3 Mpc^-3 
	    cl_xx_2_int_m[jm] = mf * b * xl;  // erg/cm^2/sr h^3 Mpc^-3

         //} else {
	 //   cl_xx_1_int_m[jm] = 0.0;
	 //   cl_xx_2_int_m[jm] = 0.0;
         //}

      }
						
      spline(mlist-1, cl_xx_1_int_m-1, nmbin, yp1, ypn, cl_xx_1_int_m2-1);
      spline(mlist-1, cl_xx_2_int_m-1, nmbin, yp1, ypn, cl_xx_2_int_m2-1);
			
      double oneh_xx, twoh_xx;
			
      //double tab_spline_and_integral(int Nbin, double *xlist, double *ylist, double *zlist)
      
      // integrate over dlogM
      oneh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_1_int_m, cl_xx_1_int_m2);
      twoh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_2_int_m, cl_xx_2_int_m2);
			
      cl_xx_1_int_z[iz] = zhere * dVdz * oneh_xx; // erg^2/cm^4/str^2
      cl_xx_2_int_z[iz] = zhere * dVdz * twoh_xx * twoh_xx * Pk; //erg^2/cm^4/str^2
      						
   }
		
    spline(zlist-1, cl_xx_1_int_z-1, nzbin, yp1, ypn, cl_xx_1_int_z2-1);
    spline(zlist-1, cl_xx_2_int_z-1, nzbin, yp1, ypn, cl_xx_2_int_z2-1);
		
    cl_xx_1 = tab_spline_and_integral(nzbin, zlist, cl_xx_1_int_z, cl_xx_1_int_z2);
    cl_xx_2 = tab_spline_and_integral(nzbin, zlist, cl_xx_2_int_z, cl_xx_2_int_z2);

    signal[i] = cl_xx_1 + cl_xx_2;

    //printf("%f %e %e %e\n", ell_bin, cl_xx_1, cl_xx_2, signal[i]);
    if(cl_xx_1 != cl_xx_1 || cl_xx_2 != cl_xx_2) signal[i]=0;
    
  }
  
  free_FFTdata();
  delete[] tab_Fourier; 
  delete[] tab_r500;

  vector<double> v;
  for(int i=0;i<Nbin;i++) v.push_back(signal[i]);
  Py_intptr_t shape[1] = { v.size() };
  npy::ndarray result = npy::zeros(1, shape, npy::dtype::get_builtin<double>());
  copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));

  return result;
}

npy::ndarray return_xx_power_alt(npy::ndarray x_input){
  int nzbin = 11;
  float zmin = 1e-4;
  float zmax = 3.0;
	
  int nmbin = 11;
  float logMvir_min= 13.0;
  float logMvir_max= 16.0;

  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  
  float z, Mvir, x;
  std::vector<float> z_fft, M_fft, xlist;
  /* set up for redshift, virial mass, radius bin */
  //redshift
  for(int i=0;i<nzbin;i++){
    z = (log(zmax)-log(zmin))/(nzbin-1)*(float)(1.*i) + log(zmin);
    z = exp(z);
    z_fft.push_back(z);
  }
	
  //virial mass (Bryan & Norman) in [Msun], not in [Msun/h]
  for(int i=0;i<nmbin;i++){
    Mvir = (logMvir_max-logMvir_min)/(nmbin-1)*(float)(1.*i) + logMvir_min;
    Mvir = pow(10.0, Mvir);
    M_fft.push_back(Mvir);
  }
	
  //radius in unit of R500, x = r/R500
  float xmin = 5e-4;
  float xmax = 50.0;
  float dlogx = (log10(xmax)-log10(xmin))/Nx;

  for(int i=0;i<Nx;i++){
    x = dlogx*(float)(1.*i+0.5) + log10(xmin);
    x = pow(10.0, x);
    xlist.push_back(x);
  }
	  
  double dlog_ell = (3.-(-5.))/(Nx-1);
  double tab_l_ls[Nx];
  double bin[Nx];
  double tab_yl_int[Nx];
  //double flux[nmbin][nzbin];
  
  //double *tab_Fourier = new double[Nx * nzbin * nmbin];
  double *tab_profile = new double[Nx * nzbin * nmbin];
  //double *tab_r500 = new double[nzbin * nmbin];
  double *tab_rs = new double[nzbin * nmbin];
  
  for(int i=0;i<Nx;i++){
    tab_l_ls[i] = -5.0+(double)(i)*dlog_ell;
    bin[i] = pow(10., tab_l_ls[i]);
  }
	  
  for(int i=0;i<nzbin;i++){
    for(int j=0;j<nmbin;j++){
      double Rs;		
      std::vector<double> emission;
      emission = calc_Flender_xray_emissivity_profile(cosm_model, z_fft[i], M_fft[j], xlist, &Rs); // ergs/s/cm^3/str
      tab_rs[j+nmbin*i] = Rs;
      for(int k=0;k<Nx;k++) {
        tab_profile[k + Nx * ( j + nmbin * i)] = emission[k];
      }
    }
  }

  /* Start computing a halo model */
  int nd = x_input.get_nd();
  if (nd != 1)
    throw std::runtime_error("a must be 1-dimensional");
  size_t nsh = x_input.shape(0);
  if (x_input.get_dtype() != npy::dtype::get_builtin<double>())
    throw std::runtime_error("a must be float64 array");

  auto xshape = x_input.get_shape();
  auto xstrides = x_input.get_strides();
  
  int Nbin=xshape[0];
  double lbin[Nbin];
  double signal[Nbin];
  for (int i = 0; i < xshape[0]; ++i) {
    lbin[i] = *reinterpret_cast<double *>(x_input.get_data() + i * xstrides[0]);
  }
  double yp1 = 1.e31;
  double ypn = 1.e31;

  int Nell = Nx;
  double tab_fc_int[Nell], tab_fc_int2[Nell];
				
  float dlnz = (log(zmax)-log(zmin))/(nzbin-1);
  float dlogm = (logMvir_max-logMvir_min)/(nmbin-1);
    
  double zlist[nzbin];
		
  double cl_xx_1_int_z[nzbin], cl_xx_1_int_z2[nzbin];
  double cl_xx_2_int_z[nzbin], cl_xx_2_int_z2[nzbin];
				
  double mlist[nmbin];
		
  double cl_xx_1_int_m[nmbin], cl_xx_1_int_m2[nmbin];
  double cl_xx_2_int_m[nmbin], cl_xx_2_int_m2[nmbin];
  
  for(int iz=0;iz<nzbin;iz++){
    cl_xx_1_int_z[iz] = 0.0;
    cl_xx_2_int_z[iz] = 0.0;
    cl_xx_1_int_z2[iz] = 0.0;
    cl_xx_2_int_z2[iz] = 0.0;

  }

  for(int jm=0;jm<nmbin;jm++){
    cl_xx_1_int_m[jm] = 0.0;
    cl_xx_2_int_m[jm] = 0.0;
    cl_xx_1_int_m2[jm] = 0.0;
    cl_xx_2_int_m2[jm] = 0.0;

  }

  for(int i=0;i<Nbin;i++){
    double ell_bin = lbin[i];						
    double cl_xx_1=0.0, cl_xx_2=0.0;
				
    for(int iz=0;iz<nzbin;iz++){
			
      double zhere = exp(dlnz*(double)(1.*iz) + log(zmin));
      zlist[iz] = log(zhere);
			
      // comoving distance in units of Mpc/h;
      double covd = chi_fast(zhere);
      double dVdz = covd*covd*C*CP.hubble/H_z(zhere); //dVdz in (Mpc/h)^3
      double calc_k = ell_bin/covd; // k in h Mpc^-1
      double gfac = (growth(1./(1.+zhere))/growth(1.0));
      // PowerSpec takes k in h/Mpc, outputs h^3 Mpc^-3
      double Pk = gfac*gfac*PowerSpec(calc_k);
			
      for(int jm=0;jm<nmbin;jm++){
	    double logMvir = dlogm*(double)(1.*jm) + logMvir_min;
	    mlist[jm] = logMvir;
	    double Mvir = pow(10., logMvir) * CP.hubble ;// in the unit of Msun/h
	    double rs = tab_rs[jm+nmbin*iz]; // in physical Mpc
	    double ells = covd/(1+zhere)/(rs *CP.hubble); // = ell_s = D_A / Rs; D_a = cchi/(1+z) in Mpc/h;
	    double l_ls = ell_bin/ells;
	    for(int k=0;k<Nx;k++){
	        tab_fc_int[k] = tab_profile[k + Nx*(jm + nmbin*iz)]; // ergs/s/cm^3/sr
	    }
	    double xl = 0.0;
	    for(int il=0;il<Nell;il++){
                double a = l_ls * xlist[il];
	        xl += dlogx * pow(xlist[il], 3.0) * tab_fc_int[il] * sinc(a);
            }
            xl *= 4.0*M_PI*rs*megapc/(ells*ells); //ergs/s/cm^2/sr
	    //double m200m = M_vir_to_M_delta(zhere, Mvir, 200.0);

	    double mf = dndlogm_fast(log10(Mvir), zhere); //input Mvir in Msun/h; mf in (h^3 Mpc^-3)
	    double b = halo_bias_fast(log10(Mvir), zhere); // b is dimensionless
								
	    cl_xx_1_int_m[jm] = mf * xl * xl; // erg^2/cm^4/sr^2 h^3 Mpc^-3 
	    cl_xx_2_int_m[jm] = mf * b * xl;  // erg/cm^2/sr h^3 Mpc^-3

      }
						
      spline(mlist-1, cl_xx_1_int_m-1, nmbin, yp1, ypn, cl_xx_1_int_m2-1);
      spline(mlist-1, cl_xx_2_int_m-1, nmbin, yp1, ypn, cl_xx_2_int_m2-1);
			
      double oneh_xx, twoh_xx;
      
      // integrate over dlogM
      oneh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_1_int_m, cl_xx_1_int_m2);
      twoh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_2_int_m, cl_xx_2_int_m2);
			
      cl_xx_1_int_z[iz] = zhere * dVdz * oneh_xx; // erg^2/cm^4/str^2
      cl_xx_2_int_z[iz] = zhere * dVdz * twoh_xx * twoh_xx * Pk; //erg^2/cm^4/str^2
      						
   }
		
    spline(zlist-1, cl_xx_1_int_z-1, nzbin, yp1, ypn, cl_xx_1_int_z2-1);
    spline(zlist-1, cl_xx_2_int_z-1, nzbin, yp1, ypn, cl_xx_2_int_z2-1);
		
    cl_xx_1 = tab_spline_and_integral(nzbin, zlist, cl_xx_1_int_z, cl_xx_1_int_z2);
    cl_xx_2 = tab_spline_and_integral(nzbin, zlist, cl_xx_2_int_z, cl_xx_2_int_z2);

    signal[i] = cl_xx_1 + cl_xx_2;

    if(cl_xx_1 != cl_xx_1 || cl_xx_2 != cl_xx_2) signal[i]=0;
    
  }
  
  delete[] tab_profile; 
  delete[] tab_rs;

  vector<double> v;
  for(int i=0;i<Nbin;i++) v.push_back(signal[i]);
  Py_intptr_t shape[1] = { v.size() };
  npy::ndarray result = npy::zeros(1, shape, npy::dtype::get_builtin<double>());
  copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));

  return result;
}

npy::ndarray return_yy_power(npy::ndarray x_input){
  int nzbin = 11;
  float zmin = 1e-3;
  float zmax = 3.0;
	
  int nmbin = 11;
  float logMvir_min= 13.0;
  float logMvir_max= 16.0;

  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  
  float z, Mvir, x;
  std::vector<float> z_fft, M_fft, xlist;
  /* set up for redshift, virial mass, radius bin */
  //redshift
  for(int i=0;i<nzbin;i++){
    z = (log(zmax)-log(zmin))/(nzbin-1)*(float)(1.*i) + log(zmin);
    z = exp(z);
    z_fft.push_back(z);
  }
	
  //virial mass (Bryan & Norman) in [Msun], not in [Msun/h]
  for(int i=0;i<nmbin;i++){
    Mvir = (logMvir_max-logMvir_min)/(nmbin-1)*(float)(1.*i) + logMvir_min;
    Mvir = pow(10.0, Mvir);
    M_fft.push_back(Mvir);
  }
	
  //radius in unit of R500, x = r/R500
  float xmin = 1e-4;
  float xmax = 100.0;
  for(int i=0;i<Nx;i++){
    x = (log10(xmax)-log10(xmin))/Nx*(float)(1.*i+0.5) + log10(xmin);
    x = pow(10.0, x);
    xlist.push_back(x);
  }
	  
  set_FFTlog_param();
		
  double dlog_ell = (3.-(-5.))/(Nx-1);
  double tab_l_ls[Nx];
  double bin[Nx];
  double tab_yl_int[Nx];
  double flux[nmbin][nzbin];
  
  double *tab_Fourier = new double[Nx * nzbin * nmbin];
  double *tab_r500 = new double[nzbin * nmbin];
  
  for(int i=0;i<Nx;i++){
    tab_l_ls[i] = -5.0+(double)(i)*dlog_ell;
    bin[i] = pow(10., tab_l_ls[i]);
  }
	  
  for(int i=0;i<nzbin;i++){
    //fprintf(stdout, ".");fflush(stdout);
    for(int j=0;j<nmbin;j++){
			
      //cout << zlist[i] << " " << Mlist[j] << endl;
      double Rs;
      std::vector<double> pressure;
      pressure = calc_Flender_pressure_profile(cosm_model, z_fft[i], M_fft[j], xlist, &Rs); 
      tab_r500[j+nmbin*i] = Rs; // in Mpc;
      
      double yp1 = 1.e31;
      double ypn = 1.e31;
			
      Nspline=0;
      for(int k=0;k<Nx;k++){
	if(pressure[k] > 0){
	  Nspline += 1;
	}
      }
			
      if(Nspline > 2){
	int id =0;
	for(int k=0;k<Nx;k++){
	  if(pressure[k] > 0){
	    xp[id] = log10(xlist[k]);
	    yp[id] = log10(factor*pressure[k]);
	    id += 1;
	  }
	}
	double xout = pow(10., xp[Nspline-1]);
	spline(xp-1, yp-1, Nspline, yp1, ypn, yp2-1);

	FFT_density_profile(tab_yl_int, bin, Nx);	
				
      }else{
	for(int k=0;k<Nx;k++) tab_yl_int[k] = 0.0;
      }

      for(int k=0;k<Nx;k++) {
        tab_Fourier[k + Nx * ( j + nmbin * i)] = tab_yl_int[k];
      }
    }
  }
  
  /* Start computing a halo model */
  int nd = x_input.get_nd();
  if (nd != 1)
    throw std::runtime_error("a must be 1-dimensional");
  size_t nsh = x_input.shape(0);
  if (x_input.get_dtype() != npy::dtype::get_builtin<double>())
    throw std::runtime_error("a must be float64 array");

  auto xshape = x_input.get_shape();
  auto xstrides = x_input.get_strides();
  
  int Nbin=xshape[0];
  double lbin[Nbin];
  double signal[Nbin];
  for (int i = 0; i < xshape[0]; ++i) {
    lbin[i] = *reinterpret_cast<double *>(x_input.get_data() + i * xstrides[0]);
  }
  double yp1 = 1.e31;
  double ypn = 1.e31;

  int Nell = Nx;
  double tab_fc_int[Nell], tab_fc_int2[Nell];
				
  float dlnz = (log(zmax)-log(zmin))/(nzbin-1);
  float dlogm = (logMvir_max-logMvir_min)/(nmbin-1);
    
  double zlist[nzbin];
		
  double cl_xx_1_int_z[nzbin], cl_xx_1_int_z2[nzbin];
  double cl_xx_2_int_z[nzbin], cl_xx_2_int_z2[nzbin];
				
  double mlist[nmbin];
		
  double cl_xx_1_int_m[nmbin], cl_xx_1_int_m2[nmbin];
  double cl_xx_2_int_m[nmbin], cl_xx_2_int_m2[nmbin];
  
  for(int iz=0;iz<nzbin;iz++){
    cl_xx_1_int_z[iz] = 0.0;
    cl_xx_2_int_z[iz] = 0.0;
    cl_xx_1_int_z2[iz] = 0.0;
    cl_xx_2_int_z2[iz] = 0.0;

  }

  for(int jm=0;jm<nmbin;jm++){
    cl_xx_1_int_m[jm] = 0.0;
    cl_xx_2_int_m[jm] = 0.0;
    cl_xx_1_int_m2[jm] = 0.0;
    cl_xx_2_int_m2[jm] = 0.0;

  }

  for(int i=0;i<Nbin;i++){
    double ell_bin = lbin[i];						
    double cl_xx_1=0.0, cl_xx_2=0.0;
				
    for(int iz=0;iz<nzbin;iz++){
			
      double zhere = exp(dlnz*(double)(1.*iz) + log(zmin));
      zlist[iz] = log(zhere);
			
      double covd = chi_fast(zhere);
      double dVdz = covd*covd *C*CP.hubble/H_z(zhere);
      double calc_k = ell_bin/covd;
      double gfac = (growth(1./(1+zhere))/growth(1.0));
      double Pk = gfac*gfac*PowerSpec(calc_k);
			
      for(int jm=0;jm<nmbin;jm++){
         //if ( flux[iz][jm] >= 0.0 ) {	
	    double logMvir = dlogm*(double)(1.*jm) + logMvir_min;
	    mlist[jm] = logMvir;
	    double Mvir = pow(10., logMvir)*CP.hubble; // in the unit of Msun/h
	    double rs = tab_r500[jm+nmbin*iz]*CP.hubble;
	    double ells = covd/(1+zhere)/rs; // = ell_500
	    double l_ls = ell_bin/ells;
										
	    for(int il=0;il<Nell;il++){
	        tab_fc_int[il] = tab_Fourier[il + Nell * (jm + nmbin * iz)] * sigma_T/m_elect *  megapc ; // 1/Mpc
	    }
				
	    double yl;
	    int index;
				
	    index = -1;
	    for(int il=0;il<Nell-1;il++){
	        if(tab_l_ls[il] < log10(l_ls) && log10(l_ls) < tab_l_ls[il+1]){
	            index = il;
	        }
	    }
				
	    if(index < 0){
	        yl = 0;
	    } else {
	        yl = (tab_fc_int[index+1]-tab_fc_int[index])/(tab_l_ls[index+1]-tab_l_ls[index])*(log10(l_ls)-tab_l_ls[index]) + tab_fc_int[index];
	        yl = yl * 4*M_PI*(rs)/ells/ells; // 
	    }

	    //double m200m = M_vir_to_M_delta(zhere, Mvir, 200.0);		
	    double mf = dndlogm_fast(log10(Mvir), zhere);
	    double b = halo_bias_fast(log10(Mvir), zhere);
		
	    cl_xx_1_int_m[jm] = mf * yl * yl;
	    cl_xx_2_int_m[jm] = mf * b * yl;

         //} else {
	 //   cl_xx_1_int_m[jm] = 0.0;
	 //   cl_xx_2_int_m[jm] = 0.0;
         //}
         //fprintf(stdout,"z, mvir, mf, b = %f %e %e %e\n", zhere, Mvir, mf,b);

      }
						
      spline(mlist-1, cl_xx_1_int_m-1, nmbin, yp1, ypn, cl_xx_1_int_m2-1);
      spline(mlist-1, cl_xx_2_int_m-1, nmbin, yp1, ypn, cl_xx_2_int_m2-1);			
      double oneh_xx, twoh_xx;
			
      //double tab_spline_and_integral(int Nbin, double *xlist, double *ylist, double *zlist)
			
      oneh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_1_int_m, cl_xx_1_int_m2);
      twoh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_2_int_m, cl_xx_2_int_m2);
			
      cl_xx_1_int_z[iz] = zhere * dVdz * oneh_xx;
      cl_xx_2_int_z[iz] = zhere * dVdz * twoh_xx * twoh_xx * Pk;
      						
      //fprintf(stdout,"iz= %d %e %e\n", iz,cl_xx_1_int_z[iz], cl_xx_2_int_z[iz] );
   }
		
    spline(zlist-1, cl_xx_1_int_z-1, nzbin, yp1, ypn, cl_xx_1_int_z2-1);
    spline(zlist-1, cl_xx_2_int_z-1, nzbin, yp1, ypn, cl_xx_2_int_z2-1);
		
    cl_xx_1 = tab_spline_and_integral(nzbin, zlist, cl_xx_1_int_z, cl_xx_1_int_z2);
    cl_xx_2 = tab_spline_and_integral(nzbin, zlist, cl_xx_2_int_z, cl_xx_2_int_z2);

    signal[i] = cl_xx_1 + cl_xx_2;

    //printf("%f %e %e\n", ell_bin, cl_xx_1, cl_xx_2);
    if(cl_xx_1 != cl_xx_1 || cl_xx_2 != cl_xx_2) signal[i]=0;
    
  }
  
  free_FFTdata();
  delete[] tab_Fourier; 
  delete[] tab_r500;

  vector<double> v;
  for(int i=0;i<Nbin;i++) v.push_back(signal[i]);
  Py_intptr_t shape[1] = { v.size() };
  npy::ndarray result = npy::zeros(1, shape, npy::dtype::get_builtin<double>());
  copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));

  return result;
}

npy::ndarray return_pressure_profile(npy::ndarray x_input, double z, double Mvir){

  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);

  cout << "computing pressure profile" << endl;

  int nd = x_input.get_nd();
  if (nd != 1)
    throw std::runtime_error("a must be 1-dimensional");
  size_t nsh = x_input.shape(0);
  if (x_input.get_dtype() != npy::dtype::get_builtin<double>())
    throw std::runtime_error("a must be float64 array");

  auto xshape = x_input.get_shape();
  auto xstrides = x_input.get_strides(); 
  //int Nbin=xshape[0];
  int Nbin = x_input.shape(0);
  /*
  double* input_ptr = reinterpret_cast<double*>(input.get_data());
  std::vector<double> v(input_size);
  for (int i = 0; i < input_size; ++i)
    v[i] = *(input_ptr + i);
  */
  
  double lbin[Nbin];
  std::vector<float> xlist(Nbin);

  for (int i = 0; i < Nbin; ++i) {
    lbin[i] = *reinterpret_cast<double *>(x_input.get_data() + i * xstrides[0]);
    xlist[i] = (float)lbin[i];
  }
  double rs;
  std::vector<double> pressure;
  pressure = calc_Flender_pressure_profile(cosm_model, z, Mvir, xlist, &rs); 

  Py_intptr_t shape[1] = { pressure.size() };
  npy::ndarray result = npy::zeros(1, shape, npy::dtype::get_builtin<double>());
  copy(pressure.begin(), pressure.end(), reinterpret_cast<double*>(result.get_data()));

  return result;

}

npy::ndarray return_density_profile(npy::ndarray x_input, double z, double Mvir){

  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);

  cout << "computing density profile" << endl;

  int nd = x_input.get_nd();
  if (nd != 1)
    throw std::runtime_error("a must be 1-dimensional");
  size_t nsh = x_input.shape(0);
  if (x_input.get_dtype() != npy::dtype::get_builtin<double>())
    throw std::runtime_error("a must be float64 array");

  auto xshape = x_input.get_shape();
  auto xstrides = x_input.get_strides(); 
  //int Nbin=xshape[0];
  int Nbin = x_input.shape(0);
  /*
  double* input_ptr = reinterpret_cast<double*>(input.get_data());
  std::vector<double> v(input_size);
  for (int i = 0; i < input_size; ++i)
    v[i] = *(input_ptr + i);
  */
  
  double lbin[Nbin];
  std::vector<float> xlist(Nbin);

  for (int i = 0; i < Nbin; ++i) {
    lbin[i] = *reinterpret_cast<double *>(x_input.get_data() + i * xstrides[0]);
    xlist[i] = (float)lbin[i];
  }


  std::vector<double> density(Nbin);
  density = calc_Flender_density_profile(cosm_model, z, Mvir, xlist); 

  Py_intptr_t shape[1] = { density.size() };
  npy::ndarray result = npy::zeros(1, shape, npy::dtype::get_builtin<double>());
  copy(density.begin(), density.end(), reinterpret_cast<double*>(result.get_data()));

  return result;

}

double return_Lx(double z, double Mvir){

  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  
  std::vector<float> xlist;

  //radius in unit of R500, x = r/R500
  float xmin = 1e-4;
  float xmax = 100.0;
  float x;
  for(int i=0;i<Nx;i++){
    x = (log10(xmax)-log10(xmin))/Nx*(float)(1.*i+0.5) + log10(xmin);
    x = pow(10.0, x);
    xlist.push_back(x);
  }
	  
  double lum;
  // lum in ergs/s
  lum = calc_Flender_xray_luminosity (cosm_model, z, Mvir, xlist);
  return lum;
}

double Mvir_to_Mdeltac(double z, double Mvir, double delta){

  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  
  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  float cvir = c_vir_DK15_fast(z, Mvir);
  nfwclus.set_conc(cvir);
  return nfwclus.get_mass_overden(delta);// Msun
	
}

double return_Mgas(double z, double Mvir){
  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  
  std::vector<float> xlist;

  //radius in unit of R500, x = r/R500
  float xmin = 1e-4;
  float xmax = 100.0;
  float x;
  for(int i=0;i<Nx;i++){
    x = (log10(xmax)-log10(xmin))/Nx*(float)(1.*i+0.5) + log10(xmin);
    x = pow(10.0, x);
    xlist.push_back(x);
  }
	  
  double Mgas;
  Mgas= calc_Flender_mgas (cosm_model, z, Mvir, xlist);
  return Mgas;	
}

double return_Tx(double z, double Mvir){
  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  
  std::vector<float> xlist;

  //radius in unit of R500, x = r/R500
  float xmin = 1e-4;
  float xmax = 100.0;
  float x;
  for(int i=0;i<Nx;i++){
    x = (log10(xmax)-log10(xmin))/Nx*(float)(1.*i+0.5) + log10(xmin);
    x = pow(10.0, x);
    xlist.push_back(x);
  }
	  
  double Tx;
  Tx= calc_Flender_xray_temperature (cosm_model, z, Mvir, xlist);
  return Tx;	
}

double return_total_xsb(){
  int nzbin = 31;
  float zmin = 1e-3;
  float zmax = 5.0;
	
  int nmbin = 31;
  float logMvir_min= 13.0;
  float logMvir_max= 16.0;

  cosmo cosm_model(CP.H0, CP.Omega_M, CP.Omega_b, CP.Omega_k, CP.wt);
  
  float z, Mvir, x;
  std::vector<float> z_fft, M_fft, xlist;
  /* set up for redshift, virial mass, radius bin */
  //redshift
  for(int i=0;i<nzbin;i++){
    z = (log(zmax)-log(zmin))/(nzbin-1)*(float)(1.*i) + log(zmin);
    z = exp(z);
    z_fft.push_back(z);
  }
	
  //virial mass (Bryan & Norman) in [Msun], not in [Msun/h]
  for(int i=0;i<nmbin;i++){
    Mvir = (logMvir_max-logMvir_min)/(nmbin-1)*(float)(1.*i) + logMvir_min;
    Mvir = pow(10.0, Mvir);
    M_fft.push_back(Mvir);
  }
	
  //radius in unit of R500, x = r/R500
  float xmin = 1e-4;
  float xmax = 100.0;
  float dlogx;
  for(int i=0;i<Nx;i++){
    x = (log10(xmax)-log10(xmin))/Nx*(float)(1.*i+0.5) + log10(xmin);
    x = pow(10.0, x);
    xlist.push_back(x);
  }
  dlogx = (log10(xmax)-log10(xmin))/Nx*0.5;
		
  /* Start computing a halo model */
  
  double yp1 = 1.e31;
  double ypn = 1.e31;

  float dlnz = (log(zmax)-log(zmin))/(nzbin-1);
  float dlogm = (logMvir_max-logMvir_min)/(nmbin-1);
    
  double zlist[nzbin];
		
  double cl_xx_1_int_z[nzbin], cl_xx_1_int_z2[nzbin];
				
  double mlist[nmbin];
		
  double cl_xx_1_int_m[nmbin], cl_xx_1_int_m2[nmbin];
  
  for(int iz=0;iz<nzbin;iz++){
    cl_xx_1_int_z[iz] = 0.0;
    cl_xx_1_int_z2[iz] = 0.0;
  }

  for(int jm=0;jm<nmbin;jm++){
    cl_xx_1_int_m[jm] = 0.0;
    cl_xx_1_int_m2[jm] = 0.0;
  }
	
  for(int iz=0;iz<nzbin;iz++){
			
      double zhere = exp(dlnz*(double)(1.*iz) + log(zmin));
      zlist[iz] = log(zhere);
			
      double covd = chi_fast(zhere);
      double dVdz = covd*covd *C*CP.H0/100.0/H_z(zhere);
			
      for(int jm=0;jm<nmbin;jm++){

	double logMvir = dlogm*(double)(1.*jm) + logMvir_min;
	mlist[jm] = logMvir;
	double Mvir = pow(10., logMvir)*CP.hubble ; // in the unit of Msun/h
	double rs;
        std::vector<double> emission;
        emission = calc_Flender_xray_emissivity_profile(cosm_model, zhere, Mvir, xlist, &rs); // ergs/s/cm^3/str; 
	double ells = covd/(1+zhere)/(rs*CP.hubble); // = ell_s
        double xsb = 0.0; 
        for(int k=0;k<Nx;k++) {
            xsb += dlogx*xlist[k]*xlist[k]*xlist[k]
                   *emission[k]*4.0*M_PI*(rs)/ells/ells * megapc; //erg/s/cm^2
        } 
		
	double mf = dndlogm_fast(log10(Mvir), zhere); // h^3 Mpc^-3
	cl_xx_1_int_m[jm] = mf * xsb;
      }
						
      spline(mlist-1, cl_xx_1_int_m-1, nmbin, yp1, ypn, cl_xx_1_int_m2-1);
      //spline(mlist-1, cl_xx_2_int_m-1, nmbin, yp1, ypn, cl_xx_2_int_m2-1);
			
      double oneh_xx;
			
      oneh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_1_int_m, cl_xx_1_int_m2);
			
      cl_xx_1_int_z[iz] = zhere * dVdz * oneh_xx;
      						
    }
		
    spline(zlist-1, cl_xx_1_int_z-1, nzbin, yp1, ypn, cl_xx_1_int_z2-1);
	
    double cl_xx_1;	
    cl_xx_1 = tab_spline_and_integral(nzbin, zlist, cl_xx_1_int_z, cl_xx_1_int_z2);

  double result = cl_xx_1;

  return result;
}


BOOST_PYTHON_MODULE( xx_power ){
  Py_Initialize();
  npy::initialize();

  py::def("init_cosmology", init_cosmology);
  py::def("free_cosmology", free_cosmology);
  py::def("set_Flender_params", set_Flender_params);
  py::def("return_xx_power", return_xx_power);
  py::def("return_yy_power", return_yy_power);
  py::def("return_xx_power_alt", return_xx_power_alt);
  py::def("return_total_xsb", return_total_xsb);
  py::def("return_Lx", return_Lx);
  py::def("return_Tx", return_Tx);
  py::def("return_Mgas", return_Mgas);
  py::def("return_pressure_profile", return_pressure_profile);
  py::def("return_density_profile", return_density_profile);
  py::def("Mvir_to_Mdeltac", Mvir_to_Mdeltac);
}

std::vector<double> calc_Flender_pressure_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x, double *Rs){
	
  float conc_norm = F.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
  
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = F.alpha0, delta_rel_n = F.n_nt, delta_rel_zslope = F.beta;
  float eps_fb = F.eps_f;
  float eps_dm = F.eps_DM;
  float fs_0 = F.f_star;
  float fs_alpha = F.S_star;

  float gamma_mod0 = F.gamma_mod0;
  float gamma_mod_zslope = F.gamma_mod_zslope;
  float x_break = F.x_break;
  float x_smooth = F.x_smooth;

  float clump0 = F.clump0;
  float clump_zslope = F.clump_zslope;
  float x_clump = F.x_clump;
  float alpha_clump1 = F.alpha_clump1;
  float alpha_clump2 = F.alpha_clump2;

  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
	
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //M500 = nfwclus.get_mass_overden(500.0);// Msun
  //R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  //Rvir = nfwclus.get_radius();
	
  float cvir;
  //nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)
  //cvir = nfwclus.get_conc();
  cvir = conc_norm * c_vir_DK15_fast(z, Mvir*h);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();
  *Rs = Rvir/cvir;
  //cout << M500 << " " << R500 << " " << Rvir << endl;
	
  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);

  double P500 = icm_mod.return_P500_arnaud10(M500, E);
	
  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = 3.0*R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, pres;
  std::vector<double> profile;

  // redshift dependence in solid angle
  double fac = 4.0*M_PI*pow(1.0+Redshift, 4.0); // in steradians

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  npoly_mod = 1.0/(gamma_mod - 1.0 );

  if (gamma_mod != 1 ) {
    npoly_mod = 1.0/(gamma_mod - 1.0 );
  } else {
    npoly_mod = 1.e30;
  }


  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*(*Rs);
    if(r >= Rmax){pres = 0.0;}
    else{
        pres = icm_mod.returnPth_mod2(r, R500, x_break, npoly_mod, x_smooth); //keV cm^-3
    } 		
		
    profile.push_back(pres);
  }
	
  return profile;
	
}

std::vector<double> calc_Flender_density_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x){
	
  float conc_norm = F.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
  
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = F.alpha0, delta_rel_n = F.n_nt, delta_rel_zslope = F.beta;
  float eps_fb = F.eps_f;
  float eps_dm = F.eps_DM;
  float fs_0 = F.f_star;
  float fs_alpha = F.S_star;

  float gamma_mod0 = F.gamma_mod0;
  float gamma_mod_zslope = F.gamma_mod_zslope;
  float x_break = F.x_break;
  float x_smooth = F.x_smooth;

  float clump0 = F.clump0;
  float clump_zslope = F.clump_zslope;
  float x_clump = F.x_clump;
  float alpha_clump1 = F.alpha_clump1;
  float alpha_clump2 = F.alpha_clump2;

  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
	
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //M500 = nfwclus.get_mass_overden(500.0);// Msun
  //R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  //Rvir = nfwclus.get_radius();
	
  float cvir;
  nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)
  //cvir = conc_norm * c_vir_DK15_fast(z, Mvir*h);
  //nfwclus.set_conc(cvir);
  cvir = nfwclus.get_conc();
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();

  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);

  double P500 = icm_mod.return_P500_arnaud10(M500, E);
	
  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = 3.0*R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, ngas;
  std::vector<double> profile;

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  npoly_mod = 1.0/(gamma_mod - 1.0 );
  if (gamma_mod != 1 ) {
    npoly_mod = 1.0/(gamma_mod - 1.0 );
  } else {
    npoly_mod = 1.e30;
  }

  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*R500;
    if(r >= Rmax){ngas = 0.0;}
    else{ 
        ngas = icm_mod.return_ngas_mod(r, R500, x_break, npoly_mod); // cm^-3
    }
    profile.push_back(ngas);
  }
	
  return profile;

}

double calc_Flender_mgas (cosmo cosm_model, float z, float Mvir, std::vector<float> x){
	
  float conc_norm = F.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
  
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = F.alpha0, delta_rel_n = F.n_nt, delta_rel_zslope = F.beta;
  float eps_fb = F.eps_f;
  float eps_dm = F.eps_DM;
  float fs_0 = F.f_star;
  float fs_alpha = F.S_star;

  float gamma_mod0 = F.gamma_mod0;
  float gamma_mod_zslope = F.gamma_mod_zslope;
  float x_break = F.x_break;
  float x_smooth = F.x_smooth;

  float clump0 = F.clump0;
  float clump_zslope = F.clump_zslope;
  float x_clump = F.x_clump;
  float alpha_clump1 = F.alpha_clump1;
  float alpha_clump2 = F.alpha_clump2;

  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
  
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)

  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();
	
  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);
	
  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, rhogas, dMgas;
  double Mgas = 0.0;

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  npoly_mod = 1.0/(gamma_mod - 1.0 );
  if (gamma_mod != 1 ) {
    npoly_mod = 1.0/(gamma_mod - 1.0 );
  } else {
    npoly_mod = 1.e30;
  }


  double dvol[x.size()]; // radial shell vol in cm^3
  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*R500; // R500 in Mpc
    if ( xi == 0 ) {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0;
    } else {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0 - dvol[xi-1];
    }
  }
  for(int xi=0;xi<x.size();xi++){
    // r in Mpc;/
    r = (double) x[xi]*R500;
    if ( r > R500 ){ 
        rhogas = 0.0;
    } else {
        double ngas,pressure, kT, clump, clump1;
        ngas = icm_mod.return_ngas_mod(r, R500, x_break, npoly_mod); //cm^-3
        
        clump1 = icm_mod.return_clumpf(r, R500, clump0, x_clump, alpha_clump1, alpha_clump2) - 1.0;
        clump1 *= pow(1.+Redshift, clump_zslope);
        clump = 1.0 + clump1;
        if (clump < 1.0) clump = 1.0;
        ngas *= sqrt(clump);
        
        rhogas = ngas * m_p * mmw; //g cm^-3
        dMgas = rhogas * dvol[xi]/Msun; // Msun
        Mgas += dMgas;		
    }
 
  }
  return Mgas;	
}

double calc_Flender_xray_temperature (cosmo cosm_model, float z, float Mvir, std::vector<float> x){
	
  float conc_norm = F.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
  
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = F.alpha0, delta_rel_n = F.n_nt, delta_rel_zslope = F.beta;
  float eps_fb = F.eps_f;
  float eps_dm = F.eps_DM;
  float fs_0 = F.f_star;
  float fs_alpha = F.S_star;

  float gamma_mod0 = F.gamma_mod0;
  float gamma_mod_zslope = F.gamma_mod_zslope;
  float x_break = F.x_break;
  float x_smooth = F.x_smooth;

  float clump0 = F.clump0;
  float clump_zslope = F.clump_zslope;
  float x_clump = F.x_clump;
  float alpha_clump1 = F.alpha_clump1;
  float alpha_clump2 = F.alpha_clump2;

  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
  
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)

  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();
	
  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
 
  assert(fs_0 >= 0); 
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);
	
  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, emi, dtemp, dlum;
  double Tx = 0.0;
  double Lx = 0.0;

  // distances in Mpc
  //double D_A = cosm_model.ang_diam(Redshift);
  //double D_L = cosm_model.lum_dist(Redshift);
  //D_L *= megapc;

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  if (gamma_mod != 1 ) {
    npoly_mod = 1.0/(gamma_mod - 1.0 );
  } else {
    npoly_mod = 1.e30;
  }

  double dvol[x.size()]; // radial shell vol in cm^3
  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*R500;
    if ( xi == 0 ) {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0;
    } else {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0 - dvol[xi-1];
    }
  }
  for(int xi=0;xi<x.size();xi++){
    // r in Mpc;/
    r = (double) x[xi]*R500;
    if(r >= Rmax or r < 0.15*R500){
        emi = 0.0;
        dtemp = 0.0;
    } else{
        double ngas,pressure, kT, clump, clump1;
        pressure = icm_mod.returnPth_mod2(r, R500, x_break, npoly_mod, x_smooth); //keV cm^-3
        ngas = icm_mod.return_ngas_mod(r, R500, x_break, npoly_mod); //cm^-3
        kT = pressure/ngas; // keV
    
        clump1 = icm_mod.return_clumpf(r, R500, clump0, x_clump, alpha_clump1, alpha_clump2) - 1.0;
        clump1 *= pow(1.+Redshift, clump_zslope);
        clump = 1.0 + clump1;
        if (clump < 1.0) clump = 1.0;
        ngas *= sqrt(clump);

        emi = icm_mod.return_xray_emissivity(ngas, kT, Redshift); // ergs/s/cm^3
        dlum = emi * dvol[xi]; // ergs/s
        dtemp = dlum * kT; 
        Lx += dlum;
        Tx += dtemp;	

    } 
    
  }

  Tx = Tx/Lx;
  return Tx;	
}

std::vector<double> calc_beta_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x, double *Rs){
	
  float conc_norm = F.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
  
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = F.alpha0, delta_rel_n = F.n_nt, delta_rel_zslope = F.beta;
  float eps_fb = F.eps_f;
  float eps_dm = F.eps_DM;
  float fs_0 = F.f_star;
  float fs_alpha = F.S_star;

  float gamma_mod0 = F.gamma_mod0;
  float gamma_mod_zslope = F.gamma_mod_zslope;
  float x_break = F.x_break;
  float x_smooth = F.x_smooth;

  float clump0 = F.clump0;
  float clump_zslope = F.clump_zslope;
  float x_clump = F.x_clump;
  float alpha_clump1 = F.alpha_clump1;
  float alpha_clump2 = F.alpha_clump2;

  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
	
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)
  //M500 = nfwclus.get_mass_overden(500.0);// Msun
  //R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  //Rvir = nfwclus.get_radius();
	
  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();

  Rscale = Rvir/cvir;
  *Rs = Rscale;

  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = R500;

  double fac = 4.0*M_PI*pow(1.0+Redshift, 4.0); // in steradians
  double r, emi, dtemp, dlum;
  double Tx = 0.0;
  double Lx = 0.0;

  Tx = 9.48*pow(M500/1e15, 0.75);
  Lx = 1.12e42/(h*h) * pow(Tx, 3.2) * (1.0+Redshift);

  // distances in Mpc
  //double D_A = cosm_model.ang_diam(Redshift);
  //double D_L = cosm_model.lum_dist(Redshift);
  //D_L *= megapc;

  double rc = 0.1; //Mpc

  std::vector<double> emission;
  double ngas[x.size()];
  double dvol[x.size()]; // radial shell vol in cm^3
  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*Rscale;
    if ( xi == 0 ) {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0;
    } else {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0 - dvol[xi-1];
    }
  }

  double enorm = 0.0;
  double emi_sum = 0.0;

  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*(Rscale);
    if(r >= R500){
        ngas[xi] = 0.0;
    }else{
        ngas[xi] = 1.0/(1.0+pow(r/rc,2.0));
    }			
    emi_sum += ngas[xi]*ngas[xi]*dvol[xi];
  }    
  enorm = Lx/emi_sum; 
  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*(Rscale);
    double emi;
    if(r >= R500){
        emi = 0.0;
    }else{
        emi = ngas[xi]*ngas[xi]*enorm;
    }
    emi = emi/fac;
    emission.push_back(emi);
  }    
 
  return emission;
	
}


double calc_Flender_xray_luminosity (cosmo cosm_model, float z, float Mvir, std::vector<float> x){
	
  float conc_norm = F.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
  
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = F.alpha0, delta_rel_n = F.n_nt, delta_rel_zslope = F.beta;
  float eps_fb = F.eps_f;
  float eps_dm = F.eps_DM;
  float fs_0 = F.f_star;
  float fs_alpha = F.S_star;

  float gamma_mod0 = F.gamma_mod0;
  float gamma_mod_zslope = F.gamma_mod_zslope;
  float x_break = F.x_break;
  float x_smooth = F.x_smooth;

  float clump0 = F.clump0;
  float clump_zslope = F.clump_zslope;
  float x_clump = F.x_clump;
  float alpha_clump1 = F.alpha_clump1;
  float alpha_clump2 = F.alpha_clump2;

  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
  
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)

  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();
	
  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);
	
  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, emi, dlum;
  double luminosity = 0.0;
  double flux = 0.0;

  // distances in Mpc
  //double D_A = cosm_model.ang_diam(Redshift);
  //double D_L = cosm_model.lum_dist(Redshift);
  //D_L *= megapc;

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  if (gamma_mod != 1 ) {
    npoly_mod = 1.0/(gamma_mod - 1.0 );
  } else {
    npoly_mod = 1.e30;
  }


  double dvol[x.size()]; // radial shell vol in cm^3
  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*R500;
    if ( xi == 0 ) {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0;
    } else {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0)/3.0 - dvol[xi-1];
    }
  }
  for(int xi=0;xi<x.size();xi++){
    // r in Mpc;/
    r = (double) x[xi]*R500;
    if(r >= Rmax ){
        emi = 0.0;
    } else{
        double ngas,pressure, kT, clump, clump1;
        pressure = icm_mod.returnPth_mod2(r, R500, x_break, npoly_mod, x_smooth); //keV cm^-3
        ngas = icm_mod.return_ngas_mod(r, R500, x_break, npoly_mod); //cm^-3
        kT = pressure/ngas; // keV
    
        clump1 = icm_mod.return_clumpf(r, R500, clump0, x_clump, alpha_clump1, alpha_clump2) - 1.0;
        clump1 *= pow(1.+Redshift, clump_zslope);
        clump = 1.0 + clump1;
        if (clump < 1.0) clump = 1.0;
        ngas *= sqrt(clump);

        emi = icm_mod.return_xray_emissivity(ngas, kT, Redshift); // ergs/s/cm^3
        dlum = emi * dvol[xi]; // ergs/s
        luminosity += dlum;		
    } 
    
  }
  //flux = luminosity/(4.0*M_PI*D_L*D_L);	//ergs/s/cm^2
  return luminosity;	
}

std::vector<double> calc_Flender_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x, double *Rs){
	
  float conc_norm = F.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
  
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = F.alpha0, delta_rel_n = F.n_nt, delta_rel_zslope = F.beta;
  float eps_fb = F.eps_f;
  float eps_dm = F.eps_DM;
  float fs_0 = F.f_star;
  float fs_alpha = F.S_star;

  float gamma_mod0 = F.gamma_mod0;
  float gamma_mod_zslope = F.gamma_mod_zslope;
  float x_break = F.x_break;
  float x_smooth = F.x_smooth;

  float clump0 = F.clump0;
  float clump_zslope = F.clump_zslope;
  float x_clump = F.x_clump;
  float alpha_clump1 = F.alpha_clump1;
  float alpha_clump2 = F.alpha_clump2;

  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
	
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)
  //M500 = nfwclus.get_mass_overden(500.0);// Msun
  //R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  //Rvir = nfwclus.get_radius();
	
  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();

  Rscale = Rvir/cvir;
  *Rs = Rscale;

  //cout << M500 << " " << R500 << " " << Rvir << endl;
	
  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);

  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = 3.0*R500;

  double r, emi;
  std::vector<double> emission;

  // redshift dependence in solid angle
  double fac = 4.0*M_PI*pow(1.0+Redshift, 4.0); // in steradians

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  npoly_mod = 1.0/(gamma_mod - 1.0 );
  if (gamma_mod != 1 ) {
    npoly_mod = 1.0/(gamma_mod - 1.0 );
  } else {
    npoly_mod = 1.e30;
  }


  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*(Rscale);
    if(r >= Rmax){emi = 0.0;}
    else{
        double ngas,pressure, kT, clump, clump1;
        ngas = icm_mod.return_ngas_mod(r, R500, x_break, npoly_mod); //cm^-3
        kT = icm_mod.returnT_mod2(r, R500, x_break, npoly_mod, x_smooth);// keV
 
        clump1 = icm_mod.return_clumpf(r, R500, clump0, x_clump, alpha_clump1, alpha_clump2) - 1.0;
        clump1 *= pow(1.+Redshift, clump_zslope);
        clump = 1.0 + clump1;
        if (clump < 1.0) clump = 1.0;

        emi = icm_mod.return_xray_emissivity(ngas, kT, Redshift); // ergs/s/cm^3
        emi *= clump;
    } 		
		
    emi = emi/fac; // ergs/s/cm^3/str

    emission.push_back(emi);
  }
	
  return emission;
	
}

std::vector<double> calc_Shaw_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x){
	
  float conc_norm = S.A_C;
  float conc_mass_norm = 1.0;
  float ad_index = 5.0; // Gamma = 1+1./ad_index in arXiv:1706.08972
  /*
    float delta_rel = 0.18, delta_rel_n = 0.8, delta_rel_zslope = 0.5; // delta_rel = alpha_0, delta_rel_n  = n_nt, delta_rel_zslope =  beta in Shaw et al 2010
    float eps_fb = 3.97e-6; // epsilon_f in arXiv:1706.08972
    float eps_dm = 0.0; // epsilon_DM in arXiv:1706.08972
    float fs_0 = 0.026; // f_star in arXiv:1706.08972
    float fs_alpha = 0.12; // S_star in arXiv:1706.08972
  */
  float delta_rel = S.alpha0, delta_rel_n = S.n_nt, delta_rel_zslope = S.beta;
  float eps_fb = S.eps_f;
  float eps_dm = S.eps_DM;
  float fs_0 = S.f_star;
  float fs_alpha = S.S_star;
	
  int pturbrad = 2;
  bool verbose = false;
  float Rvir, M500, R500, Rscale, conc, cosmic_t, cosmic_t0;
  float Omega_M = cosm_model.get_Omega_M();
  float Omega_b = cosm_model.get_Omega_b();
  float h =cosm_model.get_H0()/100.0;
  float E;
  // set cluster overdensity
  // this is the overdensity within which mass defined (i.e. \Delta)
  // set to -1.0 for virial radius, or 200 for M200 (rhocrit)
  float overden_id = -1.0; // 200 for delta=200 rho-c , -1 for delta=vir x rho-c
  int relation = 3; // concentration relation
  float rcutoff = 2.0;
	
  float Redshift = z;
  cosmic_t = cosm_model.cosmic_time(Redshift);
  cosmic_t0 = cosm_model.cosmic_time(0.0);
  E = cosm_model.Efact(Redshift);
	
  cluster nfwclus(Mvir, Redshift, overden_id, relation, cosm_model);
	
  //nfwclus.concentration(conc_norm, conc_mass_norm); // set halo concentration using M-c relation of Duffy et al (08)
  //M500 = nfwclus.get_mass_overden(500.0);// Msun
  //R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  //Rvir = nfwclus.get_radius();
	
  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();
	
  //cout << M500 << " " << R500 << " " << Rvir << endl;
	
  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);
	
  //double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  double Rmax = 3.0*R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, emi;
  std::vector<double> emission;
	
  double fac = 4.0*M_PI*180.0*3600.0/M_PI *180.0*3600.0/M_PI;
	
  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*R500;
    if(r >= Rmax){emi = 0.0;}
    else{emi = icm_mod.return_xray_emissivity(r, R500, Redshift);} 
		
    //if(emi < 1e-50){emi = 0.0;}
		
    emi = emi/fac/pow(1.+Redshift, 4.); 
		
    //cout << r << " " << emi <<endl;
		
    emission.push_back(emi);
  }
	
  return emission;
	
}

/*
double x_l_integral_int_x(double x, double l_ls){
	double res;
	double x2px;
	if(x > pow(10., xp[0]) && x < pow(10., xp[Nspline-1])){
		double logx = log10(x);
		splint(xp-1, yp-1, yp2-1, Nspline, logx, &x2px);
		x2px = pow(10., x2px) *x*x;
	}else{
		return 0;
	}
	
	double kernel;
	double lx = l_ls*x;
	if(lx < 1e-2){
		kernel = 1.0-lx*lx/6.;
	}else{
		kernel = sin(lx)/(lx);
	}
	res =x2px*kernel;
	
	return res;
}
double x_l_integral(double l_ls, double xout){
	double res,abserr;
	size_t neval;
	
	int Nint = 10;
	//Romberg
	int i,j;
	double h[Nint];
	double s[Nint][Nint];
	
	res = 0;
	int NLoop = (int)((xout)/(2*M_PI/l_ls));
	
	if(NLoop < 2){
		double x_lo = 0.0;
		double x_hi = xout;
		
		for(i=1;i<=Nint;i++){h[i-1] = (x_hi-x_lo)/pow(2.,i-1);}
		
		s[0][0] = 0.5*h[0]*(x_l_integral_int_x(x_hi, l_ls)+x_l_integral_int_x(x_lo, l_ls));
		for(i=2;i<=Nint;i++){
			s[i-1][0] = s[i-2][0];
			for(j=1;j<=pow(2.,i-2);j++){
				s[i-1][0] += h[i-2]*x_l_integral_int_x(x_lo+(2*j-1)*h[i-1], l_ls);
			}
			s[i-1][0] = 0.5*s[i-1][0];
		}
		
		for(i=2;i<=Nint;i++){
			for(j=2;j<=i;j++){
				s[i-1][j-1] = s[i-1][j-2]+(s[i-1][j-2]-s[i-2][j-2])/(pow(4.,j-1)-1);
			}
		}
		
		res += s[Nint-1][Nint-1];
	}else{
		for(int iLoop=0;iLoop<NLoop;iLoop++){
			
			double x_lo = 0.0+(double)(iLoop+0)*(xout)/NLoop;
			double x_hi = 0.0+(double)(iLoop+1)*(xout)/NLoop;
			
			if(iLoop == NLoop-1){
				x_hi = xout;
			}
			
			for(i=1;i<=Nint;i++){h[i-1] = (x_hi-x_lo)/pow(2.,i-1);}
			
			s[0][0] = 0.5*h[0]*(x_l_integral_int_x(x_hi, l_ls)+x_l_integral_int_x(x_lo, l_ls));
			for(i=2;i<=Nint;i++){
				s[i-1][0] = s[i-2][0];
				for(j=1;j<=pow(2.,i-2);j++){
					s[i-1][0] += h[i-2]*x_l_integral_int_x(x_lo+(2*j-1)*h[i-1], l_ls);
				}
				s[i-1][0] = 0.5*s[i-1][0];
			}
			
			for(i=2;i<=Nint;i++){
				for(j=2;j<=i;j++){
					s[i-1][j-1] = s[i-1][j-2]+(s[i-1][j-2]-s[i-2][j-2])/(pow(4.,j-1)-1);
				}
			}
			
			res += s[Nint-1][Nint-1];
		}
	}
	return res;
}
*/

void free_FFTdata(){
		
  delete [] r;
  delete [] a;
  delete [] k;
  delete [] wsave;
	
}

void FFT_density_profile(double *output, double *bin, int nbin){

  double mu;
	
  double dlnr;
  double logrc=(logrmin+logrmax)/2.;
  double dlogr=(logrmax-logrmin)/double(N);
  dlnr=dlogr*log(10.);
  double nc=double(N+1)/2.;
	
  double logkc=log10(kr)-logrc;

  for(int i=0;i<N;i++){
    r[i]= pow(10.,logrc+(i-nc+1)*dlogr);
    k[i]= pow(10.,logkc+(i-nc+1)*dlogr);
  }

  int index[nbin+1];
  for(int ibin=0;ibin<nbin;ibin++){
    index[ibin] = (int)((log10(bin[ibin])-log10(k[0]))/dlogr);
  }
  double bin_edge = log10(bin[nbin-1])+(log10(bin[nbin-1])-log10(bin[nbin-2]));
  bin_edge = pow(10., bin_edge);
  index[nbin] = (int)((log10(bin_edge)-log10(k[0]))/dlogr);

  mu=+0.5;      
  q =-0.5;
      	
  for(int i=0;i<N;i++){
    double prof;
    if(r[i] > pow(10., xp[0]) && r[i] < pow(10., xp[Nspline-1])){
      double logx = log10(r[i]);
      splint(xp-1, yp-1, yp2-1, Nspline, logx, &prof);
      prof = pow(10., prof);
    }else{
      prof = 0;
    }
    a[i] = r[i] * r[i] * sqrt(M_PI/2) * prof;
  }

  //CALL FORTRAN SUBROUTINE, compute Fast Hankel Transform
  int ok;
  int clogical=0;
  ok=C2FLOGICAL(clogical);
	
  fhti_(& N, & mu, & q, & dlnr, & kr, & kropt, wsave, & ok);
	
  clogical=F2CLOGICAL(ok);
	
  fhtq_(& N, a, & dir, wsave);
  //END FORTRAN CALLING
  
  for(int ibin=0;ibin<nbin;ibin++){
    //fprintf(stdout,"%d: ", ibin);
    //fprintf(stdout,".");fflush(stdout);

    // cic
    int i1 = index[ibin];
    int i2 = index[ibin]+1;

    double f1 = a[i1];
    double f2 = a[i2];

    double b1 = k[i1];
    double b2 = k[i2];

    double res = (f2-f1)/(b2-b1)*(bin[ibin]-b1)+f1;
    output[ibin] = res/bin[ibin];           
  }
  //fprintf(stdout,"\n");

}

//integral with input tables
double tab_spline_and_integral(int Nbin, double *xlist, double *ylist, double *zlist){
  int i,j;
  double x_hi = xlist[Nbin-1];
  double x_lo = xlist[0];
  int Nint = 10;
  //Romberg
  double h[Nint];
  double s[Nint][Nint];
  for(int ii=1;ii<=Nint;ii++){h[ii-1] = (x_hi-x_lo)/pow(2.,ii-1);}
	
  s[0][0] = 0.5*h[0]*(ylist[Nbin-1]+ylist[0]);
  for(int ii=2;ii<=Nint;ii++){
    s[ii-1][0] = s[ii-2][0];
    for(j=1;j<=pow(2.,ii-2);j++){
      double res, logx = x_lo+(2*j-1)*h[ii-1];
      splint(xlist-1, ylist-1, zlist-1, Nbin, logx, &res);
      s[ii-1][0] += h[ii-2]*res;
    }
    s[ii-1][0] = 0.5*s[ii-1][0];
  }
	
  for(int ii=2;ii<=Nint;ii++){
    for(j=2;j<=ii;j++){
      s[ii-1][j-1] = s[ii-1][j-2]+(s[ii-1][j-2]-s[ii-2][j-2])/(pow(4.,j-1)-1);
    }
  }
	
  return s[Nint-1][Nint-1];  
}


double sinc( double x ) {

    double y;

    if ( x <= 0.0 ){
        y = 0.0;
    } else {
        y = sin(x)/x;
    }

    return y;
}
