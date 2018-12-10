#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <time.h>

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
double zarray[nzmax]; //Solar unit
double rarray[nrmax]; 
double lambda_table[ntmax][nrmax];
double tres, zres, eres;

const double megapc = 3.0857e24; // in cm/s

using namespace std;

#define Nx 100
static double xp[Nx], yp[Nx], yp2[Nx];
static int Nspline;

std::vector<double> calc_Shaw_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x);

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

std::vector<double> calc_Flender_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x);

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

double calc_Flender_xray_flux (cosmo cosm_model, float z, float Mvir, std::vector<float> x);

void free_FFTdata();
void FFT_density_profile(double *output, double *bin, int nbin);

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

static double R500_here;

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
};

static struct cosmo_params CP;

void init_model(double H0, double Omega_M, double Omega_b, double wt, double Omega_k, double ns, double nH, char *inputPk){

  set_cosmology_halo_info(inputPk, Omega_M, Omega_b, wt, H0/100.0, ns);
  CP.H0 = H0;
  CP.Omega_M = Omega_M;
  CP.Omega_b = Omega_b;
  CP.wt = wt;
  CP.Omega_k = Omega_k;
  CP.ns = CP.ns;

  set_lambda_table(tarray,rarray,lambda_table,nH,0); 
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
  int nzbin = 31;
  float zmin = 1e-3;
  float zmax = 2.8;
	
  int nmbin = 31;
  float logMvir_min= 12.0;
  float logMvir_max= 15.8;

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
  
  double *tab_Fourier;
  tab_Fourier = (double *) malloc(sizeof(double) * Nx * nzbin * nmbin);
  if(tab_Fourier == NULL){
    fprintf(stderr, "failed malloc sizeof(double) * %d\n", Nx*nzbin*nmbin);
    exit(1);
  }

  double *tab_r500;
  tab_r500 = (double *) malloc(sizeof(double) * nzbin * nmbin);
  if(tab_r500 == NULL){
    fprintf(stderr, "failed malloc sizeof(double) * %d\n", nzbin*nmbin);
    exit(1);
  }
  
  for(int i=0;i<Nx;i++){
    tab_l_ls[i] = -5.0+(double)(i)*dlog_ell;
    bin[i] = pow(10., tab_l_ls[i]);
  }
	  
  for(int i=0;i<nzbin;i++){
    //fprintf(stdout, ".");fflush(stdout);
    for(int j=0;j<nmbin;j++){
			
      //cout << zlist[i] << " " << Mlist[j] << endl;
      //flux[i][j] = calc_Flender_xray_flux (cosm_model, z_fft[i], M_fft[j], xlist); //ergs/s/cm^2
		
      std::vector<double> emission;
      emission = calc_Flender_xray_emissivity_profile(cosm_model, z_fft[i], M_fft[j], xlist); // ergs/s/cm^3/str
      tab_r500[j+nmbin*i] = R500_here;
      
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
  double Mpc2cm = 3.0856*1e18*1e6;
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
      double dVdz = covd*covd *C*CP.H0/100.0/H_z(zhere);
      double calc_k = ell_bin/covd;
      double gfac = (growth(1./(1+zhere))/growth(1.0));
      double Pk = gfac*gfac*PowerSpec(calc_k);
			
      for(int jm=0;jm<nmbin;jm++){
         //if ( flux[iz][jm] >= 0.0 ) {	
	    double logMvir = dlogm*(double)(1.*jm) + logMvir_min;
	    mlist[jm] = logMvir;
	    double Mvir = pow(10., logMvir) * CP.H0/100; // in the unit of Msun/h
	    double r500 = tab_r500[jm+nmbin*iz];
				
	    double ells = covd/(1+zhere)/r500; // = ell_500
				
	    double l_ls = ell_bin/ells;
												
	    for(int il=0;il<Nell;il++){
	        tab_fc_int[il] = tab_Fourier[il + Nell * (jm + nmbin * iz)] * Mpc2cm; // ergs/s/cm^2/str/Mpc
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
	        xl = xl * 4*M_PI*(r500/CP.H0*100.0)/ells/ells; // ergs/s/cm^2/str
	    }

	    //double m200m = M_vir_to_M_delta(zhere, Mvir, 200.0);		
	    double mf = dndlogm_fast(log10(Mvir), zhere);
	    double b = halo_bias_fast(log10(Mvir), zhere);
								
	    cl_xx_1_int_m[jm] = mf * xl * xl;
	    cl_xx_2_int_m[jm] = mf * b * xl;
         //} else {
	 //   cl_xx_1_int_m[jm] = 0.0;
	 //   cl_xx_2_int_m[jm] = 0.0;
         //}

      }
						
      spline(mlist-1, cl_xx_1_int_m-1, nmbin, yp1, ypn, cl_xx_1_int_m2-1);
      spline(mlist-1, cl_xx_2_int_m-1, nmbin, yp1, ypn, cl_xx_2_int_m2-1);
			
      double oneh_xx, twoh_xx;
			
      //double tab_spline_and_integral(int Nbin, double *xlist, double *ylist, double *zlist)
			
      oneh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_1_int_m, cl_xx_1_int_m2);
      twoh_xx = tab_spline_and_integral(nmbin, mlist, cl_xx_2_int_m, cl_xx_2_int_m2);
			
      cl_xx_1_int_z[iz] = zhere * dVdz * oneh_xx;
      cl_xx_2_int_z[iz] = zhere * dVdz * twoh_xx * twoh_xx * Pk;
      						
   }
		
    spline(zlist-1, cl_xx_1_int_z-1, nzbin, yp1, ypn, cl_xx_1_int_z2-1);
    spline(zlist-1, cl_xx_2_int_z-1, nzbin, yp1, ypn, cl_xx_2_int_z2-1);
		
    cl_xx_1 = tab_spline_and_integral(nzbin, zlist, cl_xx_1_int_z, cl_xx_1_int_z2);
    cl_xx_2 = tab_spline_and_integral(nzbin, zlist, cl_xx_2_int_z, cl_xx_2_int_z2);

    signal[i] = cl_xx_1 + cl_xx_2;

    if(cl_xx_1 != cl_xx_1 || cl_xx_2 != cl_xx_2) signal[i]=0;
    
  }
  
  free_FFTdata();
  free(tab_Fourier); free(tab_r500);

  vector<double> v;
  for(int i=0;i<Nbin;i++) v.push_back(signal[i]);
  Py_intptr_t shape[1] = { v.size() };
  npy::ndarray result = npy::zeros(1, shape, npy::dtype::get_builtin<double>());
  copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));

  return result;
}

BOOST_PYTHON_MODULE( xx_power ){
  Py_Initialize();
  npy::initialize();

  py::def("init_model", init_model);
  py::def("free_cosmology", free_cosmology);
  py::def("set_Flender_params", set_Flender_params);
  py::def("return_xx_power", return_xx_power);
}


double calc_Flender_xray_flux (cosmo cosm_model, float z, float Mvir, std::vector<float> x){
	
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
	
  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir*h);
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
	
  double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, emi, dlum;
  double luminosity = 0.0;
  double flux = 0.0;

  // distances in Mpc
  //double D_A = cosm_model.ang_diam(Redshift);
  double D_L = cosm_model.lum_dist(Redshift);
  D_L *= megapc;

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  npoly_mod = 1.0/(gamma_mod - 1.0 );

  double dvol[x.size()]; // radial shell vol in cm^3
  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*R500;
    if ( xi == 0 ) {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0);
    } else {
        dvol[xi] = 4.0*M_PI*pow(r*megapc, 3.0) - dvol[xi-1];
    }
  }
  for(int xi=0;xi<x.size();xi++){
    // r in Mpc;
    r = (double) x[xi]*R500;
    if(r >= Rmax){
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
        //emi = icm_mod.calc_xray_emissivity(r, R500, Redshift); // ergs/s/cm^3
        dlum = emi * dvol[xi]; // ergs/s
    } 
    
    luminosity += dlum;		
  }
  flux = luminosity/(4.0*M_PI*D_L*D_L);	//ergs/s/cm^2
  return flux;	
}

std::vector<double> calc_Flender_xray_emissivity_profile(cosmo cosm_model, float z, float Mvir, std::vector<float> x){
	
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
	
  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir*h);
  nfwclus.set_conc(cvir);
  M500 = nfwclus.get_mass_overden(500.0);// Msun
  R500 = nfwclus.get_rad_overden(500.0);// (physical) Mpc
  Rvir = nfwclus.get_radius();

  R500_here = R500 * h; // physical Mpc/h
  
  //cout << M500 << " " << R500 << " " << Rvir << endl;
	
  gas_model icm_mod(delta_rel, ad_index, eps_fb, eps_dm, fs_0, fs_alpha, pturbrad, delta_rel_zslope, delta_rel_n);
	
  icm_mod.calc_fs(M500, Omega_b/Omega_M, cosmic_t0, cosmic_t);
  icm_mod.evolve_pturb_norm(Redshift, rcutoff);
  icm_mod.set_nfw_params(Mvir, Rvir, nfwclus.get_conc(), nfwclus.get_rhoi(), R500);
  icm_mod.set_mgas_init(Omega_b/Omega_M);
  icm_mod.findxs();
	
  icm_mod.solve_gas_model(verbose, 1e-5);
	
  double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
  //double Yanl = icm_mod.calc_Y(R500, Rvir, Rmax);
	
  double r, emi;
  std::vector<double> emission;

  // redshift dependence in solid angle
  double fac = 4.0*M_PI*pow(1.0+Redshift, 4.0); // in steradians

  float npoly_mod, gamma_mod;
  gamma_mod = gamma_mod0 * pow((1.0+Redshift),gamma_mod_zslope);
  npoly_mod = 1.0/(gamma_mod - 1.0 );

  for(int xi=0;xi<x.size();xi++){
    r = (double) x[xi]*R500;
    if(r >= Rmax){emi = 0.0;}
    else{
        /*
        double ngas,pressure, kT;
        pressure = icm_mod.returnPth_mod2(r, R500, x_break, npoly_mod, x_smooth);
        ngas = icm_mod.return_ngas_mod(r, R500, x_break, npoly_mod);
        kT = pressure/ngas;
        */

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
        //emi = icm_mod.calc_xray_emissivity(r, R500, Redshift); // count/s/cm^3
        //printf("r, pressure, kT, ngas, emi = %f, %e, %e, %e, %e\n", r, pressure, kT, ngas, emi);

    } 		
    //if(emi < 1e-50){emi = 0.0;}
		
    emi = emi/fac; // ergs/s/cm^3/str

    //printf("emission = %e\n", emi);		
    //cout << r << " " << emi <<endl;
		
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
	
  float cvir = conc_norm * c_vir_DK15_fast(z, Mvir*h);
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
	
  double Rmax = icm_mod.thermal_pressure_outer_rad()*R500;
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


