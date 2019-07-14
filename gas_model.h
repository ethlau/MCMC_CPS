// class for the gas model
// code originally by Suman Bhattacharya, modified by Samuel Flender
#ifndef GAS_MODEL_HEADER_INCLUDED
#define GAS_MODEL_HEADER_INCLUDED

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_dht.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include "xray.h"

using namespace std;

struct parameters {
  double A_nt;
  double B_nt;
  double gamma_nt;
  //double alpha0; // fiducial : 0.18
  //double n_nt;   // fiducial : 0.80
  //double beta;   // fiducial : 0.50
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
  double alpha_clump;
  double beta_clump;
  double gamma_clump;
};

void set_fiducial_parameters (struct parameters *params);

void set_parameters (double pzero, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16, double p17, struct parameters *params);

struct my_func_params { float a; float b; float c; double d; int e;float f;};

/*
double Shaw_model_param::conc_norm = 1.0;
double Shaw_model_param::conc_mass_norm = 1.0;
int Shaw_model_param::pturbrad = 2;
bool Shaw_model_param::verbose = false;
double Shaw_model_param::overden_id = -1.0;
int Shaw_model_param::relation = 3;
double Shaw_model_param::rcutoff = 2.0;
*/

double sx_func (double x, void * params);
double ss_func (double x, void * params);
double fx_func (double x, void * params);
double ftx_func (double x, void * params);
double tx_func (double x, void * params);
double tx_func_p (double x, void * params);
double ttx_func (double x, void * params);
double gasmod_apply_bc(const gsl_vector * x, void *p);
double yproj_func(double x, void * params);
double yint_func(double x, void * p);
double yint_func_sam(double x, void * p);
double arnaud_func(double x, void * p);
double proj_arnaud_func(double x, void * p);
double yfft_func(double x, void * p);
double kappa_integrant(double x, void* p);
double mgas500_func(double x, void * p);
double mgas500_func_mod(double x, void * p);
double mgas500_func_mod_clumped(double x, void * p);
double arnaud_func_k(double x, void * p);
double yfft_func_k(double x, void * p);
double proj_KS02_func(double x, void * p);
double gasproj_func(double x, void * p);
double gasproj_func_mod(double x, void * p);
double NFWproj_func(double x, void * p);

int gasmod_constraints(const gsl_vector * x, void *p, gsl_vector * f);
int gasmod_constraints_df(const gsl_vector *x, void *p, gsl_matrix *J);
int gasmod_constraints_fdf(const gsl_vector *x, void *p, gsl_vector *f, gsl_matrix *J);


double gxs (double x, void *p);
double dgxs (double x, void *p);
void gxs_fdf (double x, void *p, double *y, double *dy);

class gas_model {

    friend double sx_func (double x, void * params);
    friend double ss_func (double x, void * params);
    friend double fx_func (double x, void * params);
    friend double ftx_func (double x, void * params);
    friend double tx_func (double x, void * params);
    friend double tx_func_p (double x, void * params);
    friend double ttx_func (double x, void * params);
    friend double yproj_func(double x, void * params);
    friend double yint_func(double x, void * p);
    friend double yint_func_sam(double x, void * p);
    friend double yfft_func(double x, void * p);
    friend double kappa_integrant(double x, void* p);
    friend double arnaud_func(double x, void * p);
    friend double proj_arnaud_func(double x, void * p);
    friend double mgas500_func(double x, void * p);
    friend double mgas500_func_mod(double x, void * p);
    friend double mgas500_func_mod_clumped(double x, void * p);
    friend double yfft_func_k(double x, void * p);
    friend double gasproj_func(double x, void * p);
    friend double gasproj_func_mod(double x, void * p);
    friend double NFWproj_func(double x, void * p);
    friend int gasmod_constraints(const gsl_vector * x, void *p, gsl_vector * f);
    friend int gasmod_constraints_df(const gsl_vector * x, void *p, gsl_matrix * J);
    friend int gasmod_constraints_fdf(const gsl_vector * x, void *p, gsl_vector * f, gsl_matrix * J);
    friend double gxs (double x, void *p);
    friend double dgxs (double x, void *p);
    friend void quadratic_fdf (double x, void *p, double *y, double *dy);

    protected:

    //double delta_rel, delta_rel_n, n, eps, eps_dm, fs_0, fs_alpha, f_s, Mpiv, chi_turb, delta_rel_zslope;
    double A_nt, B_nt, n, eps, eps_dm, fs_0, fs_alpha, f_s, Mpiv, chi_turb, gamma_nt;
    double C, ri, rhoi, mass, radius, vcmax, mgas, Ytot, pressurebound, R500toRvir, R500toR200m;
    double xs;
    double final_beta, final_Cf, p0, rho0, T0; // need to define these
    //double Aprime, Bprime;
    double Tau_d, Tau_b, bulge_frac;
    double *x, *k;
    double *ysz, *fft_ysz, *ell;
    double *rhogas, *rr, *Tsz, *Ksz, *clumpf;
    int nrads, nell;

    double clump0, alpha_clump, beta_clump, gamma_clump;

    double m_sun, clight, mpc, G, mu_e, mmw, m_p, eV, sigma_T, m_e, charge, me_csq; 

    public:

    gas_model() {
        set_constants();
    }
    gas_model(double *p) {

        n = p[0];
        eps = p[1];
        eps_dm = p[2];
        fs_0 = p[3];
        fs_alpha = p[4];
        //delta_rel_zslope = p[6];
        //delta_rel_n = p[7];
        A_nt = p[5];
        B_nt = p[6];
        gamma_nt = p[7];

        Mpiv = 3.0e14; // in Msol
        //if (pturbrad==1) chi_turb = (n-1.0)/(-1.0*(n+1.0));
        //else 
        chi_turb = 0.0;
        set_constants();

        // stellar evolution parameters (c.f. Nagamine et al. (2006)
        pressurebound = 1.0;
        bulge_frac = 0.9;
        Tau_d = 4.5;//4.5; // in Gyr
        Tau_b = 1.5;//1.5; // in Gyr
        //assert(eps > 0);
    }

/*
void evolve_pturb_norm(double z, double outer_radius) { //compute alpha(z)
    double fmax, evo_power, evo_converge;
    if (delta_rel == 0.0) {
        delta_rel = 0.0;
    }
    else if (delta_rel_zslope<=0.0) {
        delta_rel *= pow(1.0 + z, delta_rel_zslope);
    }
    else {
        // This is the old power-law evolution
        evo_power = pow(1.0 + z, delta_rel_zslope);
        // This is the new version that asymptotes to a maximum value, preventing thermal pressure from going negative
        fmax = 1.0 / (delta_rel * pow(outer_radius*2.0, delta_rel_n)); // factor of two converts rvir-> r500
        //fmax = 1.0 / (delta_rel * pow(4.0, delta_rel_n)); // factor of two converts rvir-> r500
        // ---!!!
        // outer_radius HAS to be =2 in order to match the equations in Shaw et al (2010)
        // ---!!!
        evo_converge = (fmax - 1.0)*tanh(delta_rel_zslope*z) + 1.0;
        delta_rel *= min(evo_power, evo_converge); //v2 of model
        //delta_rel *= evo_power; //v1 of model
    }
    //cout << "delta_rel at z = " << delta_rel << endl;
}
*/

void set_nfw_params(double bmass, double bradius, double conc, double brhoi, double r500) { // set the NFW parameters
    mass = bmass; // Mvir [Msol]
    radius = bradius; // Rvir [Mpc]
    C = conc; // cvir_Mpc
    rhoi = brhoi; // NFW density at NFW scale radius [Msol/Mpc^3]
    ri = radius/C; // NFW scale radius [Mpc]
    //cout<< mass<<" "<<ri<<endl;
    ri = ri*mpc/1000.0; //in km (for later units)
    // SO after calling this, ri is always in units km!
    vcmax = sqrt(4.0*M_PI*G*rhoi*ri*ri*Gmax()); //% in km/s
    //findxs();// now can calculation radius within which stellar mass is contained
    R500toRvir= r500/radius;
}

void set_constants() {

    m_sun = 1.98892e30; //kg
    clight = 3.0e5; // in km/s
    mpc = 3.0857e22; // in m
    G = 6.67e-11*m_sun/pow(mpc,3); // in mpc^3/Msun/s^2
    //mu_e = 1.143; // SF: mean molecular weight per electron
    mu_e = 1.136; // X=0.76 assumed
    //mmw = 0.59; // mean molecular weight
    mmw = 0.58824; // X=0.76 assumed
    m_p  = 1.6726e-27;// mass proton, kg
    eV = 1.602e-19; // 1eV in J
    sigma_T = 6.652e-25/(1.0e4); // now in m^2.
    m_e = 9.11e-31; // mass electron, kg
    //charge = 1.60217646e-19;// joules
    //me_csq = m_e*clight*clight*1.0e6/(1000.0*charge); // in KeV
    me_csq = 511.0; // in KeV

}

void set_mgas_init(double baryon_frac_univ) {
    mgas = (baryon_frac_univ)*mass/(1.0+f_s);
}

void set_mass(double inp) {
    mass = inp;
}

void set_ri(double inp) {
    ri = inp;
}

void set_vcmax(double inp) {
    vcmax = inp;
}

void set_mgas(double inp) {
    mgas = inp;
}

//void set_delta_rel(double inp) {
//    delta_rel = inp;
//}

//void set_delta_rel_n(double inp) {
//    delta_rel_n = inp;
//}

void set_A_nt(double inp) {
    A_nt = inp;
}

void set_B_nt(double inp) {
    B_nt = inp;
}


void set_n(double inp) {
    n = inp;
}

void set_C(double inp){
    C = inp;
}

void set_xs(double inp){
    xs = inp;
}

void set_f_s(double inp){
    f_s = inp;
}

void set_R500toR200m(double inp){
    R500toR200m = inp;
}

double get_C(){
    return C;
}

double get_n(){
    return n;
}

//double get_delta_rel(){
//    return delta_rel;
//}

double get_A_nt(){
    return A_nt;
}


double get_f_s(){
    return f_s;
}


void calc_fs(double M500, double baryon_frac_univ, double cosm_t0, double cosm_tz) { // compute the star fraction
    //---note M500 must be in Msol

    f_s = min(fs_0 * pow(M500/Mpiv,-1.0*fs_alpha), 0.8*baryon_frac_univ); //
    if (f_s <= 0) {
        cout << f_s << endl;
        cout << fs_0 << " " << M500/Mpiv << " " << fs_alpha << endl;
        f_s = 1.e-4;
    }
    f_s = f_s / (baryon_frac_univ - f_s); // f_s is now the star formation efficiency
    
    //assert (f_s >= 0);
    //---uncomment this line for z-evolution:
    //f_s = f_s*calc_fstarz(cosm_t0, cosm_tz);

}

void set_stellar_evo_params(double inp1, double inp2, double inp3) {
    // reset stellar evolution parameters as in Nagamine et al. 2006
    Tau_b = inp1;
    Tau_d = inp2;
    bulge_frac = inp3;
}

double calc_fstarz(double cosm_t0, double cosm_tz) {
    // calculates fraction of stars formed at z = 0 that have formed by redshift z
    // Assumes Nagamine et al 06 evolution of disk and bulge populations by default
    // Use the Nagamine values for Tau_d, Tau_b and bulge_frac;
    double chi_b, chi_d, fb, fd, fstarz;

    chi_b = (1.0 - (cosm_t0/Tau_b + 1.0)*exp(-1.0*cosm_t0/Tau_b));
    chi_d = (1.0 - (cosm_t0/Tau_d + 1.0)*exp(-1.0*cosm_t0/Tau_d));
    fb = bulge_frac/chi_b*(1.0 - (cosm_tz/Tau_b + 1.0)*exp(-1.0*cosm_tz/Tau_b));
    fd = (1.0-bulge_frac)/chi_d*(1.0 - (cosm_tz/Tau_d + 1.0)*exp(-1.0*cosm_tz/Tau_d));
    fstarz = fb + fd;

    //cout<<cosm_t0<<" "<<cosm_tz<<" "<<chi_b<<" "<<chi_d<<" "<<fb<<" "<<fd<<endl;
    return fstarz;
}

// now put solvep0rho0 functions in here

double H(double x) { // eqn 6b
    return  (1/((1+x)*g(x)))*(-1*log(1+x) + (x*(1+x/2))/(1+x));
}

double g(double x) { // eqn 2b
    return log(1.0+x) - x/(1.0+x);
}

double findxs() { // solve for x_s (sec 3.1)

    int status;
    int iter = 0, max_iter = 100;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    
    double x_lo = 0.0, x_hi = 10.*C, x_guess;
    double p[2] = { C, f_s }; 

    gsl_function F;
    F.function = &gxs;
    F.params = &p;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);
    status = gsl_root_fsolver_set (s, &F, x_lo, x_hi);

    //printf ("using %s method\n", gsl_root_fsolver_name (s));
    //assert( f_s >= 0);

    do {
      iter++;
      status = gsl_root_fsolver_iterate (s);
      x_guess = gsl_root_fsolver_root (s);
      x_lo = gsl_root_fsolver_x_lower (s);
      x_hi = gsl_root_fsolver_x_upper (s);
      status = gsl_root_test_interval (x_lo, x_hi, 0, 1.e-5);
    
      //printf ("%5d [%.7f, %.7f] %.7f\n", iter, x_lo, x_hi, x_guess);

    } while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS ) {
        cout << "xs did not converge! Setting xs to C." << endl;
        xs = C;
    } else {
        xs = x_guess;
    }
    
    gsl_root_fsolver_free (s);
    //cout << "In findxs: " << endl;
    //cout << "  C = " << C << endl;
    //cout << "  f_s = " << f_s << endl;
    //cout << "  xs = " << xs << endl;
    return xs;
}

double f(double x) { // eqn 5
    if (x<=C) return log(1+x)/x - (1/(1+C));
    else if (x>C) return (C/x)*((log(1+C)/C) - (1/(1+C)));
    else return -1;
}

double Gmax() { // eqn 3b
    double xmax = 2.163; // check this number!
    return g(xmax)/xmax;
}

double delta_s() { // eqn 15
    return S_C(C) / (S_C(C) + H(C)*g(C)/pow(C,3.0));
}

double S_C(double x) { // eqn 11
    double SC;
    SC = M_PI*M_PI/2.0 - log(x)/2.0 - 1.0/(2.0*x) - 1.0/(2.0*pow(1.0+x,2)) - 3.0/(1+x);
    SC += (0.5 + 1.0/(2.0*pow(x,2)) - 2.0/x - 1.0/(1.0+x))*log(1.0+x);
    SC += (3.0/2.0)*log(1.0+x)*log(1.0+x);
    SC += 3.0*gsl_sf_dilog(-1.0*x); // gsl dilog function
    //SC = SC*g(x)*g(x);// this is the correction made to Ostriker et al 05.
    // (Mar 10) Don't think it should be here.
    return SC;
}

void test_dilog() {
    cout << gsl_sf_dilog(-3.0) << endl;
}

double S_cx(double x) { // % eqn 7b
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (2000);
    double result, error, Sx, alpha = 0.0;
    gsl_function F;
    F.function = &sx_func;
    if (x==0) x = 1.e-7; //% diverges at x = 0 
    gsl_integration_qags (&F, x, C, 0, 1.e-7, 2000, w, &result, &error);
    Sx = S_C(C) - result;
    gsl_integration_workspace_free (w);
    return Sx;
}

double K(double x) { //% eqn 16
    double Kx = (1.0/3.0)*H(x)*(1./(Gmax()*(1.0-delta_s())));
    return Kx;
}

double K_s() { // % eqn 21

    if (xs <= 0 ) {
        cout << "In K_s: xs = "<< xs << " is less than zero! " << endl;
        return 0.0;
    }

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double resulta, resultb, error, Ks, tempC = C;
    gsl_function F,FF;
    F.function = &ss_func;
    FF.function = &fx_func;
    F.params = &tempC;
    FF.params = &tempC;
    //cout << "In K_s: xs= "<< xs << endl;
    gsl_integration_qags (&F, 0.0, xs, 0, 1.e-7, 1000, w, &resulta, &error);
    gsl_integration_qags (&FF, 0.0, xs, 0, 1.e-7, 1000, w, &resultb, &error);
    Ks = (1.0/g(C))*(resulta - (2.0/3.0)*resultb);
    gsl_integration_workspace_free (w);
    return Ks;
}

double theta(double x, double beta) { // % eqn 26b
    double th;
    th = (1.0 - (beta*j(x)/(1.0+n)));
    return fabs(th);
}

double theta_mod(double x, double beta, double x_break, double npoly_mod) {
    // ---
    // modified theta (broken power-law model)
    // x_break is here in units of the NFW scale radius
    // ---
    double th;

    if (x>=x_break){
        th = (1.0 - (beta*j(x)/(1.0+n)));
    }
    else if (x<x_break){
        th = (1.0 - beta*j(x)/(1.0+npoly_mod)) * (1.0 - beta*j(x_break)/(1.0+n))/(1.0 - beta*j(x_break)/(1.0+npoly_mod));
    }
    return fabs(th);
}

double j(double x) { //% eqn 25b
    double jj;
    if (x==0.0) jj = 0.0;
    else if (x<=C) jj = 1.0 - log(1.0+x)/x;
    else jj = 1.0 - 1.0/(1.0+C) - (log(1.0+C) - C/(1.0+C))/x;
    return jj;
}

double I2(double Cf, double beta) {// % eqn 28a
    /*
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double params[5] = {delta_rel, n, pturbrad, C, beta};
    gsl_function F;
    F.function = &ftx_func;
    F.params = &params;
    if (Cf<=0) {
        cout << "Cf error! in I2" << Cf << endl;
        return 0.0;
    }
    gsl_integration_qags (&F, 1e-7, Cf, 0, 1.e-5, 10000, w, &result, &error);
    gsl_integration_workspace_free (w);
    return result;
    */
    int nxbins = 1000, i;
    double *xx, *ftx, result, dlogx;
    double lowlim = 1e-30;
    if (Cf<=lowlim) {
        cout << "Cf error! in I2, Cf=" << Cf << endl;
        return 0.0;
    }

    xx = new double [nxbins];
    ftx = new double [nxbins];
    // first need to make an array of values
    result = 0.0;
    dlogx = (log10(Cf)-log10(lowlim))/((double)(nxbins-1));

    for (i=0;i<nxbins;i++) {
        xx[i] = pow(10.0, log10(lowlim) + (double)i * dlogx);
        ftx[i] = f(xx[i])*pow(theta(xx[i], beta),n)*pow(xx[i],2);
        result += ftx[i]*xx[i]*dlogx;
    }
    delete xx;
    delete ftx;

    return result;

}

double I2spline(double Cf, double beta) {// % eqn 27 
    int nxbins = 100, i;
    if (Cf<=0) { 
        cout << "Cf error! in I2spline, Cf=" << Cf << endl;
        return 0.0;
    }
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_steffen, nxbins);
    double *xx, *ftx, result;
    xx = new double [nxbins];
    ftx = new double [nxbins];
    // first need to make an array of values
    for (i=0;i<nxbins;i++) {
        xx[i] = ((double)i * Cf / ((double)(nxbins-1)));
        ftx[i] = f(xx[i])*pow(theta(xx[i], beta),n)*pow(xx[i],2);
    }
    gsl_spline_init (spline, xx, ftx, nxbins);
    result = gsl_spline_eval_integ (spline, xx[0], xx[nxbins-1], acc);
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    delete xx;
    delete ftx;
    return result;
}

double I3(double Cf, double beta) {// % eqn 28b
    /*
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double params[5] = {delta_rel, n, pturbrad, C, beta};
    gsl_function F;
    F.function = &tx_func;
    F.params = &params;
    //cout << "In I3: Cf, beta= "<< Cf << " " << beta << endl;
    if (Cf<=0) {
        cout << "Cf error! in I3" << Cf << endl;
        return 0.0;
    }
    gsl_integration_qags (&F, 1e-7, Cf, 0, 1.e-5, 10000, w, &result, &error);
    //cout << "I3: " << result << endl;
    gsl_integration_workspace_free (w);
    return result;
    */

    int nxbins = 1000, i;
    double *xx, *tx, result, dlogx;
    double lowlim = 1e-30;
    if (Cf<=lowlim) {
        cout << "Cf error! in I3, Cf=" << Cf << endl;
        return 0.0;
    }

    xx = new double [nxbins];
    tx = new double [nxbins];
    // first need to make an array of values
    result = 0.0;
    dlogx = (log10(Cf)-log10(lowlim))/((double)(nxbins-1));
    for (i=0;i<nxbins;i++) {
        xx[i] = pow(10.0, log10(lowlim) + (double)i * dlogx);
        tx[i] = pow(theta(xx[i], beta),n+1.0)*pow(xx[i],2);
        result += tx[i]*xx[i]*dlogx;
    }
    delete xx;
    delete tx;

    return result;
}


double I3spline(double Cf, double beta) {// % eqn 27 {

    int nxbins = 1000, i;
    double lowlim = 1e-30;
    if (Cf<=lowlim) {
        cout << "Cf error! in I3spline, Cf=" << Cf << endl;
        return 0.0;
    }

    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_steffen, nxbins);
    double *xx, *tx, result, dlogx;
    xx = new double [nxbins];
    tx = new double [nxbins];
    result = 0.0;
    dlogx = (log10(Cf)-log10(lowlim))/((double)(nxbins-1));
    for (i=0;i<nxbins;i++) {
        //xx[i] = ((double)i * Cf / ((double)(nxbins-1)));
        xx[i] = pow(10.0, log10(lowlim) + (double)i * dlogx);
        tx[i] = pow(theta(xx[i], beta),n+1.0)*pow(xx[i],2);
    }
    gsl_spline_init (spline, xx, tx, nxbins);
    result = gsl_spline_eval_integ (spline, xx[0], xx[nxbins-1], acc);
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    delete xx;
    delete tx;
    return result;
}

double I3p(double Cf, double beta) {// % eqn 28b
    /*
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double params[5] = {delta_rel, n, pturbrad, C, beta};
    gsl_function F;
    if ((int)pturbrad==1) F.function = &tx_func_p;
    else F.function = &tx_func;
    F.params = &params;
    
    double lowlim = 1e-7;
    if (Cf<=lowlim) {
        cout << "Cf error! in I3p" << Cf << endl;
        return 0.0;
    }
    gsl_integration_qags (&F, 1e-7, Cf, 0, 1.e-5, 10000, w, &result, &error);
    //cout << "I3: " << result << endl;
    gsl_integration_workspace_free (w);
    if ((int)pturbrad==0) result = result*delta_rel*2.0; // check factor of 2!
    else if ((int)pturbrad==2) result = 0.0;
    return result;
    */
    int nxbins = 1000, i;
    double *xx, *tx, result, dlogx;
    double lowlim = 1e-30;
    if (Cf<=lowlim) {
        cout << "Cf error! in I3p, Cf=" << Cf << endl;
        return 0.0;
    }

    xx = new double [nxbins];
    tx = new double [nxbins];
    result = 0.0;
    dlogx = (log10(Cf)-log10(lowlim))/((double)(nxbins-1));

    for (i=0;i<nxbins;i++) {
        xx[i] = pow(10.0, log10(lowlim) + (double)i * dlogx);
        //if (pturbrad==1) tx[i] = delta_rel*pow(theta((xx[i]),beta),n-1.0)*pow((xx[i]),2);
        //else tx[i] = pow(theta((xx[i]), beta),n+1.0)*pow((xx[i]),2);
        tx[i] = pow(theta((xx[i]), beta),n+1.0)*pow((xx[i]),2);
        result += tx[i]*xx[i]*dlogx;
    }
    delete xx;
    delete tx;

    return result;

}


double I3p_spline(double Cf, double beta) {// % eqn 27 {
    // DO NOT USE THIS FOR NOW
    int nxbins = 1000, i;

    if (Cf<=0) {
        cout << "Cf error! in I3p_spline, Cf=" << Cf << endl;
        return 0.0;
    }
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_steffen, nxbins);
    double *xx, *tx, result;
    xx = new double [nxbins];
    tx = new double [nxbins];
    // first need to make an array of values

    for (i=0;i<nxbins;i++) {
        xx[i] = ((double)i * Cf / ((double)(nxbins-1)));
        //if (pturbrad==1) tx[i] = delta_rel*pow(theta((xx[i]),beta),n-1.0)*pow((xx[i]),2);
        //else tx[i] = pow(theta((xx[i]), beta),n+1.0)*pow((xx[i]),2);
        tx[i] = pow(theta((xx[i]), beta),n+1.0)*pow((xx[i]),2);
    }
    gsl_spline_init (spline, xx, tx, nxbins);
    result = gsl_spline_eval_integ (spline, xx[0], xx[nxbins-1], acc);
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    delete xx;
    delete tx;
    return result;
}

double L(double Cf, double beta) {// % eqn 27
    /*
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error, Sx;
    double params[5] = {delta_rel, n, pturbrad, C, beta};
    gsl_function F;
    F.function = &ttx_func;
    F.params = &params;
    if (Cf<=lowlim) { 
        cout << "Cf error! in L" << Cf << endl;
        return 0.0;
    }
    gsl_integration_qags (&F, 1e-7, Cf, 0, 1.e-5, 10000, w, &result, &error);
    //cout << "L: " << result << endl;
    gsl_integration_workspace_free (w);
    return result;
    */
    int nxbins = 1000, i;
    double lowlim = 1e-30;
    if (Cf<=lowlim) { 
        cout << "Cf error! in L, Cf=" << Cf << endl;
        return 0.0;
    }
    double *xx, *ttl, result, dlogx;
    xx = new double [nxbins];
    ttl = new double [nxbins];
    result = 0.0;
    dlogx = (log10(Cf)-log10(lowlim))/((double)(nxbins-1));

    for (i=0;i<nxbins;i++) {
        xx[i] = pow(10.0, log10(lowlim) + (double)i * dlogx);
        ttl[i] =  pow(theta((xx[i]), beta), n)*pow((xx[i]),2);
        result += ttl[i]*xx[i]*dlogx;
    }
    delete xx;
    delete ttl;

    return result;

}

double Lspline(double Cf, double beta) {// % eqn 27 {
    int nxbins = 1000, i;
    if (Cf<=0) {
        cout << "Cf error! in Lspline, Cf = " << Cf << endl;
        return 0.0;
    }
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_steffen, nxbins);
    double *xx, *ttl, result;
    xx = new double [nxbins];
    ttl = new double [nxbins];
    // first need to make an array of values
    for (i=0;i<nxbins;i++) {
        xx[i] = ((double)i * Cf / ((double)(nxbins-1)));
        ttl[i] =  pow(theta((xx[i]), beta), n)*pow((xx[i]),2);
    }
    gsl_spline_init (spline, xx, ttl, nxbins);
    result = gsl_spline_eval_integ (spline, xx[0], xx[nxbins-1], acc);
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    delete xx;
    delete ttl;
    return result;
}

double Lvar(double Cf, double beta) {
    // allows different outer pressure boundaries
    double gfact, Cfp, Lc, Lp;
    gfact = g(pressurebound*C) / g(C);
    if (gfact<1.0) cout << "gfact error" << endl;
    Cfp = Cf;
    //Lc = Lspline(Cf,beta);
    Lc = L(Cf,beta);
    Lp = Lc;
    // do this in three steps to speed it up
    while (Lp < gfact*Lc) {
        Cfp *= 1.2;
        //Lp = Lspline(Cfp,beta);
        Lp = L(Cfp,beta);
    }
    Cfp  = Cfp / 1.2;
    while (Lp < gfact*Lc) {
        Cfp *= 1.1;
        //Lp = Lspline(Cfp,beta);
        Lp = L(Cfp,beta);
    }
    Cfp  = Cfp / 1.1;
    while (Lp < gfact*Lc) {
        Cfp *= 1.02;
        //Lp = Lspline(Cfp,beta);
        Lp = L(Cfp,beta);
    }
    Cfp *= 1.01/1.02; // settle on half way in between
    return Cfp;
}

double Edm(double aniso_beta) { 
    //% Calculation of T + W for dark matter energy transfer (see Bode et al. 09)
    // integrate the kinetic + potential energy
    //w0 = -1*vcmax^2*mass*H(C)/Gmax; % Ostriker et al (05) Eq6a
    double winf, W0Lokas, E;
    winf = (G*pow(mpc/1000.0,3)*pow(mass,2)/ri)*pow(g(C),-2)/2.0; // Lokas & Mamon (01) eq 21
    W0Lokas = -1.0*winf*(1.0 - (1.0/pow(1.0+C,2)) - (2.0*log(1.0+C)/(1.0+C)));
    E = Ek(C, aniso_beta, winf);
    //abs(2*E/W0Lokas) % 2|T|/W (i.e. virial ratio)
    return (W0Lokas + E);
}

double Ek(double x, double aniso_beta, double Winf) {
    //% calculate total kinetic energy T in NFW halo using Lokas & Mamon 01
    double K;
    if (aniso_beta==0.0) {
        K = -3.0 + 3.0/(1.0+x) - 2.0*log(1.0+x) + x*(5.0 + 3.0*log(1.0+x));
        K = K - pow(x,2)*(7.0 + 6.0*log(1.0+x));
        K = K + pow(x,3)*(M_PI*M_PI - log(C) - log(x/C) + log(1.0+x) + 3.0*pow(log(1+x),2) + 6.0* gsl_sf_dilog(-1.0*x));
        K = K*0.5;
    }
    else if (aniso_beta==0.5) {
        K = -3.0 + 3.0/(1.0+x) - 3.0*log(1.0+x);
        K = K + 6.0*x*(1.0+log(1.0+x));
        K = K - pow(x,2)*(M_PI*M_PI + 3.0*pow(log(1.0+x),2) + 6.0*gsl_sf_dilog(-1.0*x));
        K = K/3;
    }
    else {
        K = -2.0*log(1.0+x);
        K = K + x*(M_PI*M_PI/3.0 - 1.0/(1.0+x) + pow(log(1.0+x),2) + 2.0*gsl_sf_dilog(-1.0*x));
        K = K/2.0;
    }
    return K*Winf;
}
/*
double setAprime() {
    Aprime = 1.5*(1.0+f_s)*(Gmax()*K(C)*(3.0-4.0*delta_s()) + K_s());
    Aprime +=  -1.0*(Gmax()*eps*f_s*pow(clight/vcmax,2)) - (Gmax()*eps_dm*fabs(Edm(0.0))/(mgas*pow(vcmax,2)));
    return Aprime;
}

double setBprime() {
    Bprime = (1.0+f_s)*(S_C(C)/g(C));
    return Bprime;
}
*/
double energy_constraint(double beta, double Cf) {

    double val, Lval;
    double Aprime, Bprime;
    double Ks = K_s();

    Aprime = 1.5*(1.0+f_s)*(Gmax()*K(C)*(3.0-4.0*delta_s()) + Ks);
    Aprime +=  -1.0*(Gmax()*eps*f_s*pow(clight/vcmax,2)) - (Gmax()*eps_dm*fabs(Edm(0.0))/(mgas*pow(vcmax,2)));
    Bprime = (1.0+f_s)*(S_C(C)/g(C));

    if (beta<=0.0 || beta != beta) beta = 0.1;
    if (Cf<=0.0 || Cf != Cf ) Cf = C; // (C<0) is unphysical

    val = Aprime + Bprime*(pow(Cf,3) - pow(C,3))/3.0;
    //Lval = Lspline(Cf,beta);
    Lval = L(Cf,beta);
    //val += -1.0*I2(Cf,beta)/Lval + (1.5*(I3spline(Cf,beta) + I3p(Cf,beta))/(beta*Lval));
    val += -1.0*I2(Cf,beta)/Lval + (1.5*(I3(Cf,beta) + I3p(Cf,beta))/(beta*Lval));
    if (val != val ) {
        cout << "Energy constraint failed! " << endl;
        cout << "  eps, eps_dm, fs_0, fs_alpha = " << eps << " " << eps_dm << " " << fs_0 << " " << fs_alpha << endl;
        cout << "  beta, Cf, C, xs = " << beta << " " << Cf << " " << C << " " << xs << endl;
        cout << "  A, B, Lval = " <<  Aprime  << " " << Bprime << " " << Lval << endl;
        //cout << "f_s, Gmax(), K(C), delta_s(), K_s() = " << f_s << " " << Gmax() << " " << K(C) << " " << delta_s() << " " << Ks << endl; 

        //cout << "eps, vcmax, eps_dm, |Edm|, mgas =" << eps <<  " " << vcmax << " " << eps_dm << " " << fabs(Edm(0.0)) << " " << mgas << endl;
        return 0.0;
    }
    return (val);
}

double pressure_constraint(double beta, double Cf) {

    double val, Cfp;

    if (beta<=0.0) beta = 0.1;
    if (Cf<=0.0) Cf = C;
    //if (beta<=0.0) beta = 1.e-7;
    //if (Cf<=0.0) Cf = 1.e-7;

    Cfp = Lvar(Cf, beta);
    //val = pow((1.0+f_s)*(S_C(C*pressurebound)/g(C))*beta*Lspline(Cf,beta),(1.0/(1.0+n)));
    val = pow((1.0+f_s)*(S_C(C*pressurebound)/g(C))*beta*L(Cf,beta),(1.0/(1.0+n)));
    //if (pturbrad==1) val += -1.0*pow(1.0 + delta_rel*pow(theta(Cfp,beta),-2),1.0/(1.0+n))*theta(Cfp,beta);
    //else if (pturbrad==2) val += -1.0* pow(1.0,(1.0/(1.0+n)))*(1.0 - beta*j(Cfp)/(1.0+n));
    //else val += -1.0* pow(1.0+delta_rel,(1.0/(1.0+n)))*(1.0 - beta*j(Cfp)/((1.0+n)*(1.0+delta_rel)));
    val += -1.0* pow(1.0,(1.0/(1.0+n)))*(1.0 - beta*j(Cfp)/(1.0+n));
    if (val != val) {
        return 0.0;
        cout << "Pressure constraint failed! " << endl;
    }
    return (val);
}

int solve_gas_model (bool verbose, double tolerance) {

    const gsl_multiroot_fsolver_type *T;
    gsl_multiroot_fsolver *s;

    gsl_vector *v;

    size_t i, iter = 0, max_iter=100;
    const size_t ndim = 2;
    int status;
    double size;
    double p[15] = {n, eps, eps_dm, fs_0, fs_alpha, A_nt, B_nt, gamma_nt, C, mass, vcmax, ri, mgas, xs, f_s};

    //assert(eps > 0);

    gsl_multiroot_function func = {&gasmod_constraints, ndim, &p};
    /* Starting point */
    v = gsl_vector_alloc (2);
    gsl_vector_set (v, 0, 1.0);
    gsl_vector_set (v, 1, 10*C);
    /* Initialize method and iterate */
    T = gsl_multiroot_fsolver_hybrids;
    s = gsl_multiroot_fsolver_alloc (T,2);
    gsl_multiroot_fsolver_set (s, &func, v);

    do {
        iter++;
    	status = gsl_multiroot_fsolver_iterate(s);
    	if (status)
    	    break;
    	status = gsl_multiroot_test_residual (s->f, tolerance);
    	if ((status == GSL_SUCCESS) & verbose) {
    	    printf ("converged to minimum at\n");
    	    printf ("%5d %10.4e %10.4e\n",
    		    (int)iter,
    		    gsl_vector_get (s->x, 0),
    		    gsl_vector_get (s->x, 1));
    	}
    } while (status == GSL_CONTINUE && iter < max_iter);

    final_beta = gsl_vector_get (s->x, 0);
    final_Cf = gsl_vector_get (s->x, 1);

    if (final_Cf < 0 ) {
        cout << "final Cf < 0! Setting Cf to C/10" << endl; 
        final_Cf = C/10.;
        setp0rho0(true);
    } else {
        setp0rho0(verbose);
    }

    gsl_vector_free(v);
    gsl_multiroot_fsolver_free(s);
    return status;
}

void setp0rho0(bool verbose) {

    if (verbose) cout << "eps, eps_dm, fs_0, fs_alpha = " << eps << " " << eps_dm << " " << fs_0 << " " << fs_alpha << endl;
    if (verbose) cout << "mass, conc  = " << mass << " " << C << endl;
    if (verbose) {
        cout << " final_Cf "   << final_Cf << endl;
        cout << " final_beta " << final_beta << endl;
    }
    rho0 = mgas / (4.0*M_PI*pow(ri*1000/mpc,3)*Lspline(final_Cf, final_beta)); // in Msol/Mpc^3
    p0 = rho0*vcmax*vcmax/(final_beta*Gmax()); // in Msol/Mpc^3 (km/s)^2
    T0 = (mmw*m_p*p0/rho0)*(1000.0/eV); // this is in keV

    if (verbose) {
        cout << " final model solution:" << endl;
        cout << " P0      " << "rho0       " << "T0       " << "Mgas" << endl;
        cout << p0 << " " << rho0 << " " << T0  << " "<< mgas << endl;
    }

    p0 = p0*1.0e6*m_sun/(eV*1000.0*pow(mpc,3)); // now in keV/m^3

}

void initialize_profiles(double minr, double maxr, double dr, double ellmin, double ellmax, double dlfrac) {
    int i;
    double dtheta, dl_ell_10;
    double c500=1.177;
    minr = minr*radius; //in mpc
    maxr = maxr*radius;  //in mpc
    dr = dr*radius;  //in mpc
    if (dlfrac < 1) dl_ell_10 = dlfrac/log(10.0);
    nrads = (int) ceil((maxr-minr)/dr)+1;

    if (dlfrac == 0.0) nell = 1;
    if(dlfrac < 1 && dlfrac > 0.0) nell = (int)ceil((log10(ellmax/ellmin)/dl_ell_10))+1;
    if(dlfrac >= 1) nell= (int) ceil((ellmax-ellmin)/dlfrac)+1;

    x = new double [nrads];
    ysz = new double [nrads];
    rhogas= new double [nrads];
    Tsz= new double [nrads];
    Ksz= new double [nrads];
    rr= new double [nrads];
    k= new double [nrads];
    clumpf = new double [nrads];

    for (i=0;i<nrads;i++) {
        x[i] = minr + (double)i*dr; //in Mpc
        k[i]= 2.0*M_PI/x[i]*radius/c500;
    }

    fft_ysz = new double [nell];
    ell = new double [nell];
    for (i=0;i<nell;i++) {
        if (dlfrac<1) {
            ell[i] = log10(ellmin) + ((double)i*dl_ell_10);
            ell[i] = ceil(pow(10.0, ell[i]));
        }
        if(dlfrac >= 1)  ell[i] = ellmin + (double) i*dlfrac;
        //cout<<i<<" "<<ell[i]<<endl;
    }
}

int get_nrads() {
    return nrads;
}

int get_nell() {
    return nell;
}

void clear_profiles() {
    delete[] x;
    delete[] ysz;
    delete[] fft_ysz;
    delete[] ell;
    delete[] rhogas;
    delete[] rr;
    delete[] Tsz;
    delete[] Ksz;
    delete[] clumpf;
}

double* get_ell() {
    return &ell[0];
}

double* get_yfft() {
    return &fft_ysz[0];
}

double* get_ysz() {
    return &ysz[0];
}

double* get_k(){  return &k[0];}
double* get_x() {  return (double*)&x[0];}
double* get_P() {  return &ysz[0]; }
double* get_T(){ return &Tsz[0];}
double* get_rhogas(){ return &rhogas[0];}
double* get_K() {  return &Ksz[0]; }
double* get_xx(){ return &rr[0]; }
double* get_clumpf(){ return &clumpf[0]; }

/* Pressure profile from Arnaud et al. (2010) */
double calc_pressure_Arnaud(double r, double r500, double h, double E){
  double M500, R500, pi, rho_cr, rho_cr0, h70, XH, b_HSE;
  double P, P_gnfw, P0, c500, alpha_p, beta_p, gamma_p, Mpiv, xi;

  pi = 4.0*atan(1.0);
  XH = 0.76;
  b_HSE = 0.2; // Hydrostatic bias
  rho_cr0 = 2.77536627e11*h*h; //[Msun/Mpc^3]
  rho_cr = rho_cr0*E*E;
  h70 = h/0.7;

  R500 = (double) r500/pow(1.0+b_HSE, 1./3.);
  M500 = 4.0*pi/3.0*500.0*rho_cr*pow(R500, 3.0);

/* Arnaud et al. (2010) */
  /*
  P0 = 8.403*pow(h70, -3./2.);
  c500 = 1.177;
  gamma = 0.3081;
  alpha = 1.0510;
  beta = 5.4905;
 */

/* Planck Collaboration (2013) */
  P0 = 6.41;
  c500 = 1.81;
  gamma_p = 0.31;
  alpha_p = 1.33;
  beta_p = 4.13;

  //Mpiv = 3e14*0.7; //[Msun/h]
  Mpiv = 3e14; //[Msun]

  xi = r/R500;
  P_gnfw = P0/(pow(c500*xi, gamma_p)*pow(1.0+pow(c500*xi, alpha_p), (beta_p-gamma_p)/alpha_p));
  P = 1.65e-3*pow(E, 8./3.)*pow(M500/Mpiv, 2./3.+0.12)*P_gnfw*h70*h70;
  P = P/((2.0+2.0*XH)/(3.0+5.0*XH));//convert electron pressure to thermal pressure

  return P;
}

double pntfrac ( double r, double R500 ) {

    double fnt;
    double R200m = R500/R500toR200m;
    fnt = 1.0 - A_nt * (1.0 + exp( -pow(r/(R200m*B_nt),gamma_nt) ) );
    return fnt;

}

double calc_gas_density(double r, double R500){
  double xx, rhogas;

  xx = r/(1000.0*ri/mpc);
  rhogas = rho0*pow(theta(xx,final_beta), n)/pow(mpc,3)/1.e3*m_sun; // g/cm^3

  return rhogas;
}
/*
returns thermal gas pressure (not electron pressure) in the unit of keV/cm^3.
Note that the unit does not include hubble parameter (h).
*/
double calc_gas_pressure(double r, double R500){
    double xx, ngas, T, Pgas;
    double fnt;
    xx = r/(1000.0*ri/mpc);
    ngas = rho0*pow(theta(xx,final_beta), n)/m_p/mmw/pow(mpc,3)/1.e6*m_sun; // cm^-3
    fnt = pntfrac(r, R500);
    T = T0*theta(xx,final_beta)*(1.0 - fnt); //keV

    Pgas = ngas*T; // keV/cm^3
    return Pgas;
}
    
    
double calc_gas_num_density(double r, double R500){
    double xx, ngas;
    xx = r/(1000.0*ri/mpc);
    ngas = rho0*pow(theta(xx,final_beta), n)/m_p/mmw/pow(mpc,3)/1.e6*m_sun; // cm^-3

    return ngas;
}
double calc_gas_temperature(double r, double R500){
    double xx, T, fnt;

    xx = r/(1000.0*ri/mpc);
    fnt = pntfrac(r, R500);
    T = T0*theta(xx,final_beta)*(1.0 - fnt); //keV
    //T = T0*theta(xx,final_beta)*(1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); //keV
    if ( T < 0.0 ) T = 0.0;
    return T; 
}

double calc_Y(double R500, double Rvir, double Rmax){
    int nx = 1000;
    double res, x, w;
    gsl_integration_glfixed_table *t;

    t = gsl_integration_glfixed_table_alloc(nx);

    res = 0.0;
    for(int i=0;i<nx;i++){
        gsl_integration_glfixed_point(0.0, 1.0, i, &x, &w, t);
        res += w*4.0*M_PI*x*x*calc_gas_pressure(x*Rmax, R500);
    }
    res *= pow(Rmax*mpc*1e2, 3.0);
    gsl_integration_glfixed_table_free(t);

    return res;
}

double calc_shell_Y(double R500, double Rvir, double rm, double rp){
    int nx = 100;
    double res, x, w, r;
    gsl_integration_glfixed_table *t;

    t = gsl_integration_glfixed_table_alloc(nx);

    res = 0.0;
    for(int i=0;i<nx;i++){
        gsl_integration_glfixed_point(0.0, 1.0, i, &x, &w, t);
        r = (rp-rm)*x+rm; // in Mpc
        res += w*4.0*M_PI*r*r*calc_gas_pressure(r, R500);
    }
    res *= pow(mpc*1e2, 3.0)*(rp-rm);
    gsl_integration_glfixed_table_free(t);

    return res;
}

double calc_shell_Y_Arnaud(double R500, double Rvir, double rm, double rp, double h, double E){
    int nx = 100;
    double res, x, w, r;
    gsl_integration_glfixed_table *t;

    t = gsl_integration_glfixed_table_alloc(nx);

    res = 0.0;
    for(int i=0;i<nx;i++){
        gsl_integration_glfixed_point(0.0, 1.0, i, &x, &w, t);
        r = (rp-rm)*x+rm; // in Mpc
        res += w*4.0*M_PI*r*r*calc_pressure_Arnaud(r, R500, h, E);
    }
    res *= pow(mpc*1e2, 3.0)*(rp-rm);
    gsl_integration_glfixed_table_free(t);

    return res;
}


double thermal_pressure_outer_rad() {
    // returns outermost physical radius for thermal pressure profiles (i.e. the point where the thermal pressure
    // goes to zero (in units of rvir)
    //return pow((1.0 / delta_rel), 1.0/delta_rel_n) * R500toRvir; // last factor converts from units of R500 to units of Rvir
    //return pow((1.0 / delta_rel), 1.0/delta_rel_n);//in unit of R500
    return 5.0;
}


double calc_Mhse_Mtot_ratio(double R500) {
    double delx=0.1;
    double rplus= (R500 + delx)/(1000.0*ri/mpc);
    double rminus= (R500 - delx)/(1000.0*ri/mpc);
    double diff_tot= (theta(rplus, final_beta)- theta(rminus, final_beta))/(2.0*delx/(1000.0*ri/mpc));
    //cout<< final_beta<<endl;

    double fnt = pntfrac(R500, R500);

    return 1.0 - fnt - fnt*A_nt/diff_tot*pow(theta(R500/(1000.0*ri/mpc), final_beta), 1)/(n+1)/R500*(1000.0*ri/mpc);
}

double* calc_gas_pressure_profile(double rcutoff, double xin, double xfinal, int xbinnum, double R500, double P500) {
    double xx, delx, result, xfinal_i;
    int i;
    double Vol[250], r[250], yy[250];
    xin= xin*R500;
    xfinal_i= xfinal*R500;
    delx= log(xfinal_i/xin)/xbinnum;
    // P500=1.65e-3*pow(0.7,2);
    for (i=0;i<xbinnum;i++) {
        //xx = (double)pow(2.7183, log(xin)+i*delx)/(1000.0*ri/mpc);
        xx=i*(xfinal-xin)/xbinnum * R500/(1000.0*ri/mpc);
        //ysz[i] =  (double)mmw*p0/1e6*pow(theta(xx,final_beta),(n+1.0))/mu_e;//*xx*xx; //keV/cm^3
        ysz[i] =  p0/1e6*pow(theta(xx,final_beta),(n+1.0));//*xx*xx; //keV/cm^3
        //if ((int)pturbrad==2)
        double fnt = pntfrac(xx*(1000.0*ri/mpc), R500); 
        ysz[i] = ysz[i]*(1.0 - fnt);
        r[i]= xx*(1000.0*ri/mpc)/R500;
        Vol[i]= xx*xx;
        //cout<< i<<" "<<r[i]*0.71 <<" "<< pow(r[i]*0.71,3)*ysz[i]/P500<<endl;
        cout<< i<<" "<<r[i] <<" "<< ysz[i]/P500<<endl;
    }
    return &ysz[0];
}

double* calc_gas_temp_profile(double rcutoff, double xin, double xfinal, int xbinnum, double R500) {
    double xx, delx;
    int i;

    xin= xin*rcutoff*radius;
    xfinal= xfinal*rcutoff*radius;
    delx= log(xfinal/xin)/(xbinnum-1);
    for (i=0;i<xbinnum;i++) {
        xx = (double)pow(2.7183, log(xin)+i*delx)/(1000.0*ri/mpc);
        Tsz[i] =  (double)T0*theta(xx,final_beta); //keV
        //if ((int)pturbrad==2) 
        double fnt = pntfrac(xx*(1000.0*ri/mpc), R500); 
        Tsz[i] = Tsz[i]*(1.0 - fnt);
        x[i]= xx;
        //cout<< xx*(1000.0*ri/mpc)/R500 <<" " << ysz[i]<< endl;
    }
    return &Tsz[0];
}

double* calc_gas_profile(double rcutoff, double xin, double xfinal, int xbinnum, double R500) {
    double xx, delx;
    int i;
    xin= xin*rcutoff*radius;
    xfinal= xfinal*rcutoff*radius;
    delx= log(xfinal/xin)/(xbinnum-1);

    for (i=0;i<xbinnum;i++) {
        xx = (double)pow(2.7183, log(xin)+i*delx)/(1000.0*ri/mpc);
        rhogas[i] =  (double)mmw*rho0*pow(theta(xx,final_beta), n)/mu_e/m_p/pow(mpc,3)/1.e6*m_sun; //Msol/Mpc^3
        x[i]= xx;
        //cout<< xx*(1000.0*ri/mpc)/R500 <<" " << rhogas[i]<< endl;
    }
    return &rhogas[0];
}

double return_ngas(double r){
    double xx = (double)r/(1000.0*ri/mpc); // distance from center in units of scale radius
    double rho_gas = rho0*pow(theta(xx,final_beta), n); //Msol/Mpc^3
    rho_gas *= 1e-6 * m_sun/pow(mpc,3); // now in kg/cm^3
    double n_gas = rho_gas/mmw/m_p; // in cm^-3
    return n_gas;
}


double return_ngas_mod(double r, double R500, double x_break, double npoly_mod){
    double xx = (double)r/(1000.0*ri/mpc); // distance from center in units of scale radius
    double rho_gas;
    double Rs = 1000.0*ri/mpc;
    double x_break_Rs = x_break*R500/Rs; // break radius in units of scale radius
    if(r/R500>x_break){
        rho_gas = rho0*pow(theta(xx,final_beta), n); //Msol/Mpc^3
    }
    else if(r/R500<=x_break){ // here the power law breaks!
        rho_gas = rho0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),npoly_mod);
        rho_gas *= pow(theta(x_break_Rs,final_beta), n) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),npoly_mod); //normalize such that there is no discontinuity at the break point
    }
    rho_gas *= 1e-6 * m_sun/pow(mpc,3); // now in kg/cm^3
    double n_gas = rho_gas/mmw/m_p;  // ngas in cm^-3
    return n_gas;
}

double return_clumpf ( double r, double R500, double clump0, double alpha_clump, double beta_clump, double gamma_clump ) {

    double clumpf;
    double R200m = R500/R500toR200m;
    double x = r/R200m;    

    clumpf = 1. + clump0*pow(x, alpha_clump)*pow(1+pow(x,gamma_clump), (beta_clump-alpha_clump)/(gamma_clump)) ;
    
    return clumpf;
}

double returnPth(double r, double R500){
    // returns thermal pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double fnt = pntfrac(r, R500);
    double thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - fnt);

    thisP *= 1.0e-6; //convert to keV cm^-3
    return thisP;

}

double returnP_mod(double r, double R500, double x_break, double npoly_mod, double nnt_mod){
    // returns *total* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a sharp break at x_break
    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP;
    if(r/R500>x_break){
        //thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
        thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0));
    }
    else if(r/R500<=x_break){
        thisP = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
        thisP *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break
    }

    thisP *= 1.0e-6; //convert to keV cm^-3
    return thisP;
}

double returnP_mod2(double r, double R500, double x_break, double npoly_mod, double nnt_mod, double x_smooth){
    // returns *total* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a SMOOTH transtion around x_break

    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP_outside, thisP_inside, thisP;

    //thisP_outside = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
    thisP_outside = (double)p0*pow(theta(xx,final_beta),(n+1.0));

    thisP_inside = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
    thisP_inside *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break

    //thisP_inside *= (1.0 - delta_rel*pow(r/R500, nnt_mod));  // add non-th. pressure
    //thisP_inside *= (1.0 - delta_rel*pow(x_break, delta_rel_n)) / (1.0 - delta_rel*pow(x_break, nnt_mod)); // normalize the break

    double ratio = 0.5*(1.0-tanh((r/R500-x_break)/x_smooth)); // ratio=1 inside and ratio=0 outside
    thisP = ratio*thisP_inside + (1-ratio)*thisP_outside;
    //cout<<r/R500<<" "<<ratio<<" "<<1-ratio<<endl;

    thisP *= 1.0e-6; //convert to keV cm^-3
    return thisP;
}

double returnPth_mod(double r, double R500, double x_break, double npoly_mod) {
    // returns *thermal* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a sharp break at x_break
    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP;
    double fnt = pntfrac(r, R500);
    if(r/R500>x_break){
        //thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
        thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0));
    }
    else if(r/R500<=x_break){
        thisP = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
        thisP *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break

    }

    thisP *= (1.0 - fnt);
    thisP *= 1.0e-6; //convert to keV cm^-3

    return thisP;
}


double returnPth_mod2(double r, double R500, double x_break, double npoly_mod, double x_smooth){
    // returns *thermal* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a SMOOTH transtion around x_break

    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP_outside, thisP_inside, thisP;

    double fnt = pntfrac(r, R500);

    //thisP_outside = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
    thisP_outside = (double)p0*pow(theta(xx,final_beta),(n+1.0));
    thisP_inside = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
    thisP_inside *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break

    //thisP_inside *= (1.0 - delta_rel*pow(r/R500, nnt_mod)) * mmw/mu_e;  // add non-th. pressure
    //thisP_inside *= (1.0 - delta_rel*pow(x_break, delta_rel_n)) / (1.0 - delta_rel*pow(x_break, nnt_mod)); // normalize the break

    double ratio = 0.5*(1.0-tanh((r/R500-x_break)/x_smooth)); // ratio=1 inside and ratio=0 outside
    thisP = ratio*thisP_inside + (1-ratio)*thisP_outside;
    //cout<<r/R500<<" "<<ratio<<" "<<1-ratio<<endl;
    thisP *= (1.0 - fnt);
    thisP *= 1.0e-6; //convert to keV cm^-3

    return thisP;

}

double returnT_mod(double r, double R500, double x_break, double npoly_mod ){
    // returns temperature at distance r/Mpc, in keV
    // --- this modification features a sharp break at x_break
    double thisT, thisP, ngas;

    thisP = returnPth_mod(r, R500, x_break, npoly_mod); //in keV/cm^3 
    ngas = return_ngas_mod(r, R500, x_break, npoly_mod); // in cm^-3
    thisT = thisP/ngas;
    return thisT;

}

double returnT_mod2(double r, double R500, double x_break, double npoly_mod, double x_smooth){
    // returns temperature at distance r/Mpc, in keV
    // --- this modification features a SMOOTH transtion around x_break
    double thisT, thisP, ngas;

    thisP = returnPth_mod2(r, R500, x_break, npoly_mod, x_smooth); //in keV/cm^3 
    ngas = return_ngas_mod(r, R500, x_break, npoly_mod); // in cm^-3
    thisT = thisP/ngas;
    return thisT;
}

double return_xray_emissivity(double ngas, double T, double redshift){
    double emis;
    double nH, ne;

    //xx = r/(1000.0*ri/mpc); //r in mpc, xx is r/rs
    //ngas = rho0*pow(theta(xx,final_beta), n)/m_p/mmw/pow(mpc,3)/1.e6*m_sun; // cm^-3
    //T = T0*theta(xx,final_beta)*(1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); //keV
    //ngas = return_ngas_mod(R500, r, x_break, npoly_mod);
    //T = return_T_mod(R500, r, x_break, npoly_mod);

    if ( T > 0.0) {
        emis = interpolate_emission_table (T, redshift); // in ergs cm^3 /s
        //emis = interpolate_emission_table (10.0, 0.01); // in ergs cm^3 /s
        ne = ngas * mmw / mu_e;
        nH = ne / 1.2;
        emis *= ne * nH; // in ergs /cm^3 /s
    } else {
        emis =  0.0;
    }

    return emis; 
}


double return_entropy_mod(double r, double R500, double x_break, double npoly_mod, double nnt_mod){
    // below x_break (in units of R500) switch to a model with modified polytropic index and modified n_nt
    // return gas entropy at distance r/Mpc from center, in units keV*cm^2
    // to do: R500 should be global, protected variable in this class!
    /* double convert= m_sun/pow(mpc,3.)/m_p*1.0e-6;
    double xx = (double)r/(1000.0*ri/mpc); //xx is r/Rs -- don't blame me for the poor notation, this is all based on Suman's code
    double GE = (double)T0*theta(xx,final_beta)/pow(mmw*rho0*pow(theta(xx,final_beta), n)/mu_e*convert,2./3.); // gas entropy
    GE * (1.0 - delta_rel*pow(r/R500, delta_rel_n)); // take into account non-thermal pressure
    double EE = GE * pow(mu_e/mmw,2.0/3.0); // electron entropy in keV*cm^2 */

    double xx = (double)r/(1000.0*ri/mpc);
    double rho0_alt = rho0*m_sun/pow(mpc,3); // rho0 in units kg/m^3
    double Rs = 1000.0*ri/mpc;
    double x_break_Rs = x_break*R500/Rs; // break radius in units of scale radius
    double gas_entropy;
    double fnt = pntfrac(r, R500);

    if(r/R500>x_break){
        gas_entropy = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta(xx,final_beta),1.0-(2.0/3.0)*n);
        //gas_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n));
        gas_entropy *= (1.0 - fnt);
    }
    else if(r/R500<=x_break){ // here the power law breaks

        gas_entropy = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod);
        gas_entropy *= pow(theta(x_break_Rs,final_beta),1.0-(2.0/3.0)*n) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod); // normalize at the break point
        //gas_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, nnt_mod));
        //gas_entropy *= (1.0 - delta_rel*pow(x_break,delta_rel_n))/(1.0 - delta_rel*pow(x_break, nnt_mod));
        gas_entropy *= (1.0 - fnt);
    }
    gas_entropy *= 1.e4; // units keV*cm^2
    //double electron_entropy = pow(mu_e/mmw,2.0/3.0) * gas_entropy; // units keV*cm^2
    //electron_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); // take into account non-th. pressure
    return gas_entropy;
}


double return_entropy(double r, double R500){ // new function by SF
    // return gas entropy at distance r/Mpc from center, in units keV*cm^2
    // to do: R500 should be global, protected variable in this class!
    /* double convert= m_sun/pow(mpc,3.)/m_p*1.0e-6;
    double xx = (double)r/(1000.0*ri/mpc); //xx is r/Rs -- don't blame me for the poor notation, this is all based on Suman's code
    double GE = (double)T0*theta(xx,final_beta)/pow(mmw*rho0*pow(theta(xx,final_beta), n)/mu_e*convert,2./3.); // gas entropy
    GE * (1.0 - delta_rel*pow(r/R500, delta_rel_n)); // take into account non-thermal pressure
    double EE = GE * pow(mu_e/mmw,2.0/3.0); // electron entropy in keV*cm^2 */

    double xx = (double)r/(1000.0*ri/mpc);
    double rho0_alt = rho0*m_sun/pow(mpc,3); // rho0 in units kg/m^3
    double gas_entropy = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta(xx,final_beta),1.0-(2.0/3.0)*n);
    gas_entropy *= 1e4; // units keV*cm^2
    //double electron_entropy = pow(mu_e/mmw,2.0/3.0) * gas_entropy; // units keV*cm^2
    //electron_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); // take into account non-th. pressure
    return gas_entropy;
}


double return_entropy_mod2(double r, double R500, double x_break, double npoly_mod, double nnt_mod, double x_smooth){
    // below x_break (in units of R500) switch to a model with modified polytropic index and modified n_nt
    //
    // return gas entropy at distance r/Mpc from center, in units keV*cm^2
    // to do: R500 should be global, protected variable in this class!
    /* double convert= m_sun/pow(mpc,3.)/m_p*1.0e-6;
    double xx = (double)r/(1000.0*ri/mpc); //xx is r/Rs -- don't blame me for the poor notation, this is all based on Suman's code
    double GE = (double)T0*theta(xx,final_beta)/pow(mmw*rho0*pow(theta(xx,final_beta), n)/mu_e*convert,2./3.); // gas entropy
    GE * (1.0 - delta_rel*pow(r/R500, delta_rel_n)); // take into account non-thermal pressure
    double EE = GE * pow(mu_e/mmw,2.0/3.0); // electron entropy in keV*cm^2 */

    double xx = (double)r/(1000.0*ri/mpc);
    double rho0_alt = rho0*m_sun/pow(mpc,3); // rho0 in units kg/m^3
    double Rs = 1000.0*ri/mpc;
    double x_break_Rs = x_break*R500/Rs; // break radius in units of scale radius
    double gas_entropy_inside, gas_entropy_outside, gas_entropy;

    double fnt = pntfrac(r, R500);
    gas_entropy_outside = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta(xx,final_beta),1.0-(2.0/3.0)*n);
    //gas_entropy_outside *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n));
    gas_entropy_outside *= (1.0 - fnt);

    gas_entropy_inside = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod);
    gas_entropy_inside *= pow(theta(x_break_Rs,final_beta),1.0-(2.0/3.0)*n) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod); // normalize at the break point
    //gas_entropy_inside *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, nnt_mod));
    //gas_entropy_inside *= (1.0 - delta_rel*pow(x_break,delta_rel_n))/(1.0 - delta_rel*pow(x_break, nnt_mod));
    gas_entropy_inside *= (1.0 - fnt);

    double ratio = 0.5*(1.0-tanh((r/R500-x_break)/x_smooth)); // ratio=1 inside and ratio=0 outside
    gas_entropy = ratio*gas_entropy_inside + (1-ratio)*gas_entropy_outside;

    gas_entropy *= 1e4; // units keV*cm^2
    //double electron_entropy = pow(mu_e/mmw,2.0/3.0) * gas_entropy; // units keV*m^2
    //electron_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); // take into account non-th. pressure
    return gas_entropy;
}

// -- original code by Suman - I think the following is wrong:
double* calc_gas_entropy_profile(double rcutoff, double xin, double xfinal, int xbinnum, double R500) {
    double xx, delx;
    int i;
    double convert= m_sun/pow(mpc,3.)/m_p*1.0e-6;
    //cout<< "beta="<< xin<< " "<<xfinal<<endl;
    xin= xin*rcutoff*radius;
    xfinal= xfinal*rcutoff*radius;
    delx= log(xfinal/xin)/(xbinnum-1);
    //cout<< xin<<" "<<xfinal<< R500<<" "<<rcutoff<<" "<<radius<<" "<<1000.0*ri/mpc<<endl;
    for (i=0;i<xbinnum;i++) {
        xx = (double)pow(2.7183, log(xin)+i*delx)/(1000.0*ri/mpc);
        Ksz[i] =  (double)T0*theta(xx,final_beta)/pow(mmw*rho0*pow(theta(xx,final_beta), n)/mu_e*convert,2./3.); //keV.cm^2
        //ysz[i]= mmw*p0/1e6*pow(theta(xx,final_beta),(n+1.0))/mu_e/pow(mmw*rho0*pow(theta(xx,final_beta), n)/mu_e*convert,5./3.);
        //if ((int)pturbrad==2) 
        double fnt = pntfrac(xx*(1000.0*ri/mpc), R500);
        Ksz[i] = Ksz[i]*(1.0 - fnt);
        x[i]= xx;
        //cout<< xx*(1000.0*ri/mpc)/R500 <<" " << ysz[i]<< endl;
    }
    return &Ksz[0];
}

double* calc_x(double rcutoff, double xin, double xfinal, int xbinnum) {
    double xx, delx;
    int i;
    //xin= xin*rcutoff*radius;
    //xfinal= xfinal*rcutoff*radius;
    delx= log(xfinal/xin)/(xbinnum-1);
    for (i=0;i<xbinnum;i++) {
        xx = pow(2.7183, log(xin)+i*delx);//(1000.0*ri/mpc);
        rr[i]= xx;
    }
    return &rr[0];
}

double* calc_ellspace_sz_profile(double rcutoff, double ang_diam_z, double redshift, double R500, bool verbose) {
    double units = mpc*sigma_T/me_csq;
    // note rcutoff is in units of Rvir
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error, upperlim = rcutoff*C, elli = ang_diam_z/(1000.0*ri/mpc);
    double params[9] = {A_nt, n, C, final_beta, 0.0, elli, B_nt, gamma_nt, R500/(1000.0*ri/mpc)};
    gsl_function F;
    int i;
    //if ((int)pturbrad==2) 
    upperlim = min(upperlim, thermal_pressure_outer_rad() * C);
    F.function = &yfft_func;
    F.params = &params;
    for (i=0;i<nell;i++) {
        params[4] = ell[i];
        gsl_integration_qags (&F, 0.0, upperlim, 0, 1e-7, 10000, w, &result, &error);
        result = result*4.0*M_PI*(1000.0*ri/mpc)/elli/elli;
        fft_ysz[i] = (double)mmw*units*result*p0/mu_e;
    }
    gsl_integration_workspace_free (w);
    //check_unbound(redshift, verbose);
    return &fft_ysz[0];
}

// added by SF:
double* calc_ellspace_kappa_profile(double *kappa_ell, double redshift, double chi, double Rvir, double Omega_M0, double h){
    // chi is the comoving distance to redshift z
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double rhocrit_comov = 2.7754e11*h*h; // Msol/Mpc^3
    double rhoM_comov = Omega_M0 * rhocrit_comov;
    double params[5] = {0, chi, ri*1000/mpc, rhoi, Rvir};
    double result, error;
    gsl_function F;
    F.function = &kappa_integrant;
    F.params = &params;
    //double r_test[6]={0.0,0.2,0.4,0.6,0.8,1.0};
    //cout<<chi<<" "<<ri*1000/mpc<<" "<<rhoi<<endl;
    //for(int i=0; i<6; i++){
        //cout<<kappa_integrant(r_test[i],params)<<" ";
    //}
    //cout<<endl;
    for (int i=0;i<nell;i++) {
        params[0] = ell[i];
        gsl_integration_qags (&F, 0, 1, 0, 1e-9, 10000, w, &result, &error);
        kappa_ell[i] = result/(rhoM_comov*chi*chi); // see Ma et al 2015 Eq.3.2
    }
    gsl_integration_workspace_free (w);
    return 0;
}

double* calc_ellspace_sz_profile_spline(double rcutoff, double ang_diam_z, double redshift, double R500, bool verbose) {
    double units = mpc*sigma_T/me_csq;
    // note rcutoff is in units of Rvir
    double result, error, upperlim = rcutoff*C, elli = ang_diam_z/(1000.0*ri/mpc);
    int nxbins = 100, i, j;
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, nxbins);
    double *xx, *yy;
    xx = new double [nxbins];
    yy = new double [nxbins];
    // now make absolutely sure thermal pressure never goes negative
    //if ((int)pturbrad==2) 
    upperlim = min(upperlim, thermal_pressure_outer_rad() * C);

    for (j=0;j<nxbins;j++) xx[j] = (double)j * upperlim / ((double)(nxbins-1));
    yy[0] = 0.0;
    for (i=0;i<nell;i++) {
        for (j=1;j<nxbins;j++) {
            yy[j] = pow(theta(xx[j],final_beta),n+1.0)*xx[j]*xx[j]*sin(ell[i]*xx[j]/elli)/(ell[i]*xx[j]/elli);
            //if ((int)pturbrad==2) {
            double fnt = pntfrac(xx[j]*(1000.0*ri/mpc), R500);
            yy[j] = yy[j]*(1.0 - fnt);
                //if (yy[j]<0.0) yy[j] = 0.0;
            //}
        }
        gsl_spline_init (spline, xx, yy, nxbins);
        result = gsl_spline_eval_integ (spline, xx[0], xx[nxbins-1], acc);
        result = result*4.0*M_PI*(1000.0*ri/mpc)/elli/elli;
        fft_ysz[i] = (double)mmw*units*result*p0/mu_e;
    }
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    delete[] xx;
    delete[] yy;
    return &fft_ysz[0];
}

double* calc_ellspace_nfw_profile_spline(double ang_diam_z, double redshift, double Rvir) {

    double Si_result, Ci_result, error, elli = ang_diam_z;
    int nxbins = 1000, i, j;
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, nxbins);
    double *xx, *Si, *Ci;
    xx = new double [nxbins];
    Si = new double [nxbins];
    Ci= new double [nxbins];
    double xmin=1e-3;
    double xmax=1000;
    double xbin= log(xmax/xmin)/(nxbins-1);
    // now make absolutely sure thermal pressure never goes negative

    for(j=0; j< nxbins; j++){
        xx[j]=xmin*exp(xbin*j);
        Si[j]= sin(xx[j])/xx[j];
        Ci[j]= cos(xx[j])/xx[j];
        // cout<<xx[j]<<" "<<Si[j]<<" "<<Ci[j]<<endl;
    }
    for (i=0;i<nell;i++) {
        double upperlimit=ell[i]/elli*Rvir/C*(1+C);
        double lowerlimit= ell[i]/elli*Rvir/C;
        // cout<<upperlimit<<" "<<lowerlimit<<endl;
        gsl_spline_init (spline, xx, Si, nxbins);
        Si_result = gsl_spline_eval_integ (spline, lowerlimit,upperlimit, acc);
        gsl_spline_init (spline, xx, Si, nxbins);
        Ci_result= gsl_spline_eval_integ (spline, lowerlimit,upperlimit, acc);

        fft_ysz[i] = (double) 1.0/(log(1+C)-C/(1+C))*(sin(lowerlimit)*Si_result-sin(C*lowerlimit)/upperlimit+cos(lowerlimit)*Ci_result);
        //cout<<ell[i]<<" "<<fft_ysz[i]<<endl;
    }
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    delete[] xx; delete[] Si; delete[] Ci;
    return &fft_ysz[0];
}


double* calc_kspace_sz_profile(double rcutoff, double ang_diam_z, double redshift, double R500, bool verbose) {
    double units = mpc*sigma_T/me_csq;
    // note rcutoff is in units of Rvir
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error, upperlim = rcutoff*C, elli = ang_diam_z/(1000.0*ri/mpc);
    double pt = 0.0;
    //pt= pturbrad;
    double params[9] = {A_nt, n, C, final_beta, 0.0, elli, B_nt, gamma_nt, R500/(1000.0*ri/mpc)};
    gsl_function F;
    int i;
    //if ((int)pturbrad==2) 
    upperlim = min(upperlim, thermal_pressure_outer_rad() * C);
    F.function = &yfft_func_k;
    F.params = &params;
    for (i=0;i<nrads;i++) {
        params[4] = k[i];
        gsl_integration_qags (&F, 0.0, upperlim, 0, 1e-7, 10000, w, &result, &error);
        result = result/2.0/M_PI/M_PI; //*4.0*M_PI*(1000.0*ri/mpc)/elli/elli;
        ysz[i] = (double)result; //mmw*units*result*p0/mu_e;
    }
    gsl_integration_workspace_free (w);
    //check_unbound(redshift, verbose);
    return &ysz[0];
}

double return_P500_arnaud10(double M500, double Efact){
    // this is the electron-P500, as defined in Arnaud2010
    double P500 = 1.65e-3 * pow(Efact,8.0/3.0) * pow(M500/3e14,2.0/3.0); // in units keV/cm^3
    P500 *= 1e6; // units keV/m^3
    return P500;
}

void check_unbound(double z, bool verbose) {
    // sets fft signal = 0 if model appears to result in unbound clusters
    int i;
    if ((final_Cf/C - 1.0) > 0.8) {
        if (verbose) cout << "Cluster, M = " << mass << " Msol, redshift = " << z << ", is unbound" << endl;
        for (i=0;i<nell;i++) {
            fft_ysz[i] = 0.0;
        }
    }
}

void calc_2d_sz_profile(double rcutoff, double R500, double *r, double* ysz) {
    double units = mpc*sigma_T/me_csq;
    // note rcutoff is in units of Rvir
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error, upperlim = rcutoff*C;
    //pt= (double)pturbrad;
    double params[8] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/(1000.0*ri/mpc)};
    gsl_function F;
    int i;
    F.function = &yproj_func;
    F.params = &params;

    for (i=0;i<nrads;i++) {
        xx = (double)(x[i]/(1000.0*ri/mpc));
        r[i]= xx*(1000.0*ri/mpc);//R500;
        if (xx <= upperlim) {
            params[4] = xx;
            gsl_integration_qags (&F, xx, upperlim, 0, 1e-7, 10000, w, &result, &error);

            ysz[i] = (double)units*result*p0*2.0*(1000.0*ri/mpc)/mu_e;
            //---- This gives plausible output, but I still don't understand why it's not:
            //---- ysz[i] = (double)units*result*p0*2.0*mmw/mu_e;
            //---- I think there's a factor of mmw missing
        }
        else ysz[i] = 0.0;
        // cout<<ysz[i]<<" "<<p0<<endl;
    }

    gsl_integration_workspace_free (w);
    //return &ysz[0];
}

// NEW (June 2015): return ySZ at a distance r (in Mpc) from center
// I added this function so that I can calulate the ySZ signal at only one point (pixel)
double return_ysz(double rcutoff, double R500, double this_r){
    double units = mpc*sigma_T/me_csq; // this has units Mpc*m/keV
    // the tSZ normalization (see my notes on basecamp)
    // the factor of mpc is there I think because the length element in the integral is in units of m, not Mpc.
    // note rcutoff is in units of R_Delta, where Delta is the chosen overdensity (virial, 200 or 500)
    // this_r is the proper distance to the halo center in units of Mpc

    double NFW_Rs_Mpc = (1000.0*ri/mpc); // ri has units km throughout the code; NFW_Rs_Mpc is the (proper) NFW scale radius in Mpc.

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error;
    double upperlim = rcutoff*C; // upperlim is the cutoff radius in units of the NFW scale radius
    //pt= (double)pturbrad;
    double params[8] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/NFW_Rs_Mpc};
    gsl_function F;
    int i;
    F.function = &yproj_func;
    F.params = &params;

    // at this point ri is the NFW scale radius in km. So 1000*ri/mpc is the NFW scale radius in Mpc...
    xx = this_r/NFW_Rs_Mpc; // xx = distance to center in units of NFW scale radius.
    double this_ysz=0;

    if (xx <= upperlim){ // xx<=upperlim is equivalent to (distance to center) < 3*
        params[4] = xx;
        gsl_integration_qags (&F, xx, upperlim, 0, 1e-6, 10000, w, &result, &error); // 1e7 original value
        this_ysz = units * result * p0*2.0*NFW_Rs_Mpc * mmw/mu_e;
        // I added a factor of mmw which I think was missing!
        // unit check: dropping the dim'less units, we have
        // [ysz] = [ units * p0 * NFW_Rs_Mpc] = m^3/keV/Mpc * keV/m^3 * Mpc = 1. ---good!

    }
    else this_ysz = 0.0;

    gsl_integration_workspace_free (w);
    return this_ysz;
}

//added by SF (August 2015) -- a kSZ function
double return_ksz(double vlos, double rcutoff, double R500, double this_r){
    // this function return b=-(Delta_T/T_CMB)_kSZ at location this_r
    // specify rcutoff in units of R_Delta - this is the integration limit
    // I should probably fo to 3 times R_Delta
    // proper R500 in units of Mpc
    // this_R is the proper distance to the center in Mpc
    // for converting the actual temperature use
    // Delta_T_kSZ = -b*T_CMB

    double NFW_Rs_Mpc = (1000.0*ri/mpc); // (proper) NFW scale radius in Mpc.

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error;
    double upperlim = rcutoff*C; // the upper integration limit in units of the NFW scale radius
    double pt = 0.0;
    //pt= (double)pturbrad;

    xx = (double)(this_r/NFW_Rs_Mpc); // xx = distance to center in units of NFW scale radius.
    double params[8] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/NFW_Rs_Mpc};
    double this_ksz=0;

    gsl_function F;
    F.function = &gasproj_func;
    F.params = &params;

    if (xx <= upperlim) { // xx<=upperlim is equivalent to (distance to center) < 3*
        params[4] = xx;
        //cout<<"ksz "<<upperlim<<endl;
        gsl_integration_qags (&F, xx, upperlim, 0, 1e-6, 10000, w, &result, &error); // 1e7 original value
        double integrated_gas_density = 2.0*NFW_Rs_Mpc*rho0*(double)result; // in units of Msol/Mpc^2

        this_ksz = (vlos/clight) * (sigma_T/pow(mpc,2)) * 1.0/(mu_e*m_p) * (integrated_gas_density*m_sun);

        // dimension check: these 5 factors have dimensions:
        // 1 * 1 * Mpc^2 * kg^(-1) * kg/Mpc^2 = 1 ---good!
        // the factor 1/(mu_e*m_p) is what I call zeta in my notes, in units of kg!
        // sign convention:
        // here there's no minus sign, but introduce the minus sign later:
        // a cluster with vlos>0 is moving away from the observer
        // this cluster gives a negative ksz
        // vlos>0 --> Delta_T<0 and vice versa

    }
    else this_ksz = 0.0;

    gsl_integration_workspace_free (w);
    return this_ksz;
}


//added by SF (September 2015) -- returns the proj.NFW profile at this_r from center in Mpc
double return_pnfw(double rcutoff, double this_r){

    double NFW_Rs_Mpc = (1000.0*ri/mpc); // (proper) NFW scale radius in Mpc.

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error;
    double upperlim = rcutoff*C; // the upper integration limit in units of the NFW scale radius

    xx = (double)(this_r/NFW_Rs_Mpc); // xx = distance to center in units of NFW scale radius.
    double params[1] = {xx};

    gsl_function F;
    F.function = &NFWproj_func;
    F.params = &params;

    double this_pnfw=0;

    if (xx <= upperlim) { // xx<=upperlim is equivalent to (distance to center) < 3*

        gsl_integration_qags (&F, xx, upperlim, 0, 1e-6, 10000, w, &result, &error); // 1e7 original value
        this_pnfw = 2.0*NFW_Rs_Mpc*rhoi*(double)result; // in units of Msol/Mpc^2
    }
    else this_pnfw = 0.0;

    gsl_integration_workspace_free (w);
    return this_pnfw;
}

double return_ksz_nfw(double vlos, double rcutoff, double this_r){
    // as return_ksz(), but under the assumption that baryons trace the DM NFW profile

    double NFW_Rs_Mpc = (1000.0*ri/mpc); // (proper) NFW scale radius in Mpc.

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error;
    double upperlim = rcutoff*C; // the upper integration limit in units of the NFW scale radius

    xx = (double)(this_r/NFW_Rs_Mpc); // xx = distance to center in units of NFW scale radius.
    double params[1] = {xx};
    double this_ksz=0;

    gsl_function F;
    F.function = &gasproj_func;
    F.params = &params;

    //cout<<"***"<<NFW_Rs_Mpc<<" "<<rhoi<<" "<<rho0<<endl;
    if (xx <= upperlim) { // xx<=upperlim is equivalent to (distance to center) < 3*

        gsl_integration_qags (&F, xx, upperlim, 0, 1e-6, 10000, w, &result, &error); // 1e7 original value
        double integrated_gas_density = 2.0*NFW_Rs_Mpc*rhoi*(double)result; // in units of Msol/Mpc^2
        this_ksz = (vlos/clight) * (sigma_T/pow(mpc,2)) * 1.0/(mu_e*m_p) * (integrated_gas_density*m_sun);

    }
    else this_ksz = 0.0;

    gsl_integration_workspace_free (w);
    return this_ksz;
}


// added by SF February 2016:
double return_tau0(double rcutoff, double R500){ // returns the central optical depth
    double NFW_Rs_Mpc = (1000.0*ri/mpc); // (proper) NFW scale radius in Mpc.
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error;
    double upperlim = rcutoff*C; // the upper integration limit in units of the NFW scale radius
    double pt = 0.0;
    //pt= (double)pturbrad;

    xx = 0.0; // xx = distance to center in units of NFW scale radius.
    double params[8] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/NFW_Rs_Mpc};

    gsl_function F;
    F.function = &gasproj_func;
    F.params = &params;

    params[4] = xx;
    gsl_integration_qags (&F, xx, upperlim, 0, 1e-6, 10000, w, &result, &error); // 1e7 original value
    double integrated_gas_density = 2.0*NFW_Rs_Mpc*rho0*(double)result; // in units of Msol/Mpc^2

    double tau0 = (sigma_T/pow(mpc,2)) * 1.0/(mu_e*m_p) * (integrated_gas_density*m_sun);

    gsl_integration_workspace_free (w);

    return tau0;
}

// added by SF May 2016:
double return_tau_profile(double rcutoff, double R500, double r){
    // returns the optical depth at distance r (in Mpc) from the center
    double NFW_Rs_Mpc = (1000.0*ri/mpc); // (proper) NFW scale radius in Mpc.
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error;
    double upperlim = rcutoff*C; // the upper integration limit in units of the NFW scale radius
    //double pt = 0.0;
    //pt= (double)pturbrad;

    xx = r/NFW_Rs_Mpc; // xx = distance to center in units of NFW scale radius.
    double params[8] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/NFW_Rs_Mpc};
    gsl_function F;
    F.function = &gasproj_func;
    F.params = &params;

    params[4] = xx;
    gsl_integration_qags (&F, xx, upperlim, 0, 1e-6, 10000, w, &result, &error); // integrating from xx to upperlim
    double integrated_gas_density = 2.0*NFW_Rs_Mpc*rho0*(double)result; // in units of Msol/Mpc^2

    double tau0 = (sigma_T/pow(mpc,2)) * 1.0/(mu_e*m_p) * (integrated_gas_density*m_sun);

    gsl_integration_workspace_free (w);

    return tau0;
}

double return_tau_profile_mod(double rcutoff, double R500, double r, double x_break, double npoly_mod){
    // returns the optical depth at distance r (in Mpc) from the center
    // -- this function uses a broken power-law

    double NFW_Rs_Mpc = (1000.0*ri/mpc); // (proper) NFW scale radius in Mpc.
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error;
    double upperlim = rcutoff*C; // the upper integration limit in units of the NFW scale radius
    //double pt = pturbrad;

    xx = r/NFW_Rs_Mpc; // xx = distance to center in units of NFW scale radius.
    double params[10] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/NFW_Rs_Mpc, x_break, npoly_mod};

    gsl_function F;
    F.function = &gasproj_func_mod;
    F.params = &params;

    params[4] = xx;
    gsl_integration_qags (&F, xx, upperlim, 0, 1e-6, 10000, w, &result, &error); // 1e7 original value
    double integrated_gas_density = 2.0*NFW_Rs_Mpc*rho0*(double)result; // in units of Msol/Mpc^2

    double tau0 = (sigma_T/pow(mpc,2)) * 1.0/(mu_e*m_p) * (integrated_gas_density*m_sun);

    gsl_integration_workspace_free (w);

    return tau0;
}

void calc_2d_gas_profile(double rcutoff, double R500, double *r, double* ysz) {
    double units = 2.05e-3; //1006.2828 trac et al
    // note rcutoff is in units of Rvir= R500/Rvir
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double xx, result, error, upperlim = rcutoff*C;
    double pt = 0.0;
    //pt= (double)pturbrad;
    double params[8] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/(1000.0*ri/mpc)};
    gsl_function F;
    int i;

    F.function = &gasproj_func;
    F.params = &params;
    for (i=0;i<nrads;i++) {
        xx = (double)(x[i]/(1000.0*ri/mpc));
        r[i]= xx*(1000.0*ri/mpc);//R500;
        if (xx <= upperlim) {
    	    params[4] = xx;
    	    gsl_integration_qags (&F, xx, upperlim, 0, 1e-7, 10000, w, &result, &error);
    	    ysz[i] = (double)result*sigma_T*mmw/mu_e*m_sun*rho0*2.0*(1000.0*ri/mpc)/m_p/pow(mpc,2)/clight;//1e3;  //mmw*rho0*pow(theta(xx,final_beta), n)/mu_e/m_p/
        }
        else ysz[i] = 0.0;
        // cout<<r[i]<<" "<<ysz[i]<<" "<<rho0<<endl;
    }

    gsl_integration_workspace_free (w);
    //return &ysz[0];
}

double return_Yx500(double R500){
    // returns Yx500 (X-ray observable), using R500 in Mpc
    double Tspec_to_Tmg = 1.11;
    double units = p0 * pow(mpc,3.0) * 4.0*M_PI * m_p/m_sun * mmw * pow(R500,3) * Tspec_to_Tmg;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    //double pt = (double)pturbrad;
    double params[8] = {A_nt, n, C, final_beta, 0.0, B_nt, gamma_nt, R500/(1000.0*ri/mpc)};
    gsl_function F;
    F.function = &yint_func_sam;
    F.params = &params;
    double int_upperlim = 1.0;
    gsl_integration_qags (&F, 0.0, int_upperlim, 0, 1e-6, 10000, w, &result, &error);
    Ytot = (double)units*result;
    gsl_integration_workspace_free (w);
    return Ytot;
}

double calc_mgas500(double R500) { // R500 has to be in Mpc
    double units = rho0*4.0*M_PI*pow(1000.0*ri/mpc,3);
    double NFW_Rs_Mpc = (1000.0*ri/mpc);
    double mgas500;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double params[7] = {n, C, final_beta, f_s, A_nt, B_nt, gamma_nt};
    gsl_function F;
    int i;
    F.function = &mgas500_func;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, R500/NFW_Rs_Mpc, 0, 1e-7, 10000, w, &result, &error);
    mgas500 = (double) units*result;
    gsl_integration_workspace_free (w);
    return mgas500;
}

double calc_mgas500_mod(double R500, double x_break, double npoly_break) {
    // R500 has to be in Mpc, x_break in units of R500
    double units = rho0*4.0*M_PI*pow(1000.0*ri/mpc,3);
    double NFW_Rs_Mpc = (1000.0*ri/mpc);
    double mgas500;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double params[9] = {n, C, final_beta, f_s, A_nt, B_nt, gamma_nt, x_break*R500/NFW_Rs_Mpc, npoly_break};
    gsl_function F;
    int i;
    F.function = &mgas500_func_mod;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, R500/NFW_Rs_Mpc, 0, 1e-7, 10000, w, &result, &error);
    mgas500 = (double) units*result;
    gsl_integration_workspace_free (w);
    return mgas500;
}

double calc_mgas500_mod_clumped(double R500, double x_break, double npoly_break, double gamma_clump, double gamma_clump1, double beta_clump, double clump0) {
    // R500 has to be in Mpc, x_break in units of R500
    double units = rho0*4.0*M_PI*pow(1000.0*ri/mpc,3);
    double NFW_Rs_Mpc = (1000.0*ri/mpc);
    double mgas500;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double pt = 2.0;
    double params[13]= {n, C, final_beta, f_s, A_nt, B_nt, gamma_nt, x_break*R500/NFW_Rs_Mpc, npoly_break, clump0, alpha_clump, beta_clump, gamma_clump };
    gsl_function F;
    int i;
    F.function = &mgas500_func_mod_clumped;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, R500/NFW_Rs_Mpc, 0, 1e-7, 10000, w, &result, &error);
    mgas500 = (double) units*result;
    gsl_integration_workspace_free (w);
    return mgas500;
}

double return_fgas500(double M500, double R500){
    return calc_mgas500(R500)/M500;
}

double return_fgas500_mod(double M500, double R500, double x_break, double npoly_break){
    return calc_mgas500_mod(R500, x_break, npoly_break)/M500;
}

double return_fgas500_mod_clumped(double M500, double R500, double x_break, double npoly_break, double x_clump, double alpha_clump1, double alpha_clump2, double clump0){
    return calc_mgas500_mod_clumped(R500, x_break, npoly_break,x_clump, alpha_clump1, alpha_clump2, clump0 )/M500;
}
/*
double calc_mgas2500(double R2500) { // R500 has to be in Mpc
    double units = rho0*4.0*M_PI*pow(1000.0*ri/mpc,3);
    double NFW_Rs_Mpc = (1000.0*ri/mpc);
    double mgas2500;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double params[5] = {delta_rel, n, C, final_beta, f_s};
    gsl_function F;
    int i;
    F.function = &mgas500_func;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, R2500/NFW_Rs_Mpc, 0, 1e-7, 10000, w, &result, &error);
    mgas2500 = (double) units*result;
    gsl_integration_workspace_free (w);
    return mgas2500;
}

double calc_mgas2500_mod(double R2500,double R500, double x_break, double npoly_break) { // R500 has to be in Mpc
    double units = rho0*4.0*M_PI*pow(1000.0*ri/mpc,3);
    double NFW_Rs_Mpc = (1000.0*ri/mpc);
    double mgas2500;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    double pt = 2.0;
    double params[8] = {delta_rel, n, C, final_beta, f_s, pt, x_break*R500/NFW_Rs_Mpc, npoly_break};
    gsl_function F;
    F.function = &mgas500_func_mod;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, R2500/NFW_Rs_Mpc, 0, 1e-7, 10000, w, &result, &error);
    mgas2500 = (double) units*result;
    gsl_integration_workspace_free (w);
    return mgas2500;
}

double return_fgas2500(double M2500, double R2500){
    return calc_mgas2500(R2500)/M2500;
}
double return_fgas2500_mod(double M2500, double R2500, double R500, double x_break, double npoly_break){
    return calc_mgas2500_mod(R2500,R500,x_break,npoly_break)/M2500;
}
*/
};

// -- end of class
// functions for integrals go here


#endif
