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
#include "mod_gas_model.h"

using namespace mod_gas_model;

void set_constants() {
    PI = 4.*atan(1.);
    m_sun = 1.98892e30; //kg
    clight = 3.0e5; // in km/s
    mpc = 3.0857e22; // in m
    G = 6.67e-8*m_sun/pow(mpc,3); // in mpc^3/Msun/s^2
    //mu_e = 1.143; // SF: mean molecular weight per electron
    mu_e = 1.136; // X=0.76 assumed
    //mmw = 0.59; // mean molecular weight
    mmw = 0.58824; // X=0.76 assumed
    m_p  = 1.6726e-27;// mass proton, kg
    eV = 1.602e-19; // 1eV in J
    sigma_T = 6.652e-25/1000.0; // in m^2.
    m_e = 9.11e-31; // mass electron, kg
    //q = 1.60217646e-19;// joules
    me_csq = 511.0;  // me_c^2 in keV
}

/* SF: origin of mu_e and mmw:
for the tSZ we have DeltaT/T = sigma_T/me_csq int P_e dl
now we need to convert P_e ~ n_e to P_gas ~ n_gas.
n_gas = rho_gas/mu, where mu = 4/(3+5X_H) = 0.59 is the mean atomic weight in a fully ionized gas (Suman calls it mmw)
Now,
n_e = zeta * rho_gas, with zeta = (X_H/m_H + 2*Y_He/m_He).
The factor of 2 comes from the fact that Helium is doubly ionized!
Putting everything together, we have
n_e = zeta * mu * n_gas and thus
P_e = zeta * mu * P_gas.
With Y_He=0.2477 we have zeta=0.87 (in atomic units) and mu=0.59 (in atomic units)
Suman calls mu = mmw and zeta = 1/mu_e.
Then the tSZ normalization is
zeta*mu = mmw/mu_e = 0.5133

For the kSZ the normalization is different (integrated density instead of integrated pressure)
DeltaT/T = sigma_T/c * int n_e v_los dl
now n_e = zeta * rho_gas, as above. Then we have simply
DeltaT/T = sigma_T/c * v_los * zeta * int dl rho_gas
*/

void evolve_pturb_norm(float z, float outer_radius) { //compute alpha(z)
    float fmax, evo_power, evo_converge;
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

void set_nfw_params(float bmass, float bradius, float conc, float brhoi, float r500) { // set the NFW parameters
    mass = bmass; // Mvir [Msol]
    radius = bradius; // Rvir [Mpc]
    C = conc; // cvir_Mpc
    rhoi = brhoi; // NFW density at NFW scale radius [Msol/Mpc^3]
    ri = radius/C; // NFW scale radius [Mpc]
    //cout<< mass<<" "<<ri<<endl;
    ri = ri*mpc/1000.0; //in km (for later units)
    // SO after calling this, ri is always in units km!
    vcmax = sqrt(4.0*PI*G*rhoi*ri*ri*Gmax()); //% in km/s
    //findxs();// now can calculation radius within which stellar mass is contained
    R500toRvir= r500/radius;
}

void set_mgas_init(float baryon_frac_univ) {
    mgas = (baryon_frac_univ)*mass/(1.0+f_s);
}

void set_fs(float bfs) {
    f_s = bfs;
}

float get_fs() {
    return f_s;
}

void set_Mpiv(float bMpiv) {
    Mpiv = bMpiv;
}


void calc_fs(float M500, float baryon_frac_univ, float cosm_t0, float cosm_tz) { // compute the star fraction
    //---note M500 must be in Msol

    f_s = min(fs_0 * pow(M500/Mpiv,-1.0*fs_alpha), 0.8*baryon_frac_univ); //
    f_s = f_s / (baryon_frac_univ - f_s); // f_s is now the star formation efficiency

    //---uncomment this line for z-evolution:
    //f_s = f_s*calc_fstarz(cosm_t0, cosm_tz);

}

void set_stellar_evo_params(float inp1, float inp2, float inp3) {
    // reset stellar evolution parameters as in Nagamine et al. 2006
    Tau_b = inp1;
    Tau_d = inp2;
    bulge_frac = inp3;
}

float calc_fstarz(float cosm_t0, float cosm_tz) {
    // calculates fraction of stars formed at z = 0 that have formed by redshift z
    // Assumes Nagamine et al 06 evolution of disk and bulge populations by default
    // Use the Nagamine values for Tau_d, Tau_b and bulge_frac;
    float chi_b, chi_d, fb, fd, fstarz;

    chi_b = (1.0 - (cosm_t0/Tau_b + 1.0)*exp(-1.0*cosm_t0/Tau_b));
    chi_d = (1.0 - (cosm_t0/Tau_d + 1.0)*exp(-1.0*cosm_t0/Tau_d));
    fb = bulge_frac/chi_b*(1.0 - (cosm_tz/Tau_b + 1.0)*exp(-1.0*cosm_tz/Tau_b));
    fd = (1.0-bulge_frac)/chi_d*(1.0 - (cosm_tz/Tau_d + 1.0)*exp(-1.0*cosm_tz/Tau_d));
    fstarz = fb + fd;

    //cout<<cosm_t0<<" "<<cosm_tz<<" "<<chi_b<<" "<<chi_d<<" "<<fb<<" "<<fd<<endl;
    return fstarz;
}

// now put solvep0rho0 functions in here

double findxs() { // solve for x_s (sec 3.1)
    // start here!
    vector<float> x(2000), gg(2000);
    int i;
    float mingg;
    for (i=0;i<2000;i++) {
        x[i] = 0.0 + float(i)*(2.0*C/2000.0);
        gg[i] = fabs(g(x[i]) - g(C)*f_s/(1+f_s));
    }
    mingg =  *min_element(gg.begin(), gg.end());
    i = 0;
    if (mingg>2e-3) cout << "Convergence xs error " << mingg << endl;
    while (gg[i]!=mingg) {i++;}
    xs = x[i];
    return xs;
}

double g(double x) { // eqn 2b
    return log(1.0+x) - x/(1.0+x);
}

double Gmax() { // eqn 3b
    float xmax = 2.163; // check this number!
    return g(xmax)/xmax;
}

double f(double x) { // eqn 5
    if (x<=C) return log(1+x)/x - (1/(1+C));
    else if (x>C) return (C/x)*((log(1+C)/C) - (1/(1+C)));
    else return -1;
}

double H(double x) { // eqn 6b
    return  (1/((1+x)*g(x)))*(-1*log(1+x) + (x*(1+x/2))/(1+x));
}

double sx_func (double x) {
    double f = (x - (1.0+x)*log(1.0+x)) / (pow(x,3)*pow(1.0+x,3));
    return f;
}

double S_cx(double x) { // % eqn 7b
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (2000);
    double result, error, Sx, alpha = 0.0;
    gsl_function F;
    F.function = &sx_func;
    if (x==0) x = 1e-7; //% diverges at x = 0
    gsl_integration_qags (&F, x, C, 0, 1e-7, 2000, w, &result, &error);
    Sx = S_C(C) - (double)result;
    gsl_integration_workspace_free (w);
    return (double)Sx;
}

double S_C(double x) { // eqn 11
    double SC;
    SC = pow(PI,2)/2.0 - log(x)/2.0 - 1.0/(2.0*x) - 1.0/(2.0*pow(1.0+x,2)) - 3.0/(1+x);
    SC += (0.5 + 1.0/(2.0*pow(x,2)) - 2.0/x - 1.0/(1.0+x))*log(1.0+x);
    SC += (3.0/2.0)*log(1.0+x)*log(1.0+x);
    SC += 3.0*gsl_sf_dilog(-1.0*x); // gsl dilog function
    //SC = SC*g(x)*g(x);// this is the correction made to Ostriker et al 05.
    // (Mar 10) Don't think it should be here.
    return SC;
}

double delta_s() { // eqn 15
    return S_C(C) / (S_C(C) + H(C)*g(C)/pow(C,3.0));
}

double K() { //% eqn 16
    double Kx = (1.0/3.0)*H(C)*(1./(Gmax()*(1.0-delta_s())));
    return Kx;
}

double ss_func(double x) {
  double f = S_cx(x)*pow(x,2);
  return f;
}

double fx_func(double x) {
  double g = f(x)*x/pow(1.0+x,2);
  return g;
}


double K_s() { // % eqn 21
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double resulta, resultb, error, Ks, tempC = C;
    gsl_function F,FF;
    F.function = &ss_func;
    FF.function = &fx_func;
    F.params = &tempC;
    FF.params = &tempC;
    gsl_integration_qags (&F, 0.0e-4, xs, 0, 1e-7, 10000, w, &resulta, &error);
    gsl_integration_qags (&FF, 0.0e-4, xs, 0, 1e-7, 10000, w, &resultb, &error);
    Ks = (1.0/g(C))*(resulta - (2.0/3.0)*resultb);
    gsl_integration_workspace_free (w);
    return Ks;
}

double j(double x) { //% eqn 25b
    double jj;
    if (x==0.0) jj = 0.0;
    else if (x<=C) jj = 1.0 - log(1.0+x)/x;
    else jj = 1.0 - 1.0/(1.0+C) - (log(1.0+C) - C/(1.0+C))/x;
    return jj;
}

double theta(double x, double beta) { // % eqn 26b
    double th;
    if (pturbrad==1) {
        th = (-1.0*beta*j(x)/(n+1.0) + 1.0 + chi_turb*delta_rel)/2.0;
        th = th + 0.5*sqrt(pow(1.0 + chi_turb*delta_rel - beta*j(x)/(n+1.0),2) - 4.0*chi_turb*delta_rel);
    }
    else if (pturbrad==2) th =  (double)(1.0 - (beta*j(x)/(1.0+n)));
    else th =  (double)(1.0 - (beta*j(x)/((1.0+n)*(1.0+delta_rel))));
    return (double)fabs(th);
}

double theta_mod(double x, double beta, float x_break, float npoly_mod) {
    // ---
    // modified theta (broken power-law model)
    // x_break is here in units of the NFW scale radius
    // ---
    double th;
    if (pturbrad!=2){cout<<"ERROR! -- pturbrad should be 2 for computing theta_mod!"; return -1;}

    if (x>=x_break){
        th = (double)(1.0 - (beta*j(x)/(1.0+n)));
    }
    else if (x<x_break){
        th = (double)(1.0 - (beta*j(x)/(1.0+npoly_mod))) * (1.0 - (beta*j(x_break)/(1.0+n)))/(1.0 - (beta*j(x_break)/(1.0+npoly_mod)));
    }

    return (double)fabs(th);
}

double I2spline(double Cf, double beta) {// % eqn 27 
    int nxbins = 1000, i;
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_steffen, nxbins);
    double *xx, *ftx, result;
    xx = new double [nxbins];
    ftx = new double [nxbins];
    // first need to make an array of values
    if (Cf<0) cout << "Cf error! " << Cf << endl;
    for (i=0;i<nxbins;i++) {
        xx[i] = (double)i * Cf / ((double)(nxbins-1));
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

double I3spline(double Cf, double beta) {// % eqn 27 {
    int nxbins = 100, i;
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_steffen, nxbins);
    double *xx, *tx, result;
    xx = new double [nxbins];
    tx = new double [nxbins];
    // first need to make an array of values
    if (Cf<0) cout << "Cf error! " << Cf << endl;
    for (i=0;i<nxbins;i++) {
        xx[i] = (double)i * Cf / ((double)(nxbins-1));
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

double ttx_func(double x, void * p) {
  struct my_func_params *params = (struct my_func_params *)p;
  float n = params->b;
  float C = params->c;
  double beta = params->d;
  float delta_rel = params->a;
  int pturbrad = params->e;
  float delta_rel_n= params-> f;
  double ff = pow(theta(x, beta), n)*pow(x,2);
  return ff;
}

double L(double Cf, double beta) {// % eqn 27
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error, Sx;
    struct my_func_params params = {delta_rel, n, C, beta, pturbrad };
    gsl_function F;
    F.function = &ttx_func;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, Cf, 0, 1e-5, 10000, w, &result, &error);
    //cout << "L: " << result << endl;
    gsl_integration_workspace_free (w);
    return result;
}

double Lspline(double Cf, double beta) {// % eqn 27 {
    int nxbins = 100, i;
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_steffen, nxbins);
    double *xx, *ttl, result;
    xx = new double [nxbins];
    ttl = new double [nxbins];
    // first need to make an array of values
    if (Cf<0) cout << "Cf error! " << Cf << endl;
    for (i=0;i<nxbins;i++) {
        xx[i] = (double)i * Cf / ((double)(nxbins-1));
        ttl[i] =  pow(theta(xx[i], beta), n)*pow(xx[i],2);
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
    float gfact, Cfp, Lc, Lp;
    gfact = g(pressurebound*C) / g(C);
    if (gfact<1.0) cout << "gfact error" << endl;
    Cfp = Cf;
    Lc = Lspline(Cf,beta);
    Lp = Lc;
    // do this in three steps to speed it up
    while (Lp < gfact*Lc) {
        Cfp *= 1.2;
        Lp = Lspline(Cfp,beta);
    }
    Cfp  = Cfp / 1.2;
    while (Lp < gfact*Lc) {
        Cfp *= 1.1;
        Lp = Lspline(Cfp,beta);
    }
    Cfp  = Cfp / 1.1;
    while (Lp < gfact*Lc) {
        Cfp *= 1.02;
        Lp = Lspline(Cfp,beta);
    }
    Cfp *= 1.01/1.02; // settle on half way in between
    return Cfp;
}

double ftx_func(double x, void * p) {
  struct my_func_params * params = (struct my_func_params *)p;
  float n = params->b;
  float C = params->c;
  double beta = params->d;
  float delta_rel = params->a;
  int pturbrad = params->e;
  float delta_rel_n= params->f;
  //gas_model gmod(delta_rel, n, C, 0.0, 0.0, 0.0, pturbrad, delta_rel_n);
  double ff = f(x)*pow(theta(x, beta),n)*pow(x,2);
  return ff;
}

double I2(double Cf, double beta) {// % eqn 28a
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    struct my_func_params params = { delta_rel, n, C, beta, pturbrad };
    gsl_function F;
    F.function = &ftx_func;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, Cf, 0, 1e-5, 10000, w, &result, &error);
    gsl_integration_workspace_free (w);
    return result;
}

double tx_func(double x, void * p) {
  struct my_func_params * params = (struct my_func_params *)p;
  float n = params->b;
  float C = params->c;
  double beta = params->d;
  float delta_rel = params->a;
  int pturbrad = params->e;
  float delta_rel_n= params-> f;
  //gas_model gmod(delta_rel, n, C, 0.0, 0.0, 0.0, pturbrad, delta_rel_n);
  double ff = pow(theta(x, beta),n+1.0)*pow(x,2);
  return ff;
}


double I3(double Cf, double beta) {// % eqn 28b
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    struct my_func_params params = { delta_rel, n, C, beta, pturbrad };
    gsl_function F;
    F.function = &tx_func;
    F.params = &params;
    gsl_integration_qags (&F, 0.0, Cf, 0, 1e-5, 10000, w, &result, &error);
    //cout << "I3: " << result << endl;
    gsl_integration_workspace_free (w);
    return result;
}


double Edm(float aniso_beta) { 
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

double Ek(double x, float aniso_beta, double Winf) {
    //% calculate total kinetic energy T in NFW halo using Lokas & Mamon 01
    double K;
    if (aniso_beta==0.0) {
        K = -3.0 + 3.0/(1.0+x) - 2.0*log(1.0+x) + x*(5.0 + 3.0*log(1.0+x));
        K = K - pow(x,2)*(7.0 + 6.0*log(1.0+x));
        K = K + pow(x,3)*(PI*PI - log(C) - log(x/C) + log(1.0+x) + 3.0*pow(log(1+x),2) + 6.0* gsl_sf_dilog(-1.0*x));
        K = K*0.5;
    }
    else if (aniso_beta==0.5) {
        K = -3.0 + 3.0/(1.0+x) - 3.0*log(1.0+x);
        K = K + 6.0*x*(1.0+log(1.0+x));
        K = K - pow(x,2)*(PI*PI + 3.0*pow(log(1.0+x),2) + 6.0*gsl_sf_dilog(-1.0*x));
        K = K/3;
    }
    else {
        K = -2.0*log(1.0+x);
        K = K + x*(PI*PI/3.0 - 1.0/(1.0+x) + pow(log(1.0+x),2) + 2.0*gsl_sf_dilog(-1.0*x));
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

double energy_constraint(double beta, double Cf) { //eqn (30)
    double f, Lval;
    if (Cf<=0.0) Cf = C/10.0; // (C<0) is unphysical
    f = 1.5*(1.0+f_s)*(Gmax()*K()*(3.0-4.0*delta_s()) + K_s()) 
        - 1.0*(Gmax()*eps*f_s*pow(clight/vcmax,2)) 
        - (Gmax()*eps_dm*fabs(Edm(0.0))/(mgas*pow(vcmax,2)))
        + (1.0+f_s)*(S_C(C)/g(C))*(pow(Cf,3) - pow(C,3))/3.0;
    Lval = Lspline(Cf,beta);
    f += -1.0*I2(Cf,beta)/Lval + (1.5*(I3spline(Cf,beta))/(beta*Lval));
    if (f != f ) {
        cout << "Energy constraint failed! " << endl;
        return 100.0;
    }
    return (f);
}

double pressure_constraint(double beta, double Cf) { //eqn (31)
    double f, Cfp;
    if (beta<=0.0) beta = 0.1;
    if (Cf<=0.0) Cf = C/10.0;
    Cfp = Lvar(Cf, beta);
    f = pow((1.0+f_s)*(S_C(C*pressurebound)/g(C))*beta*Lspline(Cf,beta),(1.0/(1.0+n)));
    if (pturbrad==1) f += -1.0*pow(1.0 + delta_rel*pow(theta(Cfp,beta),-2),1.0/(1.0+n))*theta(Cfp,beta);
    else if (pturbrad==2) f += -1.0* pow(1.0,(1.0/(1.0+n)))*(1.0 - beta*j(Cfp)/((1.0+n)));
    else f += -1.0* pow(1.0+delta_rel,(1.0/(1.0+n)))*(1.0 - beta*j(Cfp)/((1.0+n)*(1.0+delta_rel)));
    if (f != f) {
        return 100.0;
        cout << "Pressure constraint failed! " << endl;
    }
    return (f);
}

int solve_gas_model(bool verbose, float tolerance) {

    /* main fitting routine */

    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex;
    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *ss, *x;
    gsl_multimin_function F;
    size_t iter = 0;
    int status;
    double size;
    float adiabat_n = n;
    //setAprime();
    //setBprime();
    double p[5] = {delta_rel, adiabat_n, C, f_s, delta_rel_n};
    F.f = gasmod_apply_bc;
    F.n = 2;
    F.params = p;

    /* Starting point */
    x = gsl_vector_alloc (2);
    gsl_vector_set (x, 0, 1.0);
    gsl_vector_set (x, 1, C/2.0);
    /* Set initial step sizes to 1 */
    ss = gsl_vector_alloc (2);
    gsl_vector_set_all (ss, 1.0);
    /* Initialize method and iterate */
    s = gsl_multimin_fminimizer_alloc (T, 2);
    gsl_multimin_fminimizer_set (s, &F, x, ss);
    do {
        iter++;
    	status = gsl_multimin_fminimizer_iterate(s);
    	if (status)
    	    break;
    	size = gsl_multimin_fminimizer_size (s);
    	status = gsl_multimin_test_size (size, tolerance);
    	if ((status == GSL_SUCCESS) & verbose) {
    	    printf ("converged to minimum at\n");
    	    printf ("%5d %10.4e %10.4e f() = %7.3f size = %.3f\n",
    		    (int)iter,
    		    gsl_vector_get (s->x, 0),
    		    gsl_vector_get (s->x, 1),
    		    s->fval, size);
    	  }
    } while (status == GSL_CONTINUE && iter < 100);

    final_beta = gsl_vector_get (s->x, 0);
    final_Cf = gsl_vector_get (s->x, 1);

    if (final_Cf < 0 ) {
        cout << "final Cf < 0! Resetting Cf" << endl; 
        final_Cf = 1.e-7;
        setp0rho0(true);
    } else {
        setp0rho0(verbose);
    }

    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free (s);
    return status;
}

void setp0rho0(bool verbose) {
    if (verbose) {
        cout << "final_Cf "   << final_Cf << endl;
        cout << "final_beta " << final_beta << endl;
    }
    rho0 = mgas / (4.0*PI*pow(ri*1000/mpc,3)*Lspline(final_Cf, final_beta)); // in Msol/Mpc^3
    p0 = rho0*vcmax*vcmax/(final_beta*Gmax()); // in Msol/Mpc^3 (km/s)^2
    T0 = (mmw*m_p*p0/rho0)*(1000.0/eV); // this is in keV

    if (verbose) {
        cout << "final model solution:" << endl;
        cout << "P0      " << "rho0       " << "TO  " << "Mgas" << endl;
        cout << p0 << " " << rho0 << " " << T0  << " " << mgas << endl;
    }

    p0 = p0*1.0e6*m_sun/(eV*1000.0*pow(mpc,3)); // now in keV/m^3

}

/* Pressure profile from Arnaud et al. (2010) */
double calc_pressure_Arnaud(double r, float r500, double h, double E){
  double M500, R500, pi, rho_cr, rho_cr0, h70, XH, b_HSE;
  double P, P_gnfw, P0, c500, alpha, beta, gamma, Mpiv, xi;

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
  gamma = 0.31;
  alpha = 1.33;
  beta = 4.13;

  //Mpiv = 3e14*0.7; //[Msun/h]
  Mpiv = 3e14; //[Msun]

  xi = r/R500;
  P_gnfw = P0/(pow(c500*xi, gamma)*pow(1.0+pow(c500*xi, alpha), (beta-gamma)/alpha));
  P = 1.65e-3*pow(E, 8./3.)*pow(M500/Mpiv, 2./3.+0.12)*P_gnfw*h70*h70;
  P = P/((2.0+2.0*XH)/(3.0+5.0*XH));//convert electron pressure to thermal pressure

  return P;
}

double calc_gas_density(double r, float R500){
  double xx, rhogas;

  xx = r/(1000.0*ri/mpc);
  rhogas = rho0*pow(theta(xx,final_beta), n)/pow(mpc,3)/1.e3*m_sun; // g/cm^3

  return rhogas;
}
/*
returns thermal gas pressure (not electron pressure) in the unit of keV/cm^3.
Note that the unit does not include hubble parameter (h).
*/
double calc_gas_pressure(double r, float R500){
    double xx, ngas, T, Pgas;

    xx = r/(1000.0*ri/mpc);
    ngas = rho0*pow(theta(xx,final_beta), n)/m_p/mmw/pow(mpc,3)/1.e6*m_sun; // cm^-3
    T = T0*theta(xx,final_beta)*(1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); //keV

    Pgas = ngas*T; // keV/cm^3
    return Pgas;
}
    
    
double calc_gas_num_density(double r, float R500){
    double xx, ngas;
    xx = r/(1000.0*ri/mpc);
    ngas = rho0*pow(theta(xx,final_beta), n)/m_p/mmw/pow(mpc,3)/1.e6*m_sun; // cm^-3

    return ngas;
}

double calc_clumped_gas_density(double r, float R500){

    double xx, rhogas;
    double xb, clump;

    xx = r/(1000.0*ri/mpc); //r in Mpc, ri in km!, xx is r/ri 
    xb = x_clump * R500; //x_clump in R500, xb in units of Mpc (same as r)

    rhogas = rho0*pow(theta(xx,final_beta), n)/pow(mpc,3)/1.e3*m_sun; // g/cm^3
    
    if ( r < x_clump ) {
        clump = 1.0 + clump0*pow (r/xb, alpha_clump1);
    } 
    else {
        clump = 1.0 + clump0*pow (r/xb, alpha_clump2);
    }
    if ( clump < 1.0 ) clump = 1.0;

    rhogas *= sqrt(clump);

    return rhogas;

}

double calc_clumped_gas_num_density(double r, float R500){
    double xx, ngas;
    xx = r/(1000.0*ri/mpc);
    ngas = rho0*pow(theta(xx,final_beta), n)/m_p/mmw/pow(mpc,3)/1.e6*m_sun; // cm^-3

    double clump;
    double xb = x_clump * R500; //x_clump in R500, xb in units of Mpc (same as r)

    if ( r < x_clump ) {
        clump = 1.0 + clump0*pow (r/xb, alpha_clump1);
    } else {
        clump = 1.0 + clump0*pow (r/xb, alpha_clump2);
    }
    if ( clump < 1.0 ) clump = 1.0;
    ngas *= sqrt(clump);

    return ngas;
}

double calc_gas_temperature(double r, float R500){
    double xx, T;

    xx = r/(1000.0*ri/mpc);
    T = T0*theta(xx,final_beta)*(1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); //keV
    if ( T < 0.0 ) T = 0.0;
    return T; 
}

double calc_Y(float R500, float Rvir, double Rmax){
    int nx = 1000;
    double res, x, w;
    gsl_integration_glfixed_table *t;

    t = gsl_integration_glfixed_table_alloc(nx);

    res = 0.0;
    for(int i=0;i<nx;i++){
        gsl_integration_glfixed_point(0.0, 1.0, i, &x, &w, t);
        res += w*4.0*PI*x*x*calc_gas_pressure(x*Rmax, R500);
    }
    res *= pow(Rmax*mpc*1e2, 3.0);
    gsl_integration_glfixed_table_free(t);

    return res;
}

double calc_shell_Y(float R500, float Rvir, double rm, double rp){
    int nx = 100;
    double res, x, w, r;
    gsl_integration_glfixed_table *t;

    t = gsl_integration_glfixed_table_alloc(nx);

    res = 0.0;
    for(int i=0;i<nx;i++){
        gsl_integration_glfixed_point(0.0, 1.0, i, &x, &w, t);
        r = (rp-rm)*x+rm; // in Mpc
        res += w*4.0*PI*r*r*calc_gas_pressure(r, R500);
    }
    res *= pow(mpc*1e2, 3.0)*(rp-rm);
    gsl_integration_glfixed_table_free(t);

    return res;
}

double calc_shell_Y_Arnaud(float R500, float Rvir, double rm, double rp, double h, double E){
    int nx = 100;
    double res, x, w, r;
    gsl_integration_glfixed_table *t;

    t = gsl_integration_glfixed_table_alloc(nx);

    res = 0.0;
    for(int i=0;i<nx;i++){
        gsl_integration_glfixed_point(0.0, 1.0, i, &x, &w, t);
        r = (rp-rm)*x+rm; // in Mpc
        res += w*4.0*PI*r*r*calc_pressure_Arnaud(r, R500, h, E);
    }
    res *= pow(mpc*1e2, 3.0)*(rp-rm);
    gsl_integration_glfixed_table_free(t);

    return res;
}


double thermal_pressure_outer_rad() {
    // returns outermost physical radius for thermal pressure profiles (i.e. the point where the thermal pressure
    // goes to zero (in units of rvir)
    //return pow((1.0 / delta_rel), 1.0/delta_rel_n) * R500toRvir; // last factor converts from units of R500 to units of Rvir
    return pow((1.0 / delta_rel), 1.0/delta_rel_n);//in unit of R500
}


double calc_Mhse_Mtot_ratio(float R500) {
    double delx=0.1;
    double rplus= (R500 + delx)/(1000.0*ri/mpc);
    double rminus= (R500 - delx)/(1000.0*ri/mpc);
    double diff_tot= (theta(rplus, final_beta)- theta(rminus, final_beta))/(2.0*delx/(1000.0*ri/mpc));
    //cout<< final_beta<<endl;

    return 1.0 - delta_rel - delta_rel*delta_rel_n/diff_tot*pow(theta(R500/(1000.0*ri/mpc), final_beta), 1)/(n+1)/R500*(1000.0*ri/mpc);
}

double return_ngas(float r){
    double xx = (double)r/(1000.0*ri/mpc); // distance from center in units of scale radius
    double rho_gas = rho0*pow(theta(xx,final_beta), n); //Msol/Mpc^3
    rho_gas *= 1e-6 * m_sun/pow(mpc,3); // now in kg/cm^3
    double n_gas = rho_gas/mmw/m_p; // in cm^-3
    return n_gas;
}


double return_ngas_mod(float r, float R500, float x_break, float npoly_mod){
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

double return_clumpf ( float r, float R500, float clump0, float x_clump, float alpha_clump1, float alpha_clump2 ) {

    double clumpf, xb;
    xb = x_clump * R500;

    if (r  < xb ) {
        clumpf = 1. + clump0*pow(r/xb, alpha_clump1) ;
    } else {
        clumpf = 1. + clump0*pow(r/xb, alpha_clump1) ;
    }
    
    return clumpf;
}

double returnPth(float r, float R500){
    // returns thermal pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));

    thisP *= 1.0e-6; //convert to keV cm^-3
    return thisP;

}

double returnP_mod(float r, float R500, float x_break, float npoly_mod, float nnt_mod){
    // returns *total* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a sharp break at x_break
    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP;
    if(r/R500>x_break){
        thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
    }
    else if(r/R500<=x_break){
        thisP = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
        thisP *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break
        thisP *= (1.0 - delta_rel*pow(r/R500, nnt_mod));  // add non-th. pressure
        thisP *= (1.0 - delta_rel*pow(x_break, delta_rel_n)) / (1.0 - delta_rel*pow(x_break, nnt_mod)); // normalize the break
    }

    thisP *= 1.0e-6; //convert to keV cm^-3
    return thisP;
}

double returnP_mod2(float r, float R500, float x_break, float npoly_mod, float nnt_mod, float x_smooth){
    // returns *total* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a SMOOTH transtion around x_break

    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP_outside, thisP_inside, thisP;

    thisP_outside = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
    thisP_inside = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
    thisP_inside *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break
    thisP_inside *= (1.0 - delta_rel*pow(r/R500, nnt_mod));  // add non-th. pressure
    thisP_inside *= (1.0 - delta_rel*pow(x_break, delta_rel_n)) / (1.0 - delta_rel*pow(x_break, nnt_mod)); // normalize the break

    float ratio = 0.5*(1.0-tanh((r/R500-x_break)/x_smooth)); // ratio=1 inside and ratio=0 outside
    thisP = ratio*thisP_inside + (1-ratio)*thisP_outside;
    //cout<<r/R500<<" "<<ratio<<" "<<1-ratio<<endl;

    thisP *= 1.0e-6; //convert to keV cm^-3
    return thisP;
}

double returnPth_mod(float r, float R500, float x_break, float npoly_mod) {
    // returns *thermal* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a sharp break at x_break
    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP;
    if(r/R500>x_break){
        thisP = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
    }
    else if(r/R500<=x_break){
        thisP = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
        thisP *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break
    }

    thisP *= 1.0e-6; //convert to keV cm^-3

    return thisP;
}


double returnPth_mod2(float r, float R500, float x_break, float npoly_mod, float x_smooth){
    // returns *thermal* pressure at distance r/Mpc, in keV/cm^3
    // for electron pressure muliply this with mmw/mu_e
    // --- this modification features a SMOOTH transtion around x_break

    double xx = (double)r/(1000.0*ri/mpc); //xx is r in units of Rs
    double x_break_Rs = x_break*R500/(1000.0*ri/mpc);
    double thisP_outside, thisP_inside, thisP;

    thisP_outside = (double)p0*pow(theta(xx,final_beta),(n+1.0)) * (1.0 - delta_rel*pow(r/R500, delta_rel_n));
    thisP_inside = (double)p0*pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0));
    thisP_inside *= pow(theta(x_break_Rs,final_beta),(n+1.0)) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),(npoly_mod+1.0)); // normalize the break
    //thisP_inside *= (1.0 - delta_rel*pow(r/R500, nnt_mod)) * mmw/mu_e;  // add non-th. pressure
    //thisP_inside *= (1.0 - delta_rel*pow(x_break, delta_rel_n)) / (1.0 - delta_rel*pow(x_break, nnt_mod)); // normalize the break

    double ratio = 0.5*(1.0-tanh((r/R500-x_break)/x_smooth)); // ratio=1 inside and ratio=0 outside
    thisP = ratio*thisP_inside + (1-ratio)*thisP_outside;
    //cout<<r/R500<<" "<<ratio<<" "<<1-ratio<<endl;
    thisP *= 1.0e-6; //convert to keV cm^-3

    return thisP;

}

double returnT_mod(float r, float R500, float x_break, float npoly_mod ){
    // returns temperature at distance r/Mpc, in keV
    // --- this modification features a sharp break at x_break
    double thisT, thisP, ngas;

    thisP = returnPth_mod(r, R500, x_break, npoly_mod); //in keV/cm^3 
    ngas = return_ngas_mod(r, R500, x_break, npoly_mod); // in cm^-3
    thisT = thisP/ngas;
    return thisT;

}

double returnT_mod2(float r, float R500, float x_break, float npoly_mod, float x_smooth){
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


double return_entropy_mod(float r, float R500, float x_break, float npoly_mod, float nnt_mod){
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

    if(r/R500>x_break){
        gas_entropy = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta(xx,final_beta),1.0-(2.0/3.0)*n);
        gas_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n));
    }
    else if(r/R500<=x_break){ // here the power law breaks

        gas_entropy = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod);
        gas_entropy *= pow(theta(x_break_Rs,final_beta),1.0-(2.0/3.0)*n) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod); // normalize at the break point
        gas_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, nnt_mod));
        gas_entropy *= (1.0 - delta_rel*pow(x_break,delta_rel_n))/(1.0 - delta_rel*pow(x_break, nnt_mod));
    }
    gas_entropy *= 1.e4; // units keV*cm^2
    //double electron_entropy = pow(mu_e/mmw,2.0/3.0) * gas_entropy; // units keV*cm^2
    //electron_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); // take into account non-th. pressure
    return gas_entropy;
}

double return_entropy(float r, float R500){ // new function by SF
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

double return_entropy_mod2(float r, float R500, float x_break, float npoly_mod, float nnt_mod, float x_smooth){
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

    gas_entropy_outside = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta(xx,final_beta),1.0-(2.0/3.0)*n);
    gas_entropy_outside *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n));

    gas_entropy_inside = pow(mmw*m_p,5.0/3.0) * p0/pow(rho0_alt,5.0/3.0) * pow(theta_mod(xx,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod);
    gas_entropy_inside *= pow(theta(x_break_Rs,final_beta),1.0-(2.0/3.0)*n) / pow(theta_mod(x_break_Rs,final_beta,x_break_Rs,npoly_mod),1.0-(2.0/3.0)*npoly_mod); // normalize at the break point
    gas_entropy_inside *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, nnt_mod));
    gas_entropy_inside *= (1.0 - delta_rel*pow(x_break,delta_rel_n))/(1.0 - delta_rel*pow(x_break, nnt_mod));

    float ratio = 0.5*(1.0-tanh((r/R500-x_break)/x_smooth)); // ratio=1 inside and ratio=0 outside
    gas_entropy = ratio*gas_entropy_inside + (1-ratio)*gas_entropy_outside;

    gas_entropy *= 1e4; // units keV*cm^2
    //double electron_entropy = pow(mu_e/mmw,2.0/3.0) * gas_entropy; // units keV*m^2
    //electron_entropy *= (1.0 - delta_rel*pow(xx*(1000.0*ri/mpc)/R500, delta_rel_n)); // take into account non-th. pressure
    return gas_entropy;
}
float return_P500_arnaud10(float M500, float Efact){
    // this is the electron-P500, as defined in Arnaud2010
    float P500 = 1.65e-3 * pow(Efact,8.0/3.0) * pow(M500/3e14,2.0/3.0); // in units keV/cm^3
    P500 *= 1e6; // units keV/m^3
    return P500;
}



double gasmod_apply_bc(const gsl_vector * v, void *p) {
  double *params = (double *) p;
  float delta_rel = (float)params[0];
  float n = (float)params[1];
  float C = (float)params[2];
  //double Aprime = params[4];
  //double Bprime = params[5];
  float f_s = (float)params[3];
  float delta_rel_n= (float) params[4];
  gas_model gmod(delta_rel, n, C, f_s, delta_rel_n);
  double x0 = gsl_vector_get (v, 0); // beta
  double x1 = gsl_vector_get (v, 1); // Cf
  if (x0<0.01) x0 = 0.01;
  //if (x0>14) x0 = 14;
  if (x1<0.01) x1 = 0.01;
  //if (x1>14) x1 = 14;
  return sqrt(pow(gmod.energy_constraint(x0, x1),2)+pow(gmod.pressure_constraint(x0, x1),2));
}
