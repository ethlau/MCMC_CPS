
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xx_power, yy_power
import time

param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16, 'alpha_clump2':17}

param_label_dict = {'eps_f':r'$10^6\epsilon_f$', 'eps_DM':r'$\epsilon_{DM}$', 'f_star':r'$f_\star$', 'S_star':r'$S_\star$', 'A_C':r'$A_C$','alpha_nt':r'$\alpha_{nt}$', 'n_nt':r'$n_{nt}$', 'beta_nt':r'$\beta_{nt}$', 'gamma_mod0':r'$\Gamma_0$', 'gamma_mod_zslope':r'$\beta_\Gamma$', 'n_nt_mod':'$n_{nt,mod}$', 'clump0':r'$C_0$', 'clump_zslope':r'$\beta_C$','x_clump':r'$x_{C}$', 'alpha_clump1':r'$\alpha_{C1}$', 'alpha_clump2':r'$\alpha_{C2}$'}

param_lim_dict = {'eps_f':[3.5,4.5], 'eps_DM':1, 'f_star':[0.0250,0.0270], 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':[-0.1,0.3], 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':[0.50,0.80], 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16, 'alpha_clump2':17}


theta0 = [4.00, 3.0e-5,0.045000,0.120000,1.000000,0.180000,0.800000,0.500000,0.100000,1.720000,0.195000,0.010000,0.800000,0.670000,0.730000,1.230000,0.880000, 3.85000]

priors = { 'eps_f':2.41, 'f_star': 0.0015, 'S_star': 0.05, 'Gamma0':0.05, 'gamma_mod_zslope':0.52 }
'''
configuration    S_lim (erg/s/cm2)  P_SN

eRASS:1           4.4155700e-14      1.4075238e-22
eRASS:8           1.1188700e-14      4.3472044e-23
eRASS:8 (poles)   2.9249900e-15      8.3618594e-24
'''

#surveys = ['eRASS1', 'eRASS8', 'eRASS8poles','CMB-S4']
#surveys = ['ROSAT','eRASS1', 'eRASS8','CMB-S4']
surveys = ['Chandra', 'ROSAT']
survey_fsky = {'eRASS1':0.5, 'eRASS8':0.5, 'eRASS8poles':0.0034, 'ROSAT':0.25, 'CMB-S4':0.5, 'Chandra':0.2507}
survey_slim = {'eRASS1':1.1e-13, 'eRASS8':3.4e-14, 'eRASS8poles':1e-14, 'ROSAT':3e-14, 'Chandra':3e-14}
survey_psn = {'eRASS1':1.4075238e-22, 'eRASS8':4.3472044e-23, 'eRASS8poles':8.3618594e-24, 'ROSAT':3.e-21, 'CMB-S4':1.e-6, 'ROSAT':3.5e-22, 'Chandra':3e-21}
survey_color = {'eRASS1':'C1', 'eRASS8':'C2', 'eRASS8poles':'C3', 'Chandra':'C5', 'ROSAT':'C0'}

survey_fwhm = {'eRASS1':0.5, 'eRASS8':0.5, 'eRASS8poles':0.5, 'ROSAT':12.0, 'CMB-S4':1.0, 'Chandra':0.20}
# in arcmin

def beam (ell, fwhm=0.5) :

    #convert fwhm from arcmin to radian
    fwhm *= (np.pi/180.0)/60.0
    sigma = fwhm / (np.sqrt(8.0*np.log(2.0)))
    bl = np.exp(ell*(ell+1.0) * sigma**2)

    return bl

def cl_to_clmuK2 ( power ) :

    return ( power * Tcmb**2.0 * 1.e6 )

def clmuK2_to_cl ( power ) :

    return ( power / (Tcmb**2.0 * 1.e6) )

def xxpower (ell, theta, survey = None) :

    eps_f = theta[0]
    eps_DM = theta[1]
    f_star = theta[2]
    S_star = theta[3]
    A_C = theta[4]

    alpha0 = theta[5]
    n_nt = theta[6]
    beta = theta[7]

    gamma_mod0 = theta[8]
    gamma_mod_zslope = theta[9]
    x_break = theta[10]
    x_smooth = theta[11]
    n_nt_mod = theta[12]

    clump0 = theta[13]
    clump_zslope = theta[14]
    x_clump = theta[15]
    alpha_clump1 = theta[16]
    alpha_clump2 = theta[17]

    xx_power.set_Flender_params(alpha0, n_nt, beta, 1e-6*eps_f, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    cl = xx_power.return_xx_power(ell) # [erg cm^-2 s^-1 str^-1]^2

    if survey == None :
        psn = 0
    else :
        psn = beam(ell,fwhm=survey_fwhm[survey]) * survey_psn[survey]
        
    var = (2./((2.*ell+1)*survey_fsky[survey]))*(cl)**2

    return cl, var

def yypower (ell, theta, survey = None) :

    eps_f = theta[0]
    eps_DM = theta[1]
    f_star = theta[2]
    S_star = theta[3]
    A_C = theta[4]

    alpha0 = theta[5]
    n_nt = theta[6]
    beta = theta[7]

    gamma_mod0 = theta[8]
    gamma_mod_zslope = theta[9]
    x_break = theta[10]
    x_smooth = theta[11]
    n_nt_mod = theta[12]

    clump0 = theta[13]
    clump_zslope = theta[14]
    x_clump = theta[15]
    alpha_clump1 = theta[16]
    alpha_clump2 = theta[17]

    yy_power.set_Flender_params(alpha0, n_nt, beta, 1e-6*eps_f, eps_DM, f_star, S_star, A_C, gamma_mod0, gamma_mod_zslope, x_break, x_smooth, n_nt_mod, clump0, clump_zslope, x_clump, alpha_clump1, alpha_clump2)

    cl = yy_power.return_yy_power(ell) # [erg cm^-2 s^-1 str^-1]^2

    if survey == None :
        psn = 0
    else :
        psn = beam(ell,fwhm=survey_fwhm[survey]) * survey_psn[survey]
        
    var = (2./((2.*ell+1)*survey_fsky[survey]))*(cl)**2

    return cl, var

def xxfisher (ell, params, survey, delta = 1.e-7) :
    
    f = np.zeros([len(params), len(params)], dtype=np.float64)
    grad = np.zeros([len(params), len(ell)], dtype=np.float64)
    cl_0, var_0 = xxpower(ell, theta0, survey=survey)

    for ind, p in enumerate(params) :

        param_ind = param_ind_dict[p]
        param_value = theta0[param_ind]
        theta_less = theta0.copy()
        theta_more = theta0.copy()

        theta_less[param_ind] = (1.0 - delta)*param_value
        theta_more[param_ind] = (1.0 + delta)*param_value

        h = 2.0*delta*param_value

        cl_less, var_less = xxpower(ell, theta_less, survey=survey)
        cl_more, var_more = xxpower(ell, theta_more, survey=survey)

        diff = (cl_more-cl_less)/h

        grad[ind,:] = diff

    for ind1, p1 in enumerate(params) :
        for ind2, p2 in enumerate(params) :
            fij = 0.0
            for il, l in enumerate(ell) :
                fij += (1.0/var_0[il] * grad[ind1,il]*grad[ind2,il])
            if fij == 0.0 :
                fij = 1e-70
 
            #fij = np.sum(1.0/var_0 * grad[ind1,:]*grad[ind2,:])
            f[ind1, ind2] = fij
    
    return f

def yyfisher (ell, params, survey, delta = 1.e-7) :
    
    f = np.zeros([len(params), len(params)], dtype=np.float64)
    grad = np.zeros([len(params), len(ell)], dtype=np.float64)
    cl_0, var_0 = yypower(ell, theta0, survey=survey)

    for ind, p in enumerate(params) :

        param_ind = param_ind_dict[p]
        param_value = theta0[param_ind]
        theta_less = theta0.copy()
        theta_more = theta0.copy()

        theta_less[param_ind] = (1.0 - delta)*param_value
        theta_more[param_ind] = (1.0 + delta)*param_value

        h = 2.0*delta*param_value

        cl_less, var_less = yypower(ell, theta_less, survey=survey)
        cl_more, var_more = yypower(ell, theta_more, survey=survey)

        diff = (cl_more-cl_less)/h

        grad[ind,:] = diff

    for ind1, p1 in enumerate(params) :
        for ind2, p2 in enumerate(params) :
            fij = 0.0
            for il, l in enumerate(ell) :
                fij += (1.0/var_0[il] * grad[ind1,il]*grad[ind2,il])
            if fij == 0.0 :
                fij = 1e-70
            f[ind1, ind2] = fij
    
    return f

def extract_submat (mat, i1, i2) :
    
    subm = np.zeros([2,2])
    subm[0,0] = mat[i1,i1]
    subm[0,1] = mat[i1,i2]
    subm[1,0] = mat[i2,i1]
    subm[1,1] = mat[i2,i2]
    
    return subm

def covariance_mat ( fisher_mat, ind1, ind2 ) :

    if ind1 != ind2 :
        sub_f = fisher_mat[np.ix_([ind1,ind2],[ind1,ind2])]
    else :
        sub_f = fisher_mat[ind1,ind1]*np.identity(2)
    if np.linalg.det(sub_f) == 0 :
        cov = None
    else :
        cov = np.linalg.inv(sub_f)
    return sub_f, cov

def error_ellipse ( cov_mat ) :

    sx2 = cov_mat[0,0]
    sy2 = cov_mat[1,1]
    sxy = cov_mat[0,1]

    a2 = 0.5*(sx2+sy2) + np.sqrt(0.25*(sx2-sy2)**2 + sxy**2)
    b2 = 0.5*(sx2+sy2) - np.sqrt(0.25*(sx2-sy2)**2 + sxy**2)

    tan2theta = 2.0*sxy/(sx2-sy2)

    angle = 0.5*math.atan2(2.0*sxy,(sx2-sy2))*180.0/math.pi

    return np.sqrt(a2), np.sqrt(b2), angle

def plot_error_ellipse ( ax, cov, p1, p2, survey, nstd=2 ) :

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    from matplotlib.patches import Ellipse
    param_ind1 = param_ind_dict[p1]
    param_ind2 = param_ind_dict[p2]
    pos = [theta0[param_ind1], theta0[param_ind2]]
    pv = np.array(pos)

    vals, vecs = eigsorted(cov)
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    a,b = 2.0 * nstd * np.sqrt(vals)

    ellipse = Ellipse(xy=pos, width=a, height=b, angle=angle, edgecolor=survey_color[survey], facecolor='None', lw=1, label=survey) 
    ax.add_patch(ellipse)

    return ax

def read_data (filename) :
    ell = []
    cl = []
    var = []
    with open(filename,'r') as f:
        f.readline()
        for line in f:
            cols = line.split(' ')
            ell.append(float(cols[0]))
            cl.append(float(cols[1]))
            var.append(float(cols[2]))

    ell = np.array(ell)
    cl = np.array(cl)
    var = np.array(var)

    return ell, cl, var

def main ():

    # set cosmology and linear power spectrum
    H0=70.0
    Omega_M=0.279000
    Omega_b=0.046100
    w0=-1.000000
    Omega_k=0.000000
    n_s=0.972000
    inputPk="../input_pk/wmap9_fid_matterpower_z0.dat"
    nH = 2.4e+21

    xx_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk, 1)
    yy_power.init_cosmology(H0, Omega_M, Omega_b, w0, Omega_k, n_s, nH, inputPk)


    #param_ind_dict = {'eps_f':0, 'eps_DM':1, 'f_star':2, 'S_star':3, 'A_C':4, 'alpha_nt':5, 'n_nt':6, 'beta_nt':7, 'gamma_mod0':8, 'gamma_mod_zslope':9, 'x_break':10, 'x_smooth':11, 'n_nt_mod':12, 'clump0':13, 'clump_zslope':14, 'x_clump':15, 'alpha_clump1':16, 'alpha_clump2':17}

    #params = [ 'eps_f', 'f_star', 'S_star', 'A_C', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'n_nt_mod', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]
    #params = [ 'eps_f', 'f_star', 'clump0' ]

    #params = [ 'eps_f', 'f_star', 'S_star', 'alpha_nt', 'n_nt', 'beta_nt', 'gamma_mod0', 'gamma_mod_zslope', 'clump0', 'clump_zslope', 'x_clump', 'alpha_clump1', 'alpha_clump2' ]
    params = [ 'eps_f', 'f_star', 'clump0' ]
 
    fish = {}
    covar = {}
    for survey in surveys :
        if 'ROSAT' in survey:
            ell = np.linspace(10,700)
        elif 'Chandra' in survey:        
            ell = np.linspace(10,10000)

        fish[survey] = xxfisher (ell, params, survey)
  
        covar[survey] = np.linalg.inv(fish[survey])

        #print(covar[survey])
    '''
    for i, p in enumerate(params) :
        if p in priors.keys() :
            fish[survey] += 1.0/priors[p]**2.0
    '''
    for i1, p1 in enumerate(params) :
        for i2, p2 in enumerate(params) :
    
            f = plt.figure( figsize=(5,5) )
            ax = f.add_axes([0.20,0.18,0.72,0.72])

            if i1 != i2 :
                for survey in surveys :
                    cov = extract_submat(covar[survey],i1,i2)
                    ax = plot_error_ellipse( ax, cov, p1, p2, survey )
                    print(survey, p1, p2, cov)                              
                # recompute the ax.dataLim
                ax.relim()
                # update ax.viewLim using the new dataLim
                ax.autoscale_view()
  
                ax.set_xlabel(param_label_dict[p1])
                ax.set_ylabel(param_label_dict[p2])
                #ax.set_xlim(param_lim_dict[p1])
                #ax.set_ylim(param_lim_dict[p2])

                #ax.set_xlabel(param_label_dict[p1]+'/'+param_label_dict[p1]+r'$({\rm fid)}$')
                #ax.set_ylabel(param_label_dict[p2]+'/'+param_label_dict[p2]+r'$({\rm fid})$')
                ax.legend(loc='best')
                f.savefig("cov_"+p1+"_"+p2+".png")


            else :
                continue

if __name__ == "__main__" :
    main()
