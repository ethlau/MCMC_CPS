#ifndef _XRAY_
#define _XRAY_

/* Array lengths for lambda tables */
#define nemax       100
#define emin	    0.5
#define emax	    2.0
#define ntmax	    100
#define tmin	    0.1
#define tmax	    20.0
#define nmmax	    10
#define mlmin	    -6.00
#define mlmax	    2.0
#define rmin        1.e-6
#define rmax        3.0
#define nrmax       100

// ICM metallicity in units of Solar
#define ABUNDANCE       0.2 

extern double emission_table[ntmax][nrmax];

#ifdef __cplusplus
extern "C" void set_emission_table (int opt, double nH,  char *apec_table_file, char *eff_area_file);
extern "C" double interpolate_emission_table (double temp, double redshift);
#endif

#endif
