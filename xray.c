#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "xray.h"
#include "Apec/Apec.h"


double tarray[ntmax]; // temperature bins keV
double zarray[nmmax]; // metallicity bins in Solar unit
double rarray[nrmax]; // redshift bins
double emission_table[ntmax][nrmax];
double tres, zres, eres, rres;
double keV2erg = 1.602177e-9;

void set_emission_table (int opt, double nH, char *apec_table_file, char *eff_area_file);
double interpolate_emission_table (double temp, double redshift );

int read_effarea ( char *filename, double **energy, double **effarea ) {

    FILE *input;
    int i,neff;
    double a,b;
    char* buf[1024];

    input = fopen ( filename, "r" );
    if ( input == NULL ) {
            printf("Error opening effective area file = %s!\n",filename);
            exit (1);
    }
    i = 0;
    while( (!feof(input)) ){
        fscanf(input, "%lf %lf", &a, &b);
        //printf("%e %e\n", a, b);

        (*energy) = (double *)realloc((*energy),(i+1)*sizeof(double));
        if ( (*energy) != NULL) {
            (*energy)[i] = a;
        } else {
            free(*energy);
            printf("Error reallocating memory for energy!\n");
            exit (1);
        }
        (*effarea) = (double *)realloc((*effarea),(i+1)*sizeof(double));
        if ( (*effarea) != NULL) {
            (*effarea)[i] = b;
        } else {
            free(*effarea);
            printf("Error reallocating memory for effarea!\n");
            exit (1);
        }
        
        i++;
    }
    neff = i-1;
    fclose(input);
    return neff;
}

double wabs ( double E ) {

       /* E in keV */

    double sigma;
    int i;

    double emarray[14] = {0.1, 0.284, 0.4, 0.532, 0.707, 0.867, 1.303, 1.840,
                          2.471, 3.210, 4.038, 7.111, 8.331, 10.0};

    double c0[14] = {17.3, 34.6, 78.1, 71.4, 95.5, 308.9, 120.6, 141.3,
                         202.7, 342.7, 352.2, 433.9, 629.0, 701.2};

    double c1[14] = {608.1, 267.9, 18.8, 66.8, 145.8, -380.6, 169.3,
                        146.8, 104.7, 18.7, 18.7, -2.4, 30.9, 25.2};

    double c2[14] = {-2150., -476.1 ,4.3, -51.4, -61.1, 294.0, -47.7,
                        -31.5, -17.0, 0.0, 0.0, 0.75, 0.0, 0.0};

    for (i = 0; i < 14; i++ ) {
        if (E < emarray[i]) {
            sigma=(c0[i]+c1[i]*E+c2[i]*E*E)/pow(E,3.0) * 1.e-24;
            return sigma;
        } else {
            continue;
        }
    }
    i=13;
    sigma=(c0[i]+c1[i]*E+c2[i]*E*E)/pow(E,3.0) * 1.e-24;
    return sigma;
}

void set_emission_table (int opt, double nH, char *apec_table_file, char *eff_area_file){
    /* 
     * Tabulate X-ray emission integrated in receiver's energy range of 0.5-2.0 keV 
     * for a range of temperature and abundance   
     * Original spectrum in units of photon*cm**3/s (into 4pi angle) in emitter's frame
     * Converted to ergs*cm**3/s by multiplying E in keV and keV2erg unit conversion 
     */

    //char filename[256];
    //int neff = 0;
    //double *energy = NULL;
    //double *effarea = NULL;

    double temp, metal, redshift;
    double ebin[nemax+1],spec[nemax];
    double absorption[nemax];
    double area[nemax];
    //double total_area;

    double emission, emission_erg;
    double velocity,dem,density;
    int ne, qtherm;

    int ie, it, iz, ir, i, j;

    int re_tabulate = 0;
    int eff_area = 1;

    init_apec();
    printf("Number of bins in arrays of Energy, Temperature, Redshift = %d, %d, %d\n", nemax, ntmax, nrmax);
    printf("Energy range = [%f,%f] keV\n", emin, emax);
    printf("Temperature range = [%f,%f] keV\n", tmin, tmax);
    printf("Redshift range = [%f,%f] \n", rmin, rmax);
    eres = (emax-emin)/nemax;
    printf("Energy resolution = %f [keV]\n", eres);
    for (ie = 0; ie < nemax+1; ie++) {
        ebin[ie] = emin + (double)ie*eres;
        //printf("%d %e\n", ie, ebin[ie]);
    }

    tres = (log10(tmax)-log10(tmin))/(ntmax);
    for (it = 0; it < ntmax; it++) {
        tarray[it] = pow(10.0,log10(tmin)+(double)it*tres);
        //printf("%d %e\n", it, tarray[it]);
    }

    zres = (mlmax-mlmin)/(nmmax);
    for (iz = 0; iz < nmmax; iz++) {
        zarray[iz] = pow(10.0,mlmin+(double)iz*zres);
        //printf("%e %e %e\n", zarray[iz], mlmin, pow(10.0,mlmin+(double)iz*zres));
    }

    rres = (log10(rmax)-log10(rmin))/(nrmax);
    for (ir = 0; ir < nrmax; ir++) {
        rarray[ir] = pow(10.0, log10(rmin)+(double)ir*rres);
        //printf("%d %e\n", ir, rarray[ir]);
    }
    FILE *fd;
  
    if( opt == 1 ){

        fd = fopen(apec_table_file, "rb");
        if(fd == NULL){
            fprintf(stderr, "Cannot open file %s", apec_table_file);
            re_tabulate = 1;
        }

        fprintf(stdout, "Reading tabulated emission... ");
        int ntmax_dummy, nrmax_dummy;

        fread(&ntmax_dummy, 1, sizeof(int), fd);
        fread(&nrmax_dummy, 1, sizeof(int), fd);

        if(ntmax != ntmax_dummy || nrmax != nrmax_dummy){
            fprintf(stdout, "apec table indices do not match with input file %s!\n", apec_table_file);
            fclose(fd);
            exit(1);
        }
        for (i = 0; i < ntmax; i++) {
            for (j = 0; j < nrmax; j++) {
                fread(&emission_table[i][j], 1, sizeof(double), fd);
            }
        }
        fprintf(stdout, "... done.\n");
        fclose(fd);

    }

    if ( opt == 0 || re_tabulate == 1 ) {
        fprintf(stdout, "Tabulating emission... ");
        ne = nemax;

        for (ie = 0; ie < nemax; ie++ ) {
            if ( nH <= 0.0 ) {
                absorption[ie] = 1.0;
            } else {
                double e = (0.5*(ebin[ie]+ebin[ie+1]));
                absorption[ie] = exp( - wabs(e) * nH );
                //printf("e, absorption = %e, %e\n",e, absorption[ie]);
            }
            area[ie] = 1.0;
        }

        if (eff_area == 1) {
            int neff, je;
            double *energy = NULL;
            double *effarea = NULL;
            printf("Reading effective area file: %s\n", eff_area_file);
            neff = read_effarea (eff_area_file, &energy, &effarea );
            printf("number of energy bins in %s = %d\n", eff_area_file,neff);

            if ( energy == NULL){
                printf("energy is NULL!\n");
                exit(1);
            }
            if ( effarea == NULL){
                printf("effarea is NULL!\n");
                exit(1);
            }
            for (ie = 0; ie < nemax; ie++) {
                for (je = 0; je < neff-1; je++) {  
                    //printf("%e %e %e\n", energy[je], ebin[ie], energy[je+1]);
                    if ( ebin[ie] >= energy[je] && ebin[ie] < energy[je+1] ) {
                        area[ie] = log10(effarea[je])+(log10(effarea[je+1]/effarea[je]))/log10(energy[je+1]/energy[je])*(log10(ebin[ie]/energy[je]));
                        area[ie] = pow(10.0, area[ie]);
                    }
                }
                //double e = 0.5*(ebin[ie]+ebin[ie+1]);
                //printf("e, area = %e, %e\n",e, area[ie]);
 
            }
            free(energy);
            free(effarea);
        }

        for (i = 0; i < ntmax; i++) {
            fprintf(stdout, "."); fflush(stdout);
            for (j = 0; j < nrmax; j++) {
                for (ie = 0; ie < nemax; ie++) {
                    spec[ie] = 0.0;
                }
	        temp = tarray[i];
                metal = ABUNDANCE;
                redshift = rarray[j];

                //spec from apec is in units of photons cm^3/s/bin in receiver's frame
                //ebin is already in receiver's frame
                apec ( ebin, ne, metal, temp, redshift, spec );
                emission = 0.0;
                emission_erg = 0.0;
                for (ie = 0; ie < nemax; ie++) {
                    double e, lambda;
                    e = 0.5*(ebin[ie]+ebin[ie+1]);
                    lambda = spec[ie];
                    //lambda *= keV2erg*0.5*(ebin[ie]+ebin[ie+1]);
                    lambda *= absorption[ie];
                    emission += lambda*area[ie];
                    emission_erg += lambda*e*keV2erg;
                    //printf(" %f %e %e %e\n", e, spec[ie], absorption[ie], area[ie]);
                }   
            
                //printf("%f %f %e %e %e\n", temp, redshift, emission, emission_erg, emission_erg/emission);
                emission_table[i][j] = emission;
                assert(emission >= 0);
            }
        }


        fd = fopen(apec_table_file, "wb");
        if(fd == NULL){
            fprintf(stderr, "Cannot create file %s", apec_table_file);
        }

        int ntmax_dummy = ntmax;
        int nrmax_dummy = nrmax;

        fwrite(&ntmax_dummy, 1, sizeof(int), fd);
        fwrite(&nrmax_dummy, 1, sizeof(int), fd);
        for (i = 0; i < ntmax; i++) {
            for (j = 0; j < nrmax; j++) {
                fwrite(&emission_table[i][j], 1, sizeof(double), fd);
            }
        }
        fclose(fd);
        fprintf(stdout,"done!\n");
    }
    
    return;
}

double interpolate_emission_table (double temp, double redshift ) {

    int it,iz,ir;
    double lt1, lt2, lr1, lr2, lz1, lz2, f11, f12, f21, f22;
    double q1, q2;
    double emiss = 0.0;

    double ltemp, lredshift;

    ltemp = log10(temp);
    lredshift = log10(redshift);
    //lmetal = log10(metal);

    it = (int)((ltemp-log10(tmin))/tres);
    //iz = (int)((lmetal-mlmin)/zres);
    ir = (int)((lredshift-log10(rmin))/rres);


    //if (iz < 0 ) {
	    //iz = 0;
    //}

    if (ir < 0 ) {
	ir = 0;
    }

    if (it < 0 ) {
        it = 0;
    }

    if (ir > nrmax -1) ir = nrmax - 1;
    if (it > ntmax -1) it = ntmax - 1;

    if ( it >= 0 && it < ntmax-1  && ir >= 0 && ir < nrmax-1 ) { 

	    //printf("%d %e %e %e\n",it,tarray[it],temp,tarray[it+1]);
	    //printf("%d %e %e %e\n",iz,zarray[iz],metal,zarray[iz+1]);

	    lt1 = log10(tarray[it]);
	    lt2 = log10(tarray[it+1]);
	    lr1 = log10(rarray[ir]);
	    lr2 = log10(rarray[ir+1]);
	
	    // bilinear interpolation
        //
        
        f11 = log10(emission_table[it][ir]);
	    f12 = log10(emission_table[it][ir+1]);
	    f21 = log10(emission_table[it+1][ir]);
	    f22 = log10(emission_table[it+1][ir+1]);

	    emiss = f11/((lt2-lt1)*(lr2-lr1))*((+lt2-ltemp)*(+lr2-lredshift))
	        + f21/((lt2-lt1)*(lr2-lr1))*((-lt1+ltemp)*(+lr2-lredshift))
		    + f12/((lt2-lt1)*(lr2-lr1))*((+lt2-ltemp)*(-lr1+lredshift))
        	+ f22/((lt2-lt1)*(lr2-lr1))*((-lt1+ltemp)*(-lr1+lredshift));

	    emiss = pow(10.0,emiss);

    } else if (it == ntmax-1 && ir >= 0 && ir < nrmax-1) {

	    lr1 = log10(rarray[ir]);
	    lr2 = log10(rarray[ir+1]);

	    f11 = log10(emission_table[it][ir]);
	    f12 = log10(emission_table[it][ir+1]);
	
	    emiss = f11 + (f12-f11)/(lr2-lr1)*(lredshift-lr1); 
	    emiss = pow(10.0,emiss);

    } else if (it >= 0 && it < ntmax-1 && ir == nrmax-1) {

	    lt1 = log10(tarray[it]);
	    lt2 = log10(tarray[it+1]);

	    f11 = log10(emission_table[it][ir]);
	    f21 = log10(emission_table[it+1][ir]);
	
	    emiss = f11 + (f21-f11)/(lt2-lt1)*(ltemp-lt1); 
	    emiss = pow(10.0,emiss);

    } else if ( it == ntmax-1 && ir == nrmax-1 ) {

	    emiss = (emission_table[it][ir]);
	    //emiss = pow(10.0,emiss);
    }

    //printf("%d %d %e %e %e %e\n", it, ir, tarray[it], rarray[ir], temp, redshift);
    //printf("%d %d %e %e %e\n", it, ir, emission_table[it][ir], emission_table[it+1][ir+1], emiss);
    //assert(emiss >= 0 && emiss < 1e-3);
    if (emiss != emiss) emiss = 0.0;

    return emiss;
}


