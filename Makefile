CC = /usr/bin/gcc
CXX= /usr/bin/g++
FC = /usr/bin/gfortran 

#CC = gcc
#CXX= g++
#FC = gfortran 

#CC = icc
#CXX= icpc
#FC = ifort

#CC=/export/home/ethlau/anaconda3/bin/x86_64-conda_cos6-linux-gnu-gcc
#CXX=/export/home/ethlau/anaconda3/bin/x86_64-conda_cos6-linux-gnu-g++
#FC=/export/home/ethlau/anaconda3/bin/x86_64-conda_cos6-linux-gnu-gfortran
CFLAGS = -fPIC -Wl,--no-undefined  
CXXFLAGS = -std=c++11 -DgFortran -fPIC -Wl,--no-undefined 
FC += -fPIC

CFLAGS += -O2 -g
CXXFLAGS += -O2 -g
#CXXFLAGS += -DMPI_PARALLEL
#CXXFLAGS += -DOUTPUT_NFW_DENSITY
#CXXFLAGS += -DLONGIDS
#CXXFLAGS += -DEXCLUDE_SUBHALOS
#CXXFLAGS += -DMASS_SELECTION=1e14
#CXXFLAGS += -DROCKSTAR_CONCENTRATION
#FLINE = -D'LINE_FITS_FILE="/home/user/work/theory/atomdb_v3.0.9/apec_line.fits"'
#FCOCO = -D'COCO_FITS_FILE="/home/user/work/theory/atomdb_v3.0.9/apec_coco.fits"'
#FLINE = -D'LINE_FITS_FILE="/home/fas/nagai/etl28/programs/Xrays/atomdb/atomdb_v3.0.9/apec_line.fits"'
#FCOCO = -D'COCO_FITS_FILE="/home/fas/nagai/etl28/programs/Xrays/atomdb/atomdb_v3.0.9/apec_coco.fits"'
FLINE = -D'LINE_FITS_FILE="/home/ethlau/programs/atomdb_v3.0.9/apec_line.fits"'
FCOCO = -D'COCO_FITS_FILE="/home/ethlau/programs/atomdb_v3.0.9/apec_coco.fits"'

PYENV=/home/ethlau/anaconda3
#PYENV=/home/fas/nagai/etl28/programs/yt-conda
CFLAGS += -I${PYENV}/include 
CXXFLAGS += -I${PYENV}/include -I./DK15
CXXFLAGS += -Wall
CXXFLAGS += -I${PYENV}/include/python3.6m -I${PYENV}/include -shared
CLIBS = -L${PYENV}/lib/ -lfftw3 -lcfitsio -lgsl -lgslcblas -lm -lboost_python36 -lboost_numpy36 -lpython3.6m
CLIBS += -lgfortran

#CFLAGS += -I/home/user/gsl-2.4/include -I/home/user/local/include
#CXXFLAGS += -I/home/user/gsl-2.4/include -I/home/user/local/include -I./DK15
#CXXFLAGS += -Wall -Wunused-variable
#CXXFLAGS += -I${PYENV}/include/python3.6m -I${PYENV}/include -shared
#CLIBS = -L/home/user/gsl-2.4/lib -lgsl -lgslcblas -L/home/user/local/lib -lm -lcfitsio
#CLIBS += -lgfortran
#CLIBS += -L${PYENV}/lib -lboost_python3 -lboost_numpy3 -lpython3.6m

APEC_SRCS = Apec.c atomdb_make_spectrum.c calc_continuum.c calc_lines.c messages.c readapec.c read_continuum_data.c read_fits_spectrum.c read_line_data.c read_parameters.c gaussianLine.c
APEC_OBJS = $(patsubst %.c,Apec/%.o,$(APEC_SRCS))

DK15_SRCS = halo_info.cpp nrutil.cpp polint.cpp qromb.cpp spline.cpp trapzd.cpp
DK15_OBJS = $(patsubst %.cpp,DK15/%.o,$(DK15_SRCS))

DK15/%.o: DK15/%.cpp DK15/%.h
	$(CXX) -fPIC -O2 -I. -I./DK15 -c $< -o $@

Apec/%.o: Apec/%.c Apec/%.h
	$(CC) $(CFLAGS) $(FCOCO) $(FLINE) -I. -I./Apec -c $< -o $@

objects= fftlog.o cdgamma.o drfftb.o drfftf.o drffti.o

xray.o: xray.c xray.h
	$(CC) $(CFLAGS) -c xray.c

gas_model.o: gas_model.cpp gas_model.h
	$(CXX) $(CXXFLAGS) -I. -c gas_model.cpp

xx_power: xx_power.cpp $(objects) gas_model.o xray.o $(APEC_OBJS) $(DK15_OBJS)
	$(CXX) $(CXXFLAGS) -o xx_power.so xx_power.cpp $(objects) gas_model.o xray.o $(APEC_OBJS) $(DK15_OBJS) $(CLIBS)

yy_power: yy_power.cpp $(objects) gas_model.o xray.o $(APEC_OBJS) $(DK15_OBJS)
	$(CXX) $(CXXFLAGS) -o yy_power.so yy_power.cpp $(objects) gas_model.o xray.o $(APEC_OBJS) $(DK15_OBJS) $(CLIBS)

clean:
	/bin/rm -f *.o DK15/*.o Apec/*.o *.so *~
