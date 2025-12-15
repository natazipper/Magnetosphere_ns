#import libraries
#import psrchive
import numpy as np
import math
import optparse as op
import argparse as arg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize
from scipy.interpolate import splrep, sproot, splev
import os
#import sys
#from subprocess import Popen, PIPE

'''
p=op.OptionParser()

p.add_option('--RMlow',default=-100,type='float', help='The lowest probed RM')
p.add_option('--RMhigh',default=100,type='float', help='The highest probed RM')
p.add_option('--RMstep',default=0.1,type='float', help='The RM step')
p.add_option('--arch','-d', type='string', help='The name of archive')
p.add_option('--phi1','-l',type='float', help='The left edge of the phase of the pulse')
p.add_option('--phi2','-r',type='float', help='The right edge of the phase of the pulse')
p.add_option('--ston','-s',default=0,type='float', help='The minimum of averaged over pulse StoN of channels, the channels with StoN lower than given number are cut')
p.add_option('--init_width_factor',default=30.,type='float', help='The initial guess about the width of RM peak in the RM spectrum. RM_width=init_width_factor*theoretical error')
p.add_option('--overI', default='True', help='Use Q/I and U/I or Q and U')
p.add_option('--remove_baseline', default="True", help='Remove the baseline in Q and U. Usually it is necessary if polarisation calibration has not been applied or polarisation calibration is not perfect.')
p.add_option('--bootstrap',default=0.,type='float', help='Try bootsrapping for investigation of RM distribution. Takes some time')
p.add_option('--plot', default="False", help='Output plot')
p.add_option('--outfile', default="False", help='Output files with likelihood and spectrum')
ops,args=p.parse_args()

#Here we download the file and calculate the basics for the RM synthesis:q, u, q_err, u_err


#load archive
base=os.path.splitext(ops.arch)[0]
nameobs=ops.arch

ar = psrchive.Archive_load(ops.arch)

mjd_st=ar.get_Integration(0).get_start_time().in_days()
mjd_end=ar.get_Integration(0).get_end_time().in_days()
mjd_middle=mjd_st+(mjd_end-mjd_st)/2.
tele=ar.get_Integration(0).get_telescope()
doppler = np.sqrt(ar.get_Integration(0).get_doppler_factor())


#check if data is dedispersed
#if dedispersed, keep it as it is,
#otherwise, dedispersed
if ar.get_dedispersed()==0:
   print("Archive was not dedispersed. Dedispersing.")
   ar.dedisperse()

#remove baseline
ar.remove_baseline()

#convert stokes
#add statement abiut if stokes converted or no
if ar.get_state()=="Coherence":
   ar.convert_state('Stokes')
ar.set_rotation_measure(0)
#ar.defaraday()

#load data
data = ar.get_data()
Nbin=ar.get_nbin()
Nch=ar.get_nchan()
weights_zap = ar.get_weights()
freq_non_zap=np.asarray(np.nonzero(weights_zap[0]))[0]


#freq_lo = (ar.get_centre_frequency() - ar.get_bandwidth()/2.0)
#freq_hi = (ar.get_centre_frequency() + ar.get_bandwidth()/2.0)
#freq_list=[freq_lo+(freq_hi-freq_lo)*i/Nch for i in range(Nch)]
#freq_list=np.asarray(freq_list)[freq_non_zap]
freq_list=np.asarray([ar.get_Integration(0).get_Profile(0, ii).get_centre_frequency() for ii in range(Nch)])
freq_list=freq_list[freq_non_zap]*doppler

pipe = Popen("psrstat -j FTDp " + nameobs + "|grep on:start", shell=True, stdout=PIPE).stdout
output_start = pipe.read()
pipe = Popen("psrstat -j FTDp " + nameobs + "|grep on:end", shell=True, stdout=PIPE).stdout
output_end = pipe.read()

phi2=np.array(output_end.split()[-1].split(",")).astype(int)
phi1=np.array(output_start.split()[-1].split(",")).astype(int)

if ops.phi1:
	phi1=[int(ops.phi1*Nbin)]
if ops.phi2:
	phi2=[int(ops.phi2*Nbin)]

index_on=np.hstack([np.arange(phi1[i], phi2[i], 1) for i in range(len(phi1))])
print(index_on/np.float(Nbin))

#if ops.phi1<0.1:
#	print("ERROR:Please shift the pulsar pulse to the centre.")
#	sys.exit(0)
print("Preferably the pulse should be centered")
if ops.phi1>ops.phi2:
	print("ERROR:Oh-oh, something went wrong! Your left phase limit is larger than the right phase limit (for chossing the on pulse region)")
	sys.exit(0)
if ops.RMlow>ops.RMhigh:
	print("ERROR:Oh-oh, something went wrong! Your low RM edge is higher than high RM edge")
	sys.exit(0)
#phi1_off=int((ops.phi1-0.1)*Nbin)
#phi2_off=int((ops.phi2+0.1)*Nbin)

if ops.plot=="True":
	print(".....................Starting Classical RM Synthesis and Bayesian GLSP RM determination..............................")
	print(" ")
	print("..................................................with plots..........................................................")

if ops.plot=="False":
	print(".....................Starting Classical RM Synthesis and Bayesian GLSP RM determination..............................")
	print("..............................................without plotting........................................................ ")
	print("")
print("Warning: Preferably the pulse should be centered")
#stokes on pulse

I_on=np.transpose(np.array(data[0,0,:,index_on][:,freq_non_zap]))
Q_on=np.transpose(np.array(data[0,1,:,index_on][:,freq_non_zap]))
U_on=np.transpose(np.array(data[0,2,:,index_on][:,freq_non_zap]))
V_on=np.transpose(np.array(data[0,3,:,index_on][:,freq_non_zap]))

#choose the 0.1*1024 bin, which has minimum standard deviation
I_raw=[np.sum(np.array(data[0,0,:,i])) for i in range(len(data[0,0,1,:]))]
I_wind=np.argmin([np.std(I_raw[int(i*0.1*Nbin):int((i+1)*0.1*Nbin)]) for i in range(9)])
#choose the off-pulse region
phi1_off=int(I_wind*0.1*Nbin)
phi2_off=int((I_wind+1)*0.1*Nbin)
I_off=np.array(data[0,0,freq_non_zap,phi1_off:phi2_off])
Q_off=np.array(data[0,1,freq_non_zap,phi1_off:phi2_off])
U_off=np.array(data[0,2,freq_non_zap,phi1_off:phi2_off])
V_off=np.array(data[0,3,freq_non_zap,phi1_off:phi2_off])


#plt.imshow(np.array(data[0,0,freq_non_zap,:]), cmap="hot")
#plt.xlabel('Pulse phase')
#plt.ylabel('Frequency (MHz)')
#plt.show()


#parameters of profile
I_std=[np.std(I_off[i]) for i in range(len(I_off))]
Q_std=[np.std(Q_off[i]) for i in range(len(Q_off))]
U_std=[np.std(U_off[i]) for i in range(len(U_off))]
V_std=[np.std(V_off[i]) for i in range(len(V_off))]


#sum over on pulse
I_sum_on=sum(np.transpose(I_on))
I_sum_on50=sum(np.transpose(np.sort(I_on, axis=1)[:, int(-0.5*len(np.transpose(I_on))):]))
Q_sum_on=sum(np.transpose(Q_on))
U_sum_on=sum(np.transpose(U_on))
V_sum_on=sum(np.transpose(V_on))

#choose high s/n, avoid channels with negative I and I=0
I_sum_on_clean=[]
Q_sum_on_clean=[]
U_sum_on_clean=[]
I_std_clean=[]
Q_std_clean=[]
U_std_clean=[]
freq_list_clean=[]
clean_bin=[]
for i in range(len(I_sum_on)):
	if I_std[i]!=0:
		if I_sum_on[i]/np.shape(I_on)[-1]/I_std[i]>ops.ston and I_sum_on[i]>0.:
				I_sum_on_clean=np.append(I_sum_on_clean, I_sum_on[i]/len(index_on))
				Q_sum_on_clean=np.append(Q_sum_on_clean, Q_sum_on[i]/len(index_on))
				U_sum_on_clean=np.append(U_sum_on_clean, U_sum_on[i]/len(index_on))
				I_std_clean=np.append(I_std_clean, I_std[i]/math.sqrt(len(index_on)))
				Q_std_clean=np.append(Q_std_clean, Q_std[i]/math.sqrt(len(index_on)))
				U_std_clean=np.append(U_std_clean, U_std[i]/math.sqrt(len(index_on)))
				freq_list_clean=np.append(freq_list_clean, freq_list[i])
				clean_bin=np.append(clean_bin, i)	
#del I_std, Q_std, U_std, V_std, I_sum_on, Q_sum_on, U_sum_on, V_sum_on
I_sum_on=I_sum_on_clean
Q_sum_on=Q_sum_on_clean
U_sum_on=U_sum_on_clean
L_sum_on=np.sqrt(np.asarray(Q_sum_on)**2+np.asarray(U_sum_on)**2)
I_std=I_std_clean
Q_std=Q_std_clean
U_std=U_std_clean
freq_list=freq_list_clean
if len(freq_list)<=2:
	print("ERROR! Not enough valid frequency channels. Try to use lower StoN limit.")
	sys.exit(0)
del I_std_clean, Q_std_clean, U_std_clean, V_std, I_sum_on_clean, Q_sum_on_clean, U_sum_on_clean, V_sum_on, freq_list_clean

#constracting the dataset with I, errI, q, errq, u, erru, v, errv
if ops.overI=='True':
	q=np.divide(Q_sum_on, I_sum_on)
#        q=np.divide(Q_sum_on, L_sum_on)
	u=np.divide(U_sum_on, I_sum_on)
#        u=np.divide(U_sum_on, L_sum_on)
	q_err=[math.sqrt(Q_std[i]**2/I_sum_on[i]**2+I_std[i]**2*Q_sum_on[i]**2/I_sum_on[i]**4) for i in range(len(I_sum_on))]
	u_err=[math.sqrt(U_std[i]**2/I_sum_on[i]**2+I_std[i]**2*U_sum_on[i]**2/I_sum_on[i]**4) for i in range(len(I_sum_on))]
if ops.overI=='False':
	q=np.array([Q_sum_on[i] for i in range(len(I_sum_on))])
	u=np.array([U_sum_on[i] for i in range(len(I_sum_on))])
	q_err=np.array([Q_std[i] for i in range(len(I_sum_on))])
	u_err=np.array([U_std[i] for i in range(len(I_sum_on))])

lam2 = (299.792458/freq_list)**2
l02 = np.mean(lam2)

freq_lo = ar.get_centre_frequency() - ar.get_bandwidth()/2.0
freq_hi = ar.get_centre_frequency() + ar.get_bandwidth()/2.0
#Calculate RM and RM error by using different methods

'''

def RMsynthesis(lam2, q, u, q_err, u_err, RMlow, RMhigh, ofac=100):
	#Based on code written by G. Heald, 9 may 2008
	# subtract mean from q and u to eliminate peak at 0 rad/m^2
	q_err = np.array(q_err)
	u_err = np.array(u_err)

	lam2 = lam2
	l02 = np.mean(lam2)
	K = 1.0/len(lam2)
	R = np.array([])
	FDF = np.array([])
	RM_list = np.linspace(RMlow, RMhigh, ofac)
	
	R = []
	FDF = []
	

	for i in range(len(RM_list)):
		#Rreal = sumI*(np.sum(np.cos(-2*RM_list[i]*(lam2-l02))*I_sum_on))
		#Rimag = sumI*(np.sum(np.sin(-2*RM_list[i]*(lam2-l02))*I_sum_on))
		FDF1 = K*(np.sum((q*np.complex(1,0)/(q_err**2)+np.complex(0,1)*u/(u_err**2))*np.exp(np.complex(0,1)*-2*RM_list[i]*(lam2-l02))))
		#FDF1 = K*(num.sum((q*np.complex(1,0)+np.complex(0,1)*u)*np.exp(np.complex(0,1)*-2*phi[i]*(lam2-l02))))
		#R = np.append(R,np.complex(Rreal,Rimag))
		FDF = np.append(FDF,FDF1)
	FDFabs = np.abs(FDF)
	maxFDF = RM_list[np.array(FDFabs)==np.max(FDFabs)]
	
	return RM_list, FDFabs



def RM_bglsp(lam2, q, err_q, u, err_u, RMlow, RMhigh, ofac=100, remove_baseline = "False"):


	f = np.linspace(RMlow, RMhigh, ofac)
	err2_q = np.array(err_q) ** 2
	err2_u = np.array(err_u) ** 2

	# eq 8
	w_q = 1. / err2_q
	w_u = 1. / err2_u

	# eq 9
	W_q = w_q.sum()
	W_u = w_u.sum()

	# eq 10
	bigY_q = (w_q * np.array(q)).sum()
	bigY_u = (w_u * np.array(u)).sum()

	p = []
	constants = []
	exponents = []

	omega = f
	omega=np.array([x for x in omega if x!=0.])
	omegat = omega[:, None] * 2 * lam2[None, :]
	theta_up = ((w_q[None, :]-w_u[None, :]) * np.sin(2 * omegat)).sum(1)
	theta_bo = ((w_q[None, :]-w_u[None, :]) * np.cos(2 * omegat)).sum(1)

	theta = 0.5 * np.arctan2( theta_up, theta_bo)
	x = omegat - theta[:, None]

	cosx = np.cos(x)
	sinx = np.sin(x)
	wcosx_q = w_q[None, :] * cosx
	wsinx_q = w_q[None, :] * sinx
	wcosx_u = w_u[None, :] * cosx
	wsinx_u = w_u[None, :] * sinx
	YY=(w_q[None, :] * np.array(q)**2).sum()+(w_u[None, :] * np.array(u)**2).sum(1)

	# eq 14
	C_q = wcosx_q.sum(1)
	C_u = wsinx_u.sum(1)
	# eq 15
	S_q = wsinx_q.sum(1)
	S_u = -wcosx_u.sum(1)

	# Eq 12
	YChat = (q * wcosx_q).sum(1) + (u * wsinx_u).sum(1)
	# eq 13
	YShat = (q * wsinx_q).sum(1) - (u * wcosx_u).sum(1)
	# eq 16
	CChat = (wcosx_q * cosx).sum(1) + (wsinx_u * sinx).sum(1)
	# eq 17	
	SShat = (wsinx_q * sinx).sum(1) + (wcosx_u * cosx).sum(1)
	#ind = (CChat != 0) & (SShat != 0)
	#CChat=CChat[ind]
	#SShat=SShat[ind]

    
    	# eq
	bigF = np.zeros(CChat.shape, dtype=float)
	bigD = np.zeros(CChat.shape, dtype=float)
	bigE = np.zeros(CChat.shape, dtype=float)
	bigG = np.zeros(CChat.shape, dtype=float)
	bigJ = np.zeros(CChat.shape, dtype=float)
    
	bigO = np.zeros(CChat.shape, dtype=float)
	bigN = np.zeros(CChat.shape, dtype=float)
	bigR = np.zeros(CChat.shape, dtype=float)
	denom = np.zeros(CChat.shape, dtype=float)
    

	# below implements eqs 24, 25, & 26
	# init variables	
	K = np.zeros(CChat.shape, dtype=float)
	L = np.zeros(CChat.shape, dtype=float)
	M = np.zeros(CChat.shape, dtype=float)
	Ro = np.zeros(CChat.shape, dtype=float)
	constants = np.zeros(CChat.shape, dtype=float)
	#tmp = np.zeros(f.shape, dtype=float)
	constants_n = np.zeros(CChat.shape, dtype=float)

	# case 1
	ind = (CChat != 0) & (SShat != 0)
	tmp = 1. / (CChat[ind] * SShat[ind])
	bigD[ind] = (C_q[ind] * C_q[ind] * SShat[ind] + S_q[ind] * S_q[ind] * CChat[ind] - W_q * CChat[ind] * SShat[ind]) * 0.5 *tmp
	bigF[ind] = (C_u[ind] * C_u[ind] * SShat[ind] + S_u[ind] * S_u[ind] * CChat[ind] - W_u * CChat[ind] * SShat[ind]) * 0.5 *tmp
	bigE[ind] = (C_u[ind] * C_q[ind] * SShat[ind] + S_u[ind] * S_q[ind] * CChat[ind]) *tmp
	denom[ind] = 4.*bigD[ind]*bigF[ind]-bigE[ind]**2
	ind = (CChat != 0) & (SShat != 0) & (denom !=0) & (bigD != 0)
	tmp = 1. / (CChat[ind] * SShat[ind])
	bigD[ind] = (C_q[ind] * C_q[ind] * SShat[ind] + S_q[ind] * S_q[ind] * CChat[ind] - W_q * CChat[ind] * SShat[ind]) * 0.5 *tmp
	bigF[ind] = (C_u[ind] * C_u[ind] * SShat[ind] + S_u[ind] * S_u[ind] * CChat[ind] - W_u * CChat[ind] * SShat[ind]) * 0.5 *tmp
	bigE[ind] = (C_u[ind] * C_q[ind] * SShat[ind] + S_u[ind] * S_q[ind] * CChat[ind]) *tmp
	bigG[ind] = (C_u[ind] * YChat[ind] * SShat[ind] + S_u[ind] * YShat[ind] * CChat[ind] - bigY_u * CChat[ind] * SShat[ind]) *tmp
	bigJ[ind] = (C_q[ind] * YChat[ind] * SShat[ind] + S_q[ind] * YShat[ind] * CChat[ind] - bigY_q * CChat[ind] * SShat[ind]) *tmp
	bigO[ind] = (-bigF[ind] + (bigE[ind] * bigE[ind]) /(4*bigD[ind]))
	bigR[ind] = -bigG[ind] + (bigJ[ind] * bigE[ind]) /(2*bigD[ind])
	bigN[ind] = (bigJ[ind] * bigJ[ind]) /(4*bigD[ind])	
	M[ind] = (YChat[ind] * YChat[ind] * SShat[ind] + YShat[ind] * YShat[ind] * CChat[ind]) * 0.5 * tmp
	Ro[ind] = (bigD[ind] * bigG[ind] * bigG[ind] - bigE[ind] * bigG[ind] * bigJ[ind] + bigF[ind] * bigJ[ind] * bigJ[ind])/(bigE[ind] * bigE[ind] - 4 * bigD[ind] * bigF[ind])	
	constants = (np.sqrt(tmp / abs(4*bigD[ind] * bigF[ind]-bigE[ind]**2)))
	constants_n = 1./np.sqrt(W_q*W_u)

    # case 2
    #ind = CChat == 0
    #K[ind] = (S[ind] * S[ind] - W * SShat[ind]) / (2. * SShat[ind])
    #L[ind] = (bigY * SShat[ind] - S[ind] * YShat[ind]) / (SShat[ind])
    #M[ind] = (YShat[ind] * YShat[ind]) / (2. * SShat[ind])
    #constants[ind]  = (1. / np.sqrt(SShat[ind] * abs(K[ind])))

    #ind = SShat == 0
    #K[ind] = (C[ind] * C[ind] - W * CChat[ind]) / (2. * CChat[ind])
    #L[ind] = (bigY * CChat[ind] - C[ind] * YChat[ind]) / (CChat[ind])
    #M[ind] = (YChat[ind] * YChat[ind]) / (2. * CChat[ind])
    #constants[ind] = (1. / np.sqrt(CChat[ind] * abs(K[ind])))
    

    
	if remove_baseline == "True":
		exponents = (-YY/2.+M +Ro)
		logp = exponents[ind]+np.log(constants)
		logp_max=99999
		logp -= logp.max()
		p = np.exp(logp)
		# normalize probs
		p /= p.sum()
	if remove_baseline == "False":
		exponents = (-YY/2.+M)
		logp = exponents[ind]+1./2.*np.log(tmp)
		logp_max=99999
		logp -= logp.max()
		p = np.exp(logp)
		# normalize probs
		p /= p.sum()
                
	#exponents_n = (-YY/2.+bigY_q**2/(2*W_q)+bigY_u**2/(2.*W_u))
	eta2 = -np.asarray(exponents)*2./(2.*len(q)-4.)
	#eta2_n = -exponents_n*2./(2.*len(q)-2)
	#logp_s = -(2*len(q)-4.)*np.log(eta2[ind])+np.log(2.*math.pi)+np.log(constants)+(2-len(q))
	#logp_n=-(2*len(q)-2.)*np.log(eta2_n)+np.log(constants_n)+(1-len(q))
	#logp=logp_s-logp_n
	#plt.plot(logp)
        #plt.show()
	#logp_max=sum(logp_s)/len(logp_s)-logp_n
	return omega[ind], p, logp, np.asarray(exponents), min(eta2)

def fwhm(x, y, k=10):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    class MultiplePeaks(Exception): pass
    class NoPeaksFound(Exception): pass

    x_peak=x[np.argmax(y)]
    half_max = np.amax(y)/2.0
    s = splrep(x, y - half_max)
    roots = sproot(s)

    if len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros). Try increase the init_width_factor.")
    else:
        root2=np.sort(abs(roots-x_peak))[0:2]
        return abs(root2[0] + root2[1])/2./1.178

'''

def bootstrapping(N, q_err, u_err):
	RM_bootstr_list=[]
	for i in range(int(N)):
		I_sim=np.array([np.random.normal(I_sum_on[i], I_std[i]) for i in range(len(I_sum_on))])
		Q_sim=np.array([np.random.normal(Q_sum_on[i], Q_std[i]) for i in range(len(Q_sum_on))])
		U_sim=np.array([np.random.normal(U_sum_on[i], U_std[i]) for i in range(len(U_sum_on))])
		q=Q_sim/I_sim
		u=U_sim/I_sim
		RM_list, p, logp, exponents=RM_bglsp(lam2, q, q_err, u, u_err, ops.RMlow, ops.RMhigh, 100.)
		RM_bgls = RM_list[np.array(logp)==np.max(logp)]
		RM_list, p, logp, exponents=RM_bglsp(lam2, q, q_err, u, u_err, RM_bgls-ops.init_width, RM_bgls+ops.init_width, 100.)
		RM_bgls = RM_list[np.array(logp)==np.max(logp)]
		if RM_bgls!=0.0:
			RM_bootstr_list = np.append(RM_bootstr_list, RM_bgls)
	return RM_bootstr_list

RM_list_rmsyn, FDFabs, RM_rmsynth, err_rmsyn, StoNL=RMsynthesis(lam2, q, u, q_err, u_err, I_sum_on, ops.RMlow, ops.RMhigh, ops.RMstep)
print("RM synthesis results: %.4f+/- %.4f L/sigma: %.4f" % (RM_rmsynth, err_rmsyn, StoNL))

stdq=np.array([np.std(q)])
stdu=np.array([np.std(u)])
RM_list_bgls1, p1, logp1, exponents1, eta2s=RM_bglsp(lam2, q, 0.001*stdq*q_err, u, 0.001*stdu*u_err, ops.RMlow, ops.RMhigh, 100.)
RM_bgls = RM_list_bgls1[np.array(logp1)==np.max(logp1)]
eta2 = -exponents1[np.array(logp1)==np.max(logp1)]*2./(2.*len(q)-4)

#plotting
if ops.plot=="True":
	def func1(x):
    		q_model=x[0]*np.array([math.cos(RM_bgls*2*(lam2[i])+x[1]) +x[2] for i in range(len(lam2))])
    		return sum((np.array(q)-q_model)**2/np.array(q_err)**2)
	def func2(x):
    		u_model=x[0]*np.array([math.sin(RM_bgls*2*(lam2[i])+x[1]) +x[2] for i in range(len(lam2))])
    		return sum((np.array(u)-u_model)**2/np.array(u_err)**2)
	[A1, psi1, gamma1]=scipy.optimize.minimize(func1, [1., 1., 1.]).x
	[A2, psi2, gamma2]=scipy.optimize.minimize(func2, [1., 1., 1.]).x
	lam2_even=np.linspace(min(lam2), max(lam2), 500)
	f_even=np.sqrt(299.792458**2/lam2_even)

	q_model=A1*np.array([math.cos(RM_bgls*2*(lam2_even[i])+psi1)+gamma1 for i in range(len(lam2_even))])
	u_model=A2*np.array([math.sin(RM_bgls*2*(lam2_even[i])+psi2)+gamma2 for i in range(len(lam2_even))])

	#plt.subplots_adjust(hspace=0.4)
	ax2=plt.subplot(4,1,2)
	plt.plot(RM_list_rmsyn, FDFabs, label="RM-synth")
	plt.legend()
	plt.xlabel("RM, rad/m^2")
	plt.ylabel("RM spectrum")
	ax1=plt.subplot(4, 1, 1, sharex=ax2)
	plt.plot(RM_list_bgls1, logp1, label="Bayesian GLSP")
	plt.legend()
	plt.ylabel("Log Likelihood")
	plt.setp(ax1.get_xticklabels(), visible=False)
	#plot U results
	ax4=plt.subplot(4,1,4)
	plt.plot(f_even, u_model, color='black')
	plt.errorbar(freq_list, u, u_err, fmt='^')
	plt.ylim(-5*np.std(u), 5*np.std(u))
	plt.xlabel("f, MHz")
	plt.ylabel("Stokes U")
	#plot Q results
	ax3=plt.subplot(4,1,3, sharex=ax4)
	plt.plot(f_even, q_model, color='black')
	plt.errorbar(freq_list, q, q_err, fmt='^')
	plt.ylim(-5*np.std(q), 5*np.std(q))
	plt.ylabel("Stokes Q")
	plt.setp(ax3.get_xticklabels(), visible=False)
	plt.tight_layout()
	plt.savefig(ar.get_source()+"."+str(nameobs)+".eps")

if err_rmsyn==np.nan or err_rmsyn==0 or err_rmsyn==np.inf or np.isnan(err_rmsyn):
	err_rmsyn=1.
RM_bgls=RM_bgls[0]
RM_list_bgls, p, logp, exponents, eta2=RM_bglsp(lam2, q, q_err, u, u_err, RM_bgls-ops.init_width_factor*err_rmsyn, RM_bgls+ops.init_width_factor*err_rmsyn, 100.)
RM_bgls = RM_list_bgls[np.array(logp)==np.max(logp)]
#Error is calculated with eta
#eta2 = -exponents[np.array(logp)==np.max(logp)]*2./(2.*len(q)-4)
err_bgls=fwhm(RM_list_bgls, p)*np.sqrt(eta2)
print("RM bglsp: %.4f+/- %.4f eta2: %.2f" % (RM_bgls, err_bgls, eta2))

if ops.plot=="True":
	print(ar.get_source()+"."+str(nameobs)+".eps")

if ops.outfile=="True":
	np.savetxt("RMsynthesis.txt", np.transpose([RM_list_rmsyn, FDFabs]))
	np.savetxt("RMBGLSP.txt", np.transpose([RM_list_bgls1, logp1]))
	np.savetxt("QU.txt", np.transpose([lam2, q, q_err, u, u_err]))
	
'''
	
	
