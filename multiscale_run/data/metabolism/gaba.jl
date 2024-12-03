# psiGAD_inh_n(GLU_n), GLU_n + MET_h_c_n ⇒ GABA_inh_n + MET_co2_c_n

glutamatergic_gaba_scaling = 0.1 # 0.1 for GLU-neurons


VmaxGAD_inh_n = 0.000178  
KmGAD_inh_n = 4.0  
# from Thesis of Evelyn Rodriguez: A MATHEMATICAL MODEL OF NEUROCHEMICAL MECHANISMS IN A SINGLE GABA NEURON

#psiGAD_inh_n(GLU_n) = glutamatergic_gaba_scaling * VmaxGAD_inh_n*GLU_n/(KmGAD_inh_n + GLU_n) 
#psiGAD_inh_n1 = glutamatergic_gaba_scaling * VmaxGAD_inh_n*GLU_n0/(KmGAD_inh_n + GLU_n0) 

# psiGLU_GABA is 0.13 umol per gram tissue per minute. # Yamashita 2018: vesicular GABA uptake rate is 5–6 times slower than the glutamate uptake rate

###############################################

# psiGADcatab_inh_n(GABA_inh_n), GABA_inh_n ⇒ SUCmito_n

# combo of reactions in Bigg:
# ABUTD,4ABUTtm,ABTArm,SSALxm(r0178)

kGADcatab_inh_n = 0.0005 #0.00101 

#psiGADcatab_inh_n(GABA_inh_n) = glutamatergic_gaba_scaling*kGADcatab_inh_n * GABA_inh_n 
#psiGADcatab_inh_n1 = glutamatergic_gaba_scaling*kGADcatab_inh_n * GABA_inh_n0 

