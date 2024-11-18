# Glycogen

# based on DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151

# Glycogen synthase
#VmaxGS_a = 0.07585 # 0.00153 #0.1 #0.5 #DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 # 0.000153 in Cloutier2009 0.0001528 in cellml Cloutier2009
kL2_GS_a = 0.4502909779144298 #3.4 # JC2018 #3.33 #0.017 #0.01 # 1.53e-4 mM/s Cloutoer2009 #0.017 #mmol/L per s #DiNuzzo2010 PMID: 20827264 #3.33 prev par opt Jay 
kmL2_GS_a = 1.4 #0.57 # JC2018 

# Glycogen phosphorylase
k_L2_GS_a = 0.8008668792292503 #0.34  # JC2018 
km_L2_GS_a = 1.4  # JC2018 

VmaxGP_a = 0.001843542832727982 # 5.41521636193213E-06 # 0.00000525 # 0.000005 #0.00005 #0.0001 #0.001 #0.008 #DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 # 4.92e-5 in Cloutier2009 4.922e-5 in cellml Cloutier2009
KmGP_AMP_a = 0.1 #1e-5 # 1e-5 adj to new u0 24aug22 #0.01 # adj AMP to match glycogen dynamics in DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 # 0.016 DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151
hGPa = 1.5


#adapted from Lambeth
# Phosphoglucomutase
Vmaxfpglm_a = 1.8474212896979174 # 0.00095469565397463 # 0.64 #0.001 #0.005 #0.01 #0.1 #adj to gp flux at act # 8mM/s - 0.48 M/min Lambeth 
KmG6PPGLM_a = 0.001 #0.0024 #0.03  # adj from conc diff 0.03*0.06/0.75 # g6p = 0.75 Lambeth
KmG1PPGLM_a = 0.01 #0.0107 # adj from conc diff 0.063*0.01/0.0589 #0.063 # g1p = 0.0589  Lambeth
KeqPGLM_a = 4.9 # 6.5 # 16.62 #1.0048450801727364*6.5 #7.05 #7.01 # adj from conc ratio to flux close to GP flux at steady state #0.45

#VmaxPDE_a = 1.92e-5 #0.00105*exp(-4) with approx conc from mol atlas #1050.0  
VmaxPDE_a = 1e-3 #1.92e-5 #0.00105*exp(-4) with approx conc from mol atlas #1050.0  
Kmcamppde_a = 0.0055 #mM #5500.0  
#psiPDE_a(cAMP_a) = VmaxPDE_a*cAMP_a/(Kmcamppde_a + cAMP_a)  






# details glycogen regulation
# GSAJay
kg8_GSAJay = 5.0 
st_GSAJay = 0.003 
kmg8_GSAJay = 0.00012 
s1_GSAJay = 100.0 
kg2_GSAJay = 0.5  
kg7_GSAJay = 20.257289129639318 #1.012864456481966*20.0  # kg7_GSAJay = 20.0
kmg7_GSAJay = 0.015

# Phosphorylase kinase act  0 ⇒ PHKa 
cai0_ca_ion = 5e-05  
kg3_PHKact = 20.0  
kt_PHKact = 0.0025 
kmg3_PHKact = 0.004 
kg4_PHKact = 5.0  
kmg4_PHKact = 0.0011

# PHK
kg5_PHK = 1915.1920260832771 #20.0  
pt_PHK = 0.007  
kmg5_PHK = 0.01  
s1_PHK = 100.0 
kg2_PHK = 0.5 
kg6_PHK = 500.0 #5.0 
kmg6_PHK = 0.005  
s2_PHK = 0.001  
kgi_PHK = 10.0 
kmind_PHK = 2e-06


# PKA1,2
kgc1_PKA12 = 1e-06 
k_gc1_PKA12 = 0.01 

kgc2_PKA12 = 1e-06  
k_gc2_PKA12 = 0.01  



# GP
Vmaxfgpa = 0.001 #4.93e-5 #Cloutier2009 # 0.008 DiNuzzo2010 #0.33 

Km_GLY_b_GP_a_a = 0.15  
Ki_G1P_GP_a_a = 10.1  
Ki_GLY_fGP_a_a = 2.0  
Km_Pi_GP_a_a = 4.0 
KeqGPa = 0.42  

Ki_Pi_G_P_a_a = 4.7 
Km_GLY_fGP_a_a = 1.7  


###
Vmaxfgpb = 0.5 

KiGLYbGP_b_a = 4.4  
Km_G1P_GP_b_a = 1.5  
Ki_GLY_fGP_b_a = 15.0  
Km_Pi_GP_b_a = 0.2  
KeqGPb = 0.42  


AMPnHGPb = 1.5 #1.75  # 1.5 DiNuzzo2010
Km_AMP_GP_b_a = 0.016 #DiNuzzo2010 #1.9e-06  
Ki_Pi_G_P_b_a = 4.6  
Ki_G1P_GP_b_a = 7.4  


#### UDPGP
VmaxfUDPGP = 0.004420628605834113 #0.0045555170380946215 #0.0015762057284464041*2.890179217014261 # 1.374710748327394*1.070600857717005*0.01253505948380228 #
VmaxrUDPGP = 0.017682514423336453 #0.018222068152378486 # 0.0015762057284464041*11.560716868057044 # 1.374710748327394*1.070600857717005*0.8599116744575482* 0.026768820961076727*0.027228381148077584*2.5 
KutpUDPGP = 0.05 # KutpUDPGP
Kg1pUDPGP = 0.1  
KpiUDPGP = 0.2 #200.0 # 200 as adj for Pi instead of PPi #0.2 
KUDPglucoUDPGP_a = 0.05 # KglucoUDPGP
