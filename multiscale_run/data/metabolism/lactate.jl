# LAC
# Jolivet, Calvetti, Simpson, DiNuzzo...



# LDH
# # LDH a
# VmfLDH_a = 780.0 #Winter2017 #4160.00 #Calvetti 1110. # DiNuzzo # 1.59 #Jolivet2015
# VmrLDH_a = 32.0 #Winter2017 #3245.00 #Calvetti # 25. # DiNuzzo #0.071 # Jolivet pdf # 0.099 Joliivet matlab 
# # LDH n
# VmfLDH_n = 2000. #Winter2017 #1436.00 #Calvetti #2000. # DiNuzzo 72.3 #Jolivet2015
# VmrLDH_n = 15. #Winter2017  #1579.83 #Calvetti # 44.8 # DiNuzzo 0.72 #  Jolivet pdf   # 0.768 Joliivet matlab 

# KeqLDH = 1.62*(10^11) M^(-1) # https://doi.org/10.3389/fnins.2015.00022 # The equilibrium constant is strongly in favor of La− (1.62 × 10^11 M^−1) (Lambeth and Kushmerick, 2002)

# OBrien2007 DOI 10.1007/s11064-006-9132-9
# The main difference in LDH kinetics between the neuronal and glial preparations is found in the rates of the reverse reaction, 
# conversion of lactate to pyruvate, 
# which was more than two-fold higher in synaptosol than in astrocytes, 
# with Vmax values of 268 lmol/min/ mg protein versus 123 lmol/min/mg protein, respectively.

# LDH a: Berndt2018 (astrocytes express liver isoform LDH5)
# LDH n: Berndt2015

# LDH
# Params and mechanism from OBrien2007 DOI 10.1007/s11064-006-9132-9 eq derived according to mechanism + Calvetti2018 for redox dependence

# # LDH a
# nu_LDH1f_a = 0.1 #0.01 # adj for redox ratio diff #0.1 #Calvetti2018
# VmfLDH1_a = 7816.67 #4160.0 #Calvetti2018  # 1876 umol/min/mg protein OBrien2007 -> 1000*1876*0.25/60 = 7816.67 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
# KmLDH1pyr_a = 0.084 # OBrien2007 #6.24 #Calvetti2018

# nu_LDH1r_a = 10. #100. # adj for redox ratio diff # 10.0 #Calvetti2018
# VmrLDH1_phase1_a = 225.0 #single phased 3245.0 #Calvetti2018   # 54 umol/min/mg protein OBrien2007 -> 1000*54*0.25/60 = 225 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
# VmrLDH1_phase2_a = 512.5 # 123 umol/min/mg protein OBrien2007 -> 1000*123*0.25/60 = 512.5 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 

# KmLDH1lac_phase1_a = 1.5 # OBrien2007
# KmLDH1lac_phase2_a = 8.6 # OBrien2007


# # LDH n  LDH5_synaptic + LDH1_n
# nu_LDH5f_n = 0.1 #0.01 # adj for redox ratio diff # 0.1 #Calvetti2018
# VmfLDH5_n = 4845.83  # 1163 umol/min/mg protein OBrien2007 -> 1000*1163*0.25/60 = 4845.83  mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
# KmLDH5pyr_n = 0.0296 # OBrien2007 #6.24 #Calvetti2018

# nu_LDH5r_n = 10. #100. # adj for redox ratio diff # 10.0 #Calvetti2018
# VmrLDH5_phase1_n = 533.33  # 128 umol/min/mg protein OBrien2007 -> 1000*128*0.25/60 = 533.33 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
# VmrLDH5_phase2_n = 1116.67  # 268 umol/min/mg protein OBrien2007 -> 1000*268*0.25/60 = 1116.67 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 

# KmLDH5lac_phase1_n = 1.73  # OBrien2007
# KmLDH5lac_phase2_n = 7.77  # OBrien2007

# adj Vmax for abs val at relevant scale?

# Params and mechanism from OBrien2007 DOI 10.1007/s11064-006-9132-9 eq derived according to mechanism + Calvetti2018 for redox dependence

# LDH a
VmfLDH1_a = 7816.67 #4160.0 #7816.67 #4160.0 #Calvetti2018  # 1876 umol/min/mg protein OBrien2007 -> 1000*1876*0.25/60 = 7816.67 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
VmrLDH1_phase1_a = 225.0 #single phased 3245.0 #Calvetti2018   # 54 umol/min/mg protein OBrien2007 -> 1000*54*0.25/60 = 225 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
VmrLDH1_phase2_a = 512.5 # 123 umol/min/mg protein OBrien2007 -> 1000*123*0.25/60 = 512.5 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 

KmLDH1pyr_a = 0.084 # OBrien2007 #6.24 #Calvetti2018
KmLDH1lac_phase1_a = 1.5 # OBrien2007
KmLDH1lac_phase2_a = 8.6 # OBrien2007

nu_LDH1f_a = 0.1 #0.01 # adj for redox ratio diff #0.1 #Calvetti2018
nu_LDH1r_a = 1/nu_LDH1f_a #10. #100. # adj for redox ratio diff # 10.0 #Calvetti2018


# LDH n  LDH5_synaptic + LDH1_n
VmfLDH1_n = 7816.67 #4160.0 #7816.67 #4160.0 #Calvetti2018  # 1876 umol/min/mg protein OBrien2007 -> 1000*1876*0.25/60 = 7816.67 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
VmrLDH1_phase1_n =  225.0 #single phased 3245.0 #Calvetti2018   # 54 umol/min/mg protein OBrien2007 -> 1000*54*0.25/60 = 225 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
VmrLDH1_phase2_n =  512.5 # 123 umol/min/mg protein OBrien2007 -> 1000*123*0.25/60 = 512.5 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 

KmLDH1pyr_n = 0.084 # OBrien2007 #6.24 #Calvetti2018
KmLDH1lac_phase1_n = 1.5 # OBrien2007
KmLDH1lac_phase2_n = 8.6 # OBrien2007

nu_LDH1f_n = 0.1 #0.01 # adj for redox ratio diff #0.1 #Calvetti2018
nu_LDH1r_n = 1/nu_LDH1f_n #10. #100. # adj for redox ratio diff # 10.0 #Calvetti2018


VmfLDH5_n =  0.999637815739898*4845.83  # 1163 umol/min/mg protein OBrien2007 -> 1000*1163*0.25/60 = 4845.83  mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
VmrLDH5_phase1_n = 0.999637815739898*533.33  # 128 umol/min/mg protein OBrien2007 -> 1000*128*0.25/60 = 533.33 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
VmrLDH5_phase2_n =  0.999637815739898*1116.67  # 268 umol/min/mg protein OBrien2007 -> 1000*268*0.25/60 = 1116.67 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 


nu_LDH5f_n = 0.1 #0.01 # adj for redox ratio diff # 0.1 #Calvetti2018
nu_LDH5r_n = 1/nu_LDH5f_n #10. #100. # adj for redox ratio diff # 10.0 #Calvetti2018

KmLDH5pyr_n = 0.0296 # OBrien2007 #6.24 #Calvetti2018
KmLDH5lac_phase1_n = 1.73  # OBrien2007
KmLDH5lac_phase2_n = 7.77  # OBrien2007

# art -> cap
C_Lac_a = 0.75 #0.55 #0.55 - 0.10113109309463841 #0.6 #0.82 #0.55 #0.815 #Lac_b0 #0.82 #Lac_b0 - 0.10113109309463841 #0.82 #1.22 #1.6 #0.82 #1.222 # 0.5 #1.225 #1.22 #1.25 #1.0 #0.85 # #0.82 #0.5 Jolivet2015 # 1. #0.9 #1.0 # Leegsma-Vogt PMID: 11746404 #0.82 # set slightly higher than Lac_b from http://dx.doi.org/10.1016/j.cmet.2015.10.010 


# using simple transport equations for Lac, because we don't model precisely H which are important for equations (like equations in DiNuzzo)

# cap -> ecs
TbLac = (1/0.0275)*0.3 #0.28 #0.17 # Calvetti #0.00587 # Winter2017 0.3 #0.275 # 0.25 #Jolivet2015 0.3 # Jolivet matlab #0.29 # 0.26 ce 0.3 eb DiNuzzo2010 DiNuzzo2010_1.pdf # 0.00587 Winter2017 #0.25 #Jolivet pdf #0.17 # mM/s
KbLac = 1.0 #5. #3.5 #1.0 #1.87 #1.0 #Jolivet, Leegsma-Vogt PMID: 11746404 #5.00 # mM Calvetti # 0.5 Winter2017 # 3.5-10 mM Perez-Escuredo https://doi.org/10.1016/j.bbamcr.2016.03.013 

# ecs <-> a
#Jolivet2015, Calvetti2018
TaLac = (1/0.8)*107. # 0.035 #1.04167 #0.250 umol/min/mg Broer et al., 1997 DOI: 10.1002/jnr.20294 -> 1000*0.250*0.25/60 = 1.04167 mM/s # *0.25 from Nazaret2009    #66.67 #Calvetti #0.057 # Winter2017 # 107. #Jolivet matlab #106.1 #Jolivet pdf #66.67 Calvetti # ecsBA2a: 0.03 mM/s DiNuzzo2010, a2ecsAN: 0.04 mM/s DiNuzzo2010, ecsBA2a:1.2e-15 mmol/sec Simpson2007, a2ecsAN: 1.5e-14 mmol/sec Simpson 2007 
Km_Lac_a = 3.5 # Jolivet2015 #6.0 # 0.5 Winter2017  #3.5-10.0 mM https://doi.org/10.1016/j.bbamcr.2016.03.013 # 3.5-7.0 mM from Dienel 2019 https://doi.org/10.1002/jnr.24387   #15.0 # par[35]

# ecs <-> n
TnLac = (1/0.44)*23.5 #0.07 #66.67 #Calvetti # 0.2175 #Winter2017 # 23.5 #Jolivet matlab #24.3 Jolivet pdf #66.67 Calvetti# 0.07 mM/s DiNuzzo2010, Simpson 2007 predicted: 4.9e-15 mmol/sec
Km_LacTr_n = 0.74 #Jolivet2015 #0.6 # 0.5 Winter2017  # 0.5-0.75 mM from https://doi.org/10.1016/j.bbamcr.2016.03.013 # 0.7 from Dienel 2019 https://doi.org/10.1002/jnr.24387     # 0.025 # par[37] 

TMaxLACgc = (1/0.022)*2.43e-03
KtLACgc = 1.0

# ecs diffusion
betaLacDiff = 0 #0.001 # s-1 www.pnas.org􏱵cgi􏱵doi􏱵10.1073􏱵pnas.0605864104
#Lac_ecs0 = Lac_ecs0



VmfLDH_a = 8.74949881831907 #8.445290697870174 #0.9722314374674746*1124.6166207771603 #1486.8293841083846 # 1.71
#VmrLDH_a = 0.4331118732235805 # 0.9999030676455061*4.0693155124381875*1.107525198132424*0.099  #0.9722314374674746*67.2 #0.050693455630342395 #0.08044909945699166 # 0.099

VmfLDH_n = 241.739831262545 #226.35715419383556 #0.39617499995813404*4710.989517511934 #6059.141369229505 #78.1  
#VmrLDH_n = 0.7556440067499692 #0.999978541778008*0.9713032773071247*1.1049568944911459*0.768  #0.39617499995813404*36.18 #40.273541408198604*0.8929243326451762 #2.4067213774881893 #0.768 

KeLDH_a = 0.046994407267851 #0.051284424505696105
KeLDH_n = 0.00329708625924639 #0.0033382819705485936
