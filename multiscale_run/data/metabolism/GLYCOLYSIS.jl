# hexokinase
VmaxHK_n = 0.51617647 #0.075 #<flux0.00325  #0.050435 #<JLV #0.016 #0.01175 #0.0235 * 0.5 assuming sep half by half 0.0065 mM/s between n and a # 0.0235 is derived to have steady rate approx 0.0065 mM/s  #0.0068  w/o ATP in rate eq # Barros2007  #VmaxHK is 6 to 9 uM/sec https://doi.org/10.1093/cercor/bhs309
#VmaxHK_n = 0.050435 #<JLV #0.016 #0.01175 #0.0235 * 0.5 assuming sep half by half 0.0065 mM/s between n and a # 0.0235 is derived to have steady rate approx 0.0065 mM/s  #0.0068  w/o ATP in rate eq # Barros2007  #VmaxHK is 6 to 9 uM/sec https://doi.org/10.1093/cercor/bhs309
KmHK_n = 0.05 # Barros2007 KIATPhexn = 1.

#KIATPhex_na = 0.558001157065081 #0.6 #0.95 #1.
KIATPhex_n = 0.554 #KIATPhex_n = 0.558001157065081 #0.6 #0.95 #1.
KIATPhex_a = 0.554 #KIATPhex_a = 0.558001157065081 #0.6 #0.95 #1.

# KIATPhex_n = 0.558001157065081 
# KIATPhex_a = 0.558001157065081  #0.5575 #0.558001157065081 #0.6 #0.95 #1.
# 0.5 up
# 0.55 # slightly up, stim resp, import to cells from ecs
# 0.554 import to cells from ecs
# 0.556 import to cells from ecs
# 0.557 ANLS, a ok n slightly up
# 0.558001157065081 # almost steady, a slightly down # ANLS
# 0.6 down

nHhexn = 4.

VmaxHK_a = 0.4129411 #0.06 #<flux0.00325 #0.185 #<JLV #0.013 #0.012 #0.0101 #0.0202 * 0.5 assuming sep half by half 0.0065 mM/s between n and a #0.0202 is derived to have steady rate approx 0.0065 mM/s  #0.0068 # Barros2007 #VmaxHK is 6 to 9 uM/sec https://doi.org/10.1093/cercor/bhs309
#VmaxHK_a = 0.185 #<JLV #0.013 #0.012 #0.0101 #0.0202 * 0.5 assuming sep half by half 0.0065 mM/s between n and a #0.0202 is derived to have steady rate approx 0.0065 mM/s  #0.0068 # Barros2007 #VmaxHK is 6 to 9 uM/sec https://doi.org/10.1093/cercor/bhs309
KmHK_a = 0.05 # Barros2007 KIATPhexn = 1.
#KIATPhexa = 0.6 #0.95 #1.
nHhexa = 4.


KiHKG6P_n = 0.01021 #0.0102 #0.009206930693069307 0.0102 # Ki 0.017 from doi:10.1038/jcbfm.2010 scaled to difference of G6P steady state concentrations
KiHKG6P_a = 0.0137 #0.0132 #KiHKG6P_a = 0.0102 # Ki 0.017 from doi:10.1038/jcbfm.2010 scaled to difference of G6P steady state concentrations

# KmATPHK_n = 0.2354545455 #0.37 #0.4 mM Garfinkler1987, 0.37 Berndt2015 
# KmATPHK_a = 0.1229090909 #0.208 #mM Garfinkler1987, 0.37 Berndt2015 # #0.208 - astrocyte specific Lai 1999 PMID: 10488914 

#fluxes Hex
# n = 0.003246320878748037
# a = 0.00325030914023512

#####################################
# PGI: Cloutier2009,Berndt2015
# From Bouzier-Sore doi: 10.3389/fnagi.2015.00089 - (PGI) is a near-equilibrium enzyme that has been shown to be highly active at converting F6P into G6P in certain cells and/or tissues, such as, e.g., neurons (Gaitonde et al., 1989)

# Vmax_fPGI_n = 0.5 #Cloutier2009 pdf # 0.5 in cellml Cloutier2009 
# Vmax_rPGI_n = 0.45 #Cloutier2009 pdf #0.45 in cellml Cloutier2009
# Km_G6P_fPGI_n = 0.49 # 0.5 #0.593 # KfPGI Gaitonde PMID: 2709006  # 0.5 in cellml Cloutier2009 
# Km_F6P_rPGI_n = 0.08 #0.095 # KrPGI Gaitonde PMID: 2709006  # 0.06 in cellml Cloutier2009 

# Vmax_fPGI_a = 0.5 #Cloutier2009 pdf # 0.5 in cellml Cloutier2009 
# Vmax_rPGI_a = 0.45 #Cloutier2009 pdf #0.45 in cellml Cloutier2009
# Km_G6P_fPGI_a = 0.49 # 0.5 #0.593 # KfPGI Gaitonde PMID: 2709006  # 0.5 in cellml Cloutier2009 
# Km_F6P_rPGI_a = 0.08 #0.095 # KrPGI Gaitonde PMID: 2709006  # 0.06 in cellml Cloutier2009 

Vmax_fPGI_n = 0.5109590762636075 #0.5224916273904385 #0.6075442059699083  # 0.5 #Cloutier2009 pdf # 0.5 in cellml Cloutier2009
#Vmax_rPGI_n = 0.5467897853729176 # 0.45 #Cloutier2009 pdf #0.45 in cellml Cloutier2009
Km_G6P_fPGI_n = 0.593 # KfPGI Gaitonde PMID: 2709006  # 0.5 in cellml Cloutier2009 
Km_F6P_rPGI_n = 0.095 # KrPGI Gaitonde PMID: 2709006  # 0.06 in cellml Cloutier2009 

Vmax_fPGI_a = 0.5408672262560398 #0.5231442728059117 #0.5231442728059117 #0.6085598341799668 #0.6085598341799668 #0.5 #Cloutier2009 pdf # 0.5 in cellml Cloutier2009 
#Vmax_rPGI_a = 0.5477038507619701 #0.5477038507619701 #0.45 #Cloutier2009 pdf #0.45 in cellml Cloutier2009
Km_G6P_fPGI_a = 0.593 #0.593 # KfPGI Gaitonde PMID: 2709006  # 0.5 in cellml Cloutier2009 
Km_F6P_rPGI_a = 0.095 #0.095 # KrPGI Gaitonde PMID: 2709006  # 0.06 in cellml Cloutier2009 





#####################################

# PFK
# combo from Winter2017, Berndt2015, DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 

# From Bouzier-Sore doi: 10.3389/fnagi.2015.00089 - PFK1 in neurons represents a glycolysis bottleneck (Almeida et al., 2004). 
# PFK1 in situ activity is very low in neurons when compared with neighbor astrocytes (Almeida et al., 2004). 
# Such a low PFK1 activity is due to the virtual absence of PFKFB3 (Herrero-Mendez et al., 2009), 
# in situ PFK1 activity is ∼four-fold lower in neurons when compared with astrocytes (Almeida et al., 2004).

# VmaxPFK_n = 0.42 #0.11 # DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 # 0.44 Winter2017
# VmaxPFK_a = 0.2 #0.06 # DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 # 0.2 Winter2017
VmaxPFK_n = 0.3511362207540195 #0.35252996309065393 # 0.369795304963641 #0.435 # 0.42 #0.11 # DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 # 0.44 Winter2017
VmaxPFK_a = 0.15208109403904177 #0.14812183418703848 # 0.1554503408943446 #0.20496159880969 #0.25 #0.435 #0.235 #0.2 #0.06 # DiNuzzo2010_2_glycogen.pdf doi:10.1038/jcbfm.2010.151 # 0.2 Winter2017

KmPFKF6P_n = 0.05170785 #0.03913 # adj from Winter2017 for difference in conc: 0.03913 = 0.18*0.01/0.046  # (0.0642857 + 0.03913)/2 = 0.05170785
KmPFKF6P_a = 0.035 #0.05170785 #0.0642857 # adj from Winter2017 for difference in conc: 0.0642857 = 0.18*0.01/0.028 # (0.0642857 + 0.03913)/2 = 0.05170785

KiPFK_ATP_na = 0.666155029035142 #0.6 # adj for diiff ATP DiNuzzo2010,Winter2017,Jolivet2015
KiPFK_ATP_n = 0.666155029035142 #0.6 # adj for diiff ATP DiNuzzo2010,Winter2017,Jolivet2015
KiPFK_ATP_a = 0.666155029035142 #0.6 # adj for diiff ATP DiNuzzo2010,Winter2017,Jolivet2015

#KiPFK_ATP_a = 0.6 # adj for diiff ATP DiNuzzo2010,Winter2017,Jolivet2015
nPFKn = 4. # DiNuzzo2010,Winter2017,Jolivet2015
nPFKa = 4. # DiNuzzo2010,Winter2017,Jolivet2015

KmF26BP_PFK_a = 0.0042 #0.00485 #0.00475 #0.005 # 0.001 #adj for conc #0.0042 # Berndt2015, repeated par in pdf: 0.0042 and 0.005
nPFKf26bp_a = 5.5 # Berndt2015
KoPFK_f26bp_a = 0.55 # Berndt2015

#####################################

# PFK2 PFKFB3

# From Bolanos doi:10.1016/j.tibs.2009.10.006 - Isoform 3 (PFKFB3) displays the highest (~700 fold) kinase to bisphosphatase ratio
# its activity is almost exclusively dedicated to the generation of F2,6P2

# kinase
Vmax_PFKII_g = 0.0031731959656441096 #0.00333019366990191 #0.00333019366990191 #0.003 #0.01 #0.052 #0.06 # 0.026 # adj, no AMP in eq, Vmax opt in Berndt2015 #0.0026

Kmf6pPFKII_g = 0.027 #0.027 #0.016 # F26BP_J. Biol. Chem.-1984-Kitajima-6896-903.pdf #0.027
KmatpPFKII_g = 0.0675993571084336 #0.0675993571084336 #0.055 #0.15 # F26BP_J. Biol. Chem.-1984-Kitajima-6896-903.pdf  #0.055
#Km_act_ampPFKII_g = 0.073
Km_act_adpPFKII_g = 0.0667721029996198  #0.0667721029996198 #0.056 #0.062 # F26BP_J. Biol. Chem.-1984-Kitajima-6896-903.pdf #0.056


# Fructose-2,6-bisphosphatase # Berndt 2015
Vmax_f26pase_g = 0.052 #0.052
Km_f26bp_f_26pase_g =  0.07 #0.07 #0.001 #F26BP_J. Biol. Chem.-1984-Kitajima-6896-903.pdf #0.07
Ki_f6p_f_26_pase_g = 0.02 #0.02 #0.0015 #F26BP_J. Biol. Chem.-1984-Kitajima-6896-903.pdf #0.02

#####################################

#ALD

# Berndt2015

#Vmaxald_n = 4.01 #Vmaxald_n = 4.0 #1.5 #0.47 #4.7 #46.8  # adj
Vmaxald_n = 1.400146917265416 #1.405738558339811 #1.47463603942376 #1.42 #4.01 #4.0 #1.5 #0.47 #4.7 #46.8  # adj

#Keqald_n = 0.01 #0.005 #0.0976 # Berndt2015 
Keqald_n = 0.1 # PMID: 5357024 liver

KmfbpAld_n = 0.003
KmgapAld_n = 0.08
KmdhapAld_n = 0.03


#Vmaxald_a = 4.01 # 4.0 #1.5 # 0.47 #4.7 #46.8  # adj
Vmaxald_a = 3.2308258205047813 #3.1524640987394714 #3.313845197899958 #1.42579935322302 # 1.42 # 4.01 #4.0 #1.5 # 0.47 #4.7 #46.8  # adj

#Keqald_a = 0.01 #0.005 #0.0976 #Berndt2015 
Keqald_a = 0.0005 #0.1

KmfbpAld_a = 0.003 #0.003
KmgapAld_a = 0.08 #0.08
KmdhapAld_a = 0.03 #0.03

#####################################

# TPI

# Berndt2015

# ATTENTION!

# Vmaxtpi_n =  1000000.0
# Keqtpi_n =  0.05 #0.125 #0.05 #0.1 #0.045 # Berndt hepatokin #0.0545
# KmdhapTPI_n = 0.84
# KmgapTPI_n =   1.65

# Vmaxtpi_a =  1000000.0
# Keqtpi_a =  0.05 # 0.126 #0.125 #0.05 #0.1 #0.045 # Berndt hepatokin #0.0545
# KmdhapTPI_a = 0.84
# KmgapTPI_a = 1.65

Vmaxtpi_n = 0.9842422040175792 #0.9881749131463432 #1.03661007322829 #1.0 #1000000.0
Keqtpi_n = 20.0 # PMID: 5357024 #0.05 #0.125 #0.05 #0.1 #0.045 # Berndt hepatokin #0.0545
KmdhapTPI_n = 0.6 #0.84
KmgapTPI_n = 0.4 # 1.65

Vmaxtpi_a = 1.0263480086710068 #1.0014347584951384 #1.05267198236797 # 1.05267198236797 #1.0 #1000000.0
Keqtpi_a = 20.0 # PMID: 5357024 #0.05 # 0.126 #0.125 #0.05 #0.1 #0.045 # Berndt hepatokin #0.0545
KmdhapTPI_a = 0.6 #0.84
KmgapTPI_a = 0.4 #1.65

#####################################

# GAPDH

# Berndt2015 brain, Berndt2018 hepatokin

Vmaxgapdh_n = 182.64561255814576 #182.63166317515197 #187.048654473353 #193.6294833757099 #250.0 #1000.0 #72000.0

Keqgapdh_na = 0.2 #0.0015 #0.028 #0.0868 #0.04 #0.0868

KmnadGpdh_n = 0.00947604205269482 # 0.01  #0.027 #0.01  or 0.027
KmGapGapdh_n = 0.101
KmpiGpdh_n = 3.9 #67.8260869565 #67.8 after Pi changes to ETC-Pi # 3.9
KmnadhGapdh_n = 0.00817504585255996 #0.008
KmBPG13Gapdh_n = 0.0035


Vmaxgapdh_a = 3388.635929380235 #3301.602157609392 #3388.71025575609 #3096.9093812733613 #250.0  #1000.0 #72000.0

#Keqgapdh_a = 0.028 #0.0868 #0.04 #0.0868
KmnadGpdh_a = 0.0106952634603107 # 0.01  #0.027 #0.01  or 0.027
KmGapGapdh_a = 0.101
KmpiGpdh_a = 3.9 # 67.8260869565 #67.8 after Pi changes to ETC-Pi # 3.9
KmnadhGapdh_a = 0.008659493402079 #0.008
KmBPG13Gapdh_a = 0.0035

# Vmaxgapdh_n = 7000.0 #72000.0
# Keqgapdh_n = 0.026 #0.0868 #0.04 #0.0868
# KmnadGpdh_n =  0.01  #0.027 #0.01  or 0.027
# KmGapGapdh_n = 0.101
# KmpiGpdh_n = 3.9
# KmnadhGapdh_n = 0.008
# KmBPG13Gapdh_n = 0.0035

# Vmaxgapdh_a = 7000.0 #72000.0
# Keqgapdh_a = 0.026 #0.0868 #0.04 #0.0868
# KmnadGpdh_a =  0.01  #0.027 #0.01  or 0.027
# KmGapGapdh_a = 0.101
# KmpiGpdh_a = 3.9
# KmnadhGapdh_a = 0.008
# KmBPG13Gapdh_a = 0.0035


#####################################

# PGK: Phosphoglycerate kinase
# Berndt2015

# Vmaxpgk_n = 12. #5.5 # scaled in comparison of ratios with GAPDH Vmax #396.0
# Keqpgk_n = 1310.0
# Kmbpg13pgk_n = 0.063 # 0.0022 # Berndt2018 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_n = 0.2 #0.4 #0.25 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_n = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_n = 0.4 #0.25 #0.42 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS

# Vmaxpgk_a = 12. #5.5 # scaled in comparison of ratios with GAPDH Vmax 396.0
# Keqpgk_a =  1310.0
# Kmbpg13pgk_a = 0.063 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_a = 0.2 #0.4 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_a = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_a = 0.4 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS



# Vmaxpgk_n = 13. #5.5 # scaled in comparison of ratios with GAPDH Vmax #396.0
# Keqpgk_n = 1310.0
# Kmbpg13pgk_n = 0.063 # 0.0022 # Berndt2018 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_n = 0.2 #0.4 #0.25 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_n = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_n = 0.4 #0.25 #0.42 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS

# Vmaxpgk_a = 13. #5.5 # scaled in comparison of ratios with GAPDH Vmax 396.0
# Keqpgk_a =  1310.0
# Kmbpg13pgk_a = 0.063 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_a = 0.2 #0.4 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_a = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_a = 0.4 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS


# Vmaxpgk_n = 13. #5.5 # scaled in comparison of ratios with GAPDH Vmax #396.0
# Keqpgk_n = 2500. #1830. #1310.0
# Kmbpg13pgk_n = 0.063 # 0.0022 # Berndt2018 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_n = 0.2 #0.4 #0.25 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_n = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_n = 0.4 #0.25 #0.42 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS

# Vmaxpgk_a = 13. #5.5 # scaled in comparison of ratios with GAPDH Vmax 396.0
# Keqpgk_a = 2500. #1830. # 1310.0
# Kmbpg13pgk_a = 0.063 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_a = 0.2 #0.4 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_a = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_a = 0.4 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS


# Vmaxpgk_n = 13. #5.5 # scaled in comparison of ratios with GAPDH Vmax #396.0
# Keqpgk_n = 2600. #1830. #1310.0
# Kmbpg13pgk_n = 0.063 # 0.0022 # Berndt2018 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_n = 0.3 #0.2 #0.4 #0.25 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_n = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_n = 0.4 #0.4 #0.25 #0.42 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS


# Vmaxpgk_a = 13. #5.5 # scaled in comparison of ratios with GAPDH Vmax 396.0
# Keqpgk_a = 2600. #1830. # 1310.0
# Kmbpg13pgk_a = 0.063 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_a = 0.3 #0.2 #0.4 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_a = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_a = 0.4 #0.4 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS



# Vmaxpgk_n = 13. # 5.5 # scaled in comparison of ratios with GAPDH Vmax #396.0
# Keqpgk_n = 2600. #1830. #1310.0
# Kmbpg13pgk_n = 0.1 #0.063 # 0.0022 # Berndt2018 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_n = 0.45 #0.2 #0.4 #0.25 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_n = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_n = 0.33 #0.4 #0.25 #0.42 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS


# Vmaxpgk_a = 13. # 5.5 # scaled in comparison of ratios with GAPDH Vmax 396.0
# Keqpgk_a = 2600. #1830. # 1310.0
# Kmbpg13pgk_a = 0.1 #0.063 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_a = 0.45 #0.2 #0.4 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_a = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_a = 0.33 #0.4 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS


# Vmaxpgk_n = 13. # 5.5 # scaled in comparison of ratios with GAPDH Vmax #396.0
# Keqpgk_n = 2600. #1830. #1310.0
# Kmbpg13pgk_n = 0.1 #0.063 # 0.0022 # Berndt2018 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_n = 0.33 #0.2 #0.4 #0.25 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_n = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_n = 0.33 #0.4 #0.25 #0.42 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS


# Vmaxpgk_a = 13. # 5.5 # scaled in comparison of ratios with GAPDH Vmax 396.0
# Keqpgk_a = 2600. #1830. # 1310.0
# Kmbpg13pgk_a = 0.1 #0.063 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmadppgk_a = 0.33 #0.2 #0.4 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmpg3pgk_a = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
# Kmatppgk_a = 0.33 #0.4 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS


Vmaxpgk_n = 90.8064278816738 #100.0 #13. #5.5 # scaled in comparison of ratios with GAPDH Vmax #396.0
Keqpgk_na = 2600. #1830. #1310.0

Kmbpg13pgk_n = 0.1 #0.063 # 0.0022 # Berndt2018 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
Kmadppgk_n = 0.350477947537394 #0.3 #0.2 #0.4 #0.25 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
Kmpg3pgk_n = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
Kmatppgk_n = 0.38728441113536 #0.4 #0.4 #0.25 #0.42 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS

Vmaxpgk_a = 95.5247294642638 #100. #13. #13. #5.5 # scaled in comparison of ratios with GAPDH Vmax 396.0
#Keqpgk_a = 2600. #1830. # 1310.0
Kmbpg13pgk_a = 0.1 #0.063 # Km BPG13 = 0.063 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
Kmadppgk_a = 0.363316278693795 #0.3 #0.2 #0.4 #0.42  # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
Kmpg3pgk_a = 0.67  # Km PG3 = 0.67 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS
Kmatppgk_a = 0.431968764786817 #0.4 #0.4 #0.25 # Km f ATP = 0.42 mM, Km r ATP = 0.25 mM HARMESH K. SHARMA and MORTON ROTHSTEIN 1983 ALTERED BRAIN PHOSPHOGLYCERATE KINASE FROM AGING RATS

#####################################


# PGM
# Berndt2015

Vmaxpgm_n = 1.0002776306165402 # 0.12 #472.0 # scaled to PGK #14400.0
Keqpgm_n = 0.1814
Kmpg3pgm_n = 0.22
Kmpg2pgm_n = 0.28

Vmaxpgm_a = 1.0 #0.12 #472.0 # scaled to PGK #14400.0
Keqpgm_a = 0.1814
Kmpg3pgm_a = 0.22
Kmpg2pgm_a = 0.28


#####################################

# ENOLASE
# Berndt2015

# Vmaxenol_n = 1.8 #216000.0

# Keqenol_n = 0.7 #0.5 Berndt2015 # 1.7 Berndt2018
# Kmpg2enol_n = 0.05
# Km_pep_enol_n = 0.15

# Vmaxenol_a = 1.8 #216000.0

# Keqenol_a = 0.7 #0.5
# Kmpg2enol_a = 0.05
# Km_pep_enol_a = 0.15

Vmaxenol_n = 12.0 #1.8 #216000.0

Keqenol_n = 0.8 #0.7 #0.5 Berndt2015 # 1.7 Berndt2018
Kmpg2enol_n = 0.05
Km_pep_enol_n = 0.15

Vmaxenol_a = 12. #1.8 #216000.0
Keqenol_a = 0.8 #0.7 #0.5
Kmpg2enol_a = 0.05
Km_pep_enol_a = 0.15


#####################################

# Pyruvate Kinase PK
# Berndt2015,Jolivet2015

# Vmaxpk_n = 4. #5. #36.7 # Jolivet #23.76 # Berndt

# Km_pep_pk_n = 0.074
# Km_adp_pk_n = 0.25 #0.42
# Ki_ATP_pk_n = 1. #4.4 # 2.2 #

# Vmaxpk_a = 5. #10.0 #23.76 # 135.2 # Jolivet #23.76 # Berndt

# Km_pep_pk_a = 0.074
# Km_adp_pk_a = 0.25 #0.42
# Ki_ATP_pk_a = 1. #4.4 #2.2 #


# Vmaxpk_n = 4. #36.7 # Jolivet #23.76 # Berndt

# Km_pep_pk_n = 0.074
# Km_adp_pk_n = 0.42
# Ki_ATP_pk_n = 4.4 # 2.2 #

# Vmaxpk_a = 5.0 #23.76 # 135.2 # Jolivet #23.76 # Berndt

# Km_pep_pk_a = 0.074
# Km_adp_pk_a = 0.42
# Ki_ATP_pk_a = 4.4 #2.2 #


# Vmaxpk_n = 4. #36.7 # Jolivet #23.76 # Berndt

# Km_pep_pk_n = 0.08 #0.074
# Km_adp_pk_n = 0.5
# Ki_ATP_pk_n = 4.4 # 2.2 #

# Vmaxpk_a = 5.0 #23.76 # 135.2 # Jolivet #23.76 # Berndt

# Km_pep_pk_a = 0.08 #0.074
# Km_adp_pk_a = 0.5
# Ki_ATP_pk_a = 4.4 #2.2 #



# Vmaxpk_n = 4. #36.7 # Jolivet #23.76 # Berndt
# Km_pep_pk_n = 0.08 #0.074
# Km_adp_pk_n = 0.5
# Ki_ATP_pk_n = 4. #4.4 # 2.2 #

# Vmaxpk_a = 4.2 #23.76 # 135.2 # Jolivet #23.76 # Berndt
# Km_pep_pk_a = 0.08 #0.074
# Km_adp_pk_a = 0.5
# Ki_ATP_pk_a = 4. #4.4 #2.2 #



# Vmaxpk_n = 20.  #36.7 # Jolivet #4. #36.7 # Jolivet #23.76 # Berndt
# # Km_pep_pk_n = 0.08 #0.074
# # Km_adp_pk_n = 0.5
# # Ki_ATP_pk_n = 4. #4.4 # 2.2 #

# Vmaxpk_a = 20. #135.2 # Jolivet #4.2 #23.76 # 135.2 # Jolivet #23.76 # Berndt
# # Km_pep_pk_a = 0.08 #0.074
# # Km_adp_pk_a = 0.5
# # Ki_ATP_pk_a = 4. #4.4 #2.2 #


# Vmaxpk_n = 20.  #36.7 # Jolivet #4. #36.7 # Jolivet #23.76 # Berndt
# Km_pep_pk_n = 0.13 #0.08 #0.074
# Km_adp_pk_n = 2.3 #0.5
# Ki_ATP_pk_n = 0.9 #4.4 # 2.2 #

# Vmaxpk_a = 20. #135.2 # Jolivet #4.2 #23.76 # 135.2 # Jolivet #23.76 # Berndt
# Km_pep_pk_a = 0.13  #0.08 #0.074
# Km_adp_pk_a = 2.3 #0.5
# Ki_ATP_pk_a = 0.9 #4.4 #2.2 #

# Vmaxpk_n = 1200. #20.  #36.7 # Jolivet #4. #36.7 # Jolivet #23.76 # Berndt
# Km_pep_pk_n = 0.1 #0.5 #0.074
# Km_adp_pk_n = 0.28  #0.5
# Ki_ATP_pk_n = 1. #2.2 #

# Vmaxpk_a = 1200. #20. #135.2 # Jolivet #4.2 #23.76 # 135.2 # Jolivet #23.76 # Berndt
# Km_pep_pk_a = 0.1  #0.5 #0.074
# Km_adp_pk_a = 0.28 #0.5
# Ki_ATP_pk_a = 1. #2.2 #

# From DOI 10.1074/jbc.M508490200 
# with K: 
# VmaxPK = 1245.83 mM/s #299 umol/min/mg protein  -> 1000*299*0.25/60 = 1245.83 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
# KmPEP = 0.13 mM
# KmADP = 2.3

# VmaxPK = 1187.5 mM/s #285 umol/min/mg protein  -> 1000*285*0.25/60 = 1187.5 mM/s # *0.25 from Nazaret2009 doi:10.1016/j.jtbi.2008.09.037 
# KmPEP = 0.033 mM or 0.065 mM
# KmADP = 0.28 mM or 0.25 mM

# Keq=38900. # DOI 10.1074/jbc.M111422200



# KersonGarfinkelMildvan1966 PMID: 6022859
# fwd
# Km_pep_pk_n = 0.075 #0.1 #0.5 #0.074
# Km_adp_pk_n = 0.2 #0.28  #0.5
# Ki_ATP_pk_n = 0.34 # 0.13 in text
# KiPyrPK_n = 0.01
# # rev
# KmPyrPK_n = 13.  # 10. table
# KmATPPK_n = 0.01  # 0.86 table
# adj KersonGarfinkelMildvan1966 PMID: 6022859
# psiPK_n(PEP_n,ADP_n,ATP_n,Pyr_n) = Vmaxpk_n*PEP_n*ADP_n / ( 1 + Km_adp_pk_n/ADP_n + (Km_pep_pk_n/PEP_n)*(1 + Pyr_n/KiPyrPK_n) +  (Km_adp_pk_n/ADP_n)*(Km_pep_pk_n/PEP_n)*(1 + Pyr_n/KiPyrPK_n) )



#Mulukutla2015: ordered bi bi mechanism 

# VmfPK_n = 1245.83  # 1.0002776306165402*1245.83 # From DOI 10.1074/jbc.M508490200 #1.68  # 1200. #80. #
# VmrPK_n = 0.032 #1245.83/38900 # where Keq=38900. from DOI 10.1074/jbc.M111422200 #0.004 

# Km_pep_PK_n = 0.1 # approx avg PMID: 6022859 and DOI 10.1074/jbc.M508490200
# Km_adp_PK_n = 0.24  # approx avg PMID: 6022859 and DOI 10.1074/jbc.M508490200

# Km_pyr_PK_n = 11.5 #15.0 #12.0 # 11.5 # PMID: 6022859 avg text+table
# Km_Mg_atp_PK_n =  0.01 # text # 0.86 table  # PMID: 6022859 

# L_pk_n = 0.389  
# K_atpPK_n = 0.34 # 0.13 in text #3.39 

# VmfPK_a = 1275. #1269.1963812533515 # 1245.83  # From DOI 10.1074/jbc.M508490200 #1.68  
# VmrPK_a = 0.032 #1245.83/38900 # where Keq=38900. from DOI 10.1074/jbc.M111422200 #0.004 

# Km_pep_PK_a = 0.1 # approx avg PMID: 6022859 and DOI 10.1074/jbc.M508490200
# Km_adp_PK_a = 0.24  # approx avg PMID: 6022859 and DOI 10.1074/jbc.M508490200

# Km_pyr_PK_a =  10.363500082938396 # 11.5 # PMID: 6022859 avg text+table
# Km_Mg_atp_PK_a =  0.01 # text # 0.86 table  # PMID: 6022859 

# L_pk_a = 0.389  
# K_atpPK_a = 0.34 # 0.13 in text #3.39 

# Berndt2015 + opt
Vmaxpk_n = 7.48969758041304 #7.843117959791727 #23.76

Km_pep_pk_n = 0.074 # 0.1 # approx avg PMID: 6022859 and DOI 10.1074/jbc.M508490200
Km_adp_pk_n = 0.562062013433244 #0.42 # 0.24  # approx avg PMID: 6022859 and DOI 10.1074/jbc.M508490200
Ki_ATP_pk_n = 1.88029571803631 #2.2 #4.4

Vmaxpk_a = 8.88447689329492 #8.789368481405708 #23.76

Km_pep_pk_a = 0.074
Km_adp_pk_a = 0.510068084908044 #0.42
Ki_ATP_pk_a = 1.76791009718649 #2.2 #4.4


