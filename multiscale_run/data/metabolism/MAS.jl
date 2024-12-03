# AAT mito n (GOT mito n): psiAAT_n(ASPmito_n,AKGmito_n,OXAmito_n,GLUmito_n), AKGmito_n + ASPmito_n ⇒ OXAmito_n + GLUmito_n

#Mulukutla2015:  ping pong bi bi mechanism 

VfAAT_n = 1.2829135841551298  # after opt #0.1 #0.3 #32.  # Berndt2015 #0.13 

KiAKG_AAT_n = 1.9 # Recasens1980 #26.5 
KeqAAT_n = 0.14 # 0.147 in Berndt2015     # because AAT/GOT near-equilibrium DOI: 10.1002/iub.2367    

KmAKG_AAT_n = 1.3 # Magee1971  #0.344 # WILCOCK1973 #3.54 # Recasens1980 #3.22  
KmASP_AAT_n = 0.5 # Magee1971  #1.56 # WILCOCK1973 #0.58 # Recasens1980 #0.89  
KiGLU_AAT_n = 10.7  

KiASP_AAT_n = 263.0 # Recasens1980 #3.9  
KmOXA_AAT_n = 0.1 # Magee1971  # 0.1-0.25 Quang Khai HUYNH 1980 #0.088  
KmGLU_AAT_n = 3.5 # Magee1971  #16.66 # 10.0-25.0 Quang Khai HUYNH 1980 #32.5  


#AAT/GOT was psiMAAT_n     
#alpha_AAT_n = (1.0 + AKGmito_n/KiAKG_AAT_n)


#before 27 may2020 psiAAT_n = VfAAT_n*(ASPmito_n*AKGmito_n - OXAmito_n*GLUmito_n/KeqAAT_n) /  ( KmAKG_AAT_n*ASPmito_n +  KmASP_AAT_n*alpha_AAT_n*AKGmito_n + ASPmito_n*AKGmito_n + KmASP_AAT_n*AKGmito_n*GLUmito_n/KiGLU_AAT_n + (  KiASP_AAT_n*KmAKG_AAT_n/(KmOXA_AAT_n*KiGLU_AAT_n)  )*  ( KmGLU_AAT_n*ASPmito_n*OXAmito_n/KiASP_AAT_n + OXAmito_n*GLUmito_n +  KmGLU_AAT_n*alpha_AAT_n*OXAmito_n + KmOXA_AAT_n*GLUmito_n )  )

#psiAAT_n  = VmaxmitoAAT_n*(ASPmito_n*AKGmito_n-OXAmito_n*GLUmito_n/KeqmitoAAT_n) # it was psiMAAT_n

 
# GOT2 and MDH2, enzymes that are usually assumed to be close to equilibrium  DOI: 10.1002/iub.2367


# psiAAT_n1 = VfAAT_n*(ASPmito_n0*AKGmito_n0 - OXAmito_n0*GLUmito_n0/KeqAAT_n) /  ( KmAKG_AAT_n*ASPmito_n0 +  KmASP_AAT_n*(1.0 + AKGmito_n0/KiAKG_AAT_n)*AKGmito_n0 + ASPmito_n0*AKGmito_n0 + KmASP_AAT_n*AKGmito_n0*GLUmito_n0/KiGLU_AAT_n + (  KiASP_AAT_n*KmAKG_AAT_n/(KmOXA_AAT_n*KiGLU_AAT_n)  )*  ( KmGLU_AAT_n*ASPmito_n0*OXAmito_n0/KiASP_AAT_n + OXAmito_n0*GLUmito_n0 +  KmGLU_AAT_n*(1.0 + AKGmito_n0/KiAKG_AAT_n)*OXAmito_n0 + KmOXA_AAT_n*GLUmito_n0 )  )


###############################

# cMDH n: psicMDH_n(MAL_n,NAD_n,OXA_n,NADH_n), MAL_n + NAD_n ⇒ OXA_n + NADH_n 
# MAS_r01_cMDH_n

VmaxcMDH_n = 11.927187547739674 # after opt #10. #0.16 # 

Keqcmdh_n = 3.15e-5   # because MDH near-equilibrium DOI: 10.1002/iub.2367 #0.000402 # 
Kmmalcmdh_n = 0.77 # Berndt2015 #0.35 #  
Kmnadcmdh_n = 0.05 # Berndt2015 
Kmoxacmdh_n = 0.04 # Berndt2015
Kmnadhcmdh_n = 0.05 # Berndt2015

# # GOT2 and MDH2, enzymes that are usually assumed to be close to equilibrium  DOI: 10.1002/iub.2367


###############################

# psiCAAT_n(ASP_n,AKG_n,OXA_n,GLU_n), ASP_n + AKG_n ⇒ OXA_n + GLU_n
# MAS_r02_cAAT_n

VfCAAT_n = 0.6248605383756786 #0.023594697230144117 # after opt #0.3 #32. 

KiAKG_CAAT_n = 1. #17.0 # Recasens1980 #0.73 #26.5  # 1-3  KRISTA1972
KeqCAAT_n = 0.358 #0.36    # because AAT/GOT near-equilibrium DOI: 10.1002/iub.2367 #2.5 #1.56  

KmAKG_CAAT_n = 0.15 # Magee1971 #0.085 # WILCOCK1973 #0.54 # Recasens1980 # 0.06-0.1 Quang Khai HUYNH 1980 # 0.17 KRISTA1972 #3.22  

KmASP_CAAT_n = 6.7 # Magee1971 #1.55 # WILCOCK1973 #1.13  Recasens1980 # 1.81-2.5 Quang Khai HUYNH 1980 # 2.0  KRISTA1972  #0.89 

KiGLU_CAAT_n =  10.7 

KiASP_CAAT_n = 21.0 # Recasens1980 #3.9  

KmOXA_CAAT_n = 0.11 # Magee1971 #0.5 #0.33-0.5  Quang Khai HUYNH 1980 #0.088 

KmGLU_CAAT_n = 5.0 # Magee1971  #12.5  #Quang Khai HUYNH 1980 #32.5 

#psiCAAT_n = VmaxcAAT_n*(ASP_n*AKG_n-OXA_n*GLU_n/KeqcAAT_n) #changed to below 27may2020
#alpha_CAAT_n = (1.0 + AKG_n/KiAKG_CAAT_n)


###############################

# AGC (aralar,citrin) n: psiAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n,MitoMembrPotent_n),  ASPmito_n + GLU_n ⇒ GLUmito_n + ASP_n

# MAS_r03_AGC_n

# Berndt2015 

Vmaxagc_n = 0.0001830479496983698  # after opt #0.0069 
Km_aspmito_agc_n = 0.05  
Km_glu_agc_n = 5.6 
Km_asp_agc_n = 0.05  
Km_glumito_agc_n = 2.8 

# Aspartate/glutamate carrier [64] Berndt 2015  
# Asp_mito + glu_cyt + h_cyt ↔ Asp_cyt + glu_mito + h_mito
#Vmaxagc_n = 3200.0
#Keqagc_n =  0.968 # from NEDERGAARD 1991: 7.01/7.24 # hcyt/hext  # was 1.737 in Berndt 2015 - mistake in Berndt2015 ??  # Hcyt_n/Hext #
#Vmm_n = -0.14 #-140.0 mV
#Km_aspmito_agc_n = 0.05
#Km_glu_agc_n = 2.8
#Km_asp_agc_n = 0.05
#@reaction_func VAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n) = Vmaxagc_n*(ASPmito_n*GLU_n - ASP_n*GLUmito_n/ exp(-Vmm_n)^(F/(R*T))*Keqagc_n ) / ((ASPmito_n+Km_aspmito_agc_n)*(GLU_n+Km_glu_agc_n) + (ASP_n+Km_asp_agc_n)*(GLUmito_n+Km_glu_agc_n))

# From DOI: 10.1002/iub.2367:
# cytosolic Ca stimulation of the aspartate–glutamate transporter 
# It is possible that this calcium stimulation plays a physiological role in heart and brain

# psiAGC_n1 = Vmaxagc_n*(ASPmito_n0*GLU_n0 - ASP_n0*GLUmito_n0 / ((exp(MitoMembrPotent_n0)^(F/(R*T))) *  (C_H_cyt_n/C_H_mito_n)) ) / ((ASPmito_n0+Km_aspmito_agc_n)*(GLU_n0+Km_glu_agc_n) + (ASP_n0+Km_asp_agc_n)*(GLUmito_n0+Km_glumito_agc_n))      


###############################

# MAKGC n: psiMAKGC_n(MAL_n,AKGmito_n,MALmito_n,AKG_n), AKGmito_n + MAL_n ⇒ MALmito_n + AKG_n 

# MAS_r04_MAKGC_n

Vmaxmakgc_n = 0.000262718660265385  # after opt # 0.0005 #0.0004267 #

#KeqMakgc_n = 2.28  
Km_mal_mkgc_n = 0.4  
Km_akgmito_mkgc_n = 0.2  
Km_malmito_mkgc_n = 0.71  
Km_akg_mkgc_n = 0.1  

# psiMAKGC_n1 = Vmaxmakgc_n*( MAL_n0*AKGmito_n0 - MALmito_n0*AKG_n0) / ((MAL_n0+Km_mal_mkgc_n)*(AKGmito_n0+Km_akgmito_mkgc_n)+(MALmito_n0+Km_malmito_mkgc_n)*(AKG_n0+Km_akg_mkgc_n))     
