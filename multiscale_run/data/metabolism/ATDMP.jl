# Jolivet2015 ATP-ADP
qAK  = 0.92
vATPasesn   = 0.0 #0.0033489929892437376 #0.003348829040159755 #0.010188268272998515 # 0.0 # 0.5*0.1388 #0.0832;     % mmol/L/sec
vATPasesg   = 0.0 # 0.5*0.1351 #0.1061;     % mmol/L/sec
ATDPtot_n = 1.4449961078157665 #1.4482032555861983
ATDPtot_a = 1.3434724532826217 #1.3452032555861981

# ADK
# check DiNuzzo PMID: 19888285 and ref there adenylate kinase (Rapoport et al, 1976)
# Winter2017, Lambeth 2002, JayGliaExpand

#psiADK_n(ATP_n,AMP_n,ADP_n), ATP_n + AMP_n + MET_h_c_n ⇒ 2ADP_n
    
Vmaxfadk_n = 1.0 #14.67  #kcat_fadk_n = 14018.6 #Vmaxfadk 14.67  

KmADPADK_n = 0.35 # KADPADK_n
KmATPADK_n = 0.27 # KATPADK_n
KmAMPADK_n = 0.32 # KAMPADK_n
KeqADK_n = 2.21  

# psiADK_n1 = ((Vmaxfadk_n*(ATP_n0*AMP_n0)/(KmATPADK_n*KmAMPADK_n) - (((Vmaxfadk_n*(KmADPADK_n^2)) / (KmATPADK_n*KmAMPADK_n*KeqADK_n))*(((ADP_n0)^2)/(KmADPADK_n^2)))) / (1 + ATP_n0/KmATPADK_n + AMP_n0/KmAMPADK_n + (ATP_n0*AMP_n0)/(KmATPADK_n*KmAMPADK_n) + (2*ADP_n0)/KmADPADK_n + ((ADP_n0)^2)/(KmADPADK_n^2)))   


############################################

# ADK 
# check DiNuzzo PMID: 19888285 and ref there adenylate kinase (Rapoport et al, 1976)
# Winter2017, Lambeth 2002, JayGliaExpand

#psiADK(ATP_a,AMP_a,ADP_a), ATP_a + AMP_a + MET_h_c_a ⇒ 2ADP_a
    
Vmaxfadk_a = 14.67  #kcat_fadk_a = 14018.6 #Vmaxfadk 14.67 

KmADPADK_a = 0.35 # KADPADK
KmATPADK_a = 0.27 # KATPADK
KmAMPADK_a = 0.32 # KAMPADK
KeqADK_a = 2.21 


# psiADK_a1 = ((Vmaxfadk_a*(ATP_a0*AMP_a0)/(KmATPADK_a*KmAMPADK_a) - (((Vmaxfadk_a*(KmADPADK_a^2)) / (KmATPADK_a*KmAMPADK_a*KeqADK_a))*(((ADP_a0)^2)/(KmADPADK_a^2)))) / (1 + ATP_a0/KmATPADK_a + AMP_a0/KmAMPADK_a + (ATP_a0*AMP_a0)/(KmATPADK_a*KmAMPADK_a) + (2*ADP_a0)/KmADPADK_a + ((ADP_a0)^2)/(KmADPADK_a^2)))    


############################################

# AC and NEneuromod

# # ATP_a ⇒  cAMP_a + PPi_a + MET_h_c_a
    
# # From J. Biol. Chem.-1997-Dessauer-27787-95.pdf

# VmaxfAC_a = 3.0 #kcat_fAC_a = 109151.4 #VmaxfAC 3.0 #30.0  
# KmACATP_a = 0.34 # KACATP
# KicAMPAC_a = 2.3 

# VmaxrAC_a = 0.1 #kcat_rAC_a = 3638.4 #VmaxrAC_a  0.1 #1.0 
# KmpiAC_a = 0.31 #0.12 # KpiAC
# KmcAMPAC_a = 2.3 # KcAMPAC

# VmaxfAC_a = 0.3 #3.0 #kcat_fAC_a = 109151.4 #VmaxfAC 3.0 #30.0  
# KmACATP_a = 0.4 # 0.34 # KACATP
# KicAMPAC_a = 0.045 #2.3 

# VmaxrAC_a = 0.01 #0.1 #kcat_rAC_a = 3638.4 #VmaxrAC_a  0.1 #1.0 
# KmpiAC_a = 0.01 #0.31 #0.12 # KpiAC
# KmcAMPAC_a = 2.0 #2.3 # KcAMPAC


#VmaxfAC_a = 0.3 #3.0 #kcat_fAC_a = 109151.4 #VmaxfAC 3.0 #30.0  
#KmACATP_a = 0.4 # 0.34 # KACATP
#KicAMPAC_a = 0.04533 #0.045 #0.0465 #2.3 

# 0.045 -> -2.2e-7
# 0.0453  -> -1.783216198374761e-8
# 0.0454 -> 4.93348351140038e-8
# 0.0456  -> 1.8329278509024457e-7
# 0.046 -> 4.4971786195144035e-7

#VmaxrAC_a = 0.01 #0.1 #kcat_rAC_a = 3638.4 #VmaxrAC_a  0.1 #1.0 
KmpiAC_a = 0.01 #0.31 #0.12 # KpiAC
#KmcAMPAC_a = 2.0 #2.3 # KcAMPAC

#
VmaxfAC_a = 0.003 #3.0 #kcat_fAC_a = 109151.4 #VmaxfAC 3.0 #30.0  
KmACATP_a = 0.8 #0.34 # 0.34 # KACATP
KicAMPAC_a = 0.045 #0.256 #0.04533 #0.045 #0.0465 #2.3 

VmaxrAC_a = 0.0001 #0.1 #kcat_rAC_a = 3638.4 #VmaxrAC_a  0.1 #1.0 
KmcAMPAC_a = 0.4 #2.7 #8.8 #2.3 #8.8 #2.7 #2.0 #2.3 # KcAMPAC


# Adenylate cyclase; ATP ⇒ cAMP + Pi #Ppi
#assuming competitive inh by cAMP in f dir
#psiAC(ATP_a,cAMP_a,Pi_a) = (a[11]/cai0_ca_ion)*( ((kcat_fAC_a*concentration_enzyme_transporter_AC_a*ATP_a/(KmACATP*(1+cAMP_a/KicAMPAC)) - kcat_rAC_a*concentration_enzyme_transporter_AC_a*cAMP_a*PPi_a/(KmpiAC*KmcAMPAC))/(1 + ATP_a/(KmACATP*(1+cAMP_a/KicAMPAC) )+ cAMP_a/KmcAMPAC + (cAMP_a*PPi_a)/(KmcAMPAC*KmpiAC) + PPi_a/KmpiAC))  )

# # without cai for optimiz; but with PPi_a instead of Pi_a for gem
# psiAC_a1 = ((VmaxfAC_a*ATP_a0/(KmACATP_a*(1+cAMP_a0/KicAMPAC_a)) - VmaxrAC_a*cAMP_a0*PPi_a0/(KmpiAC_a*KmcAMPAC_a))/(1 + ATP_a0/(KmACATP_a*(1+cAMP_a0/KicAMPAC_a) )+ cAMP_a0/KmcAMPAC_a + (cAMP_a0*PPi_a0)/(KmcAMPAC_a*KmpiAC_a) + PPi_a0/KmpiAC_a))

# #psiAC(NE_neuromod,ATP_a,cAMP_a,PPi_a)







