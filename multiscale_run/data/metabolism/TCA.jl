# PDH

# r01 PDH

# “lower pyruvate dehydrogenase activity in astrocytes” (Glucose utilization: still in the synapse Stoessl)

#Berndt2012, Zhang2018
# VmaxPDHCmito_n = 0.719 #0.85 #1. #10000. 
# # AmaxCaMitoPDH_n = 1.7  
# # Km_a_CaMito_PDH_n = 0.001  
# KmPyrMitoPDH_n = 0.045 #0.09     # 0.002 - 0.25 Brenda relevant species
# KmNADmitoPDH_na = 0.046 #0.036           # 0.022 - 0.07 Brenda relevant species
# KmCoAmitoPDH_n = 0.008 #0.0047   # 0.004-0.0761 Brenda relevant species


VmaxPDHCmito_n = 2.67222326259307 #2.8 # 2.9164794313209406 # 2.9164794313209406 # 2.916873266724163 # 2.798321254695664 # 2.15 #2.17 #2.18good #2.1 - slightly lower #2.2 - almost, slightly higher #2.0 -a bit too low #0.9999567804941919*2.4 -a bit too high # 2.4 almost #3.0 #3.112889016753674 #4.705913926611023 # 3.0 #2.0 #1.0 #0.1 # 
#4.705913926611023 # 3.112889016753674  # 4.575 #4.57 slightly down #4.56 down # 4.58 slightly up #4.6 # 4.7 #3.04 #4.25 #4.22 #0.825 #0.719 #0.85 #1. #10000. 

# AmaxCaMitoPDH_n = 1.7  
# Km_a_CaMito_PDH_n = 0.001  

KmPyrMitoPDH_n = 0.068 #0.045 #0.09 #0.045 #0.09    # 0.002 - 0.25 Brenda relevant species
KmNADmitoPDH_na = 0.036 #0.046 #0.036           # 0.022 - 0.07 Brenda relevant species

KmCoAmitoPDH_n = 0.007684422893475319 # 0.0047 #0.008 #0.0047   # 0.004-0.0761 Brenda relevant species



VmaxPDHCmito_a = 2.79810789599674 #2.8 # 3.06 # 3.06  # from opt: 2.6496686498982145 # 3.06 #3.0575 #3.065 #3.1 #3.1987080122030234 #3.112889016753674 #2.6496686498982145  # before opt: 4.0  #0.677 #0.8 #0.5 #1. #10000. #4.0  #0.677 #0.8 #0.5 #1. #10000. 
# AmaxCaMitoPDH_a = 1.7  
# Km_a_CaMito_PDH_a = 0.001  
KmPyrMitoPDH_a = 0.068 # 0.02 # from opt: 0.012933056341060813 0.02 # 0.0189 #0.03 #0.012933056341060813 # before opt: 0.04 #0.0252         # 0.002 - 0.25 Brenda relevant species
#KmNADmitoPDH_na = 0.046 #0.036 #0.035          # 0.022 - 0.07 Brenda relevant species
KmCoAmitoPDH_a = 0.0047 # 0.0149         # 0.004-0.0761 Brenda relevant species


# # psiPDH_n1 = VmaxPDHCmito_n*(1.0+AmaxCaMitoPDH_n*CaMito_n0/(CaMito_n0 + Km_a_CaMito_PDH_n)) * (PYRmito_n0/(PYRmito_n0+KmPyrMitoPDH_n)) * (NADmito_n0/(NADmito_n0 + KmNADmitoPDH_na)) * (CoAmito_n0/(CoAmito_n0 + KmCoAmitoPDH_n)) 
# # psiPDH_a1 = VmaxPDHCmito_a*(1.0+AmaxCaMitoPDH_a*CaMito_a0/(CaMito_a0 + Km_a_CaMito_PDH_a)) * (PYRmito_a0/(PYRmito_a0+KmPyrMitoPDH_a)) * (NADmito_a0/(NADmito_a0 + KmNADmitoPDH_na)) * (CoAmito_a0/(CoAmito_a0 + KmCoAmitoPDH_a))

# psiPDH_n1 = VmaxPDHCmito_n* (PYRmito_n0/(PYRmito_n0+KmPyrMitoPDH_n)) * (NADmito_n0/(NADmito_n0 + KmNADmitoPDH_na)) * (CoAmito_n0/(CoAmito_n0 + KmCoAmitoPDH_n)) 
# psiPDH_a1 = VmaxPDHCmito_a* (PYRmito_a0/(PYRmito_a0+KmPyrMitoPDH_a)) * (NADmito_a0/(NADmito_a0 + KmNADmitoPDH_na)) * (CoAmito_a0/(CoAmito_a0 + KmCoAmitoPDH_a))

# println(psiPDH_n1)
# println(psiPDH_a1)


##############################################

# r02: Citrate synthase: Oxa + AcCoA -> Cit # Berndt 2015

# VmaxCSmito_n =  0.2 #0.4833  # 0.17 #0.165 #0.2 #0.4833 #0.2 #adj #1280.0 #0.116 # doi:10.1046/j.1471-4159.2003.01871.x 0.116 umol/min/mg  
# KmOxaMito_n = 0.005   # 0.005 mM brain https://pubmed.ncbi.nlm.nih.gov/4201777/
# KiCitMito_n = 1.0 #3.7 #1.6 # 3.7  # 0.037  
# KmAcCoAmito_n = 0.0048   # 0.0048 mM brain https://pubmed.ncbi.nlm.nih.gov/4201777/  
# KiCoA_n = 0.02 # 0.025 # Berndt 2012,2015 #0.067 #0.00025  

VmaxCSmito_n = 0.4689427553439143 #0.4645 # 0.4722162450277844 # 0.4724918447819282 # 0.5112293698278352 # 0.4645 # 0.57 #0.55 #0.65 #0.8 #0.4968890850459*1.0 #0.2 #0.4833 # 1.0 #0.2 #0.4833   # 0.17 #0.165 #0.2 #0.4833 #0.2 #adj #1280.0 #0.116 # doi:10.1046/j.1471-4159.2003.01871.x 0.116 umol/min/mg  

KmOxaMito_n = 0.005   # 0.005 mM brain https://pubmed.ncbi.nlm.nih.gov/4201777/
KiCitMito_n = 1.0 #3.7 #1.6 # 3.7  # 0.037  
KmAcCoAmito_n = 0.0048   # 0.0048 mM brain https://pubmed.ncbi.nlm.nih.gov/4201777/  
KiCoA_n = 0.02 # 0.025 # Berndt 2012,2015 #0.067 #0.00025  


VmaxCSmito_a = 0.7181459358580017 #0.27867260457961346*1.9413118399434741 # 1.9413118399434741 # < from opt a 2.0 #0.2 #0.4833  # 0.17 #0.165 # 0.2 # 0.4833 #0.116 umol/min/mg   # doi:10.1046/j.1471-4159.2003.01871.x  #0.2 #adj  #1280.0 

KmOxaMito_a = 0.004 #0.005   # 0.005 mM brain https://pubmed.ncbi.nlm.nih.gov/4201777/ 
KiCitMito_a = 1.0  #3.7 #1.6 # Berndt2018 #3.7
KmAcCoAmito_a = 0.0048   # 0.0048 mM brain https://pubmed.ncbi.nlm.nih.gov/4201777/  
KiCoA_a = 0.02 # 0.025 # Berndt 2012,2015


# psiCS_n1 = VmaxCSmito_n*(OXAmito_n0/(OXAmito_n0 + KmOxaMito_n*(1.0 + CITmito_n0/KiCitMito_n))) * (AcCoAmito_n0/(AcCoAmito_n0 + KmAcCoAmito_n*(1.0+CoAmito_n0/KiCoA_n)))  
# psiCS_a1 = VmaxCSmito_a*(OXAmito_a0/(OXAmito_a0 + KmOxaMito_a*(1.0 + CITmito_a0/KiCitMito_a))) * (AcCoAmito_a0/(AcCoAmito_a0 + KmAcCoAmito_a*(1.0+CoAmito_a0/KiCoA_a)))

# println(psiCS_n1)
# println(psiCS_a1)

##############################################

# r03: aconitase, CITmito_a  ⇒ ISOCITmito_a

# VmaxAco_n = 2. #5. #10. #5. #100. #200. #16000.0
# KeqAco_na = 0.11 #0.067 #Berndt2015 #0.1 #Berndt2018 
# KmCitAco_n = 0.48 # GUARRIERO-BOBYLEVA 1973   
# KmIsoCitAco_n = 0.12 # GUARRIERO-BOBYLEVA 1973  

VmaxAco_n = 25.611147830094392 # 25.782707422203963 # 25.797754993950043 # 29. # 23.433 #29. #28.0 #2.35 # 2.3 #2. #5. #10. #5. #100. #200. #16000.0 #29. #28.0 #2.35 # 2.3 #2. #5. #10. #5. #100. #200. #16000.0
KeqAco_na = 0.11 #0.067 #Berndt2015 #0.1 #Berndt2018 
KmCitAco_n = 0.48 # GUARRIERO-BOBYLEVA 1973   
KmIsoCitAco_n = 0.12 # GUARRIERO-BOBYLEVA 1973  


# Cit <-> IsoCit # Berndt 2015
VmaxAco_a = 9.438075110105698 #26.99 # 29. #2. #5. # 10. #5. # 100. #200. #16000.0 
#KeqAco_na = 0.11 #0.067 #0.1 #Berndt2018  
KmCitAco_a = 0.48 # GUARRIERO-BOBYLEVA 1973  
KmIsoCitAco_a = 0.12 # GUARRIERO-BOBYLEVA 1973  


# VmaxAco_n = 2. #16000.0
# KeqAco_na = 0.067 # par[511]
# KmCitAco_n = 0.48 # GUARRIERO-BOBYLEVA 1973   
# KmIsoCitAco_n = 0.12 # GUARRIERO-BOBYLEVA 1973  


# # Cit <-> IsoCit # Berndt 2015
# VmaxAco_a = 2. #16000.0 
# KeqAco_na = 0.067
# KmCitAco_a = 0.48 # GUARRIERO-BOBYLEVA 1973  
# KmIsoCitAco_a = 0.12 # GUARRIERO-BOBYLEVA 1973  



# #psiACO_n1 = VmaxAco_n*(CITmito_n0-ISOCITmito_n0/KeqAco_na) / (KmCitAco_n + CITmito_n0 + KmCitAco_n*ISOCITmito_n/KmIsoCitAco_n)  
# psiACO_n1 = VmaxAco_n*(CITmito_n0-ISOCITmito_n0/KeqAco_na) / (1.0+CITmito_n0/KmCitAco_n + ISOCITmito_n0/KmIsoCitAco_n)
# psiACO_a1 = VmaxAco_a*(CITmito_a0-ISOCITmito_a0/KeqAco_na) / (1.0+CITmito_a0/KmCitAco_a + ISOCITmito_a0/KmIsoCitAco_a)

# println(psiACO_n1)
# println(psiACO_a1)

##############################################

# r04: IDH: ISOCITmito_a + NADmito_a  ⇒ AKGmito_a + NADHmito_a #  ISOCITmito_a  ⇒ AKGmito_a #

# key rate-limiting enzyme in TCA !!!

# a
#CO2_mito_a = 1.2 # par[349]

# NAD-dependent isocitrate dehydrogenase (IDH); ISOCITmito + NADmito  ⇒ AKGmito + NADHmito
#VmaxIDH_a = 64.0 #### CHECK IT!!!! UNITS!!!!
#nIsoCitmitoIDH_a = 1.9
#KmIsoCitmito1IDH_a = 0.11
#KmIsoCitmito2IDH_a = 0.06
#KaCaidhIDH_a = 0.0074
#nCaIdhIDH_a = 2.0
#KmNADmitoIDH_a = 0.091
#KiNADHmitoIDH_a = 0.041

######### IDH_a
#Wu2007
# VmaxfIDH_a = 425.0 #mM/s
# Km_a_NAD_IDH_a = 0.074
# Km_b_ISOCIT_IDH_a = 0.183 # 0.059, 0.055, 0.183
# nH_IDH_a = 2.5 # 2.67 #3.0
# KibIDH_a = 0.0238 # 0.00766, 0.0238
# KiqIDH_a = 0.029
# Ki_atp_IDH_a = 0.091
# Ka_adp_IDH_a = 0.05
# Keq0_IDH_a = 3.5e-16 #3.5*(10^(-16))
#Pakg_IDH_a = 1.0
#Keq_IDH_a = 30.5 #-Mulukutla2015  #Keq0_IDH_a*(Pakg_IDH_a*Pnadh_IDH_a*Pco2tot_IDH_a)/(C_H_mitomatr*Pnad_IDH_a*Picit_IDH_a)


# Berndt + doi:10.1016/j.bbapap.2008.07.001 + Mulukutla 

# # NAD-dependent isocitrate dehydrogenase (IDH); ISOCITmito + NADmito  ⇒ AKGmito + NADHmito
# VmaxIDH_n = 0.1249913793 # VmaxCSmito_n/3.8666666667   #3.8666666667 is calc from rates ratio of doi:10.1046/j.1471-4159.2003.01871.x  Table 2 #4.25 #64.0
# nIsoCitmito_n = 1.9   # 3 
# KmIsoCitmito1_n = 0.2 #0.11
# KmIsoCitmito2_n = 0.06
# Km_a_Ca_idh_n = 0.074 #0.0074
# nCaIdh_n = 2.0
# KmNADmito_n = 0.091 
# KiNADHmito_n = 0.0041 # 4.1 uM PMID: 7359132 #0.05 #0.041
# #KmIsoCitmito = KmIsoCitmito1/(1+(CaMito/KaCaidh)^nIsoCitmito) + KmIsoCitmito2


# VmaxIDH_a = 0.2708146552 # VmaxCSmito_a/1.7846153846 # 1.7846153846 is calc from rates ratio of doi:10.1046/j.1471-4159.2003.01871.x  Table 2 Specific activities of LDH, CS, and ICDH in total lysates of neural cell cultures
# nIsoCitmito_a = 1.9  # 3 
# KmIsoCitmito1_a = 0.2 #0.11
# KmIsoCitmito2_a = 0.06
# Km_a_Ca_idh_a = 0.074 #0.0074
# nCaIdh_a = 2.0
# KmNADmito_a = 0.091 
# KiNADHmito_a = 0.05 #0.041

# psiIDH_n1 = VmaxIDH_n*((ISOCITmito_n0^nIsoCitmito_n) / (ISOCITmito_n0^nIsoCitmito_n + (KmIsoCitmito1_n/(1+(CaMito_n0/Km_a_Ca_idh_n)^nCaIdh_n) + KmIsoCitmito2_n)^nIsoCitmito_n) ) * (NADmito_n0/(NADmito_n0 + KmNADmito_n*(1+NADHmito_n0/KiNADHmito_n)))
# psiIDH_a1 = VmaxIDH_a*((ISOCITmito_a0^nIsoCitmito_a) / (ISOCITmito_a0^nIsoCitmito_a + (KmIsoCitmito1_a/(1+(CaMito_a0/Km_a_Ca_idh_a)^nCaIdh_a) + KmIsoCitmito2_a)^nIsoCitmito_a) ) * (NADmito_a0/(NADmito_a0 + KmNADmito_a*(1+NADHmito_a0/KiNADHmito_a)))

######  with eq from doi:10.1016/j.bbapap.2008.07.001

VmaxIDH_n = 1194.49868869589 #1109.856835123883 # 1117.155367232157 # 1117.8073730616716 # 1033.507542656321 # 1090. #1080. #1078.5 #1076. #950. #910. #905.055  #1076.0 # 891.34 #500. #80.0 #5.2 #2.5 #0.1249913793 # n/a = 0.030/0.065 = 0.4615384615384615 ratio of doi:10.1046/j.1471-4159.2003.01871.x  #1076.0 # 891.34 #500. #80.0 #5.2 #2.5 #0.1249913793 # n/a = 0.030/0.065 = 0.4615384615384615 ratio of doi:10.1046/j.1471-4159.2003.01871.x  Table 2 #4.25 #64.0  #  1000*69.2*0.25/60 = 288.33 where 69.2 from doi:10.1016/j.bbapap.2008.07.001  #
KiNADmito_na = 0.14 #0.0776 # Kia
KmIsoCitIDHm_n = 0.15 #0.35 #0.45 #0.1489 # Kmb
KmNADmito_na = 0.5033 # Kma
KiNADHmito_na = 0.00475 ## Kiq
nIDH = 3 #1.9 #3


VmaxIDH_a = 332.052098136637 #344.93939819582073 # 1087.7 # 1076.0 # 891.34 #500. #80.0 #5.42 #0.2708146552 # n/a = 0.030/0.065 = 0.4615384615384615 ratio of doi:10.1046/j.1471-4159.2003.01871.x Table 2 Specific activities of LDH, CS, and ICDH in total lysates of neural cell cultures
#KiNADmito_na = 0.14 #0.0776 # Kia
KmIsoCitIDHm_a = 0.15 # 0.35 #0.45 # 0.1489 # Kmb
#KmNADmito_na = 0.5033 # Kma
#KiNADHmito_na = 0.00475 ## Kiq
nIDH = 3 #1.9 #3

#psiIDH_a(ISOCITmito_a,NADmito_a,NADHmito_a) = VmaxIDH_a*(NADmito_a/KiNADmito_a)*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH ) /  (1.0 + NADmito_a/KiNADmito_a + (KmNADmito_a/KiNADmito_a)*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH) + NADHmito_a/KiNADHmito_a + (NADmito_a/KiNADmito_a)*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH) +   ((KmNADmito_a*NADHmito_a)/(KiNADmito_a*KiNADHmito_a))*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH) )

# psiIDH_n(ISOCITmito_n,NADmito_n,NADHmito_n) = VmaxIDH_n*((NADmito_n*ISOCITmito_n)/(KiNADmito_n*KmIsoCitIDHm_n)) / 
# (1.0 + NADmito_n/KiNADmito_n + (KmNADmito_n*ISOCITmito_n)/(KiNADmito_n*KmIsoCitIDHm_n) + NADHmito_n/KiNADHmito_n + (NADmito_n*ISOCITmito_n)/(KiNADmito_n*KmIsoCitIDHm_n) +    
#     (KmNADmito_n*ISOCITmito_n*NADHmito_n)/(KiNADmito_n*KmIsoCitIDHm_n*KiNADHmito_n) )

#psiIDH_n(ISOCITmito_n,NADmito_n,NADHmito_n) = VmaxIDH_n*(NADmito_n/KiNADmito_n)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH ) /  (1.0 + NADmito_n/KiNADmito_n + (KmNADmito_n/KiNADmito_n)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) + NADHmito_n/KiNADHmito_n + (NADmito_n/KiNADmito_n)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) +   ((KmNADmito_n*NADHmito_n)/(KiNADmito_n*KiNADHmito_n))*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) )

# add ADP-activation
# psiIDH_n(ISOCITmito_n,NADmito_n,NADHmito_n) = VmaxIDH_n*(NADmito_n/KiNADmito_n)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH ) / 
# (1.0 + NADmito_n/KiNADmito_n + (KmNADmito_n/KiNADmito_n)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) + NADHmito_n/KiNADHmito_n + (NADmito_n/KiNADmito_n)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) +    
#    ((KmNADmito_n*NADHmito_n)/(KiNADmito_n*KiNADHmito_n))*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) )

#
# psiIDH_n(ISOCITmito_n,CaMito_n,NADmito_n,NADHmito_n) = VmaxIDH_n*((ISOCITmito_n^nIsoCitmito_n) / (ISOCITmito_n^nIsoCitmito_n + (KmIsoCitmito1_n/(1+(CaMito_n/Km_a_Ca_idh_n)^nCaIdh_n) + KmIsoCitmito2_n)^nIsoCitmito_n) ) * (NADmito_n/(NADmito_n + KmNADmito_n*(1+NADHmito_n/KiNADHmito_n)))
# psiIDH_a(ISOCITmito_a,CaMito_a,NADmito_a,NADHmito_a) = VmaxIDH_a*((ISOCITmito_a^nIsoCitmito_a) / (ISOCITmito_a^nIsoCitmito_a + (KmIsoCitmito1_a/(1+(CaMito_a/Km_a_Ca_idh_a)^nCaIdh_a) + KmIsoCitmito2_a)^nIsoCitmito_a) ) * (NADmito_a/(NADmito_a + KmNADmito_a*(1+NADHmito_a/KiNADHmito_a)))

# psiIDH_n1 = VmaxIDH_n*(NADmito_n0/KiNADmito_n)*((ISOCITmito_n0/KmIsoCitIDHm_n)^nIDH ) /  (1.0 + NADmito_n0/KiNADmito_n + (KmNADmito_n/KiNADmito_n)*((ISOCITmito_n0/KmIsoCitIDHm_n)^nIDH) + NADHmito_n0/KiNADHmito_n + (NADmito_n0/KiNADmito_n)*((ISOCITmito_n0/KmIsoCitIDHm_n)^nIDH) +   ((KmNADmito_n*NADHmito_n0)/(KiNADmito_n*KiNADHmito_n))*((ISOCITmito_n0/KmIsoCitIDHm_n)^nIDH) )
# psiIDH_a1 = VmaxIDH_a*(NADmito_a0/KiNADmito_a)*((ISOCITmito_a0/KmIsoCitIDHm_a)^nIDH ) /  (1.0 + NADmito_a0/KiNADmito_a + (KmNADmito_a/KiNADmito_a)*((ISOCITmito_a0/KmIsoCitIDHm_a)^nIDH) + NADHmito_a0/KiNADHmito_a + (NADmito_a0/KiNADmito_a)*((ISOCITmito_a0/KmIsoCitIDHm_a)^nIDH) +   ((KmNADmito_a*NADHmito_a0)/(KiNADmito_a*KiNADHmito_a))*((ISOCITmito_a0/KmIsoCitIDHm_a)^nIDH) )

# println(psiIDH_n1)
# println(psiIDH_a1)

###################################################

# r05: aKGDH: AKGmito_a + NADmito_a + CoAmito_a ⇒ SUCCOAmito_a + NADHmito_a # AKGmito_a + CoAmito_a ⇒ SUCCOAmito_a  

# VmaxKGDH_n = 10. #1.344  
# #KiCa2KGDH_n = 0.0012 #1.2e-05  
# Km1KGDHKGDH_n = 0.67 #0.025  # 0.67 # doi:10.1098/rstb.2005.1764
# Km2KGDHKGDH_n = 0.16 #0.0016 
# #Ki_KG_Ca_KGDH_n = 1.33e-7 #1.33e-09  
# KiNADHKGDHKGDH_n = 0.0045 #4.5e-05  # 0.0045 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
# KmNADkgdhKGDH_n = 0.021  #0.00021 # 0.021  # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
# Km_CoA_kgdhKGDH_n = 0.0027 #1.3e-05  # 0.0027 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
# Ki_SucCoA_kgdhKGDH_n = 0.0039 # in brain! Luder 1990 # 0.0069 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase

# aKG dehydrogenase (KGDH); AKGmito + NADmito + CoAmito ⇒ SUCCOAmito + NADHmito      ### Berndt2012
### but in future for more details and regulation can consider Detailed kinetics and regulation of mammalian 2- oxoglutarate dehydrogenase 
# 2011 Feng Qi1,2, Ranjan K Pradhan1, Ranjan K Dash1 and Daniel A Beard

# VmaxKGDH_a = 10. #1.344  
# #KiCa2KGDH_a = 0.0012 # McCormack 1979, Mogilevskaya 2006
# Km1KGDHKGDH_a = 0.67 #2.5 # Berndt 2012 # 0.67 # doi:10.1098/rstb.2005.1764
# Km2KGDHKGDH_a = 0.16 # McCormack 1979
# #Ki_KG_Ca_KGDH_a = 1.33e-7 # calculated from McCormack 1979
# KiNADHKGDHKGDH_a = 0.0045 # Smith 1974 # 0.0045 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
# KmNADkgdhKGDH_a = 0.021 # Smith 1974 # 0.021  # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
# Km_CoA_kgdhKGDH_a = 0.0027 #0.0013 # Smith 1974 # 0.0027 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
# KiSucCoAkgdhKGDH_a = 0.0039 # in brain! Luder 1990  # # 0.0069 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase

# # psiKGDH_n(CaMito_n,AKGmito_n,NADHmito_n,NADmito_n,CoAmito_n,SUCCOAmito_n) = VmaxKGDH_n*(1-CaMito_n/(CaMito_n+KiCa2KGDH_n))*(AKGmito_n/(AKGmito_n+(Km1KGDHKGDH_n/(1+CaMito_n/Ki_KG_Ca_KGDH_n)+Km2KGDHKGDH_n)*(1+NADHmito_n/KiNADHKGDHKGDH_n))) *   (NADmito_n/(NADmito_n+KmNADkgdhKGDH_n*(1+NADHmito_n/KiNADHKGDHKGDH_n))) * (CoAmito_n/(CoAmito_n + Km_CoA_kgdhKGDH_n*(1+SUCCOAmito_n/Ki_SucCoA_kgdhKGDH_n)))     
# # psiKGDH_a(CaMito_a,AKGmito_a,NADHmito_a,NADmito_a,CoAmito_a,SUCCOAmito_a) = VmaxKGDH_a*(1-CaMito_a/(CaMito_a+KiCa2KGDH_a))*(AKGmito_a/(AKGmito_a+(Km1KGDHKGDH_a/(1+CaMito_a/Ki_KG_Ca_KGDH_a)+Km2KGDHKGDH_a)*(1+NADHmito_a/KiNADHKGDHKGDH_a))) *  (NADmito_a/(NADmito_a+KmNADkgdhKGDH_a*(1+NADHmito_a/KiNADHKGDHKGDH_a))) * (CoAmito_a/(CoAmito_a + Km_CoA_kgdhKGDH_a*(1+SUCCOAmito_a/KiSucCoAkgdhKGDH_a)))

# psiKGDH_n(CaMito_n,AKGmito_n,NADHmito_n,NADmito_n,CoAmito_n,SUCCOAmito_n) = VmaxKGDH_n*(1-CaMito_n/(CaMito_n+KiCa2KGDH_n))*(AKGmito_n/(AKGmito_n+(Km1KGDHKGDH_n/(1+CaMito_n/Ki_KG_Ca_KGDH_n)+Km2KGDHKGDH_n)*(1+NADHmito_n/KiNADHKGDHKGDH_n))) *   (NADmito_n/(NADmito_n+KmNADkgdhKGDH_n*(1+NADHmito_n/KiNADHKGDHKGDH_n))) * (CoAmito_n/(CoAmito_n + Km_CoA_kgdhKGDH_n*(1+SUCCOAmito_n/Ki_SucCoA_kgdhKGDH_n)))     
# psiKGDH_a(CaMito_a,AKGmito_a,NADHmito_a,NADmito_a,CoAmito_a,SUCCOAmito_a) = VmaxKGDH_a*(1-CaMito_a/(CaMito_a+KiCa2KGDH_a))*(AKGmito_a/(AKGmito_a+(Km1KGDHKGDH_a/(1+CaMito_a/Ki_KG_Ca_KGDH_a)+Km2KGDHKGDH_a)*(1+NADHmito_a/KiNADHKGDHKGDH_a))) *  (NADmito_a/(NADmito_a+KmNADkgdhKGDH_a*(1+NADHmito_a/KiNADHKGDHKGDH_a))) * (CoAmito_a/(CoAmito_a + Km_CoA_kgdhKGDH_a*(1+SUCCOAmito_a/KiSucCoAkgdhKGDH_a)))

# psiKGDH_n1 = VmaxKGDH_n*(AKGmito_n0/(AKGmito_n0+(Km1KGDHKGDH_n+Km2KGDHKGDH_n)*(1+NADHmito_n0/KiNADHKGDHKGDH_n))) *  (NADmito_n0/(NADmito_n0+KmNADkgdhKGDH_n*(1+NADHmito_n0/KiNADHKGDHKGDH_n))) * (CoAmito_n0/(CoAmito_n0 + Km_CoA_kgdhKGDH_n*(1+SUCCOAmito_n0/Ki_SucCoA_kgdhKGDH_n)))     
# psiKGDH_a1 = VmaxKGDH_a*(AKGmito_a0/(AKGmito_a0+(Km1KGDHKGDH_a+Km2KGDHKGDH_a)*(1+NADHmito_a0/KiNADHKGDHKGDH_a))) *  (NADmito_a0/(NADmito_a0+KmNADkgdhKGDH_a*(1+NADHmito_a0/KiNADHKGDHKGDH_a))) * (CoAmito_a0/(CoAmito_a0 + Km_CoA_kgdhKGDH_a*(1+SUCCOAmito_a0/KiSucCoAkgdhKGDH_a)))


# Mogilevskaya 

VmaxKGDH_n = 28.6907036435173 #28.91517141816992 # 28.932047191118823 # 29.791351583908167 #28.496668132945448 # 30.41167501394481 #30. #28. #27. #26. #25.524 #23.4 #30.0 #35.0 # 40.0 #50.415 #166.22 #10. #1.344   23.4 #30.0 #35.0 # 40.0 #50.415 #166.22 #10. #1.344  
KiADPmito_KGDH_n = 0.6 # 0.56 # 0.1 # both values are in Mogilevskaya, 0.56 is fitted, 0.1 is lit
KiATPmito_KGDH_n = 0.0108261270854904 #0.01 # 0.1 # both values are in Mogilevskaya, 0.01 is fitted, 0.1 is lit
Km1KGDHKGDH_n = 0.8 #0.67 #0.025  # 0.67 # doi:10.1098/rstb.2005.1764
Km_CoA_kgdhKGDH_n = 0.005 # 0.0027 #1.3e-05  # 0.0027 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
KmNADkgdhKGDH_na = 0.021  #0.00021 # 0.021  # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
KiCa2KGDH_n = 1.2e-05  # 0.0012 #1.2e-05  
KiNADHKGDHKGDH_na = 0.0045 #4.5e-05  # 0.0045 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
Ki_SucCoA_kgdhKGDH_n = 0.0039 # in brain! Luder 1990 # 0.0069 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase

VmaxKGDH_a = 7.74815022676304 #7.4236647140402 #18.792728253785338  # 23.86 #23.93  #24.0 #up #23.8 down  #24.256852208624576 up #23.4 #47.681 #166.22 #10. #1.344  
KiADPmito_KGDH_a = 0.5046016363070087 #KiADPmito_KGDH_a = 0.6 #0.57 #0.56 # 0.1 # both values are in Mogilevskaya, 0.56 is fitted, 0.1 is lit
KiATPmito_KGDH_a = 0.054580855381112 #0.053818392618842754 # 0.01 # 0.1 # both values are in Mogilevskaya, 0.01 is fitted, 0.1 is lit
Km1KGDHKGDH_a = 0.6287013099563603 # 0.8 #0.67 #0.025  # 0.67 # doi:10.1098/rstb.2005.1764
Km_CoA_kgdhKGDH_a = 0.005 #0.0027 #1.3e-05  # 0.0027 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase
KiCa2KGDH_a = 1.2e-05  #0.0012 #1.2e-05  
Ki_SucCoA_kgdhKGDH_a = 0.0039 # in brain! Luder 1990 # 0.0069 # SMITH1973 Regulation of Mitochondrial a=Ketoglutarate Metabolism by Product Inhibition at a-Ketoglutarate Dehydrogenase


# psiKGDH_n1 = 
# (VmaxKGDH_n*(1 + ADPmito_n0/KiADPmito_KGDH_n)*(AKGmito_n0/Km1KGDHKGDH_n)*(CoAmito_n0/Km_CoA_kgdhKGDH_n)*(NADmito_n0/KmNADkgdhKGDH_n) ) / 
# ( ( (CoAmito_n0/Km_CoA_kgdhKGDH_n)*(NADmito_n0/KmNADkgdhKGDH_n)*(AKGmito_n0/Km1KGDHKGDH_n + (1 + ATPmito_n0/KiATPmito_KGDH_n)/(1 + CaMito_n0/KiCa2KGDH_n)) ) +
#     ( (AKGmito_n0/Km1KGDHKGDH_n)*(CoAmito_n0/Km_CoA_kgdhKGDH_n + NADmito_n0/KmNADkgdhKGDH_n)*(1 + NADHmito_n0/KiNADHKGDHKGDH_n + SUCCOAmito_n0/Ki_SucCoA_kgdhKGDH_n) )  )

# psiKGDH_a1 = 
# (VmaxKGDH_a*(1 + ADPmito_a0/KiADPmito_KGDH_a)*(AKGmito_a0/Km1KGDHKGDH_a)*(CoAmito_a0/Km_CoA_kgdhKGDH_a)*(NADmito_a0/KmNADkgdhKGDH_a) ) / 
# ( ( (CoAmito_a0/Km_CoA_kgdhKGDH_a)*(NADmito_a0/KmNADkgdhKGDH_a)*(AKGmito_a0/Km1KGDHKGDH_a + (1 + ATPmito_a0/KiATPmito_KGDH_a)/(1 + CaMito_a0/KiCa2KGDH_a)) ) +
#     ( (AKGmito_a0/Km1KGDHKGDH_a)*(CoAmito_a0/Km_CoA_kgdhKGDH_a + NADmito_a0/KmNADkgdhKGDH_a)*(1 + NADHmito_a0/KiNADHKGDHKGDH_a + SUCCOAmito_a0/Ki_SucCoA_kgdhKGDH_a) )  )

# println(psiKGDH_n1)
# println(psiKGDH_a1)




###################################################


# r06: SCS: SUCCOAmito_n + ADPmito_n ⇒ SUCmito_n + ATPmito_n + CoAmito_n

VmaxSuccoaATPscs_n = 410.83664327655714 #409.916218893484 #433.58091004733 # 433.8339610456146 # 437.42815951086135 # 458.29727009401 # 476.615 # 400. #396. #395. #394.5 #394.21 #395. #400.0 #1.0067 #192.0 #400.0 #1.0067 #192.0 
AmaxPscs_n = 1.2 #  Berndt 2015
npscs_n = 3.0 # Berndt 2015 #2.5 #  
Keqsuccoascs_na = 3.8 # Berndt 2015
Km_succoa_scs_n = 0.02852096394334664 #0.041 # Berndt2012,2015 #0.086 #  0.024 # Mogilevskaya
Km_succ_scs_n = 0.836589467070621 #1.6 #PMID: 1869044 #0.49 # 
Km_pi_scs_na = 2.5 # 0.72,2.5 # Berndt 2012,2015
Km_coa_scs_n = 0.056  #PMID: 1869044 
Km_atpmito_scs_n = 0.0156620432157514 #0.017 # PMID: 1869044 # 0.72 # 
Km_ADPmito_scs_n = 0.25 #  Berndt 2012,2015


VmaxSuccoaATPscs_a = 182.496590618396 #171.24090767446916 # 360.0 #490.78642491794716 #400.0 #429.06807326391527 #367.77263422621314 #490.36351230161745 #576.4491390247175 #490.7864249179471 #370.0 #360.0 #415.0 # 410 down # 420 up #360.0 #400.0 #0.9236 #192.0  #360.0 #370.0 # 360.0 # 400.0 #0.9236 #192.0 
AmaxPscs_a = 1.2 #  Berndt 2015
npscs_a = 3.0 # Berndt 2015 #2.5 #  
#Keqsuccoascs_na = 3.8 # Berndt 2015
Km_succoa_scs_a = 0.041 # Berndt2012,2015 #0.086 #  0.024 # Mogilevskaya
Km_succ_scs_a = 1.6 #PMID: 1869044 #0.49 # 
#Km_pi_scs_na = 2.5 # 0.72,2.5 # Berndt 2012,2015
Km_coa_scs_a = 0.056  #PMID: 1869044 
Km_atpmito_scs_a = 0.0164177843454365 #0.017 # PMID: 1869044 # 0.72 # 
Km_ADPmito_scs_a = 0.25 #  Berndt 2012,2015

###################################################

# !!! J_DH from ETC instead

# # r07: SDH: SUCmito_n + Qmito_n  ⇒ FUMmito_n + QH2mito_n

# # kcat_SDH_n = 79274.6 #Vf_SDH_n 16.1389 # par[547]

# # #Mulukutla2015 
# # KiOXA_SDH_n = 0.0015 # par[544]
# # KaSUC_SDH_n = 0.45 # par[545]
# # KaFUM_SDH_n = 0.375 # par[546]
# # Keq_SDH_n = 1.21 # par[548]
# # KiSUC_SDH_n = 0.12 # par[549]
# # KmQ_SDH_n = 0.48 # par[550]
# # KmSuc_SDH_n = 0.467 # par[551]
# # KiFUM_SDH_n = 1.275 # par[552]
# # KmQH2_SDH_n = 0.00245 # par[553]
# # KmFUM_SDH_n = 1.2 # par[554]

# # psiSDH_n(SUCmito_n,Qmito_n,QH2mito_n,FUMmito_n,OXAmito_n) = kcat_SDH_n*concentration_enzyme_transporter_TCA_r07_SDH_n*( SUCmito_n*Qmito_n - QH2mito_n*FUMmito_n/Keq_SDH_n  ) / ( KiSUC_SDH_n*KmQ_SDH_n*((1.0 + OXAmito_n/KiOXA_SDH_n + SUCmito_n/KaSUC_SDH_n + FUMmito_n/KaFUM_SDH_n ) / (1.0 + SUCmito_n/KaSUC_SDH_n + FUMmito_n/KaFUM_SDH_n  ) )   +    KmQ_SDH_n*SUCmito_n + KmSuc_SDH_n*((1.0 + OXAmito_n/KiOXA_SDH_n + SUCmito_n/KaSUC_SDH_n + FUMmito_n/KaFUM_SDH_n ) / (1.0 + SUCmito_n/KaSUC_SDH_n + FUMmito_n/KaFUM_SDH_n  ) ) *Qmito_n + SUCmito_n*Qmito_n + KmSuc_SDH_n*Qmito_n*FUMmito_n/KiFUM_SDH_n  +    (KiSUC_SDH_n*KmQ_SDH_n/(KiFUM_SDH_n*KmQH2_SDH_n) )*( KmFUM_SDH_n*((1.0 + OXAmito_n/KiOXA_SDH_n + SUCmito_n/KaSUC_SDH_n + FUMmito_n/KaFUM_SDH_n ) / (1.0 + SUCmito_n/KaSUC_SDH_n + FUMmito_n/KaFUM_SDH_n  ) ) *QH2mito_n + KmQH2_SDH_n*FUMmito_n + KmFUM_SDH_n*SUCmito_n*QH2mito_n/KiSUC_SDH_n + QH2mito_n*FUMmito_n ) )        # eto_mito_n_scaled*


# # # Succinate dehydrogenase (SDH); SUCmito + Qmito  ⇒ FUMmito + QH2mito  # Mogilevskaya 2006  ## try eq from IvanChang for this reaction
# # #kfSDH_a = 10000.0
# # #Kesucsucmito_a = 0.01
# # #Kmqmito_a = 0.0003
# # #krSDH_a = 102.0
# # #Kefumfummito_a = 0.29
# # #Kmqh2mito_a = 0.0015
# # #Kmsucsdh_a = 0.13
# # #Kmfumsdh_a = 0.025
# # #@reaction_func VSDH(SDHmito,SUCmito,Qmito,FUMmito,QH2mito) = SDHmito*(kfSDH*(SUCmito/Kesucsucmito)*(Qmito/Kmqmito) - krSDH*(FUMmito/Kefumfummito)*(QH2mito/Kmqh2mito)) / (1+(SUCmito/Kesucsucmito)+(Qmito/Kmqmito)*(Kmsucsdh/Kesucsucmito)+(SUCmito/Kesucsucmito)*(Qmito/Kmqmito)+(FUMmito/Kefumfummito)+(QH2mito/Kmqh2mito)*(Kmfumsdh/Kefumfummito) + (FUMmito/Kefumfummito)*(QH2mito/Kmqh2mito))
# # ####################################################################################

# # # Succinate dehydrogenase (SDH); SUCmito + Qmito  ⇒ FUMmito + QH2mito  # IvanChang
# # #VmaxDHchang_a = 0.28
# # #KrDHchang_a = 0.100
# # #pDchang_a =0.8

# # #Mulukutla2015
# # kcat_SDH_a = 82609.7 #Vf_SDH_a 16.14 #58100.0/3600.0
# # Keq_SDH_a = 1.21
# # KmSuc_SDH_a = 0.467
# # KmQ_SDH_a = 0.48
# # KmQH2_SDH_a = 0.00245
# # KmFUM_SDH_a = 1.2
# # KiSUC_SDH_a = 0.12
# # KiFUM_SDH_a = 1.275
# # KiOXA_SDH_a = 0.0015
# # KaSUC_SDH_a = 0.45
# # KaFUM_SDH_a = 0.375

# # #psiSDH_a  VmaxDHchang_a*((NADmito_a/NADHmito_a)/(NADmito_a/NADHmito_a+KrDHchang_a))

# # #Mulukutla2015 
# # #alpha_SDH_a ((1.0 + OXAmito_a/KiOXA_SDH_a + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a ) / (1.0 + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a  ))
# # psiSDH_a(SUCmito_a,Qmito_a,QH2mito_a,FUMmito_a,OXAmito_a) = kcat_SDH_a*concentration_enzyme_transporter_TCA_r07_SDH_a*( SUCmito_a*Qmito_a - QH2mito_a*FUMmito_a/Keq_SDH_a  ) /     ( KiSUC_SDH_a*KmQ_SDH_a*((1.0 + OXAmito_a/KiOXA_SDH_a + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a ) / (1.0 + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a  ))  + KmQ_SDH_a*SUCmito_a + KmSuc_SDH_a*((1.0 + OXAmito_a/KiOXA_SDH_a + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a ) / (1.0 + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a  ))*Qmito_a + SUCmito_a*Qmito_a + KmSuc_SDH_a*Qmito_a*FUMmito_a/KiFUM_SDH_a  +    (KiSUC_SDH_a*KmQ_SDH_a/(KiFUM_SDH_a*KmQH2_SDH_a) )*( KmFUM_SDH_a*((1.0 + OXAmito_a/KiOXA_SDH_a + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a ) / (1.0 + SUCmito_a/KaSUC_SDH_a + FUMmito_a/KaFUM_SDH_a  ))*QH2mito_a + KmQH2_SDH_a*FUMmito_a + KmFUM_SDH_a*SUCmito_a*QH2mito_a/KiSUC_SDH_a + QH2mito_a*FUMmito_a   )   )


# # Berndt2012 
# # ZEYLEMAKER1969 for OXA inh 1-s2.0-0005274469902423-main.pdf KiOXA = 4 uM
# # http://dx.doi.org/10.1016/j.bbabio.2016.06.002 for BRAIN OXA inhibition of SDH KdOXA ~ 1e-8 M

# VmaxSDH_n = 12.1 # 5000. # approx Mogilevskaya #2.9167 #brain approx 1000*0.7*0.25/60 from Shaw1981 #2000. # approx from LW fig in MISHRA1993 # 0.33056 # approx by 1000*4.76*0.25/3600 from RAMESH REDDY 1989 # #20.0 # approx 
# Keq_SDH_n = 0.06 #2.547923498331544 # calc Berndt2012# 1.21 #0.06 # calc as in Berndt2015 fad #1.21 #0.0102 # approx 102/10000
# KmSuc_SDH_n = 1.6 #0.00769 #1.0 # approx Nakae1995 KmSDH solub and membrane bound are diff #0.00769  #1.6 # Berndt2012 # 2.5 # RAMESH REDDY 1989 # 0.26 #doi:10.1016/j.bbabio.2010.10.009  # 0.00769 approx from LW fig in MISHRA1993  # 0.048 from https://doi.org/10.1186/s13765-021-00626-1
# #KiMAL_SDH_n = 2.2 # Berndt2012
# KiOXASDH_n = 0.004 # ZEYLEMAKER1969

# VmaxSDH_a = 11.7 #5000. # approx Mogilevskaya  2.9167 #brain approx 1000*0.7*0.25/60 from Shaw1981  # 2000. # approx from LW fig in MISHRA1993 #0.33056 # approx by 1000*4.76*0.25/3600 from RAMESH REDDY 1989 #  #20.0 # approx 
# Keq_SDH_a = 0.06 #2.547923498331544 #1.21 #0.06 # calc as in Berndt2015 fad  #1.21 #0.0102 # approx 102/10000
# KmSuc_SDH_a = 1.6 #0.00769 #1.0 # approx Nakae1995 KmSDH solub and membrane bound are diff # 0.00769  #1.6 # Berndt2012 # 2.5 # RAMESH REDDY 1989  # 0.26 #doi:10.1016/j.bbabio.2010.10.009  # 0.00769 approx from LW fig in MISHRA1993  # 0.048 from https://doi.org/10.1186/s13765-021-00626-1
# #KiMAL_SDH_a = 2.2 # Berndt2012
# KiOXASDH_a = 0.004 # ZEYLEMAKER1969


# # #psiSDH_n1 = VmaxSDH_n*( SUCmito_n0*Qmito_n0 - (QH2mito_n0*FUMmito_n0)/Keq_SDH_n  ) / (SUCmito_n0 + KmSuc_SDH_n*(1 + MALmito_n0/KiMAL_SDH_n) )
# # #psiSDH_a1 = VmaxSDH_a*( SUCmito_a0*Qmito_a0 - (QH2mito_a0*FUMmito_a0)/Keq_SDH_a  ) / (SUCmito_a0 + KmSuc_SDH_a*(1 + MALmito_a0/KiMAL_SDH_a) )

# # psiSDH_n1 = VmaxSDH_n*( SUCmito_n0*Qmito_n0 - (QH2mito_n0*FUMmito_n0)/Keq_SDH_n  ) / (SUCmito_n0 + KmSuc_SDH_n*(1 + OXAmito_n0/KiOXASDH_n) )
# # psiSDH_a1 = VmaxSDH_a*( SUCmito_a0*Qmito_a0 - (QH2mito_a0*FUMmito_a0)/Keq_SDH_a  ) / (SUCmito_a0 + KmSuc_SDH_a*(1 + OXAmito_a0/KiOXASDH_a) )

# # println(psiSDH_n1)
# # println(psiSDH_a1)



###################################################


# r08: FUM: FUMmito_a  ⇒ MALmito_a

# kr = 1.4*kf # doi:10.1111/febs.14782 The FEBS Journal - 2019 - Ajalla Aleixo - Structural  biochemical and biophysical characterization of recombinant human.pdf

# Vmaxfum_n = 20. #0.8109 # 90721*2.27*1e-4=20.59 Mogilevskaya 
# Keqfummito_na = 5.5 #4.4 #4.3 #doi:10.1111/febs.14782 #4.4  Berndt 2015
# Km_fummito_n = 0.14 #0.2 # doi:10.1111/febs.14782 #0.14  Berndt 2015
# Km_malmito_n = 0.3 #1.4 #doi:10.1111/febs.14782 #0.3  Berndt 2015


Vmaxfum_n = 129.92082588239924 # 129.9966516254743 # 131.89863784638874 # 132.91320856708984 # 133. #130. #139. #132. #140. #155. #180. #130. #260. #0.8109 # 90721*2.27*1e-4=20.59 Mogilevskaya 

Keqfummito_na = 6.0 #5.6 #5.5 #4.4 #4.3 #doi:10.1111/febs.14782 #4.4  Berndt 2015
Km_fummito_n = 0.15102517071248805 #0.14 #0.2 # doi:10.1111/febs.14782 #0.14  Berndt 2015
Km_malmito_n = 0.3 #1.4 #doi:10.1111/febs.14782 #0.3  Berndt 2015


# Fumarase (FUM); FUMmito  ⇒  MALmito  based on Berndt 2015
Vmaxfum_a = 42.24027032956515 #231.87304452250146 #260. # 20. #1.496  # 90721*2.27*1e-4=20.59 Mogilevskaya 
Km_fummito_a = 0.14 #0.2 # doi:10.1111/febs.14782 #0.14  Berndt 2015
Km_malmito_a = 0.3 #1.4 #doi:10.1111/febs.14782 #0.3  Berndt 2015


# psiFUM_n1 = Vmaxfum_n*(FUMmito_n0 - MALmito_n0/Keqfummito_n)/(1.0+FUMmito_n0/Km_fummito_n+MALmito_n0/Km_malmito_n)   
# psiFUM_a1 = Vmaxfum_a*(FUMmito_a0 - MALmito_a0/Keqfummito_a)/(1.0+FUMmito_a0/Km_fummito_a+MALmito_a0/Km_malmito_a)

# println(psiFUM_n1)
# println(psiFUM_a1)


###################################################


# r09: MDH: MALmito_a + NADmito_a ⇒ OXAmito_a + NADHmito_a # MALmito_a  ⇒ OXAmito_a #

# VmaxMDHmito_n = 3261.5989 #20. #10.  # approx #32000.0 #Berndt #0.53 
# Keqmdhmito_na = 0.022 #0.02 #0.02 #0.01 #4e-5 #1e-3 #1.2e-3 #Berndt2012 or 1e-4 Berndt2015  #0.000402 # 
# Km_mal_mdh_n = 0.4 #0.33 #0.145 #Berndt #0.167 #0.0145  # 0.33 Berndt2018
# Km_nad_mdh_na = 0.06 #Berndt #0.056 #0.006 
# Km_oxa_mdh_n = 0.1 #0.08 #0.017 #0.04 #Bernstein1978 #0.017 #Berndt #0.055 #0.0017  
# Km_nadh_mdh_na = 0.044 #Berndt #0.026 #0.0044  

VmaxMDHmito_n = 390.370655175952 #417.5043146364773 # 417.7479828450139 # 418.7013772682505 # 450.58427655768554 #450. # 445. #390. #475. #800.0 #70.07614130634326 #3261.5989 #20. #10.  # approx #32000.0 #Berndt #0.53 

Keqmdhmito_na = 0.025 #0.022 #0.02 #0.02 #0.01 #4e-5 #1e-3 #1.2e-3 #Berndt2012 or 1e-4 Berndt2015  #0.000402 # 
Km_mal_mdh_n = 0.4 #0.33 #0.145 #Berndt #0.167 #0.0145  # 0.33 Berndt2018
Km_nad_mdh_na = 0.06 #Berndt #0.056 #0.006 
Km_oxa_mdh_n = 0.1 #0.08 #0.017 #0.04 #Bernstein1978 #0.017 #Berndt #0.055 #0.0017  
Km_nadh_mdh_na = 0.044 #Berndt #0.026 #0.0044  



# Malate dehydrogenase; MALmito + NADmito ⇒ OXAmito + NADHmito  based on Berndt 2015
VmaxMDHmito_a = 107.964870005644 #118.4814599996424 # 343.03071799750387 #300.0 #343.03071799750387 #800.0 #53.57 #20. # approx #32000.0 #Berndt 
#Keqmdhmito_na = 0.022 #0.02 #0.01  #4e-5 #1e-3 #1.2e-3 #Berndt  0.0001
Km_mal_mdh_a = 0.4 #0.33 #0.145 #Berndt #0.0145 #0.145  # 0.33 Berndt2018
#Km_nad_mdh_na = 0.06 #Berndt #0.006 #0.06
Km_oxa_mdh_a = 0.1 #0.08 #0.017 #0.04 #Bernstein1978 0.017 #Berndt #0.0017 #0.017
#Km_nadh_mdh_na = 0.044 #Berndt #0.0044 #0.044



# psiMDH_n1 = VmaxMDHmito_n*(MALmito_n0*NADmito_n0-OXAmito_n0*NADHmito_n0/Keqmdhmito_n) / ((1.0+MALmito_n0/Km_mal_mdh_n)*(1.0+NADmito_n0/Km_nad_mdh_n)+(1.0+OXAmito_n0/Km_oxa_mdh_n)*(1.0+NADHmito_n0/Km_nadh_mdh_n))   # eto_mito_n_scaled*
# psiMDH_a1 = VmaxMDHmito_a*(MALmito_a0*NADmito_a0-OXAmito_a0*NADHmito_a0/Keqmdhmito_a) / ((1.0+MALmito_a0/Km_mal_mdh_a)*(1+NADmito_a0/Km_nad_mdh_a)+(1+OXAmito_a0/Km_oxa_mdh_a)*(1+NADHmito_a0/Km_nadh_mdh_a))     


# println(psiMDH_n1)
# println(psiMDH_a1)

