# bHB art-cap

C_bHB_a = 0.3 #0.18

# JbHBTrArtCap1 = (2*(C_bHB_a - bHB_b0)/eto_b)*(global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(100-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(100-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))) 

################################

# MCT bhb b

# MCT1_bHB_b(bHB_b,bHB_ecs), bHB_b ⇒ ∅
# (eto_b/eto_ecs)*(bHB_b,bHB_ecs), ∅ ⇒ bHB_ecs  # eto_b/eto_ecs = 0.0275

VmaxMCTbhb_b = 0.35559905936750225 #0.29 #mM/s # Neves 2012  
KmMCT1_bHB_b = 12.5 #mM # Perez-Escuredo 2016 Table 2 
#KmMCT1_bHB_a = 6.03 #mM - astrocyte #Jay's alt from Achanta and Rae 2017, with subref Tildon 1994 

# MCT1_bHB_b1 = VmaxMCTbhb_b*(bHB_b0/(bHB_b0 + KmMCT1_bHB_b) - bHB_ecs0/(bHB_ecs0 + KmMCT1_bHB_b)) 


################################

# MCT bhb a: MCT1_bHB_a(bHB_ecs,bHB_a), bHB_ecs ⇒ bHB_a

VmaxMCTbhb_a = 0.29 #mM/s # Neves 2012  
KmMCT1_bHB_a = 6.03 #mM astrocytes Achanta and Rae 2017, with subref Tildon 1994  #12.5 #mM # Perez-Escuredo 2016 Table 2 
#KmMCT1_bHB_a = 6.03 #mM #Jay's alt from Achanta and Rae 2017, with subref Tildon 1994 

## Jay comment: besides MCT1, there is also the pyruvate transporter for bHB: betaOHB is poor substrate for the 
## mitochondrial pyruvate carrier (KM = 5.6 mM; although its metabolite acetoacetate is carried with 
## reasonable affinity (0.56 mM)"" Achanta and Rae 2017, Halestrap  1978 Biochem J. 
## Not sure if we'll ever need this mechanism for bHB, maybe for AcAc


# MCT1_bHB_a1 = VmaxMCTbhb_a*(bHB_ecs0/(bHB_ecs0 + KmMCT1_bHB_a) - bHB_a0/(bHB_a0 + KmMCT1_bHB_a)) 


################################

# MCT bhb n: MCT2_bHB_n(bHB_ecs,bHB_n), bHB_ecs ⇒ bHB_n

VmaxMCTbhb_n = 0.29032861767118245 #0.29 #mM/s # Neves 2012  
KmMCT2_bHB_n = 1.2 #mM # Perez-Escuredo 2016 https://doi.org/10.1016/j.bbamcr.2016.03.013 Table 2 # Ronowska 2018

# MCT2_bHB_n1 = VmaxMCTbhb_n*(bHB_ecs0/(bHB_ecs0 + KmMCT2_bHB_n) - bHB_n0/(bHB_n0 + KmMCT2_bHB_n)) 



################################

# bHBDH_n(NAD_n,bHB_n), NAD_n + bHB_n  ⇒  AcAc_n + NADH_n


# beta-hydroxybutyrate dehydrogenase is exclusively located in mitochondria (DOI: 10.1002/iub.2367)
# but here I was using cytosolic conc just as approximation


Vmax_bHBDH_f_n = 0.05139599967731973 #0.532 #0.532 # 0.9063430122643024 #1.2 # 0.532 # Nielsen 1973, check units
Vmax_bHBDH_r_n = 0.012848999919329931 #0.665 #0.665 # 0.5358569280430245 #0.4 # 0.665 # Nielsen 1973, check units
#Keq_bHBDH_n  0.033 # Nielsen 1973
                                          
# Km_AcAc_BHBD_n = 0.39 # 0.26784852162653866 #0.39 mM   # brain mito Dombrowski 1977                                                     
# Km_NADH_BHBD_n = 0.05 #mM   # brain mito Dombrowski 1977                                                   
# Km_NAD_B_HBD_n = 0.39 # 0.17691521110119537 # 0.39 mM # brain mito, similar in brain and liver Dombrowski 1977                                         
# Km_betaHB_BHBD_n = 0.45 # 0.4574168440640211 #1.98 #mM #brain mito, similar in brain and liver Dombrowski 1977                                        
# Ki_NADH_BHBD_r_n = 0.3 # 0.38858801604311016 # 0.22 # mM in direction AcAc to bHB # brain mito Dombrowski 1977  
# Ki_NAD_B_HBD_f_n = 0.45 # 1.0508306759981987 # 1.5 # mM Nielsen 1973         

Km_AcAc_BHBD_n = 0.2 #0.45 #0.39 # 0.26784852162653866 #0.39 mM   # brain mito Dombrowski 1977                                                     
Km_NADH_BHBD_n = 0.05 #mM   # brain mito Dombrowski 1977                                                   
Km_NAD_B_HBD_n = 0.2 #0.39 # 0.17691521110119537 # 0.39 mM # brain mito, similar in brain and liver Dombrowski 1977                                                      
Km_betaHB_BHBD_n = 0.45 # 0.4574168440640211 #1.98 #mM #brain mito, similar in brain and liver Dombrowski 1977                                                    
Ki_NADH_BHBD_r_n = 0.39 #0.3 # 0.38858801604311016 # 0.22 # mM in direction AcAc to bHB # brain mito Dombrowski 1977  
Ki_NAD_B_HBD_f_n = 0.45 # 1.0508306759981987 # 1.5 # mM Nielsen 1973         


################################

# SCOT_n: SCOT_n(SUCCOAmito_n,AcAc_n,AcAcCoA_n,SUCmito_n), SUCCOAmito_n + AcAc_n  ⇒  AcAcCoA_n + SUCmito_n

VmaxfSCOT_n = 2.6842893020795207 #1.68 # 1.67 #mM/s # calc from kcat from WhiteJencks and conc of SCOT (gene OXCT1) in mol atlas                                          
VmaxrSCOT_n = 0.08787851881807955 #0.055 #0.08 # 0.08 adj for conc #0.12  # mM/s # calc from kcat from WhiteJencks and conc of SCOT (gene OXCT1) in mol atlas                                                                                                                                           

Km_AcAc_SCOT_n = 0.25 #0.16 #mM    # or 0.2 mM in Hersh (ref from White Jencks 1975)                                                                    
Km_AcAcCoA_SCOT_n = 0.19 #mM   # or 0.93 mM in Hersh (ref from White Jencks 1975)                                                                     
Km_SUC_SCOT_n = 23.0 #mM   # Km_Succinate_SCOT_n or 36 mM in Hersh (ref from White Jencks 1975)                                                                     
Km_SucCoA_SCOT_n = 4.2 #mM                                                                        
Ki_AcAc_SCOT_n = 0.78 #0.29 #mM  # or 0.78 mM in Hersh (ref from White Jencks 1975)                                                                       
Ki_AcAcCoA_SCOT_n = 0.033 #mM     # or 0.17 mM in Hersh (ref from White Jencks 1975)                                                                   
Ki_SUC_SCOT_n = 0.54 #mM    # Ki_Succinate_SCOT_n or 1.0 mM in Hersh (ref from White Jencks 1975)                                                                    
Ki_SucCoA_SCOT_n= 2.4 #mM   # or 1.9 mM in Hersh (ref from White Jencks 1975)                                                                     


# combo f r as SCOT_n() for TD chem pot bigg: OCOAT1m


################################

# thiolase (gene ACAT1) 

# thiolase_n(CoAmito_n,AcAcCoA_n), CoAmito_n + AcAcCoA_n ⇒ 2AcCoAmito_n 
#bigg: 2.0 accoa_m → aacoa_m + coa_m

# also can adjust for brain thiolase activity is 8% of heart thiolase activity # table 1 from Yang 1987

# Vmax_thiolase_f_n = 2.13 #0.5 # approx #2.13 # mM/s # Gilbert: kf is 360 sec-1 # 360*exp(1.78)*1e-3

# #Vmax_thiolase_r_n = 2.37e-5 #Gilbert: kr is 4e-3 sec-1  # 4e-3*exp(1.78)*1e-3

# #Km_AcCoA_thiolase_r_n = 0.237 #mM (rat liver) # Huth 1982 
# #Km_AcAcCoA_thiolase_r_n = 0.035 #mM #35 uM (peroxisomes), 80 uM (cytosol) # Antonenkov2000 FEBS                              

# Km_AcAcCoA_thiolase_f_n = 0.021 #0.021 #mM #21.0 uM # Gilbert thiolase II heart (ketone bodies utilization) # 0.01 mM Huth 1982 # 9 uM (peroxisomes), 16 uM (cytosol) # Antonenkov2000 FEBS
# #Km_AcCoA_thiolase_f_n = 0.0085 #mM #8.5 uM # Gilbert thiolase II heart (ketone bodies utilization) # 0.09 mM (rat liver) # Huth 1982                      
# Km_CoA_thiolase_f_n = 0.056 #0.056 #mM #56.0 uM    # Gilbert thiolase II heart (ketone bodies utilization) #0.025 mM Huth 1982 # 8 uM (peroxisomes), 20 uM (cytosol) # Antonenkov2000 FEBS
# Ki_CoA_thiolase_f_n =  0.05 #mM Huth 1982
# #Ki_AcAcCoA_thiolase_r_n = 0.0016 #mM Huth 1982

Vmax_thiolase_f_n = 1.0 #2.13 #0.5 # approx #2.13 # mM/s # Gilbert: kf is 360 sec-1 # 360*exp(1.78)*1e-3

#Vmax_thiolase_r_n = 2.37e-5 #Gilbert: kr is 4e-3 sec-1  # 4e-3*exp(1.78)*1e-3

#Km_AcCoA_thiolase_r_n = 0.237 #mM (rat liver) # Huth 1982 
#Km_AcAcCoA_thiolase_r_n = 0.035 #mM #35 uM (peroxisomes), 80 uM (cytosol) # Antonenkov2000 FEBS                              

Km_AcAcCoA_thiolase_f_n = 0.021 #0.021 #mM #21.0 uM # Gilbert thiolase II heart (ketone bodies utilization) # 0.01 mM Huth 1982 # 9 uM (peroxisomes), 16 uM (cytosol) # Antonenkov2000 FEBS
#Km_AcCoA_thiolase_f_n = 0.0085 #mM #8.5 uM # Gilbert thiolase II heart (ketone bodies utilization) # 0.09 mM (rat liver) # Huth 1982                      
Km_CoA_thiolase_f_n = 0.056 #0.056 #mM #56.0 uM    # Gilbert thiolase II heart (ketone bodies utilization) #0.025 mM Huth 1982 # 8 uM (peroxisomes), 20 uM (cytosol) # Antonenkov2000 FEBS
Ki_CoA_thiolase_f_n =  0.05 #mM Huth 1982
#Ki_AcAcCoA_thiolase_r_n = 0.0016 #mM Huth 1982



# eq type based on eq18 https://www.qmul.ac.uk/sbcs/iubmb/kinetics/ek4t6.html#p52 

# thiolase_n1 = Vmax_thiolase_f_n*CoAmito_n0*AcAcCoA_n0 / ( Ki_CoA_thiolase_f_n * Km_AcAcCoA_thiolase_f_n + Km_AcAcCoA_thiolase_f_n*CoAmito_n0 +     Km_CoA_thiolase_f_n*AcAcCoA_n0 + CoAmito_n0*AcAcCoA_n0)


