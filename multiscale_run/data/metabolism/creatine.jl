# Calvetti2018
# # psiCrKinase_n(Cr_n,ATP_n,ADP_n,PCr_n), Cr_n + ATP_n ⇒ ADP_n + PCr_n + MET_h_c_n

# V_Cr_n = 10.0 #16666.67  #kcat_Cr_n = 935138.5 #V_Cr_n  16666.67  
# mu_Cr_n = 0.01  
# Km_Cr_n = 495.0  

# V_PCr_n = 10.0 #16666.67   #kcat_PCr_n = 935138.5 #V_PCr_n  16666.67  
# mu_PCr_n = 100.0 
# Km_PCr_n = 528.0  

# # psiCrKinase_n1 = V_Cr_n  * (Cr_n0 / (Cr_n0 + Km_Cr_n)) * ( (ATP_n0/ADP_n0)/ ( mu_Cr_n + ATP_n0/ADP_n0 )) - ( V_PCr_n * (PCr_n0 / (PCr_n0 + Km_PCr_n)) * ( ADP_n0/ATP_n0 ) / ( mu_PCr_n + ADP_n0/ATP_n0 ) )


# #######################################################

# #psiCrKinase_a(Cr_a,ATP_a,ADP_a,PCr_a), Cr_a + ATP_a ⇒ ADP_a + PCr_a + MET_h_c_a
    
# V_Cr_a = 10.0 #16666.67 #kcat_Cr_a = 668433.5 #V_Cr_a  16666.67 
# mu_Cr_a = 0.01 # par[281]
# Km_Cr_a = 495.0 # par[282]

# V_PCr_a = 10.0 #16666.67 #kcat_PCr_a = 668433.5 #V_PCr_a  16666.67 
# mu_PCr_a = 100.0 
# Km_PCr_a = 528.0  

# # psiCrKinase_a1 = V_Cr_a * (Cr_a0 / (Cr_a0 + Km_Cr_a)) * ( (ATP_a0/ADP_a0) / ( mu_Cr_a + ATP_a0/ADP_a0 ))  -  (V_PCr_a * (PCr_a0 / (PCr_a0 + Km_PCr_a)) * ( ADP_a0/ATP_a0 ) / ( mu_PCr_a + ADP_a0/ATP_a0 ))    


# Jolivet2015 + opt par
Crtot = 10.0 #5.0 #10.0 in Jlv

kCKnps = 0.0214016075483191 #0.027854224286550353
#kCKnms = 0.0008979238618658251

kCKgps = 0.00178189983486018  #0.00178189983486018 #0.0003859785697371309
#kCKgms = 1.2499574733414296e-5

# creatine upd for better opt: 
# kCKnms -> KeqCKnpms*kCKnps
# kCKgms -> KeqCKgpms*kCKgps

# psiCrKinase_n1(PCr_n,ATP_n,ADP_n)  =  kCKnps*PCr_n*ADP_n - KeqCKnpms*kCKnps*(Crtot - PCr_n)*ATP_n # kCKnps*PCr_n*ADP_n - kCKnms*(Crtot - PCr_n)*ATP_n
# psiCrKinase_a1(PCr_a,ATP_a,ADP_a)  =  kCKgps*PCr_a*ADP_a - KeqCKgpms*kCKgps*(Crtot - PCr_a)*ATP_a # kCKgps*PCr_a*ADP_a - kCKgms*(Crtot - PCr_a)*ATP_a

KeqCKnpms = 0.04265840286184623 #0.0453633356986354 #0.03223654166881234 #0.0008979238618658251/0.027854224286550353 #kCKnms/kCKnps
KeqCKgpms = 0.034 #KeqCKgpms = 0.0375 #0.0283089638639555 #0.032384115889975654 #1.2499574733414296e-5/0.0003859785697371309 #kCKgms/kCKgps
# ini: 0.032236541668812340.032384115889975654

