# PYRCARB pyruvate carboxylase:  PYRmito_a + ATPmito_a ⇒ OXAmito_a + ADPmito_a

VmPYRCARB_a = 0.00755985436706299 #0.00770594375034992 # < from opt a #0.1 #11.97  


muPYRCARB_a = 0.01  
CO2_mito_a = 1.2  
KeqPYRCARB_a = 1.0  
KmPYR_PYRCARB_a = 0.05638211229110231 # < from opt a #0.22  
KmCO2_PYRCARB_a = 3.2 

# psiPYRCARB_a1 = ( (ATPmito_a0/ADPmito_a0)/(muPYRCARB_a +  (ATPmito_a0/ADPmito_a0)))*VmPYRCARB_a*(PYRmito_a0*CO2_mito_a - OXAmito_a0/KeqPYRCARB_a)/(  KmPYR_PYRCARB_a*KmCO2_PYRCARB_a +  KmPYR_PYRCARB_a*CO2_mito_a + KmCO2_PYRCARB_a*PYRmito_a0 + CO2_mito_a*PYRmito_a0)    # PYRmito_a + ATPmito_a ⇒ OXAmito_a + ADPmito_a
