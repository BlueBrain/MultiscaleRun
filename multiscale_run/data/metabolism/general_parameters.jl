R = 8.31 # J/(K*mol)
T = 310.0 # Kelvin, 37 C 
F = 9.64853e04  #  
RTF     =   26.73 # mV

eto_n =  0.45 # Jolivet2015 # 0.4 # volume fraction neuron Calvetti2018      #
eto_a = 0.25 #0.3 # volume fraction astrocyte Calvetti2018   # 0.25 Jolivet2015
eto_ecs = 0.2 #0.3 # volume fraction ecs Calvetti2018       # 0.2 Jolivet2015
eto_b = 0.0055 #0.04 # volume fraction blood Calvetti2018      # 0.0055 Jolivet2015, Winter2017
beta = eto_n/eto_ecs #1.33 # Calvetti2018; in Cressman2011 it was set to 7.0

# Compare J_C4 from Theurey with vMitooutn from Jolivet: 3.5926631697208284 times higher respiration (C4) in Theurey as compared to Jolivet (vMitooutn) 
# 1/3.5926631697208284 = 0.2783450473253539 = T2Jcorrection 
# make J_C4 (with proper 1000* /W_) be same as 0.6*vMitooutn (see du for O2, see enzymes_preBigg/OXPHOS_ETC_Theurey2019de_mMmod_n2test.ipynb ) and correspondingly change all other mito rates

T2Jcorrection = 0.2783450473253539 #1.3*0.2783450473253539 #1.3049872091414738*0.2783450473253539  # 0.2783450473253539