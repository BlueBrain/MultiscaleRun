# # Na/K-ATPase
# kPumpn  =   2.49e-06;       # cm (mmol/L)-1 sec-1
# kPumpg  =   4.64e-07;       # cm (mmol/L)-1 sec-1
KmPump  =   0.5;            # mmol/L
# vPumpg0 =   0.0708;         # mmol/L sec-1

SmVn    =   2.5e04;         # cm-1
SmVg    =   2.5e04;         # cm-1
# gNan    =   0.0155;         # mS cm-2
# gNag    =   0.00623;        # mS cm-2

# Ratio of excitatory conductance
glia        =   50.0;         # mV
#
# Neuronal parameters
Cm      =   1e-03;          # mF/cm2
gL      =   0.02;           # mS/cm2

gNa     = 56. # 40.0;             # mS/cm2
gK      = 6. # 18.0;             # mS/cm2
g_M = 0.075 # not Jlv, enzymes/enzymes_preBigg/gen_mix.ipynb

gCa     =   0.02;           # mS/cm2
gmAHP   =   6.5;            # mS/cm2
KD      =   30e-03;         # mmol/L
tauCa   =   150e-03;        # sec
Ca0     =   0.5e-04;        # mmol/L
EK      =   -80.0;            # mV
ECa     =   120.0;            # mV
Ee      =   0.0;              # mV
Ei      =   -80.0;            # mV
phi    =   4.0;     # phih         #
#phin    =   4.0;              #
## 

kPumpn  = 2.2e-06;
gNan    = 0.0136;
gNag    = 0.0061;
gKpas   = 0.04 #0.05 #0.2035;  ############
kPumpg  = 4.5e-07;
vPumpg0 = 0.0687;

gKn = 0.05  # approx between Calv, Jlv, enzymes/enzymes_preBigg/gen_mix.ipynb
gKg = 0.05 # approx between Calv, Jlv, enzymes/enzymes_preBigg/gen_mix.ipynb

# qAK  = 0.92;                                            
# A    = 2.212;                                           # mmol/L

tau_max = 0.6 # not jlv, approx ms-s, 608ms /enzymes/enzymes_preBigg/gen_mix.ipynb



RateDecayK_a = 0.07

ImaxNaKa = 1.45
INaKaKThr = 20. # 16 is estim in W2013 # !!!!!!!!!!!!
INaKaNaThr = 30. # 1 is estim in W2013 # !!!!!!!!!!!!

gKirV = 6.75*gKpas # 0.1*2* #7*gKpas
VKirAV = 31.15

gKirS = 40*gKpas #4*gKpas # 0.1* #40*gKpas
VKirS = 26.8

gleakA = gKpas # 0.1*
VleakA = -40.

###
VprodEET_a = 72.0
kdeg_EET_a = 7.1
CaMinEET_a = 5e-5 #5e-6 #1e-4  # !!!!!!!!!!!!
EETshift = 2000.0 #mV 4.0 mV is [EET]*EETshift  #0.002  # 2. # approx  # !!!!!!!!!!!!

Ca3BK = 0.0004
Ca4BK = 0.00015

v4BK = 14.5
v5BK = 8.
v6BK = -13.

psiBK = 2.664 #0.02 #  !!!!!!!!!!!!
gBK = 54.0*gKpas # 200. ##1000*gKpas # 200. #0.2 # 200pS  # !!!!!!!!!!!!

EBK = -80. #-95. # !!!!!!!!!!!!

KGlutSyn = 4. #8.82  ###########
deltaGlutSyn = 0.001235
rhIP3a = 0.0048
kdegIP3a = 1.25

konhIP3Ca_a = 2000. #2e-3 # 2
khIP3Ca_aINH = 1e-4
Pleak_CaER_a = 1e-3 #2.5e-3 #5.2e-3 # # !!!!!!!!!!!!

beta_Ca_a =  0.025 #0.0244              #in W2013 !!!!!!!!!!!




VCa_pump_a = 0.055 #0.05worked #0.02  # !!!!!!!!!!!!
KpCa_pump_a = 0.00019 #0.000192 




ImaxIP3_a = 1.0 #2.88 in W2013 !!!!!!!!!!!
KIIP3_a = 1e-5 #3.0e-5 in W2013 !!!!!!!!!!!
KCaactIP3_a = 0.00017 

Ca_perivasc = 5.0e-3
tauTRPCa_perivasc = 20. #1.5 #2.0 #0.9 #in W2013 !!!!!!!!!!!




gTRP = 10*gKpas #w  #11.5*gKpas #10*gKpas worked #13.5*gKpas  # 1e-5* !!!!!!!!!!!!
VTRP = 6.

r0TRPVsinf = 0.0237 #0.23wwithe2TRPVsinf = 0.25 #0.032w with e2TRPVsinf = 0.1 #0.034f #0.032w #0.03 # 0.0237  # !!!!!!!!!!!!

#eTRPVsinf = ((rTRPVsinf-r0TRPVsinf)/r0TRPVsinf)
e2TRPVsinf = 0.046 #0.045 #0.05f #0.04r #0.06f #0.02r #0.025 #0.1w
kTRPVsinf = 0.101 #0.102r #0.1 # !!!!!!!!!!!!

gammaCaaTRPVsinf = 5e-5 #5e-5worked #5.0e-7  #1e-6 #1e-5 #5.0e-5 # 0.2uM in W2012 # 0.01uM in W2013 !!!!!!!!!!!
gammaCapTRPVsinf = 0.32 #w # 0.3r #0.35 #w  #0.2worked
v1TRPsinf_a = 120.
v2TRPsinf_a = 13. 




epsilon = 1.0 #9.33 # 1/s # Calvetti2018 #epsilon=1.333333333 # Cressman2011 # 1.2 Cressman2009
mu_glia_ephys = 0.1 #0.01  #0.1 # Calvetti2018
#mu_pump_ephys = 0.1

kbath = 4.0 #6.3 #4.0 # #kbath=4.0 Cressman2011 #3. CapoRangel ... Calvetti2019 #5.03 #4.2 #6.3 # mM # Calvetti2018

gamma = 5.99 #but gamma-ephys isn't in use, because different ephys model in the end; gamma is only used for mito-ETC functions #5. #0.045
glia_c = 30. #20.75 # mM/s # Calvetti2018 #glia=66.666666666 # Cressman2011

Na_n2_baseNKA = 18. #15.5 #15.7 #10.0 #14.70 # 18                                # Jglia

K_n_Rest = 120. #130. #140.0 
#Na_ecs_Rest = 130. #140. #144.0 
Na_n_Rest = 8. #9.5 #10.5 #9.5 #10.5 #8. #10. #10.5 


# Jolivet conc
Na_out = 150. # 153.5 #
# Na_n0 = 8.5
# Na_a0 = 15.

synInput = 1e-12

gbar    = 7.8e-06;  # mS cm-2 sec
finf    = 3.5;  # Hz
f0      = 32;   # Hz
tau     = 2.5
Ne = 240.0

# synaptic glutamate
amplGR = 1.1 #25.
tauG= 2. #2.7
KsynGlutPulse = 0.4; #8.8

