# THIS IS NOT REALLY U0. It is a series of parameters used to build up the system


VNeu0 = -65.0 #-71 NMC -73.0 Jolivet -70.0;-56.1999;-75.0 #mV # -56.1999 mV from Calvetti2018  # -61.5 in ephys 
Va0 = -90.0 # mV  # -0.09 V # Breslin2018 microdomains
m0 = 0.05 #calc based on Pospischil 2008  DOI 10.1007/s00422-008-0263-8 in workflow_sim/coupled_sim_gen_Ephys_Markram.ipynb #0.0054 #0.1*(VNeu0+30.0)/(1.0-exp(-0.1*(VNeu0+30.0))) / (  0.1*(VNeu0+30.0)/(1.0-exp(-0.1*(VNeu0+30.0))) + 4.0*exp(-(VNeu0+55.0)/18.0) )  # (alpha_m + beta_m)
h0 = 0.6 #0.9002 # Calvetti2018, but 0.99 in Jolivet2015
n0 = 0.32 #0.1558 # Calvetti2018, but 0.02 in Jolivet2015
pgate0 = 0.0266 #calc 

ksi0 = 0.001

Conc_Cl_out0 = 125.0 #130.0 #;140.0;110.0 #140 mM # Novel determinants of the neuronal Cl− concentration Eric Delpire 2014 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4215762/
Conc_Cl_in0 = 10.0 #6.0 #8.0;6.0 # mM # Novel determinants of the neuronal Cl− concentration Eric Delpire 2014 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4215762/

Na_n0 = 10.0 #11.5604 # mM

Na_out0 = 153.5
K_n0 = 140.0

K_out0 = 4.1 #;6.2773;2.3;5.9 #mM +-1.8 (mouse,rat) - Takaneshi, Manaka, Sano 1981 # Consider using 3mM for K_out (see ErecinskaSilverIonsMetabolism.pdf)  #  6.2773 # mM
K_a0 = 100.0 #52.0 #;54.0;100.0;110.0;113.0;130.0 #Witthoft2013 # 100.0 #Flanagan2018  #  # 130.0 # approx  ## in intracellular K+ concentration [K+]in (from 110 to ~113 mM) dissipated over several seconds after [K+]out returned to 3 mM
Na_a0 = 13.5 # 13.5 mM is derived from 10-17 mM range of literature values #10.0 #17.0;10.0 #Witthoft2013 supp figure S2 in pdf # 15.0 in Jolivet2015
Ca_a0 = 5.1e-5 #7.5e-5;8.0e-5;5.0e-5 # 5.0e-5 is from Jay Coggan 2020 https://doi.org/10.1016/j.jtbi.2019.110123 #75 nM # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 # [Ca2+]i of 50−80 nM and [Na+]i of 15 mM, the ENCX could be as negative as about −85 to −90 mV, being thus very close (or even slightly more negative) to the resting membrane potential of astrocytes.
Ca_n0 = 5.1e-5 # Jlv2015


nBK_a0 = 1e-5
mGluRboundRatio_a0 = 0.0
IP3_a0 = 1e-5 #5e-5
hIP3Ca_a0 = 1.0 # approx
Ca_r_a0 = 0.4 # mM doi:10.1016/j.jtbi.2007.08.024  
sTRP_a0 = 0.001255 #f prestim, r after #0.00126r #0.00125f  #0.00127r #0.0013r #0.0015r #0.002r 0.001f #0.005sg #0.001 #0.1 #0.001
EET_a0 = 5e-4

O2_ecs0 = 0.04 #;0.03;0.05 #0.04 mM Calvetti2018

Glc_a0 = 1.2 # FelipeBarros2017doi:10.1002/jnr.23998 #;1.25;1.0;0.8;1.5;1.8;2.0;2.5 # ~1.5-2.5 mM From Molecules to Networks 2014 ######################################################
ATP_a0 = 1.3 # 0.7-1.3 mM (acutely isolated cortical slices ) 1.5 mM ( primary cultures of cortical astrocytes from mice) doi: 10.3389/fncel.2020.565921 Kohler 2020 #2.17 #1.4;2.17;2.2 #2.17  # or 1.4 ?  # atpc/adpc = 29 -> for 1.4 atp -> adp=0.05
ADP_a0 = 0.045 #  atpc/adpc = 29 -> for 1.3 atp -> adp=0.045

NADH_a0 = 0.10386865243459381 # Jlv 0.00078 #0.00078 derived from 2 eq: NAD+NADH=0.212 in Jolivet, Nad/NADH = 270 # Nad/NADH = 670-715  #NADH/NAD = [0.001,0.01] # Neves 2012 # mean NAD+/NADH value in hippocampal neurons was ∼660 and in hippocampal astrocytes was ∼270  https://doi.org/10.1089/ars.2015.6593 
NAD_a0 = 0.212 #0.21122 # derived from 2 eq: NAD+NADH=0.212 in Jolivet, Nad/NADH = 270  #0.16 #0.5 #;0.2 # 0.162061 Copasi Winter2017

# SPLIT GLC_ECS INTO basal_lamina and interstitial as in FelipeBarros2017doi:10.1002/jnr.23998 
#Glc_ecs0 = 1.3 #;1.25;0.23;2.4;2.48 #2.48 mM Jolivet2015 #1.25 # 2.4 mM SilverErecinsca 1994 Table 1 brain_glucose_SilverErecinska.pdf  #try to split ECS_n and ECS_a see fig 2 in Felipe Barros 2017 https://doi.org/10.1002/jnr.23998 # or 0.23 mM from 10.1007/978-1-4614-1788-0 p394 : table 13.4 : rat brain interstitial : 0.23+-0.12 # 0.2-0.25 from 10.1007/978-1-4614-1788-0  p679 + (Silver and Erecinska 1994) # 0.33 mM from Cloutier2009
Glc_ecsBA0 = 1.3 # 1.3 to Glc_b # endothel-to-astrocyte (basal lamina) FelipeBarros2017doi:10.1002/jnr.23998 # 2.7 DiNuzzo 2010 1
Glc_ecsAN0 = 1.0 # astrocyte-to-neuron (interstitial space) FelipeBarros2017doi:10.1002/jnr.23998 
Glc_t_t0 = 1.4 # endothelium DOI: 10.1002/jnr.23998

Lac_ecs0 = 0.55 # MaechlerCell Lac gradient # 0.6 Jolivet #1.3 #1.4;1.3;1.2;0.6 #1.3 #Calvetti # 0.4 mM from Cloutier2009

O2_a0 = 0.03 # mM Calvetti2018 # 0.102 Cloutier2009

G6P_a0 = 0.06 #Kauffman1969,From Molecules to Networks 2014 #0.2 #;0.675;0.53;0.75;0.06;0.1 # #0.675 #mM-Park2016-worked #0.53 #0.75 # 0.53 also ok # 0.06-0.2 mM From Molecules to Networks 2014 (book)
#G6P_a0 = 0.2 # in between exp data #0.1 # Winter2017 #0.6 Poliquin, # 0.7 Cloutier #0.06 # 0.07 Winter2017 #0.2 #;0.675;0.53;0.75;0.06;0.1 # #0.675 #mM-Park2016-worked #0.53 #0.75 # 0.53 also ok # 0.06-0.2 mM From Molecules to Networks 2014 (book)

# Mulukutla2014 (bistability): F6P = 0.09 - 0.3 mM 

F6P_a0 = 0.01 #Kauffman1969,From Molecules to Networks 2014 #;0.01;0.0969;0.228 # 0.01 - 0.02 mM From Molecules to Networks 2014 (book) #0.0969#-worked #mM -Park2016 # 0.228 mM -Lambeth2002 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#F6P_a0 = 0.02 #;0.01;0.0969;0.228 # 0.01 - 0.02 mM From Molecules to Networks 2014 (book) #0.0969#-worked #mM -Park2016 # 0.228 mM -Lambeth2002 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
f26bp_a0 = 0.015 #;0.005;0.025 #0.005-0.025 #mM #Mulukutla supp fig plot 6 # 2.6-12.4 nmol/g wet weight ErecinskaSilverIonsMetabolism.pdf  p15/35  -> 1.19*1e-3*(2.6-12.4) mM -> 0.003094 - 0.014756


FBP_a0 = 0.03 #approx Berndt2015 #0.1 #;0.01;1.52;0.0723 # 0.01 - 0.1 mM From Molecules to Networks 2014 (book) #1.52 #-worked #0.0723 # Lambeth # Jay Glia expand  # 1.52 #mM -Park2016 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Lac_a0 = 0.6 #0.6 #Jolivet #1.3 #1.4;1.3;0.6 #1.3 # mM Calvetti2018; Lambeth   ## 0.602393 # Jay 181130 # From AMPK 2020 CellRep https://doi.org/10.1016/j.celrep.2020.108092 : Lactate is 10–50 times more abundant than pyruvate in healthy cells; Lac/PYR = 10-15 (Siesjo 1978)

AMP_a0 = 1e-5 ##0.01 #;0.03;2e-5;1e-5;0.05 #2e-5 #0.03 #-Mulukutla2015 #2e-5 # Lambeth # 0.01 # 2e-5 # 1e-5 # from 1e-5 to 0.05 # 0.00017 to 0.05 mM PMID: 2642915
Pi_a0 = 20.0 # Theurey2019 #1.15 # estim from new ATP and  ATP/Pi = 1.13-1.7; PCr/Pi = 1.87-3.74; ATP/PCr = 1.5-2.23 PMID: 2642915 #4.1 #;1.0;40.0;31.3 # Lambeth # 40.0 # 4.1 # 31.3 # 4.1 # 4.0 Anderson&Wright 1979 # wide range from 1 to 40 mM

# Mulukutla2014 (bistability): GAP = 0.02 mM, DHAP = 0.04 mM
GAP_a0 = 0.005 #Kauffman1969,Jolivet2015,Tiveci2005. #0.05 #;0.04;0.0355;0.0574386;0.0046;0.141 # 0.04 - 0.05 mM From Molecules to Networks 2014 (book) # 0.141 #mM -Park2016  #0.0355 # Lambeth # 0.0574386 # 0.0574386 is from latest Jay's data; it was 0.0046 in Jay Glia expand  !!!!!!!!!!!!!!!!!!!!!!!!!!
DHAP_a0 = 0.04 #Kauffman1969,From Molecules to Networks 2014 #0.03 #;0.01;1.63;0.0764 # 0.01 - 0.03 mM From Molecules to Networks 2014  #1.63 #mM  -Park2016 # 0.0764 # Lambeth # Jay Glia
#GAP_a0 = 0.06 # in between Cloutier2009 and From Molecules to Networks 2014   #;0.04;0.0355;0.0574386;0.0046;0.141 # 0.04 - 0.05 mM From Molecules to Networks 2014 (book) # 0.141 #mM -Park2016  #0.0355 # Lambeth # 0.0574386 # 0.0574386 is from latest Jay's data; it was 0.0046 in Jay Glia expand  !!!!!!!!!!!!!!!!!!!!!!!!!!
#DHAP_a0 = 0.03 #;0.01;1.63;0.0764 # 0.01 - 0.03 mM From Molecules to Networks 2014  #1.63 #mM  -Park2016 # 0.0764 # Lambeth # Jay Glia expand   !!!!!!!!!!!!!!!!!!!!!!!!!!

BPG13_a0 = 0.04 #approx Shestov2014  DOI: 10.7554/eLife.03342 , erythrocyte, Lambeth 0.065 #mM Lambeth
PG3_a0 = 0.1 #approx Berndt2015 # 0.375 #;0.52;0.0168 #0.52 #0.0168 ####0.375 #-Park # 0.052 #mM Lambeth  #!!!!!!!!!!!!!
PG2_a0 = 0.01 #approx Berndt2015 #0.00949 #;0.02;0.05;0.00256;0.005 ####0.00949 #mM #-Park #0.02 #mM -Berndt #0.005  #mM Lambeth # !!!!!!!!!!!!!!!!!!!!

PEP_a0 = 0.005 #0.015 #0.005;0.004;0.0194;0.0142;0.028;0.017;0.015 # 0.015 # mM Jolivet2015 # 0.004-0.005 mM From Molecules to Networks 2014  # 0.0194 # Lambeth # 0.014203938186866 # Jay 181130 this value taken from Jolivet PEPg # 0.0279754 # was working with 0.0170014 # was working with 0.0279754 - Glia_170726(1).mod # was working 0.015 # glia expand in between n and g ### check it

Pyr_a0 = 0.033 #Arce-Molina2019  #0.1 #0.04;0.033;0.15;0.1;0.2;0.35;0.0994;0.202 #mM 0.033-0.04 mM Arce-Molina2019 astrocyte cytosol #0.15 #0.1–0.2 mM -Lajtha 2007 book  NeurochemistryEnergeticsBook.pdf  #0.35 # mM Calvetti2018  # 0.0994 # Lambeth # 0.202024 # !!!!! From AMPK 2020 CellRep https://doi.org/10.1016/j.celrep.2020.108092 : Lactate is 10–50 times more abundant than pyruvate in healthy cells

G1P_a0 = 0.01 #From Molecules to Networks 2014 (book) #0.0589 #mM  -Lambeth #u59

################# glycogen_scaled[i]*5.0 here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! LAYER_SPECIFIC!!!!!!!!!!!
GLY_a0 = 14.0 # DiNuzzo2010 doi:10.1038/jcbfm.2010.151  #5.0 #1.4;1.12;4.2;5.0;14mM DiNuzzo2010 doi:10.1038/jcbfm.2010.151 #glycogen_scaled[1]*5.0;glycogen_scaled[1]*30.0;glycogen_scaled[1]*1.12 #my main googledocs table # 1.12 # mM-Waitt2017 (up to 100 times lower than liver,skeletal muscle); Cloutier2009: 3mM #u60 # 1.4 to 4.2 mM Cloutier2009

GPa_a0 = 0.0016099978488516124 #0.00699  #0.0699071 #Jay
GPb_a0 = 0.06839000215114839 #0.010485 #;0.000025  #1.5*GPa_a;0.000025 #0.000025 #Jay
UDPgluco_a0 = 0.1 # from ratio with G1P #0.589 #mM
GS_a0  = 0.0029999 #0.01 # ;0.003 #0.003 # 0.0111569 # or 0.003 - both are in Jay's most recent # GSa_a0
UTP_a0 = 0.23 #;1.76 #0.23 # too low 5.17e-15 # 5170 pmol/10^6cells Lazarowski&Harden  # or 0.23 mM - Anderson&Wright

cAMP_a0 = 0.04  #0.0449285 # approx Jay mod Coggan20188 wide range of conc starting from low conc 
PKAa_a0 = 5.0e-8 #0.0018 #;1.4339 # from Jay 181130 = CC# 1.4339 # or 0.0018
PKAb_a0 = 0.00025 #;0.082 # 0.0823673 # or  0.00025

PHKa_a0 = 4.0e-7 #0.0025 #;0.0089 # or 0.00899953
PHKb_a0 = 0.000025

PP1_a0 = 1.0e-6 #0.00025
R2CcAMP2_a0 = 1.0 #;0.3584 # Jay 181130  # 0.3584 # 1.0 # 0.3584
R2CcAMP4_a0 = 1.0 #;1.55948 # Jay 181130 #1.55948 # 1.0 # 1.55948

Glc_b0 = 4.5 #5.0 # 5.5 glc blood in DOI 10.1002/glia so use slightly less in cap than art #4.5 #6.15 #mM Nehlig 1992 #4.51 # mM Calvetti2018 # 4.5 mM Jolivet2015
Lac_b0 = 0.81 # mM #http://dx.doi.org/10.1016/j.cmet.2015.10.010 #0.5 #0.55 Jolivet # #1.24 # mM Calvetti2018
O2cap0 = 7.0 #O2_b0 6.67;7.3 #7.3 #Winter2017 # 7.0 Jolivet2015 # 6.67 # mM Calvetti2018 # ?is it total or free?
q0 = 0.0067 #mL/s #0.4 # mL/min denoted as Q in table but seems to be q0 (baseline bloodflow) according to text and equations; used for callback

PCr_a0 = 4.9242059083825644 #3.0 # close to 1.5 Cloutier2009, estim from new ATP and ATP/Pi = 1.13-1.7; PCr/Pi = 1.87-3.74; ATP/PCr = 1.5-2.23 PMID: 2642915  #4.9 #10.32;4.9 # mM Calvetti2018 #34.67 # Lambeth # 4.9 mM Jay 2020 # The ratio of creatine phosphate to creatine in isolated nerve terminals is about 0.5:1 (Erecinska and Nelson, 1994) Lajtha 2007 book  NeurochemistryEnergeticsBook.pdf  but the same source: If the creatine kinase activity is sufficient for maintaining the reaction close to equilibrium then the ATP/ADP ratio should be in the region of 90. This is far higher than is measured in the same preparation by conventional extraction and enzymatic analysis, where values close to 5 are generally reported (Erecinska and Nelson, 1994). 
Cr_a0 = 2.0 # to sum to 5.0 as in Cloutier2009  #1.1e-3 # mM Calvetti2018 # 40-34.67 # Lambeth #

###### GLU_GLN astrocyte
#  Consequently, vesicular glutamate release from astrocytes creates localized extracellular glutamate accumulations of 1–100 μM [42] # Astrocyte glutamine synthetase: pivotal in health and disease  Rose,  Verkhratsky and Vladimir Parpura
GLUT_out0 = 2.5e-5 # mM # 25 nM  # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 #The extracellular concentration of glutamate in resting conditions is around 25 nM (677)  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/
GLUT_a0 = 0.3 #mM #0.3 mM (227) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/ # Physiology of Astroglia. Verkhratsky and Nedergaard 2018   and the same value in Savtchenko-Rusakov ASTRO 2018
GLN_out0 = 0.2 #;0.3;0.13;0.5 #0.3 #mM # from 0.13 mM to 0.5 mM # Broer Brookes 2001 Transfer of glutamine between astrocytes and neurons #estimates of extracellular glutamine vary from 0.13 mm to 0.5 mm, the upper limit of this range being close to the mean around which plasma and CSF levels ¯uctuate (Jacobson et al. 1985; ErecinÂska and Silver 1990; Xu et al. 1998)
GLN_a0 = 0.25 #;2.0;0.3;0.2 #0.2-2.0 mM Hertz2017 #0.25 #mM estimate based on "a bit lower than GLUT_a"   ##Broer Brookes 2001  Astrocytes and neurons, cultured in medium containing 2 mm glutamine, generate intracellular glutamine concentrations of 20 mm or more, when estimated on the basis of a solute-accessible water content of 4 mL per mg protein (Patel and Hunt 1985;Brookes 1992a)

GLUT_syn0 = 5e-3 # mM #Breslin2018 table2 ref52   # Scimemi,Beato: DOI 10.1007/s12035-009-8087-7 glutamate concentration in synaptic vesicles is ~60 mM  !!!!!!!!!!!!!!!!!!!!!!!!!!

# steady-state release rate of glutamate (1.2 molecules per EAAC1 per second) is predicted to increase 20-fold at short times after the depolarization   -- Transport direction determines the kinetics of substrate transport by the glutamate transporter EAAC1 2007 Zhou Zhang, Zhen Tao, Armanda Gameiro, Stephanie Barcelona, Simona Braams, Thomas Rauen, and Christof Grewer
#  Broer Brookes 2001 The overall concentration of glutamine in normal mammalian brain is an estimated 5±9 nmol/mg wet weight, equivalent to 6±11 mM, with little regional variation(ErecinÂska and Silver 1990)
# GLUT_n_cyt = 1−10 mM. # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/ # the cytosolic concentration of glutamate in neurons is usually assumed to be in a range of 1−10 mM.
# # Physiology of Astroglia. Verkhratsky and Nedergaard 2018 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6050349/ Kinetics of the glutamate translocation by EAATs is relatively slow; both EAAT1 and EAAT2 transport ~30 molecules of glutamate per second (1286, 1956). The glutamate binding to the transporters (Km ~20 μM) is however much faster, and hence glutamate transporters concentrated at the perisynaptic processes act as almost instant buffers for glutamate. The higher is the density of transporters, the higher is their buffering capacity (1792). The EAAT2 in cultured hippocampal astrocytes has a remarkable lateral mobility regulated by glutamate, possibly allowing a continuous exchange of glutamate-bound and unbound transporters to maintain high buffer capacity of perisynaptic zones (1177).

################ neuron
Lac_n0 = 0.5 #gradient in http://dx.doi.org/10.1016/j.cmet.2015.10.010 0.55 #0.6 Jolivet #0.8 #1.3;0.6 # From AMPK 2020 CellRep https://doi.org/10.1016/j.celrep.2020.108092 : Lactate is 10–50 times more abundant than pyruvate in healthy cells
Glc_n0 = 0.9 # FelipeBarros2017doi:10.1002/jnr.23998 #1.2 #;1.5;2.5 # ~1.5-2.5 mM From Molecules to Networks 2014
O2_n0 = 0.03 # mM Calvetti2018 # 0.102 Cloutier2009


ATP_n0 = 1.400216 # 00216 because of ETC # 1.4 # Non-canonical... paper #2.18 #1.4;0.84;2.18;2.2 #0.8435861272283393 #1.4 # 2.18 Calvetti2018
ADP_n0 = 0.047784 # because of ETC  #0.048 #6.3e-3 #0.03;6.3e-3  #6.3e-3 # mM Calvetti2018 # atpc/adpc = 29 -> for 1.4 atp -> adp=0.048 # atp/adp in range from 1 to >100 doi: 10.1038/ncomms3550 (2013) # 0.021-0.56 mM PMID: 2642915

#NADH_n = 1.2e-3 # mM Calvetti2018    # Nad/NADH = 670-715  ### check !!
#NAD_n = 0.03 # mM Calvetti2018  #NAD_n = 0.20574782239567735 # calc from Jay 181130 Rminusn 0.00625245  / 0.0303889 # was working with 0.5 # Jay Glia_170726(1).mod
NADH_n0 = 0.006243658877436395 #0.0003 # derived from 2 eq: NAD+NADH=0.212 in Jolivet, Nad/NADH = 700  # Nad/NADH = 670-715  #NADH/NAD = [0.001,0.01] # Neves 2012
NAD_n0 = 0.212 #0.2117 # derived from 2 eq: NAD+NADH=0.212 in Jolivet, Nad/NADH = 700 

PEP_n0 = 0.005 #0.015 #;0.005;0.004 # 0.004-0.005 mM From Molecules to Networks 2014 (book) # 0.015 # mM Jolivet2015
Pyr_n0 =  0.06 #0.17 #0.3;0.05;0.2;0.38 #0.05 #0.05-0.2 mM From Molecules to Networks 2014 (book) #  #0.38 # mM Calvetti2018 # From AMPK 2020 CellRep https://doi.org/10.1016/j.celrep.2020.108092 : Lactate is 10–50 times more abundant than pyruvate in healthy cells; Lac/PYR = 10-15 (Siesjo 1978 Brain energy metabolism. New York, NY: John Wiley & Sons)

PCr_n0 = 4.946022342318186 #3.0 # close to Cloutier2009 2.5 mM PCrn, 3.0 is estim from new ATP and ATP/Pi = 1.13-1.7; PCr/Pi = 1.87-3.74; ATP/PCr = 1.5-2.23 PMID: 2642915   #4.9 #10.33;4.9 # 10.33 mM Calvetti2018 # 4.9 mM Jay 2020
Cr_n0 = 2.0 # to sum to 5.0 as in Cloutier2009 #3.0e-4 # mM Calvetti2018

G6P_n0 = 0.06 #Kauffman1969,From Molecules to Networks 2014 #0.15 #;0.06;0.2;0.7 #was 0.2 before 27may2020 #0.06-0.2 mM From Molecules to Networks 2014 #0.7 #was working with 0.1 #  # both n and a values worked here # # 0.7 = a  ### 0.1 # approx from figure, so check it, Berndt 2015
#G6P_n0 = 0.2 # in between exp data #0.1 # Winter2017 #0.6 Poliquin, # 0.7 Cloutier  #0.06 #0.15 #;0.06;0.2;0.7 #was 0.2 before 27may2020 #0.06-0.2 mM From Molecules to Networks 2014 #0.7 #was working with 0.1 #  # both n and a values worked here # # 0.7 = a  ### 0.1 # approx from figure, so check it, Berndt 2015

# Mulukutla2014 (bistability): F6P = 0.09 - 0.3 mM 

F6P_n0 = 0.01 #Kauffman1969,From Molecules to Networks 2014 #;0.01;0.035;0.228 # 0.01 - 0.02 mM From Molecules to Networks 2014 #0.035#-worked # 0.228 # 0.228  = a ### 0.035 # approx from figure, so check it, Berndt 2015
#F6P_n0 = 0.03 # 0.03 is between Winter2017 and From Molecules to Networks2014 #0.02 # 0.1 - Cloutier, Poliquin # #;0.01;0.035;0.228 # 0.01 - 0.02 mM From Molecules to Networks 2014 #0.035#-worked # 0.228 # 0.228  = a ### 0.035 # approx from figure, so check it, Berndt 2015
FBP_n0 = 0.03 #approx Berndt2015 #0.1 #;0.01;1.52 # 0.01 - 0.1 mM From Molecules to Networks 2014  #0.035 #-approx from figure, so check it, Berndt 2015 # 1.52 #mM -Park2016 # !!!!

# Mulukutla2014 (bistability): GAP = 0.02 mM, DHAP = 0.04 mM
GAP_n0 = 0.005 #Kauffman1969,Jolivet2015,Tiveci2005 #0.05 #;0.04;0.0046;0.057 # 0.04 - 0.05 mM From Molecules to Networks 2014  # 0.0574386 # Jay 181130 # was working with 0.00460529 ## 0.0574386 # 0.0574386  = a ### 0.00460529 # Jay Glia_170726(1).mod
DHAP_n0 = 0.04 #Kauffman1969,From Molecules to Networks 2014 #0.03 #;0.01;0.05 # 0.01 - 0.03 mM From Molecules to Networks 2014  # 0.05  # 0.0764 = a  ### 0.05 # approx from figure, so check it,  Berndt 2015
#GAP_n0 = 0.04 # Cloutier2009 and From Molecules to Networks 2014 #;0.04;0.0046;0.057 # 0.04 - 0.05 mM From Molecules to Networks 2014  # 0.0574386 # Jay 181130 # was working with 0.00460529 ## 0.0574386 # 0.0574386  = a ### 0.00460529 # Jay Glia_170726(1).mod
#DHAP_n0 = 0.03 #;0.01;0.05 # 0.01 - 0.03 mM From Molecules to Networks 2014  # 0.05  # 0.0764 = a  ### 0.05 # approx from figure, so check it,  Berndt 2015

Pi_n0 = 20.0 # Theurey2019  1.15 # estim from new ATP and ATP/Pi = 1.13-1.7; PCr/Pi = 1.87-3.74; ATP/PCr = 1.5-2.23 PMID: 2642915  #4.1  #1.0-worked before 18feb2020 # 4.1 # check it Jay Glia expand
BPG13_n0 = 0.04 #approx Shestov  DOI: 10.7554/eLife.03342 , erythrocyte, Lambeth  #0.065 # check it Jay Glia expand #0.01 approx doi: 10.15255/CABEQ.2014.2002 ecoli

PG3_n0 = 0.1 #approx Berndt2015 #0.07 #0.375=latest astrocyte  # 0.052 =a  ### 0.07 # approx from figure, so check it,  Berndt 2015
PG2_n0 = 0.01 #approx Berndt2015 #0.009 # 0.005 #;0.02;0.00949 #0.00949 =latest astrocyte # Lambeth  # 0.02 # 0.005 # 0.005  = a ### 0.02 # approx from figure, so check it,  Berndt 2015

f26bp_n0 = 0.015 #0.005;0.015;0.025 #0.015 #mM 0.005-0.025 mM Mulukutla2014 supp fig plot 6 #2.6-12.4 nmol/g wet weight ErecinskaSilverIonsMetabolism.pdf  p15/35  -> 1.19*1e-3*(2.6-12.4) mM


# Berndt 2015 # fig S1 #mM and other ref, specified below
PYRmito_n0 = 0.05 #inferred from PYRcyt to PYRmito diff in astro where conc are measured by Arce-Molina 0.025  #0.14 # 0.14 Nazaret 2009 doi:10.1016/j.jtbi.2008.09.037 #0.1-0.6 Berndt 2015 #0.05 #inferred from PYRcyt to PYRmito diff in astro where conc are measured by Arce-Molina 0.025 
CITmito_n0 = 0.35 # 0.2-0.4 Ronowska 2018 10.3389/fncel.2018.00169 # 0.4 Nazaret 2009 10.1016/j.jtbi.2008.09.037 #;1.25 #1.25 # approx from figure, so check it,  Berndt 2015
ISOCITmito_n0 = 0.035 # 0.1CIT Frezza 2017 https://doi.org/10.1098/rsfs.2016.0100 #;0.09 #0.09 # approx # approx from figure, so check it,  Berndt 2015
AKGmito_n0 = 0.25 #0.25 Nazaret 2009 10.1016/j.jtbi.2008.09.037 #;0.6 #0.6 # approx # approx from figure, so check it,  Berndt 2015
SUCCOAmito_n0 = 0.0025 #0.05 # approxm # approx from figure, so check it,  Berndt 2015 # SUCCOAmito < OXAmito
SUCmito_n0 = 0.5 #;1.25 ### 1.25 # Berndt 2015; or  0.5 mM from Succinate, an intermediate in metabolism, signal transduction, ROS, hypoxia, and tumorigenesis Laszlo Trette  2016
FUMmito_n0 = 0.04 #Fink 2018 10.1074/jbc.RA118.005144 #0.055 #;0.35;1.94 #0.35 # in Berndt2015, but 1.94 in Mogilevskaya 2006
MALmito_n0 = 0.22 # Fink 2018 10.1074/jbc.RA118.005144 and Biochemistry book 2012 Reginald H. Garrett, Charles M. Grisham #;0.03;2.0 #2.0 # approx from figure, so check it,  Berndt 2015 ############## 0.03 by Chen ##########################################
OXAmito_n0 = 0.005  # 0.005 Nazaret 2009 10.1016/j.jtbi.2008.09.037 #;0.0001;0.08;0.1 #0.0001 #mM Williamson 1967 #0.08 # OAmito # approx from figure, so check it,  Berndt 2015  # 0.1 #mM Shestov 2007

AcCoAmito_n0 = 0.074 # Poliquin doi:10.1371/journal.pone.0069146  # 0.07 Nazaret 2009 10.1016/j.jtbi.2008.09.037 #0.01 #0.07  # 0.05 - estimated from fig 1e (Intracellular concentrations of metabolites in HCT116 cells after incubation in [13C6]-glucose medium for 5 hr. - Liu 2018 Cell. Acetate Production from Glucose and Coupling to Mitochondrial Metabolism in Mammals #0.01  # The Regulatory Effects of Acetyl-CoA Distribution in the Healthy and Diseased Brain Ronowska 2018: the acetyl-CoA concentrations in neuronal mitochondrial and cytoplasmic compartments are in the range of 10 and 7 μmol/L, respectively # very small and not visible on plot in Berndt
CoAmito_n0 = 0.002 #Poliquin 0.16 #;0.37;0.001 #0.37  # 0.37 = a ### 0.001 - lead to domain error # Rock 2000 Pantothenate Kinase Regulation of the Intracellular Concentration of Coenzyme A  + very small and not visible on plot in Berndt

#
#
CaMito_n0 = 0.0001 #uM-mM Brocard2001  J Physiol #Lajtha 2007 book  NeurochemistryEnergeticsBook.pdf  # 1e-7 M Martinez-Serrano 1992 PMID: 1550964 #;5.1e-5 # from Mogilevskaya 2006 # 5e-5 # Calcium  = 5.10258e-5 in  Jay, but check which part is mito....  see my pencil notes on printed Calvetti  p 242 about Ca in ddifferent organelles


###
ASPmito_n0 = 1.4 # approx set as cyt, Maletic-Savatic PMID:19022759 and Nazaret 2009 10.1016/j.jtbi.2008.09.037 #0.1 #;2.0;2.6;1.5  #2.0 #Shestov2007 #2.6 From Molecules to Networks 2014 (book) #1.5 ### Chen Abs Quant Big mmc1 # check and set neuronal!!!!
GLUmito_n0 = 10.0 #Roberg PMID: 10942715 #5.3 # Nazaret 2008 #0.057 # by Chen # check and set neuronal!!!!


### MAS cyto-mito
MAL_n0 = 0.45 #;2.0;0.6 # 2.0 #cytosol # 0.6 by Chen # 5.0 # check and set neuronal!!!! just assumption  from # Nonactivating behavior is observed at concentrations between 0.02 and 0.15 mM L-malate and activating behavior is observed between 0.15 and 0.5 mM L-malate.(Malate dehydrogenase. Kinetic studies of substrate activation of supernatant enzyme by L-malate. Mueggler PA, Wolfe RG.)
OXA_n0 = 0.01 #0.1 #;0.005;0.01 #cytosol #0.005 # check and set neuronal, just assumption from heart data now # 0.01 mM from 10.1007/978-1-4614-1788-0 and Williamson 1967 # Berndt 2015 re Indiveri C, Dierks T, Kramer R, Palmieri F. Reaction-Mechanism of the Reconstituted Oxoglutarate Carrier from Bovine Heart-Mitochondria
ASP_n0 = 1.4 #;6.0 #6.0 #cytosol  ### Chen Abs Quant Big mmc1  # 5.0 # 2.0 # 1.19*1.2=1.4 mM  1.0–1.4 mmol/kg ww Maletic-Savatic PMID:19022759
AKG_n0 = 0.2 #;1.2;0.265 #1.2 #cytosol # by Chen  #0.265 Pritchard 1995 # check and set neuronal!!!!


# ### PPP #  Kauffman 1969 (https://www.jbc.org/article/S0021-9258(18)83418-4/pdf) - Normal Brain Levels: In general, the levels of pentose phosphate pathway metabolites in normal mouse brain are extremely low (Table II).   # [umol/g]*1.04g/ml*0.001 -> mM  # 1e-6/1e3 -> 1e-9 mol/g -> 1e-6 mmol/g -> ~1e-3mM
# GL6P_n0 =  3.0e-06 # gluconolactone 6-phosphate 3.00121e-06 Copasi Winter2017 #mM from Sabate 1995 # 7.62e-06  mM Nakayama 2005 rbc #0.45366 mM Winter 2017
# GO6P_n0 = 0.0097 # gluconate 6-phosphate Kauffman1969 # 0.00288326 Copasi Winter2017 #2.72  Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM

NADP_n0 = 0.0003 #2.21464e-09 # 2.21464e-09 Copasi Winter2017  #mM from Sabate 1995  #8.06e-05 # Nakayama 2005 rbc #0.45366 mM Winter 2017
NADPH_n0 = 0.03 #order of magnitude lower than NADH Berndt2015 #0.003 #Bradshaw 2019 10.3390/nu11030504  #0.291226 # 0.291226 Copasi Winter2017  # mM from Sabate 1995  #6.58e-02 mM Nakayama 2005 rbc #0.45366 mM Winter 2017

# RU5P_n0 = 0.0072 # Kauffman1969 # 0.000674378 Copasi Winter2017 #1.48e-04 Nakayama 2005 rbc  #### from Sabate 1995: 0.012 mM
# X5P_n0 = 0.014  # Kauffman1969 # 0.0206819 Copasi Winter2017 #mM calc from Kauffman1969 #4.3e-04  # Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM
# R5P_n0 = 0.025 # Kauffman1969 in text, R5P+S7P=0.064   #2.70062e-05 Copasi Winter2017  #mM from Sabate 1995 #2.81e-04 # Nakayama 2005 rbc #### +S7P calc from Kauffman1969: 0.0643 mM +S7P
# S7P_n0 = 0.04 # Kauffman1969 in text, R5P+S7P=0.064 # 0.2 Winter2017pdf 1.15524 Copasi Winter2017 #mM calc from Kauffman1969 #0.0749 # Nakayama 2005 rbc #### +R5P calc from Kauffman1969: 0.0643 mM #### from Sabate 1995:  0.068 mM
# E4P_n0 = 0.002  # Kauffman1969  # 0.00651212 Copasi Winter2017 # mM calc from Kauffman1969 #1.17 Nakayama 2005 rbc ####  calc from Kauffman1969: <0.002 mM #### from Sabate 1995: 0.004 mM

# ### PPPa
# GL6P_a0 = 3.0e-06 # gluconolactone 6-phosphate 2.99776e-06 Copasi Winter2017 #mM from Sabate 1995 # 7.62e-06  mM Nakayama 2005 rbc  #u147
# GO6P_a0 = 0.0097 # gluconate 6-phosphate Kauffman1969  #0.00180362  Copasi Winter2017 #mM calc from Kauffman1969 #2.72  Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM #u148

NADP_a0 = 0.0003 #2.75699e-09 # 2.75699e-09 Copasi Winter2017 #mM from Sabate 1995  #8.06e-05 # Nakayama 2005 rbc #u149
NADPH_a0 = 0.03 #order of magnitude lower than NADH Berndt2015 #0.003 #Bradshaw 2019 10.3390/nu11030504 #0.291226 #0.291226 Copasi Winter2017 # mM from Sabate 1995  #6.58e-02 mM Nakayama 2005 rbc #u150

# RU5P_a0 = 0.0072 # Kauffman1969 # 0.000674347 Copasi Winter2017  #mM calc from Kauffman1969 #1.48e-04 Nakayama 2005 rbc  #### from Sabate 1995: 0.012 mM #u151
# X5P_a0 = 0.014  # Kauffman1969 # 0.0206807 Copasi Winter2017 #mM calc from Kauffman1969 #4.3e-04  # Nakayama 2005 rbc #### from Sabate 1995: 0.018 mM#u152
# R5P_a0 = 0.025 # Kauffman1969 in text, R5P+S7P=0.064  # 2.601e-05 Copasi Winter2017 #mM from Sabate 1995 #2.81e-04 # Nakayama 2005 rbc #### +S7P calc from Kauffman1969: 0.0643 mM +S7P  #u153
# S7P_a0 = 0.04 # Kauffman1969 in text, R5P+S7P=0.064  #0.276691 Copasi Winter2017 #mM calc from Kauffman1969 #0.0749 # Nakayama 2005 rbc #### +R5P calc from Kauffman1969: 0.0643 mM #### from Sabate 1995:  0.068 mM #u154
# E4P_a0 = 0.002  # Kauffman1969 # 0.00569938 Copasi Winter2017 # mM calc from Kauffman1969 #1.17 Nakayama 2005 rbc ####  calc from Kauffman1969: <0.002 mM #### from Sabate 1995: 0.004 mM#u155

# Winter2017
GL6P_n0 = 3.001213795e-6
GL6P_a0 = 2.997760015e-6
GO6P_n0 = 0.002883261904
GO6P_a0 = 0.001803623341
RU5P_n0 = 0.000674378444
RU5P_a0 = 0.0006743472507
R5P_n0 = 2.700617026e-5
R5P_a0 = 2.600996357e-5
X5P_n0 = 0.02068192382
X5P_a0 = 0.02068072623
S7P_n0 = 1.155239863
S7P_a0 = 0.2766906117
E4P_n0 = 0.006512123226
E4P_a0 = 0.005699383152


#!conc is not involved explicitly in the current version of the model (15oct2021) SDHmito_n0 = 0.05 # Mogilevskaya 2006  # levels of sdh expression in astrocytes and neurons are almost the same by Sharma's data

### mito astrocyte
PYRmito_a0 = 0.021 # Arce-Molina2019 ;0.0235 #0.025 #0.0235 #mM  Arce-Molina2019 astrocyte mito #u120 !!!!!!!!!!!!
CITmito_a0 = 0.35 #;0.2;0.4 #0.25-0.35 From molecules to networks 2014; 0.2-0.4 Ronowska2018 # # u121
ISOCITmito_a0 = 0.035 #;0.02 #Frezza2017 Cit/Isocit =10; 0.02 From molecules to networks 2014 # u122
AKGmito_a0 = 0.2 #From molecules to networks 2014 # u123 #0.2 most of files, # 0.6 - v8_v9_v10_K and DATA - approx from Berndt
SUCCOAmito_a0 = 0.002 #approx from Berndt2015 # SUCCOAmito < OXAmito #0.0068 #0.05 #0.0068 #Park2016 # u124
SUCmito_a0 = 0.45 #;0.7 # 0.7 #0.5 #0.45-0.7 From molecules to networks 2014 #u125
FUMmito_a0 = 0.04 # Fink 2018 10.1074/jbc.RA118.005144 #0.055 #in between of Fink20018 and From molecules to networks 2014 #u126
MALmito_a0 = 0.22 # Fink 2018 10.1074/jbc.RA118.005144 and Biochemistry book 2012 Reginald H. Garrett, Charles M. Grisham #;0.45 #0.45 #in between of googledoc sources and From molecules to networks 2014  # u127
OXAmito_a0 = 0.004 #From molecules to networks 2014 (book) #u128 # several diff values: 0.004; 0.04; 0.08; 2.0
AcCoAmito_a0 = 0.0045 #mM Ronowska 2018  #0.07 #0.0045 #Nazaret2008 #0.0045 #From molecules to networks 2014; Ronowska2018 #u129
CoAmito_a0 = 0.002 #Poliquin #0.16 #;0.03 # in between of lit values #0.16 #Mogilevskaya #0.003  #From molecules to networks 2014 #u130

CaMito_a0 = 0.0001 #set same as neuronal # Lajtha 2007 book  NeurochemistryEnergeticsBook.pdf  #0.001 Mogilevskaya 2006 #u133


#Pimito_a0 = 2.7 #;5.0;2.0 #2.5 #5.0 #2.7 #2.0 #From molecules to networks 2014 #  #u136



ASPmito_a0 = 1.4 # approx set as cyt, Maletic-Savatic PMID:19022759 and Nazaret 2009 10.1016/j.jtbi.2008.09.037  #0.1 #-most of files #2.0 - fn8,9,10 set same as neuron value, noref
GLUmito_a0 = 10.0 #neur Roberg PMID: 10942715 + astro 17mM doi:10.1088/1742-6596/1141/1/012028 #5.3 # Nazaret 2008 #0.057 # Chen # check cell type specificity #u142 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MAL_a0 = 0.45 #From Molecules to Networks2014   #u143   ### check subcell specificity
OXA_a0 = 0.01 # 0.1 #;0.01 #cytosol #0.01 #mM Williamson 1967  #u144    ### check subcell specificity
ASP_a0 = 1.4 #u145    ### check subcell specificity
AKG_a0 = 0.2 #From molecules to networks 2014 ##u146      ### check subcell specificity


# Glutathione Astrocyte
GSH_a0 = 4.3 # 4.3 estim average, also considering neuron/astrocyte ratio # 0.91 mM PMID: 16624809 #2.6 #McBean2017: astrocytes contain one of the highest cytosolic GSH conc 8-10 mM #0.5–10 mM -  Reed2008 doi:10.1186/1742-4682-5-8 -GSH/GSSG ~100; 90% total glutathione in cytosole is reduced (GSH) # ~2.6 mM From molecules to networks 2014 #reduced
GSSG_a0 = 0.043 #;2.88 #Reed2008 doi:10.1186/1742-4682-5-8 GSH/GSSG ~100; 90% total glutathione in cytosole is reduced (GSH) #0.43*GSH_a #Raps1989 #GSH_a/0.9 #GSH/GSSG = 90% McBean2017 # glutathione disulfide
# Glutathione Neuron Koga2011: neuronal glutathione concentrations ranging from 0.2 to 1 mM, though some have suggested concentrations as high as 10 mM
GSH_n0 = 1.2 # 1.2 estim from average # 0.21 mM PMID: 16624809  #2.0 #0.57 # 1-3 mM from Vali doi:10.1016/j.neuroscience.2007.08.028 # ~2.6 mM From molecules to networks 2014 #reduced
GSSG_n0 = 0.012 #0.022 #0.003*GSH_n #0.3% of totGSH # Reed2008 GSH/GSSG ~100 # GSH_n/0.9 #GSH/GSSG = 90% McBean2017 # glutathione disulfide


GLU_n0 = 10.0 #11.6 #10.0;11.6 # from MAS # 11.6 - From molecules to networks 2014 #check

NE_neuromod0 = 0.0 # define in callback

AMP_n0 = 1e-5 #0.01 #;0.03;2.0e-5 #2e-5 #0.03 #-Mulukutla2015 #2e-5 # Lambeth # 0.01 # 2e-5 # 1e-5 # from 1e-5 to 0.05 # 0.00017 to 0.05 mM PMID: 2642915

GLN_n0 = 0.4 # mM Shestov 2007 # for now no separation between cyto and mito GLN

GABA_n0 = 1.0 #mM check! Diff for diff n types
GABA_inh_n0 = 5.0 # mM # Yamashita et al., 2018 : suggesting that endogenous GABA concentrations in BCs are 5 mM, as in other inhibitory neurons (Apostolides and Trussell, ) ... 1-10 mM Yamashita et al., 2018  # 3mM released and  https://doi.org/10.3389/fnsyn.2018.00040

vV0 = 0.0237 # ml global_par_Compartment_9 0.02396 - in pdf # 0.0237 # - in Winter2017 matlab file  # 0.02 in Jolivet2015
ddHb0 = 0.05 #0.058 #0.0478 #0.058 #mM Jolivet2015 # 0.0478 #Winter2017 # 0.12 Aubert2007 doi 10.1073/pnas.0605864104 # 0.063 Auebrt 2002 doi:10.1006/nimg.2002.1224 and Buxton et al. (1998a,b)

### ketones

bHB_n0 = 0.00156 # mM #from brain-to-plasma bHB ratio and bHB_b 0.078*0.02 # beta-HydroxyButyrate   # brain-to-plasma bHB ratio = 0.078 # Fig5 Chowdhury doi:10.1038/jcbfm.2014.77
# bHB_n0 = 0.02 to 0.18 mM # Cobelli_ketoModelHuman # 0.15 mM Cornille
AcAc_n0 = 0.00312 #inferred from ratio 2*0.00156 ratio of AcetoAcetate to bHb in the brain: 2 to 4.5  # Nehlig 1992 # AcetoAcetate
# AcAc_n0 = 0.03 to 0.1 mM # Cobelli_ketoModelHuman # 0.3 mM Cornille
# AcAc_n0 = 0.063 to 1.6 mM # Berndt2018 Hepatokin1
AcAcCoA_n0 = 0.0006 #mM Berndt2018 Hepatokin1 Menahan1981 # AcetoAcetyl-CoA
#astrocyte
bHB_a0 = 0.00156 # set to the same value as in the neuron, need to search for more precise data # beta-HydroxyButyrate
AcAc_a0 = 0.00312 #inferred from ratio 2*0.00156 ratio of AcetoAcetate to bHb in the brain: 2 to 4.5  # Nehlig 1992 # AcetoAcetate
# AcAc_a0 = 0.03 to 0.1 mM # Cobelli_ketoModelHuman # 0.3 mM Cornille
# AcAc_a0 = 0.063 to 1.6 mM # Berndt2018 Hepatokin1
#HMGCoA_a0 = # 3-hydroxy-3-methylglutaryl-CoA
AcAcCoA_a0 = 0.0006 #mM Berndt2018 Hepatokin1 Menahan1981  # AcetoAcetyl-CoA
# ecs
bHB_ecs0 = 0.002 # # 0.05/25 using plasma:ecs ratio from Achanta and Rae (2017) # #0.05 # set to the same value as in the blood plasma, need to search for more precise data # beta-HydroxyButyrate
#blood
#bHB_b0 = 0.05 #mM in blood plasma Ronowska 2018
bHB_b0 = 0.18 #mM in adult wide range for diff develop stages in Table 2 Nehlig 1992 #0.02 mM # Nehlig 2003 # beta-HydroxyButyrate    # 1.27 mM for awake rats fassted for 2 days, 9.1-12 mM is for hyperketonemia # Chowdhury doi:10.1038/jcbfm.2014.77
# bHB0 in plasma >= 0.2 mM -> hyperketonaemia Robinson&williamson 1980 (Evans 2020)
fattyAcids0 = 0.14 #mM Persson 2010 human #0.43 # mM # 0.13 mM at birth rat, which represent ~ 30% of adult value (Yeh and Zee 1976) - Nehlig 1992

##### concentrations as constant parameters
# from optim/param_estim/from_desktop/TCA_astro_17feb2020.ipynb
C_Mg_a = 0.369  #0.7 # mM  # 0.369 mM 0.4 mM -Mulquiney1999  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
C_Mg_n = 0.369  #0.7 # mM  # 0.369 mM 0.4 mM -Mulquiney1999  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
BPG23_a = 0.237 #mM -Park2016 #3.1 # mM #for now fixed; check value  #C_23BPG_a = 3.1 # mM #BPG23
BPG23_n = 0.237 #mM -Park2016 #3.1 # mM #for now fixed; check value  #C_23BPG_a = 3.1 # mM #BPG23
GBP_a = 0.01 #0.6 #0.3 #0.1 #mM  #for now fixed; check value # G16bp #Quick1974: 10 μm and 600 μm #C_g16bp_a = 0.1 # mM #GBP
GBP_n = 0.01 #0.6 #0.3 #0.1 #mM  #for now fixed; check value # G16bp #Quick1974: 10 μm and 600 μm #C_g16bp_a = 0.1 # mM #GBP
C_ALA_a = 0.65 #-From Molecules to networks 2014 #1.0 #mM -Mulukutla2015  Alanine ### check more specific neuron and astroyte values
C_ALA_n = 0.65 #-From Molecules to networks 2014 #1.0 #mM -Mulukutla2015  Alanine ### check more specific neuron and astroyte values

C_H_cyt_a = 3.981071705534969e-5 #3.981e-5 #6.31e-05 #1e3*(10^(-7.4)) #mM #Arce-Molina2019 biorxiv #10^(-7.3) # 10^(-7.01-7.4) M
C_H_cyt_n = 3.981071705534969e-5 #3.981e-5 #6.31e-05 #1e3*(10^(-7.4)) #mM #Arce-Molina2019 biorxiv #10^(-7.3) # 10^(-7.01-7.4) M
C_H_mito_a = 1.58e-05 #1e3*(10^(-7.8)) #mM #Arce-Molina2019 biorxiv #10^(-8) #10^(-7.8)
C_H_mito_n = 1.58e-05 #1e3*(10^(-7.8)) #mM #Arce-Molina2019 biorxiv #10^(-8) #10^(-7.8)

#Hin_n = 7.01 #7.0-7.4 Wiki # # from NEDERGAARD 1991: 7.01/7.24 # hcyt/hext  #
#Hmito_n = 7.8 # Mito mattrix Wiki # check it more precisely    ############################################################# check!!

C_O2_mito_a = 0.01 #mM # Zhang2018 supp, consistent with whole cell O2 in Calvetti2018
C_O2_mito_n = 0.01 #mM # Zhang2018 supp, consistent with whole cell O2 in Calvetti2018
PPi_a0 = 0.0062 #mM pyrophosphate
PPi_n0 = 0.0062 #mM pyrophosphate
CO2_a = 1.2 #mM in cytosol Physiology of astroglia... book Verkhratsky #0.001 # mM from Sabate 1995 # PPP #CO2_a = 0.001 # mM from Sabate 1995  # 0.012 mM Winter2017
CO2_mito_a = 1.2 #21.4 #Wu2007 #or 1.2 as cytosol astrocyte?
CO2_n = 1.2 #mM in cytosol Physiology of astroglia... book Verkhratsky #0.001 # mM from Sabate 1995 # PPP #CO2_a = 0.001 # mM from Sabate 1995  # 0.012 mM Winter2017
CO2_mito_n = 1.2 #21.4 #Wu2007 #or 1.2 as cytosol astrocyte?
h2o_m_a = 1.0 # H2O is not taken for the analysis but only participate in rns to balance stoichiometries
h2o_m_n = 1.0 # H2O is not taken for the analysis but only participate in rns to balance stoichiometries


###################
# Theurey2019
###################

C_H_mitomatr_n0 = 1.82084728579186E-05 #Theurey2019 #1e-05 #mM # pH=8.0 -> C = 10^(-8) M
C_H_mitomatr_a0 = 1.82084728579186E-05 #Theurey2019 #1e-05 #mM # pH=8.0 -> C = 10^(-8) M

K_x_n0 = 54.8851390930353
K_x_a0 = 54.8851390930353
Mg_x_n0  =  0.413984791104507
Mg_x_a0  =  0.413984791104507

NADHmito_n0 = 0.35662010403786 #mito matrix #0.07 
NADHmito_a0 = 0.35662010403786 #mito matrix #0.07 

QH2mito_n0 = 0.03858504874339510 
QH2mito_a0 = 0.03858504874339510 

CytCredmito_n0 = 0.182721863756137 
CytCredmito_a0 = 0.182721863756137


ATPmito_n0 = 1.074997491836194   # ATP_x # !!!!!!!!
ADPmito_n0 = 1.5250025081638081  # ADP_x # !!!!!!!!

ATPmito_a0 = 1.074997491836194   # ATP_x # 1.5mM CLARKE AND NICKLASS 1970 ratBrainMito.pdf
ADPmito_a0 = 1.5250025081638081  # ADP_x

ATP_mx_n0   =  0.7343956789485272  # Matrix ATP bound to magnesium
ADP_mx_n0   =  0.8296195300562087  # Matrix ADP bound to magnesium
ATP_mx_a0   =  0.7343956789485272  # Matrix ATP bound to magnesium
ADP_mx_a0   =  0.8296195300562087  # Matrix ADP bound to magnesium

Pimito_n0 = 16.099237382306928 # Pi_x #1.0 # mM https://doi.org/10.1016/j.mbs.2021.108646
Pimito_a0 = 16.099237382306928 # Pi_x #1.0 # mM https://doi.org/10.1016/j.mbs.2021.108646

ATP_i_n0   =  2.3183097760668474  
ADP_i_n0   =  0.2912160581175691 
ATP_i_a0   =  2.3183097760668474  
ADP_i_a0   =  0.2912160581175691 

AMP_i_n0   = 1e-5 #because diff ATP_n #0.0
AMP_i_a0   = 1e-5 #because diff ATP_a #0.0

ATP_mi_n0   =  2.295947266871272    # IMS ATP bound to magnesium
ADP_mi_n0   =  0.28617845665402003  # IMS ADP bound to magnesium
ATP_mi_a0   =  2.295947266871272    # IMS ATP bound to magnesium
ADP_mi_a0   =  0.28617845665402003  # IMS ADP bound to magnesium

Pi_i_n0   =  19.99966122867552 
Pi_i_a0   =  19.99966122867552 

MitoMembrPotent_n0 = 149.170131272782
MitoMembrPotent_a0 = 149.170131272782

Ctot_n0   = 2.7
Ctot_a0   = 2.7

Qtot_n0   = 1.35
Qtot_a0   = 1.35

C_H_ims_n0 = 3.98107170548711E-05 # Theurey2019 #1e-05 1e-04 # 1e3*10^(-7) #mM # pH=7.0 -> C = 10^(-7) M
C_H_ims_a0 = 3.98107170548711E-05 # Theurey2019#1e-05  #1e-04 # 1e3*10^(-7) #mM # pH=7.0 -> C = 10^(-7) M

#################
# ###
# C_H_mitomatr_n  =  1.82084728579186E-05
# K_x_n  =  54.8851390930353
# Mg_x  =  0.4139847911045070
# NADH_x  =  0.3566201040378600
# QH2  =  0.03858504874339510
# Cred  =  0.18272186375613700
# O2  =  0.03  
# ATP_x  =  1.074997491836194  
# ADP_x  =  1.5250025081638081  

# ATP_mx   =  0.7343956789485272  
# ADP_mx   =  0.8296195300562087  
# Pi_x   =  16.099237382306928

# ATP_i   =  2.3183097760668474  
# ADP_i   =  0.2912160581175691 
# AMP_i   = 0.0

# ATP_mi   =  2.295947266871272  
# ADP_mi   =  0.28617845665402003  

# Pi_i   =  19.99966122867552 

# dPsi   = 149.17013127278200

# Ctot   = 2.7
# Qtot   = 1.35

# H_i   = 3.98107170548711E-05

# # ATP_e_n0   =  1.4 #1.400216  
# # ADP_e_n0   =  0.048 #0.047784 
# # ATP_e_a0 = 1.3 # == Init[23]     # ATP_c, cytosolic ATP astrocyte
# # ADP_e_a0 = 0.045 # == Init[24]   # ADP_c, cytosolic ADP astrocyte

#################

# tmp, replaced with tot-red
NADmito_n0 = 0.36937989596213994 # 0.0 #mito matrix #0.14
NADmito_a0 = 0.36937989596213994 #0.0 #mito matrix #0.14  

Qmito_n0 = 1.3114149512566051 #0.0 #0.004 #mM rat brain  https://academicjournals.org/article/article1380896866_Abdallah%20et%20al.pdf PMID: 20385196 

Qmito_a0 = 1.3114149512566051 #0.0 #0.004 #mM rat brain  https://academicjournals.org/article/article1380896866_Abdallah%20et%20al.pdf PMID: 20385196  


#CytCoxmito_n0 = #0.0588 # ~80% Cooper PMID: 9620863 #ratio from PMID: 19029908, sum from JOHN B. CLARKE AND WILLIAM J. NICKLASS 1970 ratBrainMito.pdf #also CytCredox and CuCCoxidase redox are close: doi:10.1371/journal.pcbi.1000212  # 0.063 # CytCmito total = 0.07 mM JOHN B. CLARKE AND WILLIAM J. NICKLASS 1970 ratBrainMito.pdf and 10% cytc red from fig 5 febs.14151.pdf Heiske doi:10.1111/febs.14151  #0.02 #;0.21  #Zhang2018  #0.21 #0.315006*(33.3506/50.0) = 0.21011278207200001 #0.09# #0.136275*(33.3506/50.0)= 0.09089706030000001 inferred from IvanChang steady state #0.0186

#CytCredmito+CytCoxmito=30 in IvanChang(default,but commented) -> 0.03
# ratio from Cytochrome c is rapidly reduced in the cytosol after mitochondrial outer membrane permeabilization Ripple 2010: 62% mito cyt C oxidized
# CytCredmito+CytCoxmito=50 not commented -> 0.136275 scaled to Jay
#CytCmito total = 0.07 mM JOHN B. CLARKE AND WILLIAM J. NICKLASS 1970 ratBrainMito.pdf

#CytCoxmito_a0 = #0.0588 # 80% Cooper PMID: 9620863  #ratio from PMID: 19029908, sum from JOHN B. CLARKE AND WILLIAM J. NICKLASS 1970 ratBrainMito.pdf  #also CytCredox and CuCCoxidase redox are close: doi:10.1371/journal.pcbi.1000212  #0.063 # CytCmito total = 0.07 mM JOHN B. CLARKE AND WILLIAM J. NICKLASS 1970 ratBrainMito.pdf and 10% cytc red from fig 5 febs.14151.pdf Heiske doi:10.1111/febs.14151 #0.02 #;0.21  #0.02 Zhang2018 # 0.21  #u140 #0.315006*(33.3506/50.0) = 0.21011278207200001 #0.09# #0.136275*(33.3506/50.0)= 0.09089706030000001 inferred from IvanChang steady state #0.0186


