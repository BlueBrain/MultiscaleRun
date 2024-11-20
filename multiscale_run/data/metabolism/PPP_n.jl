### PPP: Winter2017, Stincone 2015; Nakayama 2005; Sabate 1995 rat liver; Kauffman1969 mouse brain; Mulukutla(2015) Multiplicity of Steady States in Glycolysis and Shift of Metabolic State in Cultured Mammalian Cells. PLoS ONE; Cakir 2007 


# PPP_r01_G6PDH

# ZWF = Glucose 6-phosphate Dehydrogenase (G6PDH):  Glucose 6-phosphate(G6P) + NADP+(NADP) ↔ 6-phospho-glucono-1,5-lactone(GL6P) + NADPH + H+ ###
VmaxG6PDH_n = 2413.389592436078 #0.150144617884625 # 0.130010113583145 #0.53895900801198 #0.586458
KeqG6PDH_n = 0.00674385476573124 #0.00548183269281042 #0.005002081708789302 #22906.4 
K_G6P_G6PDH_n = 0.0109183101365999 #6.91392e-05 #0.01  #0.0142523987811961 # 0.0158229604461531 #0.0261692706915093 #6.91392e-05
K_NADP_G6PDH_n = 0.0835423944062894 #0.10608175747939 # 0.117925904076472 #0.0621260938598377 #1.31616e-05
K_GL6P_G6PDH_n = 0.0180932 #0.01
K_NADPH_G6PDH_n = 5.7841990084678E-05 # 2.87650878408365E-05 # 2.51285420418258E-05 #1.05119349399397E-05 #0.00050314


# r02: 6PGL

# SOL = 6-Phosphogluconolactonase (6PGL): 6-Phosphoglucono-1,5-lactone(GL6P) + H2 O → 6-phosphogluconate(GO6P) 
Vmax6PGL_n = 8711.029873731239 #0.00162744832170299 #0.0779750236619484 #0.300002217429445 #0.373782
Keq6PGL_n = 961.0 #933.79300277648 #2116.69452134578 #531174.
K_GL6P_6PGL_n = 0.0180932
K_GO6P_6PGL_n = 2.28618


# r03: GND = 6-Phosphogluconate Dehydrogenase (6PGDH): 6-Phosphogluconate(GO6P) + NADP+ → ribulose 5-phosphate(RU5P) + CO2 +NADPH+H+

# GND = 6-Phosphogluconate Dehydrogenase (6PGDH): 6-Phosphogluconate(GO6P) + NADP+ → ribulose 5-phosphate(RU5P) + CO2 +NADPH+H+
Vmax6PGDH_n = 80.41773386353813 #2.39373833940569 #2.50428994017528 #2.0000015929249 #2.6574
Keq6PGDH_n = 23.4988291155184 #23.2831288483441 #23.4988291155184 #23.389469865437693 #4.0852e+07
K_GO6P_6PGDH_n = 3.23421e-05
K_NADP_6PGDH_n = 0.0330619252781597 #0.0576096039029965 #0.199999612436424 #3.11043e-06
K_RU5P_6PGDH_n = 0.0537179
K_NADPH_6PGDH_n = 5.2529395568622E-05 #0.000312185665026002 #2.00000197818523E-05 #0.00050314


# r04: RKI =  Ribose Phosphate Isomerase (RPI): Ribulose 5-phosphate(RU5P) ↔ ribose 5- phosphate(R5P)
# RKI =  Ribose Phosphate Isomerase (RPI): Ribulose 5-phosphate(RU5P) ↔ ribose 5- phosphate(R5P)
VmaxRPI_n = 0.013057366069767283 #2.27562615941395E-06 #1.52507680529037E-05 #0.0010000001580841 #0.00165901
KeqRPI_n = 0.587090549959032 #19.3244443714851 #35.4534 
K_RU5P_RPI_n = 0.0537179
K_R5P_RPI_n = 0.778461


# r05: Ribulose Phosphate Epimerase (RPE): Ribulose 5-phosphate(RU5P) ↔ xylulose 5-phosphate(X5P)
# R5PE has about 10‐fold higher activity in mammalian tissue than does R5PI (Novello and McLean, 1968) - book Neurochemistry Lajtha 2007 NeurochemistryEnergeticsBook.pdf 
# Ribulose Phosphate Epimerase (RPE): Ribulose 5-phosphate(RU5P) ↔ xylulose 5-phosphate(X5P)
VmaxRPE_n = 0.2039985464986835 #3.78141492912804E-05 #0.000613070392724124 #0.0100000519904446 #0.0156605
KeqRPE_n = 34.9820646657167 #32.2532017229088 #39.2574
K_RU5P_RPE_n = 0.0537179
K_X5P_RPE_n = 0.603002


# r06  Transketolase1

# Transketolase TKL 1,2
VmaxTKL1_n = 0.004950133634259906 #4.99999381232524E-05 #9.94132171852896E-05 #0.000100000000723543 #0.000493027
KeqTKL1_n = 9860.43282385396 #23251.6522242321 #6.136173332909908e6 #1652870.0 
K_X5P_TKL1_n = 0.000173625
K_R5P_TKL1_n = 0.000585387
K_GAP_TKL1_n = 0.00499217405386077 #0.0199199125588176 #0.100000072700887 #0.168333
K_S7P_TKL1_n = 0.192807


# r07  Transketolase2

VmaxTKL2_n = 0.030468555594855798 #5.61436923030841E-06 #1.39858900748084E-05 #0.000121191639770347 #0.000276758
KeqTKL2_n = 0.142052917466236 #0.0585732500846428 #0.07821611436297593 #0.07776450379275156 #0.0777764
K_F6P_TKL2_n = 1.38310526172865 #1.46286766935271 # 0.458132513897045 #0.0799745
K_GAP_TKL2_n = 0.123421366286679 #0.0998077937106258 #0.892390735815738 #0.168333
K_X5P_TKL2_n = 0.603002
K_E4P_TKL2_n = 0.109681


# r08 Transaldolase (TAL): S7P + GAP ↔ E4P + F6P

# Transaldolase (TAL): S7P + GAP ↔ E4P + F6P

VmaxTAL_n = 77.12873595245502 #0.000672290592103076 #0.0010141429158538 #0.00500000013544425 #0.0162259
KeqTAL_n = 3.0 # 0.0649345422904358 #0.0153189035204859 #0.015129601518014119 #0.323922
K_GAP_TAL_n = 1.434093758861 #1.14526908778254 #0.899999415343457 #0.168333
K_S7P_TAL_n = 0.192807
K_F6P_TAL_n = 0.00045967540870832 #0.00557726396440652 #0.015000019459948 #0.0799745
K_E4P_TAL_n = 0.109681


# NADPH oxidase 
k1NADPHox_n = 0.0009369376443550519 #5.53110470372277E-06 #3.78431669786565E-05 #0.000423283 #0.00281384
