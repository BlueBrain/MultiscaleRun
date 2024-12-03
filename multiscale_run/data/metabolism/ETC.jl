# diff from Theurey2019:

# ADTPtot; total Adenosine phosphates in cytosol  # needs opt
# state_fact # needs opt
# Init[7] # O2  # Theurey2019 works with my O2


#ADTP_tot_n    = 1.448e-3 #    #1e-3*(ATP_n0 + ADP_n0) #2.6e-3  # total Adenosine phosphates in cytosol, calculated from Evans_77         
#state_fact  = 0.967  #10/11      #ATP_n0/(ATP_n0 + ADP_n0)  #10/11 

# ATP_e       = state_fact*ADTP_tot  # ini ATP level
# ADP_e       = ADTP_tot-ATP_e       # ini ADP level


######################################################


# fixed par
etcF       = 0.096484 # kJ mol^{-1} mV^{-1}
etcR       = 8314e-6  # Universal gas constant (kJ * mol^{-1} * K^{-1}]
etcT       = (273.15 + 37)             # Temperature (K), 37 degree
etcRT      = etcR*etcT                       # kJ  mol^{-1}


dG_C1o  = -69.37                    # kJ mol^{-1}
dG_C3o  = -32.53                    # kJ mol^{-1}
dG_C4o  = -122.94                   # kJ mol^{-1}
dG_F1o  = 36.03                     # kJ mol^{-1}
n_A     = 3.0                       # numbers of proteins used by ATP synthase

# concentrations and pH
#pH_e    = 7.4            # External pH (cytosol)    
#H_e     = 10^(-pH_e)  == C_H_cyt_n   #  cytosolic hydrogen concentration (Molar)
K_i    = 120e-3         # K_i IM potassium-concentration
Mg_tot  = 20e-3          # Mg_tot;  IM magnesium-concentration
#Pi_e    = 20e-3          # Pi_e    IM phosphate-concentration

NADtot      = 726e-6     # NADtot mito   
#Ctot0       = 2.70e-3    # Cyt-c (total IMS Cyt-C, Cred+Cox, M]  # Cyt-c from Beard. Total IMS Cyt-c, Cred+Cox, molar
#Qtot0       = 1.35e-3    # total IMS Ubiquinol, Q+QH2, M




W_c     = 1/0.0575  # Volume fraction cytosol/mitochondria # 0.0575 is mito vol fraction from supp of Santuy2018 doi: 10.1093/cercor/bhy159 #1/0.06 0.06 is mitochondrial fraction Ward 2007
W_m     = 0.143/0.20;               # mitochondrial water space (ml water / ml mito]
W_x     = 0.9*W_m;                  # Matrix water space (ml water / ml mito]
W_i     = 0.1*W_m;                  # IM water space (ml water / ml mito]




# Potassium-Hydrogen-Antiport
x_KH   =  2.9802e7            # x_KH      # K+ / H+ antiporter activity

# Matrix buffer and membrane capacitance
CIM     = 6.7568e-6           # CIM       # Inner Membrane capacitance

#c_inc   = 1               # Cyt-c after release set to 0.1#
#t12cyto = 90;                       # cyt-c release half time Waterhouse 2000 (seconds)
#Cfin    = Ctot0*W_i/W_c*c_inc;      # final cytochrome c after release in IMS given by re-equilibration of the
# IMS water space with the rest of the cell (0.1#)
# c_inc is used to modify OXPHOS-contributing cyt-c after release

# potassium uniporter and adenylate kinase neglected
x_K    = 0;                         # Passive potassium transporter activity
x_AK   = 0                       # AK activity
K_AK   = 0;                        # Adenelyte Kinase switched off


# Parameters for complex I
x_C1   =  1.0200e+003          # x_C1       # Complex I activity

# Parameters for complex III
x_C3   =  0.2241              # x_C3      # Complex III activity
k_Pi3  =  0.192e-3            # k_Pi3     # Complex III / Pi parameter
k_Pi4  =  25.31e-3            # k_Pi4     # Complex III / Pi parameter

# Parameters for complex IV
x_C4   =  3.2131e-004         # x_C4      # Complex IV activity
k_O2    = 1.2e-4              # k_O2      # kinetic constant for complex IV M


# Parameters for OM transporters
gamma   = 5.99                # gamma     # MOM area per unit volume micron^{-1}
x_A     = 85.0                # x_A       # MOM permeability to nucleotides (micron s^{-1}) 
x_Pi2   = 327.0                 # x_Pi2     # MOM permeability to phosphate (micron s^{-1})

# Phosphate-Hydrogen-Cotransport
k_dHPi  = 10^(-6.75)          # k_dHPi    # H/Pi co-transpor Molar form factor 1 binding
k_PiH   = 0.45082e-3          # k_PiH     # H+/Pi co-transport activity form factor 2 Michaelis constant
x_Pi1   = 3.85e5              # x_Pi1     # H+/Pi co-transport activity

# Parameters for ATP synthase and Mg-binding to ATP?ADP
x_MgA  =  1e6                 # x_MgA     # Mg2+ binding activity

# Parameters for input function
k_Pi1  =  0.13413e-3   # k_Pi1      # Dehydrogenase flux input
k_Pi2  =  0.677e-3     # k_Pi2      # Dehydrogenase flux input

######################################################

# Proton leak activity
x_Hle  =  149.61265600173826 #150.0               # x_Hle     # Proton leak activity
x_Ht    = 2044.5370854408536 #2000.0               # x_Ht      # MOM permeability to protons (micron s^{-1})

# adj for eq: 0.9654561819314937
r_DH   = 4.35730398512522 # 4.3          # r_DH       # Input-flux: Initial disturbance of equilibrium

x_buff =  1437.0652678883314 # 200/(0.5*T2Jcorrection) # #100                 # x_buff    # Inner Matrix hydrogen buffer capacity


# optimised parameters

# Parameters for input function
x_DH   = 0.0478558445772299 # 0.04702218860608453 #0.05896      # x_DH       # Dehydrogenase activity

# Parameters for Adenosine Transferase
x_ANT  = 0.0647001029937813 # 0.07649900124686723 #0.0020              # x_ANT     # ANT activity
k_mADP = 1.03045032866899E-05 # 1.0588243052994988e-5 #3.50e-6             # k_mADP    # ANT Michaelis-Menten constant


# Cytosolic ATP production and consumption
# K_ADTP_dyn = 0.5912763309268644 #3.42      # K_ADTP_dyn; Cytosolic ATP production
# x_ATPK   = 0.30499319022776666 # 0.504    # x_ATPK 
# K_ADTP_cons = 1.1895556584725608 #1.0     # K_ADTP_cons; Cytosolic ATP consumption 



# Parameters for ATP synthase and Mg-binding to ATP?ADP
x_F1   =  7099.66908851658 #6912.715469614502 #6829.4        # x_F1      # F1Fo ATPase activity
K_DT    = 0.00166023273526023 #0.001855689326651793 #0.000192             # K_DT      # Mg/ATP binding constant (M)
K_DD    = 0.000409766157053928 #0.00036972016989755054 #347e-6              # K_DD      # Mg/ADP binding constant (M)

######################################################
# Astrocyte

#ADTP_tot_a    = 1.345e-3 #    #1e-3*(ATP_n0 + ADP_n0) #2.6e-3  # total Adenosine phosphates in cytosol, calculated from Evans_77         

# ATP_e0 = 1.3 # == Init[23]     # ATP_c, cytosolic ATP 
# ADP_e0 = 0.045 # == Init[24]   # ADP_c, cytosolic ADP 


# only params that may differ from neuronal

# Proton leak activity
x_Hle_a  = 149.61265600173826 # 150.0               # x_Hle     # Proton leak activity

# Potassium-Hydrogen-Antiport
x_KH_a   =  2.9802e7            # x_KH      # K+ / H+ antiporter activity

# Matrix buffer and membrane capacitance
x_buff_a =   1437.0652678883314 # 200/(0.5*T2Jcorrection) #  100                 # x_buff    # Inner Matrix hydrogen buffer capacity


# potassium uniporter and adenylate kinase neglected
x_K_a    = 0;                         # Passive potassium transporter activity
x_AK_a   = 0                       # AK activity
K_AK_a   = 0;                        # Adenelyte Kinase switched off


# Parameters for complex I
x_C1_a   =  1.0200e+003          # x_C1       # Complex I activity

# Parameters for complex III
x_C3_a   =  0.2241              # x_C3      # Complex III activity
k_Pi3_a  =  0.192e-3            # k_Pi3     # Complex III / Pi parameter
k_Pi4_a  =  25.31e-3            # k_Pi4     # Complex III / Pi parameter

# Parameters for complex IV
x_C4_a   =  3.2131e-004         # x_C4      # Complex IV activity
k_O2_a    = 1.2e-4              # k_O2      # kinetic constant for complex IV M


# Parameters for OM transporters
x_Ht_a    = 2044.5370854408536 #2000.0               # x_Ht      # MOM permeability to protons (micron s^{-1})
gamma_a   = 5.99                # gamma     # MOM area per unit volume micron^{-1}
x_A_a     = 85.0                # x_A       # MOM permeability to nucleotides (micron s^{-1}) 
x_Pi2_a   = 327.0                 # x_Pi2     # MOM permeability to phosphate (micron s^{-1})

# Phosphate-Hydrogen-Cotransport
k_dHPi_a  = 10^(-6.75)          # k_dHPi    # H/Pi co-transpor Molar form factor 1 binding
k_PiH_a   = 0.45082e-3          # k_PiH     # H+/Pi co-transport activity form factor 2 Michaelis constant
x_Pi1_a   = 3.85e5              # x_Pi1     # H+/Pi co-transport activity

# Parameters for ATP synthase and Mg-binding to ATP?ADP
x_MgA_a  =  1e6                 # x_MgA     # Mg2+ binding activity

# Parameters for input function
k_Pi1_a  =  0.13413e-3   # k_Pi1      # Dehydrogenase flux input
k_Pi2_a  =  0.677e-3     # k_Pi2      # Dehydrogenase flux input


######################################################

# optimised parameters

# adj for eq: 0.9654561819314937
r_DH_a   = 4.51551099390501 # 4.3          # r_DH       # Input-flux: Initial disturbance of equilibrium


# Parameters for input function
#x_DH_a   =  0.04702218860608453 #0.05896      # x_DH       # Dehydrogenase activity
x_DH_a = 0.014771869267445427 #0.014885075252276352 #0.0151125164947073 #0.014882801669692953 # 0.041477965015268535 # 0.0412 # 0.04150217281642099 # 0.04702218860608453

# Parameters for Adenosine Transferase
x_ANT_a  = 0.0881948452324846 # 0.07649900124686723 #0.0020              # x_ANT     # ANT activity
k_mADP_a = 1.07960040439672E-05 # 1.0588243052994988e-5 #3.50e-6             # k_mADP    # ANT Michaelis-Menten constant


# Cytosolic ATP production and consumption
# K_ADTP_dyn_a = 0.5912763309268644 #3.42      # K_ADTP_dyn; Cytosolic ATP production
# x_ATPK_a   = 0.30499319022776666 # 0.504    # x_ATPK 
# K_ADTP_cons_a = 1.1895556584725608 #1.0     # K_ADTP_cons; Cytosolic ATP consumption 



# Parameters for ATP synthase and Mg-binding to ATP?ADP
x_F1_a  = 6306.50439347496 # 6912.715469614502 #6829.4        # x_F1      # F1Fo ATPase activity
K_DT_a  = 0.00194323827864375 #0.001855689326651793 #0.000192             # K_DT      # Mg/ATP binding constant (M)
K_DD_a  = 0.000328048467890892 #0.00036972016989755054 #347e-6              # K_DD      # Mg/ADP binding constant (M)

