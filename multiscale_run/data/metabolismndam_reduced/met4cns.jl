function metabolism(du,u,p,t)
    
    """
        This function defines the metabolism model in a system of ordinary differential equations. It is called from main.py in multiscale_run software. It can also be used independently of main.py from pure Julia programming language code script/notebook. 
    
        It is based on Calvetti 2018 paper https://doi.org/10.1016/j.jtbi.2018.02.029
    
        For the details on Julia Differential Equations please see https://docs.sciml.ai/DiffEqDocs/stable/ (this webpage also provides the tutorial to help you run simulation with this function from pure Julia without multiscale_run if need be).


    Args: 

                      Formally, the args are du,u,p,t. In fact, du is a placeholder to keep the solution of the system. It should be the same size as u, which is an initial values list. Next, p is a list of parameters and t is a time span. Please see the details on arguments in https://docs.sciml.ai/DiffEqDocs/stable/  In diffeqpy interface used in multiscale run we just define u,t,p as input args.


    Returns:

                      Formally, there is no "return" of this function, but the function itself is being used for metabolism dynamics simulation from main.py as shown below:
                      prob_metabo = de.ODEProblem(metabolism, vm, tspan_m, param)
    In a way the function used with de.ODEProblem it defines an ODE system problem to be solved with
    sol = de.solve( prob_metabo, de.Rosenbrock23(autodiff=False), reltol=1e-8, abstol=1e-8, saveat=1, maxiters=1e6, )

    """
    
    
    ina_density,ik_density,mito_scale, notBigg_FinDyn_W2017, notBigg_Fout_W2017, u_notBigg_vV_b_b  = @. p # p[]
    
    
    # constants

    R = 8.31 # J/(K*mol)
    T = 310.0 # Kelvin, temperature 310K is 37 C 
    F = 96.485 # 96485.0  C/mol 

    Cm = 1.0 # uF/cm2 Calvetti2018 capacitance
    phi = 3.0 # 1/ms Calvetti2018 time constant

    # Conductance param from Calvetti2018 and Cressman2011
    gNa = 100.0 # mS/cm2
    gK = 40.0 # mS/cm2
    gNa0leak = 0.0175 # mS/cm2
    gK0leak = 0.05 # mS/cm2
    gCl = 0.05 # mS/cm2

    beta = 1.33 # Calvetti2018; in Cressman2011 it was set to 7.0
    ksi_rest = 0.06 #0.15 # 0.06 is for resting system # 2.5 #  Calvetti2018; external stimuli, 2.5 should give 90 Hz # this can be decreased for smaller Hz
    ksi_act =  2.5 #  Calvetti2018; external stimuli, 2.5 should give 90 Hz # this can be decreased for smaller Hz
    gamma = 0.0445 # Calvetti2018 mM*cm2/uC

    #rho=1.25 # Cressman2011
    #epsilon=1.333333333 # Cressman2011
    #kbath=4.0 # Cressman2011
    #glia=66.666666666 # Cressman2011
    rho=13.83 # mM/s # Calvetti2018
    epsilon=9.33 # 1/s # Calvetti2018
    kbath=6.3 # mM # Calvetti2018
    glia=20.75 # mM/s # Calvetti2018

    s_metabo_atpase = 0.15 # Calvetti2018 tuned this param from OGI
    H1_metabo_atpase = 0.071667 # mM/s # 4.3 #mM/min
    sigma_atpase = 103.0 # assumed by Calvetti2018


    mu_pump_ephys = 0.1 # Calvetti2018
    mu_glia_ephys = 0.1 # Calvetti2018

    # volume fractions and blood related parameters (table 1) Calvetti2018
    eto_n = 0.4 # volume fraction neuron
    eto_a = 0.3 # volume fraction astrocyte
    eto_ecs = 0.3 # volume fraction ecs
    eto_b = 0.04 # volume fraction blood

    Hct = 0.45 # is the volume percentage (vol%) of red blood cells in blood.
    Hb = 5.18
    KH = 0.0364 #36.4*10^(-3) mM

    C_Glc_a = 5.0 # mM arterial concentration glucose
    C_Lac_a = 1.1 # mM arterial concentration lactate
    C_O_a = 9.14 # mM arterial concentration oxygen
    Fr_blood = 2/3 # from Bayesian flux balance analysis applied to a skeletal muscle metabolic model 2007 Heino .. Calvetti

    # table 2 Calvetti2018
    TbGlc = 0.02 # mM/s  
    TbLac = 0.17 # mM/s  
    lambda_b_O2 = 0.04 # mM^(1-k)/s  
    TnGlc = 83.33 # mM/s  
    TnLac = 66.67 # mM/s  
    lambda_n_O2 = 0.94 # 1/s  
    TaGlc = 83.33 # mM/s  
    TaLac = 66.67 # mM/s 
    lambda_a_O2 = 0.68 # 1/s  

    tau = 1000.0 # Calvetti2018 

    KbGlc = 4.60 # mM
    KbLac = 5.00 # mM
    KnGlc = 5.00 # mM
    KnLac = 0.4 # mM
    KaGlc = 12500.0 # mM
    KaLac = 0.4 # mM

    # oxidative phosphorylation
    V_oxphos_n = 8.18 # mM/s  
    V_oxphos_a = 2.55 # mM/s  
    K_oxphos_n = 1.0 #mM
    K_oxphos_a = 1.0 #mM
    mu_oxphos_n = 0.01
    mu_oxphos_a = 0.01
    nu_oxphos_n = 0.10
    nu_oxphos_a = 0.10

    # glycolysis
    V_Glc_n = 0.26 # mM/s  
    V_Glc_a = 0.25 # mM/s 
    K_Glc_n = 4.6 # mM
    K_Glc_a = 3.1 # mM
    mu_Glc_n = 0.09
    mu_Glc_a = 0.09
    nu_Glc_n = 10.00
    nu_Glc_a = 10.00

    V_LDH1_n = 1436.0 # mM/s 
    V_LDH1_a = 4160.0 # mM/s 
    K_LDH1_n = 2.15 # mM
    K_LDH1_a = 6.24 # mM
    nu_LDH1_n = 0.1
    nu_LDH1_a = 0.1

    V_LDH2_n = 1579.83 # mM/s
    V_LDH2_a = 3245.0 # mM/s
    K_LDH2_n = 23.7 # mM
    K_LDH2_a = 48.66 # mM
    nu_LDH2_n = 10.00
    nu_LDH2_a = 10.00

    # Krebs
    V_TCA_n = 0.03 # mM/s  
    V_TCA_a = 0.01 # mM/s 
    K_TCA_n = 0.01 # mM
    K_TCA_a = 0.01 # mM
    mu_TCA_n = 0.01
    mu_TCA_a = 0.01
    nu_TCA_n = 10.00
    nu_TCA_a = 10.00

    V_Cr_n = 16666.67 # mM/s  
    V_Cr_a = 16666.67 # mM/s  
    K_Cr_n = 495.00 # mM
    K_Cr_a = 495.00 # mM
    mu_Cr_n = 0.01
    mu_Cr_a = 0.01

    V_PCr_n = 16666.67 # mM/s  
    V_PCr_a = 16666.67 # mM/s  
    K_PCr_n = 528.00 # mM
    K_PCr_a = 528.00 # mM
    mu_PCr_n = 100.00
    mu_PCr_a = 100.00


    #INa = p[1] # same as INaact !!!
    Na_in = u[7] #p[4] 
    #Kin = p[5] # Kin from Ndam
    IK = ik_density #p[6] #current density, uA/cm2 in Calvetti  #gK*(n^4)*(V-VK) + gKleak*(V-VK)  # based on Calvetti2018 gKleak depends on ksi and glutamate activation
    K_out = u[8] #p[7] #u8 before coupling with ndam
    
    #V = p[3]  #u[1]
    m = u[2]
    h = u[3]
    n = u[4]
    ksi = ksi_rest #p[2] # u[36] # placeholder for callback activation related to glu
    u[12] = notBigg_FinDyn_W2017
    
    JGlc = TbGlc*( u[9]/(u[9] + KbGlc) - u[13]/(u[13] + KbGlc) )
    JLac = TbLac*( u[10]/(u[10] + KbLac) - u[14]/(u[14] + KbLac) )
    JO2 = lambda_b_O2*( (u[11] + 4*Hct*Hb*(u[11]^2.5)/((KH^2.5)+(u[11]^2.5))  )^(-1) - u[15])^0.1 # k=0.1 eq3 Calvetti2018
    #JO2 = lambda_b_O2*(clamp( (u[11] + 4*Hct*Hb*(u[11]^2.5)/((KH^2.5)+(u[11]^2.5))  ),1e-12,(u[11] + 4*Hct*Hb*(u[11]^2.5)/((KH^2.5)+(u[11]^2.5))  ))^(-1) - u[15])^0.1 # k=0.1 eq3 Calvetti2018 # clamped to avoid /0
    
    
    # ECS
    
    jGlc_n = TnGlc*( u[13]/(u[13] + KnGlc) - u[18]/(u[18] + KnGlc) )
    jLac_n = TnLac*( u[14]/(u[14] + KnLac) - u[20]/(u[20] + KnLac) )
    jGlc_a = TaGlc*( u[13]/(u[13] + KaGlc) - u[19]/(u[19] + KaGlc) )
    jLac_a = TaLac*( u[14]/(u[14] + KaLac) - u[21]/(u[21] + KaLac) )
    
    jO2_n = lambda_n_O2*(u[15]- u[16])
    jO2_a = lambda_a_O2*(u[15]- u[17])
    
    
    # metabolism
    
    p_n_ratio = u[28]/u[30] #clamp(u[28],1e-12,u[28])/clamp(u[30],1e-12,u[30]) #u28 = ATP_n # u30 = ADP_n
    p_a_ratio = u[29]/u[31] #clamp(u[29],1e-12,u[29])/clamp(u[31],1e-12,u[31])
    r_n_ratio = u[32]/u[34] #clamp(u[32],1e-12,u[32])/clamp(u[34],1e-12,u[34]) # u32 = NADH_n # u34 = NAD_n
    r_a_ratio = u[33]/u[35] #clamp(u[33],1e-12,u[33])/clamp(u[35],1e-12,u[35])
    
    
    psiOxphos_n = V_oxphos_n * ((1/p_n_ratio) / ( mu_oxphos_n + (1/p_n_ratio) )) * (r_n_ratio / (nu_oxphos_n + r_n_ratio) ) * (u[16] / (u[16] + K_oxphos_n))
    psiOxphos_a = V_oxphos_a * ((1/p_a_ratio) / ( mu_oxphos_a + (1/p_a_ratio) )) * (r_a_ratio / (nu_oxphos_a + r_a_ratio) ) * (u[17] / (u[17] + K_oxphos_a))
    
    
    
    #######################
    psiGlc_n = V_Glc_n * ((1/p_n_ratio) / ( mu_Glc_n + (1/p_n_ratio) )) * ( (1/r_n_ratio) / (nu_Glc_n + (1/r_n_ratio)) ) * (u[18] / (u[18] + K_Glc_n))
    psiGlc_a = V_Glc_a * ((1/p_a_ratio) / ( mu_Glc_a + (1/p_a_ratio) )) * ((1/r_a_ratio) / (nu_Glc_a + (1/r_a_ratio) ) ) * (u[19] / (u[19] + K_Glc_a)) # u19 = Glc_a
    
    
    psiLDH1_n = V_LDH1_n *  (r_n_ratio / (nu_LDH1_n + r_n_ratio) ) * (u[22] / (u[22] + K_LDH1_n))
    psiLDH1_a = V_LDH1_a * (r_a_ratio / (nu_LDH1_a + r_a_ratio) ) * (u[23] / (u[23] + K_LDH1_a))
    
    psiLDH2_n = V_LDH2_n *  ( (1/r_n_ratio) / (nu_LDH2_n + (1/r_n_ratio) ) ) * (u[20] / (u[20] + K_LDH2_n))
    psiLDH2_a = V_LDH2_a * ( (1/r_a_ratio) / (nu_LDH2_a + (1/r_a_ratio) ) ) * (u[21] / (u[21] + K_LDH2_a))

    
    psiTCA_n = V_TCA_n * ((1/p_n_ratio) / ( mu_TCA_n + (1/p_n_ratio) )) * ( (1/r_n_ratio) / (nu_TCA_n + (1/r_n_ratio)) ) * (u[22] / (u[22] + K_TCA_n))
    psiTCA_a = V_TCA_a * ((1/p_a_ratio) / ( mu_TCA_a + (1/p_a_ratio) )) * ((1/r_a_ratio) / (nu_TCA_a + (1/r_a_ratio) ) ) * (u[23] / (u[23] + K_TCA_a))
    
    psiCr_n = V_Cr_n * (p_n_ratio/ ( mu_Cr_n + p_n_ratio )) * (u[26] / (u[26] + K_Cr_n))
    psiCr_a = V_Cr_a * (p_a_ratio / ( mu_Cr_a + p_a_ratio ))* (u[27] / (u[27] + K_Cr_a))
    
    psiPCr_n = V_PCr_n * ((1/p_n_ratio) / ( mu_PCr_n + (1/p_n_ratio) ))  * (u[24] / (u[24] + K_PCr_n))
    psiPCr_a = V_PCr_a * ((1/p_a_ratio) / ( mu_PCr_a + (1/p_a_ratio) ))  * (u[25] / (u[25] + K_PCr_a))
    

    # ATPase
    H2_metabo_atpase= 0.833*H1_metabo_atpase
    
    #JpumpNa = (p_n_ratio/(mu_pump_ephys + p_n_ratio)) * (rho/(1+exp((25.0 -u[7])/3))) * (1/(1+exp(5.5 - u[8])))
    JpumpNa = (p_n_ratio/(mu_pump_ephys + p_n_ratio)) * (rho/(1+exp((25.0 -Na_in)/3))) * (1/(1+exp(5.5 - u[8])))
    
    JgliaK = (p_a_ratio/(mu_glia_ephys + p_a_ratio)) * (glia/(1+exp((18.0 - u[8])/2.5)))
    JdiffK = epsilon*(u[8] - kbath)
    
    #Naout = 144.0 - beta*(u[7] - 11.5) # mM  
    Naout = 144.0 - beta*(Na_in - 11.5) # mM  
    
    #VNa = 26.64*log(Naout/u[7]) #26.64*log(clamp(Naout,1e-12,Naout)/clamp(u[7],1e-12,u[7])) # u7 = Na0in
    VNa = 26.64*log(Naout/Na_in) #26.64*log(clamp(Naout,1e-12,Naout)/clamp(u[7],1e-12,u[7])) # u7 = Na0in
    
    gNaleak = (1+ksi)*gNa0leak
    
    INaleak = gNaleak*(u[1]-VNa)
    INaleak0 = gNa0leak*(u[1]-VNa) # check if this is correct, not given explicitly in Calvetti2018
    INaact = ina_density #p[1] # INaleak - INaleak0
    
    #psiNKA_n = H1_metabo_atpase + s_metabo_atpase*(eto_n*JpumpNa + 0.33*(gamma/sigma_atpase)* INaact ) # INaact because ksi>0 in a model 
    psiNKA_a = H2_metabo_atpase + s_metabo_atpase*(eto_ecs*JgliaK/2.0 + 2.33*(gamma/sigma_atpase)* INaact )
    
    
    
    
    du[1]=0
    du[2]=0
    du[3]=0
    du[4]=0
    du[5]=0
    du[6]=0
    du[7]=0
    du[8]=(1/tau)*(gamma*beta*IK-2*beta*JpumpNa-JgliaK-JdiffK)
    
    du[9] = (1/eto_b) * ( (u[12]/Fr_blood)*( C_Glc_a -  u[9]) - JGlc ) # Glc_b
    du[10] = (1/eto_b) * ( (u[12]/Fr_blood)*( C_Lac_a -  u[10]) - JLac ) # Lac_b
    du[11] = (1/eto_b) * ( (u[12]/Fr_blood)*( C_O_a -  u[11]) - JO2 ) # O2_b
    
    du[12]=0 # placeholder for callback (q=A(t)*Q0, blood flow)
    
    du[13] = (1/eto_ecs) * ( JGlc - jGlc_n - jGlc_a ) # Glc_ecs
    du[14] = (1/eto_ecs) * ( JLac - jLac_n - jLac_a ) # Lac_ecs
    du[15] = (1/eto_ecs) * ( JO2 - jO2_n - jO2_a ) # O2_ecs
    
    du[16] = (1/eto_n) * ( jO2_n + (-1)*psiOxphos_n ) # O2_n
    du[17] = (1/eto_a) * ( jO2_a + (-1)*psiOxphos_a ) # O2_a
    
    du[18] = (1/eto_n) * (jGlc_n + (-1)*psiGlc_n)  # u[18] = Glc_n
    du[19] = (1/eto_a) * (jGlc_a + (-1)*psiGlc_a)    # u[19] = Glc_a
    
    du[20] = (1/eto_n) * (jLac_n + psiLDH1_n + (-1)*psiLDH2_n)  # u[20] = Lac_n
    du[21] = (1/eto_a) * (jLac_a + psiLDH1_a + (-1)*psiLDH2_a)  # u[21] = Lac_a
    

    du[22] = (1/eto_n) * ( 2*psiGlc_n  + (-1)*psiLDH1_n + psiLDH2_n + (-1)*psiTCA_n )  # u22 = Pyr_n
    du[23] = (1/eto_a) * ( 2*psiGlc_a  + (-1)*psiLDH1_a + psiLDH2_a + (-1)*psiTCA_a )  # u23 = Pyr_a
    
    du[24] = (1/eto_n) * ( (-1)*psiPCr_n + psiCr_n )  # u24 = PCr_n
    du[25] = (1/eto_a) * ( (-1)*psiPCr_a + psiCr_a )  # u25 = PCr_a
    
    du[26] = (1/eto_n) * ( (-1)*psiCr_n + psiPCr_n )  # u26 = Cr_n
    du[27] = (1/eto_a) * ( (-1)*psiCr_a + psiPCr_a )  # u27 = Cr_a
    
    du[28] = (1/eto_n) * ( 2*psiGlc_n + psiTCA_n + 5*psiOxphos_n + psiPCr_n + (-1)*psiCr_n) #no psiNKA_n because it's already accounted in Ndam and ini ATP is from ndam + (-1)*psiNKA_n ) # ATP_n
    du[29] = (1/eto_a) * ( 2*psiGlc_a + psiTCA_a + 5*psiOxphos_a + psiPCr_a + (-1)*psiCr_a + (-1)*psiNKA_a ) # ATP_a
    
    du[30] = (1/eto_n) * ( (-2)*psiGlc_n + (-1)*psiTCA_n + (-5)*psiOxphos_n + (-1)*psiPCr_n + psiCr_n) #no psiNKA_n because it's already accounted in Ndam and ini ADP is adjusted in py from ndam  + psiNKA_n  ) # ADP_n
    du[31] = (1/eto_a) * ( (-2)*psiGlc_a + (-1)*psiTCA_a + (-5)*psiOxphos_a + (-1)*psiPCr_a + psiCr_a + psiNKA_a ) # ADP_a
  
   
    
    du[32] = (1/eto_n) * ( 2*psiGlc_n + (-1)*psiLDH1_n + psiLDH2_n + 5*psiTCA_n + (-2)*psiOxphos_n )  # u32 = NADH_n
    du[33] = (1/eto_a) * ( 2*psiGlc_a + (-1)*psiLDH1_a + psiLDH2_a + 5*psiTCA_a + (-2)*psiOxphos_a )  # u33 = NADH_a
 
    
    du[34] = (1/eto_n) * ( -2*psiGlc_n + psiLDH1_n + (-1)*psiLDH2_n -5*psiTCA_n + 2*psiOxphos_n )  # u34 = NAD_n
    du[35] = (1/eto_a) * ( -2*psiGlc_a + psiLDH1_a + (-1)*psiLDH2_a -5*psiTCA_a + 2*psiOxphos_a )  # u35 = NAD_a

    du[36] = 0 # placeholder for callback ksi
    du[37]=0 
    
    #return nothing

end
