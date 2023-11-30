function metabolism!(du,u,p,t)
    
    ina_density,ik_density,mito_scale,glutamatergic_gaba_scaling,outs_r_to_met  = @. p 
    synInput = 0. 
    Iinj = 0. 
    
    Pi_n = Pi_n0;    Pi_a = Pi_a0
    NADHmito_n = u[4];ATPmito_n = u[8];ADPmito_n = u[9];Pimito_n = u[12];
    ATP_n = u[23];
    FUMmito_n = u[25];MALmito_n = u[26];OXAmito_n = u[27];SUCmito_n = u[28];SUCCOAmito_n = u[29];CoAmito_n = u[30];AKGmito_n = u[31];CaMito_n = u[32];ISOCITmito_n = u[33];CITmito_n = u[34];AcCoAmito_n = u[35];AcAc_n = u[36];AcAcCoA_n = u[37];PYRmito_n = u[38];bHB_n = u[39];bHB_ecs = u[40];bHB_a = u[41];bHB_b = u[42];ASPmito_n = u[43];ASP_n = u[44];GLUmito_n = u[45];MAL_n = u[46];OXA_n = u[47];AKG_n = u[48];GLU_n = u[49];NADH_n = u[50];NADHmito_a = u[54];ATPmito_a = u[58];ADPmito_a = u[59];Pimito_a = u[62];
    ATP_a = u[73];
    FUMmito_a = u[75];MALmito_a = u[76];OXAmito_a = u[77];SUCmito_a = u[78];SUCCOAmito_a = u[79];CoAmito_a = u[80];AKGmito_a = u[81];CaMito_a = u[82];ISOCITmito_a = u[83];CITmito_a = u[84];AcCoAmito_a = u[85];AcAc_a = u[86];AcAcCoA_a = u[87];PYRmito_a = u[88];C_H_mitomatr_n = u[1];C_H_mitomatr_nM = 1e-3*u[1];K_x_nM= 1e-3*u[2];Mg_x_nM   = 1e-3*u[3];
    NADHmito_nM = 1e-3*u[4]; QH2mito_nM= 1e-3*u[5]; CytCredmito_nM   = 1e-3*u[6]; O2_nM = 1e-3*u[7]; O2_n = u[7]; ATPmito_nM  = 1e-3*u[8];ADPmito_nM  = 1e-3*u[9]
    
    PCr_n = u[174];  PCr_a = u[176];  Cr_n = Crtot - PCr_n;  Cr_a = Crtot - PCr_a
    cAMP_a = u[177]; NE_neuromod = u[178]
    UDPgluco_a = u[179]; UTP_a = u[180]; GS_a = u[181]; GPa_a = u[182]; GPb_a = u[183]; 
    #PKAa_a = u[184]; PKAb_a = u[185]; PHKa_a = u[186]; R2CcAMP2_a = u[187]; R2CcAMP4_a = u[188]; PP1_a = u[189];  PP1_GPa_a = u[190]
    
    
    ADP_n = ATP_n/2*(-qAK+sqrt(qAK*qAK+4*qAK*(ATDPtot_n/ATP_n-1))) 
    ADP_a = ATP_a/2*(-qAK+sqrt(qAK*qAK+4*qAK*(ATDPtot_a/ATP_a-1)))
    
    j_un          =   qAK*qAK+4*qAK*(ATDPtot_n/ATP_n-1)
    j_ug          =   qAK*qAK+4*qAK*(ATDPtot_a/ATP_a-1)

    dAMPdATPn   =   -1+qAK/2-0.5*sqrt(j_un)+qAK*ATDPtot_n/(ATP_n*sqrt(j_un))
    dAMPdATPg   =   -1+qAK/2-0.5*sqrt(j_ug)+qAK*ATDPtot_a/(ATP_a*sqrt(j_ug))
    
    
    ATP_nM  = 1e-3*ATP_n;   ADP_nM  = 1e-3*ADP_n
    ATP_aM  = 1e-3*ATP_a;   ADP_aM  = 1e-3*ADP_a  
    
    ATP_mx_nM = 1e-3*u[10]; ADP_mx_nM = 1e-3*u[11] # Matrix ATDP bound to magnesium
    Pimito_nM   = 1e-3*u[12]                     # Matrix inorganic phosphate

    ATP_i_n = u[13]
    ATP_i_nM  = 1e-3*u[13]; ADP_i_nM  = 1e-3*u[14]; ADP_i_n  = u[14]       # IMS  ATDP
    AMP_i_nM  = 1e-3*u[15]                     # IMS  AMP

    ATP_mi_nM = 1e-3*u[16];  ADP_mi_nM = 1e-3*u[17]     # IMS ATP bound to magnesium
    Pi_i_nM   = 1e-3*u[18]                     # IMS inorganic phosphate

    MitoMembrPotent_n   = u[19]                     # Mitochondrial membrane potential

    Ctot_nM   = 1e-3*u[20]                     # total cyt-c in IMS
    Qtot_nM   = 1e-3*u[21]                     # total ubiquinol

    C_H_ims_nM    = 1e-3*u[22]                     # IMS hydrogen concentration

    AMP_nM  = 0  # cytosolic ATDMP, M
    
    # Balancing moeities
    NAD_x_n  = NADtot - NADHmito_nM;    NADmito_n = 1000*NAD_x_n # mM ################
    Q_n      = Qtot_nM   - QH2mito_nM;    Qmito_n = 1000*Q_n # mM ################
    Cox_n    = Ctot_nM   - CytCredmito_nM;

    # Other concentrations computed from the state variables
    ATP_fx_n  = ATPmito_nM - ATP_mx_nM
    ADP_fx_n  = ADPmito_nM - ADP_mx_nM
    ATP_fi_n  = ATP_i_nM - ATP_mi_nM
    ADP_fi_n  = ADP_i_nM - ADP_mi_nM

    # ADP/Mg/K binding in E space
    ADP_me_n  = ( (K_DD + ADP_nM + Mg_tot) - sqrt((K_DD + ADP_nM + Mg_tot)^2 - 4*(Mg_tot*ADP_nM)) )/2;
    #ADP_fe_n  = ADP_nM - ADP_me_n;
    Mg_i_n    = Mg_tot - ADP_me_n;  # Mg_i_n    = Mg_e;  # Mg,K in IM space same as external/cytosolic

    # Calculating Membrane proton motive force and respiration fluxes
    dG_H_n    = etcF*MitoMembrPotent_n + 1*etcRT*log(C_H_ims_nM/C_H_mitomatr_nM);   # Protomotive force
    dG_C1op_n = dG_C1o - 1*etcRT*log(C_H_mitomatr_nM/1e-7);
    dG_C3op_n = dG_C3o + 2*etcRT*log(C_H_mitomatr_nM/1e-7);
    dG_C4op_n = dG_C4o - 2*etcRT*log(C_H_mitomatr_nM/1e-7);
    dG_F1op_n = dG_F1o - 1*etcRT*log(C_H_mitomatr_nM/1e-7);

    #a
    C_H_mitomatr_a = u[51];    C_H_mitomatr_aM = 1e-3*u[51]   # Matrix hydrogen concentration
    K_x_aM    = 1e-3*u[52]            # Matrix potassium concentration
    Mg_x_aM   = 1e-3*u[53]            # Matrix magnesium concentration

    NADHmito_aM = 1e-3*u[54]        # Reduced NADH in matrix
    QH2mito_aM    = 1e-3*u[55]        # Reduced ubiquinol in matrix
    CytCredmito_aM   = 1e-3*u[56]                      # Reduced cyt-c in IMS
    O2_aM     = 1e-3*u[57]                      # O2
    O2_a     = u[57]  # O2_a
    
    ATPmito_aM  = 1e-3*u[58]                      # Matrix  ATP 
    ADPmito_aM  = 1e-3*u[59]                      # Matrix  ADP 

    ATP_mx_aM = 1e-3*u[60]                     # Matrix ATP bound to magnesium
    ADP_mx_aM = 1e-3*u[61]                     # Matrix ADP bound to magnesium
    Pimito_aM   = 1e-3*u[62]                   # Matrix inorganic phosphate

    ATP_i_a = u[63]
    ATP_i_aM  = 1e-3*u[63]                     # IMS  ATP
    ADP_i_aM  = 1e-3*u[64]; ADP_i_a  = u[64]                     # IMS  ADP
    AMP_i_aM  = 1e-3*u[65]                     # IMS  AMP
    ATP_mi_aM = 1e-3*u[66]                     # IMS ATP bound to magnesium
    ADP_mi_aM = 1e-3*u[67]                     # IMS ADP bound to magnesium
    Pi_i_aM   = 1e-3*u[68]                     # IMS inorganic phosphate

    MitoMembrPotent_a   = u[69]                     # Mitochondrial membrane potential

    Ctot_aM   = 1e-3*u[70]                     # total cyt-c in IMS
    Qtot_aM   = 1e-3*u[71]                     # total ubiquinol
    C_H_ims_aM    = 1e-3*u[72]                     # IMS hydrogen concentration

    AMP_aM  = 0                   # cytosolic AMP concentration (Molar), Nicholls and others
    #AMP_e  = 0;                         # cytosolic AMP concentration (Molar], Nicholls and others

    # Balancing moeities
    NAD_x_a  = NADtot - NADHmito_aM;
    NADmito_a = 1000*NAD_x_a # mM ################
    Q_a      = Qtot_aM   - QH2mito_aM;
    Qmito_a = 1000*Q_a # mM ################
    Cox_a    = Ctot_aM   - CytCredmito_aM;

    # Other concentrations computed from the state variables
    ATP_fx_a  = ATPmito_aM - ATP_mx_aM
    ADP_fx_a  = ADPmito_aM - ADP_mx_aM
    ATP_fi_a  = ATP_i_aM - ATP_mi_aM
    ADP_fi_a  = ADP_i_aM - ADP_mi_aM

    # ADP/Mg/K binding in E space
    ADP_me_a  = ( (K_DD_a + ADP_aM + Mg_tot) - sqrt((K_DD_a + ADP_aM + Mg_tot)^2 - 4*(Mg_tot*ADP_aM)) )/2;
    #ADP_fe_a  = ADP_aM - ADP_me_a;
    Mg_i_a    = Mg_tot - ADP_me_a;  # Mg_i_a    = Mg_e;  # Mg,K in IM space same as external/cytosolic

    # Calculating Membrane proton motive force and respiration fluxes
    dG_H_a    = etcF*MitoMembrPotent_a + 1*etcRT*log(C_H_ims_aM/C_H_mitomatr_aM);   # Protomotive force
    dG_C1op_a = dG_C1o - 1*etcRT*log(C_H_mitomatr_aM/1e-7);
    dG_C3op_a = dG_C3o + 2*etcRT*log(C_H_mitomatr_aM/1e-7);
    dG_C4op_a = dG_C4o - 2*etcRT*log(C_H_mitomatr_aM/1e-7);
    dG_F1op_a = dG_F1o - 1*etcRT*log(C_H_mitomatr_aM/1e-7);

    ###############
    # GLT-GLN
    GLN_n = u[89]; GLN_out = u[90];    GLN_a = u[91];    GLUT_a = u[92];    Va = u[93];    Na_a = u[94];    K_a = u[95];    K_out = u[96];    GLUT_syn = u[97]
    
    ##############
    # cyto
    VNeu = u[98]; Na_n = u[99]; h = u[100]; n = u[101]; Ca_n = u[102]; pgate = u[103]; nBK_a = u[104]; mGluRboundRatio_a = u[105]; IP3_a = u[106]; hIP3Ca_a = u[107]; Ca_a = u[108]; Ca_r_a = u[109]; sTRP_a = u[110]
    vV = u[111]; EET_a = u[112]; ddHb = u[113]; O2cap = u[114]; Glc_b = u[115]; glc_D_ecsEndothelium = u[116]; Glc_ecsBA = u[117]; Glc_a = u[118]; Glc_ecsAN = u[119]; Glc_n = u[120]; G6P_n = u[121]; G6P_a = u[122]
    F6P_n = u[123]; F6P_a = u[124]; FBP_n = u[125]; FBP_a = u[126]; f26bp_a = u[127]; GLY_a = u[128]; AMP_n = u[129]; AMP_a = u[130]; G1P_a = u[131]; GAP_n = u[132]; GAP_a = u[133]; DHAP_n = u[134]; DHAP_a = u[135]; BPG13_n = u[136]; BPG13_a = u[137]; NADH_a = u[138]
    PG3_n = u[141]; PG3_a = u[142]; PG2_n = u[143]; PG2_a = u[144]; PEP_n = u[145]; PEP_a = u[146]; Pyr_n = u[147]; Pyr_a = u[148]; Lac_b = u[149]; Lac_ecs = u[150]; Lac_a = u[151]; Lac_n = u[152]
    NADPH_n = u[153]; NADPH_a = u[154]; GL6P_n = u[155]; GL6P_a = u[156]; GO6P_n = u[157]; GO6P_a = u[158]; RU5P_n = u[159]; RU5P_a = u[160]; R5P_n = u[161]; R5P_a = u[162]; X5P_n = u[163]; X5P_a = u[164]; S7P_n = u[165]; S7P_a = u[166]; E4P_n = u[167]; E4P_a = u[168]; GSH_n = u[169]; GSH_a = u[170]; GSSG_n = u[171]; GSSG_a = u[172]
    
    NADP_n = 0.0303 - NADPH_n;    NADP_a = 0.0303 - NADPH_a
    
    # NAD+ + NADH = 0.212 # Jolivet2015
    NAD_n = 0.212 - NADH_n;    NAD_a = 0.212 - NADH_a
    
    ################
    # Rates n
    J_DH_n    = x_DH*(r_DH*NAD_x_n - NADHmito_nM)*((1+Pimito_nM/k_Pi1)/(1+Pimito_nM/k_Pi2));
    J_C1_n    = x_C1*(exp(-(dG_C1op_n+4*dG_H_n)/etcRT)*NADHmito_nM*Q_n - NAD_x_n*QH2mito_nM);
    J_C3_n    = x_C3*((1+Pimito_nM/k_Pi3)/(1+Pimito_nM/k_Pi4))*(exp(-(dG_C3op_n+4*dG_H_n-2*etcF*MitoMembrPotent_n)/(2*etcRT))*Cox_n*QH2mito_nM^0.5 - CytCredmito_nM*Q_n^0.5);
    J_C4_n    = x_C4*(O2_nM/(O2_nM+k_O2))*(CytCredmito_nM/Ctot_nM)*(exp(-(dG_C4op_n+2*dG_H_n)/(2*etcRT))*CytCredmito_nM*(O2_nM^0.25) - Cox_n*exp(etcF*MitoMembrPotent_n/etcRT));
    J_F1_n    = x_F1*(exp(-(dG_F1op_n-n_A*dG_H_n)/etcRT)*(K_DD/K_DT)*ADP_mx_nM*Pimito_nM - ATP_mx_nM);

    # ATP transferase Korzeniewski 1998, Theurey 2019
    J_ANT_n  = x_ANT*( ADP_fi_n/(ADP_fi_n + ATP_fi_n*exp(-etcF*(0.35*MitoMembrPotent_n)/etcRT))  - ADP_fx_n/(ADP_fx_n + ATP_fx_n*exp(-etcF*(-0.65*MitoMembrPotent_n)/etcRT)) )*(ADP_fi_n/(ADP_fi_n+k_mADP));

    # Calculating ionic fluxes
    H2PIi_n       = Pi_i_nM*C_H_ims_nM/(C_H_ims_nM + k_dHPi);
    H2PIx_n       = Pimito_nM*C_H_mitomatr_nM/(C_H_mitomatr_nM + k_dHPi);
    
    J_Pi1_n       = x_Pi1*(C_H_mitomatr_nM*H2PIi_n - C_H_ims_nM*H2PIx_n)/(H2PIi_n+k_PiH);
    J_Hle_n       = x_Hle*MitoMembrPotent_n*(C_H_ims_nM*exp(etcF*MitoMembrPotent_n/etcRT) - C_H_mitomatr_nM)/(exp(etcF*MitoMembrPotent_n/etcRT) - 1);
    J_KH_n        = x_KH*( K_i*C_H_mitomatr_nM - K_x_nM*C_H_ims_nM );
    J_K_n         = x_K*MitoMembrPotent_n*(K_i*exp(etcF*MitoMembrPotent_n/etcRT) - K_x_nM)/(exp(etcF*MitoMembrPotent_n/etcRT) - 1);
    J_AKi_n       = x_AK*( K_AK*ADP_i_nM*ADP_i_nM - AMP_i_nM*ATP_i_nM );
    J_AMP_n       = gamma*x_A*(AMP_nM - AMP_i_nM);

    J_ADP_n       = gamma*x_A*(ADP_nM - ADP_i_nM);
    J_ATP_n       = gamma*x_A*(ATP_nM - ATP_i_nM);

    J_Pi2_n       = gamma*x_Pi2*(1e-3 * Pi_n - Pi_i_nM);
    J_Ht_n        = gamma*x_Ht*(1e-3 * C_H_cyt_n - C_H_ims_nM);

    J_MgATPx_n    = x_MgA*(ATP_fx_n*Mg_x_nM - K_DT*ATP_mx_nM);
    J_MgADPx_n    = x_MgA*(ADP_fx_n*Mg_x_nM - K_DD*ADP_mx_nM);
    J_MgATPi_n    = x_MgA*(ATP_fi_n*Mg_i_n - K_DT*ATP_mi_nM);
    J_MgADPi_n    = x_MgA*(ADP_fi_n*Mg_i_n - K_DD*ADP_mi_nM);

    # J_ATPK_n   = x_ATPK * (K_ADTP_cons*ATP_nM - K_ADTP_dyn*ADP_nM);

    #######################
    # TCA n
    psiPDH_n(PYRmito_n,NADmito_n,CoAmito_n) = VmaxPDHCmito_n* (PYRmito_n/(PYRmito_n+KmPyrMitoPDH_n)) * (NADmito_n/(NADmito_n + KmNADmitoPDH_na)) * (CoAmito_n/(CoAmito_n + KmCoAmitoPDH_n)) 
    psiCS_n(OXAmito_n,CITmito_n,AcCoAmito_n,CoAmito_n) = VmaxCSmito_n*(OXAmito_n/(OXAmito_n + KmOxaMito_n*(1.0 + CITmito_n/KiCitMito_n))) * (AcCoAmito_n/(AcCoAmito_n + KmAcCoAmito_n*(1.0+CoAmito_n/KiCoA_n)))
    psiACO_n(CITmito_n,ISOCITmito_n) = VmaxAco_n*(CITmito_n-ISOCITmito_n/KeqAco_na) / (1.0+CITmito_n/KmCitAco_n + ISOCITmito_n/KmIsoCitAco_n)
    psiIDH_n(ISOCITmito_n,NADmito_n,NADHmito_n) = VmaxIDH_n*(NADmito_n/KiNADmito_na)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH ) /  (1.0 + NADmito_n/KiNADmito_na + (KmNADmito_na/KiNADmito_na)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) + NADHmito_n/KiNADHmito_na + (NADmito_n/KiNADmito_na)*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) +   ((KmNADmito_na*NADHmito_n)/(KiNADmito_na*KiNADHmito_na))*((ISOCITmito_n/KmIsoCitIDHm_n)^nIDH) )
    psiKGDH_n(ADPmito_n,ATPmito_n,CaMito_n,AKGmito_n,NADHmito_n,NADmito_n,CoAmito_n,SUCCOAmito_n) = (VmaxKGDH_n*(1 + ADPmito_n/KiADPmito_KGDH_n)*(AKGmito_n/Km1KGDHKGDH_n)*(CoAmito_n/Km_CoA_kgdhKGDH_n)*(NADmito_n/KmNADkgdhKGDH_na) ) / ( ( (CoAmito_n/Km_CoA_kgdhKGDH_n)*(NADmito_n/KmNADkgdhKGDH_na)*(AKGmito_n/Km1KGDHKGDH_n + (1 + ATPmito_n/KiATPmito_KGDH_n)/(1 + CaMito_n/KiCa2KGDH_n)) ) +    ( (AKGmito_n/Km1KGDHKGDH_n)*(CoAmito_n/Km_CoA_kgdhKGDH_n + NADmito_n/KmNADkgdhKGDH_na)*(1 + NADHmito_n/KiNADHKGDHKGDH_na + SUCCOAmito_n/Ki_SucCoA_kgdhKGDH_n) )  ) 
    psiSCS_n(Pimito_n,SUCCOAmito_n,ADPmito_n,SUCmito_n,CoAmito_n,ATPmito_n)  = VmaxSuccoaATPscs_n*(1+AmaxPscs_n*((Pimito_n^npscs_n)/((Pimito_n^npscs_n)+(Km_pi_scs_na^npscs_n)))) * (SUCCOAmito_n*ADPmito_n*Pimito_n -  SUCmito_n*CoAmito_n*ATPmito_n/Keqsuccoascs_na)/((1+SUCCOAmito_n/Km_succoa_scs_n)*(1+ADPmito_n/Km_ADPmito_scs_n)*(1+Pimito_n/Km_pi_scs_na)+(1+SUCmito_n/Km_succ_scs_n)*(1+CoAmito_n/Km_coa_scs_n)*(1+ATPmito_n/Km_atpmito_scs_n))  
    psiFUM_n(FUMmito_n,MALmito_n) = Vmaxfum_n*(FUMmito_n - MALmito_n/Keqfummito_na)/(1.0+FUMmito_n/Km_fummito_n+MALmito_n/Km_malmito_n)   
    psiMDH_n(MALmito_n,NADmito_n,OXAmito_n,NADHmito_n) = VmaxMDHmito_n*(MALmito_n*NADmito_n-OXAmito_n*NADHmito_n/Keqmdhmito_na) / ((1.0+MALmito_n/Km_mal_mdh_n)*(1.0+NADmito_n/Km_nad_mdh_na)+(1.0+OXAmito_n/Km_oxa_mdh_n)*(1.0+NADHmito_n/Km_nadh_mdh_na))    

    # Ketones
    SCOT_n(SUCCOAmito_n,AcAc_n,AcAcCoA_n,SUCmito_n) = (VmaxfSCOT_n*SUCCOAmito_n*AcAc_n/( Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n*(SUCCOAmito_n/Ki_SucCoA_SCOT_n + Km_SucCoA_SCOT_n*AcAc_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n) + AcAcCoA_n/Ki_AcAcCoA_SCOT_n + Km_AcAcCoA_SCOT_n*SUCmito_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n) + SUCCOAmito_n*AcAc_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n) +      SUCCOAmito_n*AcAcCoA_n/(Ki_SucCoA_SCOT_n*Ki_AcAcCoA_SCOT_n) +    Km_SucCoA_SCOT_n*AcAc_n*SUCmito_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n*Ki_SUC_SCOT_n) +    AcAcCoA_n*SUCmito_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n)        ) ))   -  (VmaxrSCOT_n*AcAcCoA_n*SUCmito_n/( Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n*(SUCCOAmito_n/Ki_SucCoA_SCOT_n +     Km_SucCoA_SCOT_n*AcAc_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n) + AcAcCoA_n/Ki_AcAcCoA_SCOT_n + Km_AcAcCoA_SCOT_n*SUCmito_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n) +    SUCCOAmito_n*AcAc_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n) + SUCCOAmito_n*AcAcCoA_n/(Ki_SucCoA_SCOT_n*Ki_AcAcCoA_SCOT_n) +     Km_SucCoA_SCOT_n*AcAc_n*SUCmito_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n*Ki_SUC_SCOT_n) + AcAcCoA_n*SUCmito_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n)        ) ))   
    thiolase_n(CoAmito_n,AcAcCoA_n) = Vmax_thiolase_f_n*CoAmito_n*AcAcCoA_n / ( Ki_CoA_thiolase_f_n * Km_AcAcCoA_thiolase_f_n + Km_AcAcCoA_thiolase_f_n*CoAmito_n +     Km_CoA_thiolase_f_n*AcAcCoA_n + CoAmito_n*AcAcCoA_n)
    JbHBTrArtCap(t,bHB_b) = (2*(C_bHB_a - bHB_b)/eto_b)*(global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))) 
    MCT1_bHB_b(bHB_b,bHB_ecs) = VmaxMCTbhb_b*(bHB_b/(bHB_b + KmMCT1_bHB_b) - bHB_ecs/(bHB_ecs + KmMCT1_bHB_b)) 
    MCT2_bHB_n(bHB_ecs,bHB_n) = VmaxMCTbhb_n*(bHB_ecs/(bHB_ecs + KmMCT2_bHB_n) - bHB_n/(bHB_n + KmMCT2_bHB_n)) 
    MCT1_bHB_a(bHB_ecs,bHB_a) = VmaxMCTbhb_a*(bHB_ecs/(bHB_ecs + KmMCT1_bHB_a) - bHB_a/(bHB_a + KmMCT1_bHB_a)) 
    bHBDH_n(NADmito_n,NADHmito_n,bHB_n,AcAc_n) = Vmax_bHBDH_f_n*NADmito_n*bHB_n/(Ki_NAD_B_HBD_f_n*Km_betaHB_BHBD_n + Km_betaHB_BHBD_n*NADmito_n + Km_NAD_B_HBD_n*bHB_n + NADmito_n*bHB_n )   -     (Vmax_bHBDH_r_n*NADHmito_n*AcAc_n/(Ki_NADH_BHBD_r_n*Km_AcAc_BHBD_n + Km_AcAc_BHBD_n*NADHmito_n + Km_NADH_BHBD_n*AcAc_n + NADHmito_n*AcAc_n ))   
    
    #######################
    # MAS 
    # AAT mito n (GOT mito n): psiAAT_n(ASPmito_n,AKGmito_n,OXAmito_n,GLUmito_n), AKGmito_n + ASPmito_n ⇒ OXAmito_n + GLUmito_n
    psiAAT_n(ASPmito_n,AKGmito_n,OXAmito_n,GLUmito_n) = VfAAT_n*(ASPmito_n*AKGmito_n - OXAmito_n*GLUmito_n/KeqAAT_n) /  ( KmAKG_AAT_n*ASPmito_n +  KmASP_AAT_n*(1.0 + AKGmito_n/KiAKG_AAT_n)*AKGmito_n + ASPmito_n*AKGmito_n + KmASP_AAT_n*AKGmito_n*GLUmito_n/KiGLU_AAT_n + (  KiASP_AAT_n*KmAKG_AAT_n/(KmOXA_AAT_n*KiGLU_AAT_n)  )*  ( KmGLU_AAT_n*ASPmito_n*OXAmito_n/KiASP_AAT_n + OXAmito_n*GLUmito_n +  KmGLU_AAT_n*(1.0 + AKGmito_n/KiAKG_AAT_n)*OXAmito_n + KmOXA_AAT_n*GLUmito_n )  )
    # cMDH n: psicMDH_n(MAL_n,NAD_n,OXA_n,NADH_n), MAL_n + NAD_n ⇒ OXA_n + NADH_n 
    psicMDH_n(MAL_n,NAD_n,OXA_n,NADH_n) = VmaxcMDH_n*(MAL_n*NAD_n-OXA_n*NADH_n/Keqcmdh_n)/ ((1+MAL_n/Kmmalcmdh_n)*(1+NAD_n/Kmnadcmdh_n) + (1+OXA_n/Kmoxacmdh_n)*(1+NADH_n/Kmnadhcmdh_n)-1)  
    # psiCAAT_n(ASP_n,AKG_n,OXA_n,GLU_n), ASP_n + AKG_n ⇒ OXA_n + GLU_n
    psiCAAT_n(ASP_n,AKG_n,OXA_n,GLU_n) = VfCAAT_n*(ASP_n*AKG_n - OXA_n*GLU_n/KeqCAAT_n) /  ( KmAKG_CAAT_n*ASP_n +  KmASP_CAAT_n*(1.0 + AKG_n/KiAKG_CAAT_n)*AKG_n +  ASP_n*AKG_n + KmASP_CAAT_n*AKG_n*GLU_n/KiGLU_CAAT_n + (  KiASP_CAAT_n*KmAKG_CAAT_n/(KmOXA_CAAT_n*KiGLU_CAAT_n)  )*    ( KmGLU_CAAT_n*ASP_n*OXA_n/KiASP_CAAT_n + OXA_n*GLU_n +  KmGLU_CAAT_n*(1.0 + AKG_n/KiAKG_CAAT_n)*OXA_n + KmOXA_CAAT_n*GLU_n )  )
    # AGC (citrin) n: psiAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n,MitoMembrPotent_n),  ASPmito_n + GLU_n ⇒ GLUmito_n + ASP_n
    psiAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n,MitoMembrPotent_n) = Vmaxagc_n*(ASPmito_n*GLU_n - ASP_n*GLUmito_n/ ((exp(MitoMembrPotent_n)^(F/(R*T)))*  (C_H_cyt_n/C_H_mito_n)) ) / ((ASPmito_n+Km_aspmito_agc_n)*(GLU_n+Km_glu_agc_n) + (ASP_n+Km_asp_agc_n)*(GLUmito_n+Km_glumito_agc_n))      
    # MAKGC n: psiMAKGC_n(MAL_n,AKGmito_n,MALmito_n,AKG_n), AKGmito_n + MAL_n ⇒ MALmito_n + AKG_n 
    psiMAKGC_n(MAL_n,AKGmito_n,MALmito_n,AKG_n) = Vmaxmakgc_n*( MAL_n*AKGmito_n - MALmito_n*AKG_n) / ((MAL_n+Km_mal_mkgc_n)*(AKGmito_n+Km_akgmito_mkgc_n)+(MALmito_n+Km_malmito_mkgc_n)*(AKG_n+Km_akg_mkgc_n))     

    
    ################
    # Rates a
    J_DH_a    = x_DH_a*(r_DH_a*NAD_x_a - NADHmito_aM)*((1+Pimito_aM/k_Pi1)/(1+Pimito_aM/k_Pi2));
    J_C1_a    = x_C1*(exp(-(dG_C1op_a+4*dG_H_a)/etcRT)*NADHmito_aM*Q_a - NAD_x_a*QH2mito_aM);
    J_C3_a    = x_C3*((1+Pimito_aM/k_Pi3)/(1+Pimito_aM/k_Pi4))*(exp(-(dG_C3op_a+4*dG_H_a-2*etcF*MitoMembrPotent_a)/(2*etcRT))*Cox_a*QH2mito_aM^0.5 - CytCredmito_aM*Q_a^0.5);
    J_C4_a    = x_C4*(O2_aM/(O2_aM+k_O2))*(CytCredmito_aM/Ctot_aM)*(exp(-(dG_C4op_a+2*dG_H_a)/(2*etcRT))*CytCredmito_aM*(O2_aM^0.25) - Cox_a*exp(etcF*MitoMembrPotent_a/etcRT));
    J_F1_a    = x_F1_a*(exp(-(dG_F1op_a-n_A*dG_H_a)/etcRT)*(K_DD_a/K_DT_a)*ADP_mx_aM*Pimito_aM - ATP_mx_aM);

    # ATP transferase Korzeniewski 1998, Theurey 2019 
    J_ANT_a  = x_ANT_a*( ADP_fi_a/(ADP_fi_a + ATP_fi_a*exp(-etcF*(0.35*MitoMembrPotent_a)/etcRT))  - ADP_fx_a/(ADP_fx_a + ATP_fx_a*exp(-etcF*(-0.65*MitoMembrPotent_a)/etcRT)) )*(ADP_fi_a/(ADP_fi_a+k_mADP_a));

    # Calculating ionic fluxes
    H2PIi_a       = Pi_i_aM*C_H_ims_aM/(C_H_ims_aM + k_dHPi);
    H2PIx_a       = Pimito_aM*C_H_mitomatr_aM/(C_H_mitomatr_aM + k_dHPi);
    J_Pi1_a       = x_Pi1*(C_H_mitomatr_aM*H2PIi_a - C_H_ims_aM*H2PIx_a)/(H2PIi_a+k_PiH);
    J_Hle_a       = x_Hle*MitoMembrPotent_a*(C_H_ims_aM*exp(etcF*MitoMembrPotent_a/etcRT) - C_H_mitomatr_aM)/(exp(etcF*MitoMembrPotent_a/etcRT) - 1);
    J_KH_a        = x_KH*( K_i*C_H_mitomatr_aM - K_x_aM*C_H_ims_aM );
    J_K_a         = x_K*MitoMembrPotent_a*(K_i*exp(etcF*MitoMembrPotent_a/etcRT) - K_x_aM)/(exp(etcF*MitoMembrPotent_a/etcRT) - 1);
    J_AKi_a       = x_AK*( K_AK*ADP_i_aM*ADP_i_aM - AMP_i_aM*ATP_i_aM );
    J_AMP_a       = gamma*x_A*(AMP_aM - AMP_i_aM);

    J_ADP_a       = gamma*x_A*(ADP_aM - ADP_i_aM);
    J_ATP_a       = gamma*x_A*(ATP_aM - ATP_i_aM);

    J_Pi2_a       = gamma*x_Pi2*(1e-3 * Pi_a - Pi_i_aM);
    J_Ht_a        = gamma*x_Ht*(1e-3 * C_H_cyt_a - C_H_ims_aM);

    J_MgATPx_a    = x_MgA*(ATP_fx_a*Mg_x_aM - K_DT_a*ATP_mx_aM);
    J_MgADPx_a    = x_MgA*(ADP_fx_a*Mg_x_aM - K_DD_a*ADP_mx_aM);
    J_MgATPi_a    = x_MgA*(ATP_fi_a*Mg_i_a - K_DT_a*ATP_mi_aM);
    J_MgADPi_a    = x_MgA*(ADP_fi_a*Mg_i_a - K_DD_a*ADP_mi_aM);

    # J_ATPK_a   = x_ATPK * (K_ADTP_cons*ATP_aM - K_ADTP_dyn*ADP_aM);

    ####################### 
    # TCA
    psiPDH_a(PYRmito_a,NADmito_a,CoAmito_a) = VmaxPDHCmito_a* (PYRmito_a/(PYRmito_a+KmPyrMitoPDH_a)) * (NADmito_a/(NADmito_a + KmNADmitoPDH_na)) * (CoAmito_a/(CoAmito_a + KmCoAmitoPDH_a)) 
    psiCS_a(OXAmito_a,CITmito_a,AcCoAmito_a,CoAmito_a) = VmaxCSmito_a*(OXAmito_a/(OXAmito_a + KmOxaMito_a*(1.0 + CITmito_a/KiCitMito_a))) * (AcCoAmito_a/(AcCoAmito_a + KmAcCoAmito_a*(1.0+CoAmito_a/KiCoA_a)))
    psiACO_a(CITmito_a,ISOCITmito_a) = VmaxAco_a*(CITmito_a-ISOCITmito_a/KeqAco_na) / (1.0+CITmito_a/KmCitAco_a + ISOCITmito_a/KmIsoCitAco_a)
    psiIDH_a(ISOCITmito_a,NADmito_a,NADHmito_a) = VmaxIDH_a*(NADmito_a/KiNADmito_na)*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH ) /  (1.0 + NADmito_a/KiNADmito_na + (KmNADmito_na/KiNADmito_na)*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH) + NADHmito_a/KiNADHmito_na + (NADmito_a/KiNADmito_na)*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH) +   ((KmNADmito_na*NADHmito_a)/(KiNADmito_na*KiNADHmito_na))*((ISOCITmito_a/KmIsoCitIDHm_a)^nIDH) )
    psiKGDH_a(ADPmito_a,ATPmito_a,CaMito_a,AKGmito_a,NADHmito_a,NADmito_a,CoAmito_a,SUCCOAmito_a) = (VmaxKGDH_a*(1 + ADPmito_a/KiADPmito_KGDH_a)*(AKGmito_a/Km1KGDHKGDH_a)*(CoAmito_a/Km_CoA_kgdhKGDH_a)*(NADmito_a/KmNADkgdhKGDH_na) ) / ( ( (CoAmito_a/Km_CoA_kgdhKGDH_a)*(NADmito_a/KmNADkgdhKGDH_na)*(AKGmito_a/Km1KGDHKGDH_a + (1 + ATPmito_a/KiATPmito_KGDH_a)/(1 + CaMito_a/KiCa2KGDH_a)) ) +    ( (AKGmito_a/Km1KGDHKGDH_a)*(CoAmito_a/Km_CoA_kgdhKGDH_a + NADmito_a/KmNADkgdhKGDH_na)*(1 + NADHmito_a/KiNADHKGDHKGDH_na + SUCCOAmito_a/Ki_SucCoA_kgdhKGDH_a) )  ) 
    psiSCS_a(Pimito_a,SUCCOAmito_a,ADPmito_a,SUCmito_a,CoAmito_a,ATPmito_a)  = VmaxSuccoaATPscs_a*(1+AmaxPscs_a*((Pimito_a^npscs_a)/((Pimito_a^npscs_a)+(Km_pi_scs_na^npscs_a)))) * (SUCCOAmito_a*ADPmito_a*Pimito_a -  SUCmito_a*CoAmito_a*ATPmito_a/Keqsuccoascs_na)/((1+SUCCOAmito_a/Km_succoa_scs_a)*(1+ADPmito_a/Km_ADPmito_scs_a)*(1+Pimito_a/Km_pi_scs_na)+(1+SUCmito_a/Km_succ_scs_a)*(1+CoAmito_a/Km_coa_scs_a)*(1+ATPmito_a/Km_atpmito_scs_a))  
    psiFUM_a(FUMmito_a,MALmito_a) = Vmaxfum_a*(FUMmito_a - MALmito_a/Keqfummito_na)/(1.0+FUMmito_a/Km_fummito_a+MALmito_a/Km_malmito_a)   
    psiMDH_a(MALmito_a,NADmito_a,OXAmito_a,NADHmito_a) = VmaxMDHmito_a*(MALmito_a*NADmito_a-OXAmito_a*NADHmito_a/Keqmdhmito_na) / ((1.0+MALmito_a/Km_mal_mdh_a)*(1.0+NADmito_a/Km_nad_mdh_na)+(1.0+OXAmito_a/Km_oxa_mdh_a)*(1.0+NADHmito_a/Km_nadh_mdh_na))    
    
     psiPYRCARB_a(ATPmito_a,ADPmito_a,PYRmito_a,OXAmito_a) = ( (ATPmito_a/ADPmito_a)/(muPYRCARB_a +  (ATPmito_a/ADPmito_a)))*VmPYRCARB_a*(PYRmito_a*CO2_mito_a - OXAmito_a/KeqPYRCARB_a)/(  KmPYR_PYRCARB_a*KmCO2_PYRCARB_a +  KmPYR_PYRCARB_a*CO2_mito_a + KmCO2_PYRCARB_a*PYRmito_a + CO2_mito_a*PYRmito_a)    # PYRmito_a + ATPmito_a ⇒ OXAmito_a + ADPmito_a
    
    ######################
    # GLT-GLN 
    psiSNAT_GLN_n(GLN_out,GLN_n) = TmaxSNAT_GLN_n*(GLN_out-GLN_n/coeff_gln_ratio_n_ecs)/(KmSNAT_GLN_n+GLN_n)  
    psiGLS_n(GLN_n,GLUmito_n) = VmGLS_n*( GLN_n - GLUmito_n/KeqGLS_n )/ (KmGLNGLS_n*(1.0 + GLUmito_n/KiGLUGLS_n) + GLN_n  ) 
    #psiEAAT12(Va,Na_a,GLUT_syn,GLUT_a,K_a,K_out) = - ((1/(2*F*1e-3))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(Va - ((R*T/(2*F*1e-3))*log( ((Na_syn_EAAT/Na_a)^3) *  (H_syn_EAAT/H_ast_EAAT)  *   (GLUT_syn/GLUT_a)   *   (K_a/K_out )  )))) )) 
    psiEAAT12(Va,Na_a,GLUT_syn,GLUT_a,K_a,K_out) = - ((1/(2*F*1e-3)) * (  -alpha_EAAT*exp(-beta_EAAT*(Va - ((R*T/(2*F*1e-3))*log( ((Na_syn_EAAT/Na_a)^3) *  (H_syn_EAAT/H_ast_EAAT)  *   (GLUT_syn/GLUT_a)   *   (K_a/K_out )  )))) )) 
    
    #psiGDH_simplif_a(NADmito_a,GLUmito_a,NADHmito_a,AKGmito_a) = VmGDH_a*(NADmito_a*GLUmito_a - NADHmito_a*AKGmito_a/KeqGDH_a)/ (KiNAD_GDH_a*KmGLU_GDH_a + KmGLU_GDH_a*NADmito_a + KiNAD_GDH_a*GLUmito_a + GLUmito_a*NADmito_a + GLUmito_a*NADmito_a/KiAKG_GDH_a +         KiNAD_GDH_a*KmGLU_GDH_a*NADHmito_a/KiNADH_GDH_a +     KiNAD_GDH_a*KmGLU_GDH_a*KmNADH_GDH_a*AKGmito_a/(KiAKG_GDH_a*KiNADH_GDH_a) + KmNADH_GDH_a*GLUmito_a*NADHmito_a/KiNADH_GDH_a +           KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*NADHmito_a/(KiAKG_GDH_a*KiNADH_GDH_a) +  KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*AKGmito_a*NADHmito_a/(KiAKG_GDH_a*KiNADH_GDH_a)   +         GLUmito_a*NADmito_a*AKGmito_a/KiAKG_GDH_a + KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a/KiAKG_GDH_a +  KiNAD_GDH_a*KmGLU_GDH_a*GLUmito_a*NADHmito_a*AKGmito_a/(KiGLU_GDH_a*KiAKG_GDH_a*KiNADH_GDH_a) +        KiNAD_GDH_a*KmGLU_GDH_a*AKGmito_a*NADHmito_a/(KiAKG_GDH_a*KiNADH_GDH_a)  +  KmNADH_GDH_a*KmGLU_GDH_a*AKGmito_a*NADmito_a/(KiAKG_GDH_a*KiNADH_GDH_a) )
    # simplif: no mito-cyto compart for glutamate in astro
    psiGDH_simplif_a(NADmito_a,GLUT_a,NADHmito_a,AKGmito_a) = VmGDH_a*(NADmito_a*GLUT_a - NADHmito_a*AKGmito_a/KeqGDH_a)/ (KiNAD_GDH_a*KmGLU_GDH_a + KmGLU_GDH_a*NADmito_a + KiNAD_GDH_a*GLUT_a + GLUT_a*NADmito_a + GLUT_a*NADmito_a/KiAKG_GDH_a +         KiNAD_GDH_a*KmGLU_GDH_a*NADHmito_a/KiNADH_GDH_a +     KiNAD_GDH_a*KmGLU_GDH_a*KmNADH_GDH_a*AKGmito_a/(KiAKG_GDH_a*KiNADH_GDH_a) + KmNADH_GDH_a*GLUT_a*NADHmito_a/KiNADH_GDH_a +           KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*NADHmito_a/(KiAKG_GDH_a*KiNADH_GDH_a) +  KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*AKGmito_a*NADHmito_a/(KiAKG_GDH_a*KiNADH_GDH_a)   +         GLUT_a*NADmito_a*AKGmito_a/KiAKG_GDH_a + KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a/KiAKG_GDH_a +  KiNAD_GDH_a*KmGLU_GDH_a*GLUT_a*NADHmito_a*AKGmito_a/(KiGLU_GDH_a*KiAKG_GDH_a*KiNADH_GDH_a) +        KiNAD_GDH_a*KmGLU_GDH_a*AKGmito_a*NADHmito_a/(KiAKG_GDH_a*KiNADH_GDH_a)  +  KmNADH_GDH_a*KmGLU_GDH_a*AKGmito_a*NADmito_a/(KiAKG_GDH_a*KiNADH_GDH_a) )

    # psiGLNsynth_a(GLUT_a,ATP_a,ADP_a) = VmaxGLNsynth_a*(GLUT_a/( KmGLNsynth_a  +  GLUT_a))*( (ATP_a/ADP_a)/(muGLNsynth_a + ATP_a/ADP_a ) )   
    psiGLNsynth_a(GLUT_a,ATP_a,ADP_a) = VmaxGLNsynth_a*(GLUT_a/( KmGLNsynth_a  +  GLUT_a))*( (1/(ATP_a/ADP_a))/(muGLNsynth_a + (1/(ATP_a/ADP_a)) ) )   

    psiSNAT_GLN_a(GLN_a,GLN_out) = TmaxSNAT_GLN_a*(GLN_a-GLN_out)/(KmSNAT_GLN_a+GLN_a) 
    #synGlutRelease(V) =  glut_vesicle_deltaConc*exp(-((V+v_thr)/coeff_synGlutRelease)^2) / (coeff_synGlutRelease * 1.772 )

    #du[28] = 1000*(+J_DH_a - J_C1_a)/W_x #+ psiMDH_a(MALmito_a,NADmito_a,OXAmito_a,NADHmito_a) + psiKGDH_a(ADPmito_a,ATPmito_a,CaMito_a,AKGmito_a,NADHmito_a,NADmito_a,CoAmito_a,SUCCOAmito_a) + psiIDH_a(ISOCITmito_a,NADmito_a,NADHmito_a) + psiPDH_a(PYRmito_a,NADmito_a,CoAmito_a) + psiGDH_simplif_a(NADmito_a,GLUT_a,NADHmito_a,AKGmito_a) # NADHmito_a
    
    ######################
    # Cyto
    # Winter2017, Jolivet2015
    FinDyn_W2017(t) = global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3)))))) # is global_par_F_in
    Fout_W2017(vV,t) = global_par_F_0*((vV/vV0)^2 + (vV/vV0)^(-0.5)*global_par_tau_v/vV0*   (  global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))  )  )/(1+global_par_F_0*(vV/vV0)^(-0.5)*global_par_tau_v/vV0)           

    JdHbin(O2cap,t) = 2*(C_O_a - O2cap) * (  global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))  ) 
    JdHbout(vV,t,ddHb) = (ddHb/vV) * (  global_par_F_0*((vV/vV0)^2 +(vV/vV0)^(-0.5)*global_par_tau_v/vV0*   (  global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))  )  )/(1+global_par_F_0*(vV/vV0)^(-0.5)*global_par_tau_v/vV0)  )  

    #JO2art2cap(O2cap,t) = 2*((C_O_a  -  O2cap)/eto_b) *  (  global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))  )    
    JO2art2cap(O2cap,t) = (1/eto_b)*2*(C_O_a  -  O2cap) *  (  global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))  )    
    JO2fromCap2a(O2cap,O2_a) = (PScapA*(KO2b*(HbOP/O2cap - 1.)^(-1/param_degree_nh) - O2_a ))   
    JO2fromCap2n(O2cap,O2_n) = (PScapNAratio*PScapA*(KO2b*(HbOP/O2cap - 1.)^(-1/param_degree_nh) - O2_n )) 

    # Winter2017, Jolivet2015
    trGLC_art_cap(t,Glc_b) = (1/eto_b)*(2*(C_Glc_a-Glc_b))*(global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))) 

    # DiNuzzo2010, but denom corr to a common form formula given in main text pdf instead of supp pdf
    JGlc_b(Glc_b,glc_D_ecsEndothelium) = TmaxGLCce*( Glc_b*(KeG + glc_D_ecsEndothelium) - glc_D_ecsEndothelium*(KeG + Glc_b) ) /  ( KeG^2 + KeG*ReGoi*Glc_b + KeG*ReGio*glc_D_ecsEndothelium + ReGee*Glc_b*glc_D_ecsEndothelium )
    JGlc_e2ecsBA(glc_D_ecsEndothelium,Glc_ecsBA) = TmaxGLCeb*( glc_D_ecsEndothelium*(KeG2 + Glc_ecsBA) - Glc_ecsBA*(KeG2 + glc_D_ecsEndothelium) ) / ( KeG2^2 + KeG2*ReGoi2*glc_D_ecsEndothelium + KeG2*ReGio2*Glc_ecsBA + ReGee2*glc_D_ecsEndothelium*Glc_ecsBA )
    JGlc_ecsBA2a(Glc_ecsBA,Glc_a) = TmaxGLCba*( Glc_ecsBA*(KeG3 + Glc_a) - Glc_a*(KeG3 + Glc_ecsBA) ) /  ( KeG3^2 + KeG3*ReGoi3*Glc_ecsBA + KeG3*ReGio3*Glc_a + ReGee3*Glc_ecsBA*Glc_a )
    JGlc_a2ecsAN(Glc_a,Glc_ecsAN) = TmaxGLCai*( Glc_a*(KeG4 + Glc_ecsAN) - Glc_ecsAN*(KeG4 + Glc_a) ) /  ( KeG4^2 + KeG4*ReGoi4*Glc_a + KeG4*ReGio4*Glc_ecsAN + ReGee4*Glc_a*Glc_ecsAN )
    JGlc_ecsAN2n(Glc_ecsAN,Glc_n) = TmaxGLCin*( Glc_ecsAN*(KeG5 + Glc_n) - Glc_n*(KeG5 + Glc_ecsAN) ) /  ( KeG5^2 + KeG5*ReGoi5*Glc_ecsAN + KeG5*ReGio5*Glc_n + ReGee5*Glc_ecsAN*Glc_n )
    JGlc_diffEcs(Glc_ecsBA,Glc_ecsAN) = kGLCdiff*(Glc_ecsBA - Glc_ecsAN)

    # Glycogen
    # Glycogen synthase: UDPgluco + GLY ⇒ 2*GLY
    #psiGS_a(GLY_a) = VmaxGS_a*( GLY_a0/GLY_a - 1)
    psiGS_a(GS_a,UDPgluco_a) = kL2_GS_a*GS_a*UDPgluco_a / (kmL2_GS_a+UDPgluco_a) 
    
    #psiGPa_a(GPa_a,GPb_a,AMP_a) = (GPa_a/(GPa_a+GPb_a))*VmaxGP_a*(1/(1 + (KmGP_AMP_a^hGPa)/(AMP_a^hGPa)))
    #psiGPa_a(GPa_a,GPb_a,cAMP_a,GLY_a) = (GPa_a/(GPa_a+GPb_a))*VmaxGP_a*GLY_a*(1/(1 + (KmGP_AMP_a^hGPa)/(cAMP_a^hGPa)))
    psiGPa_a(GPa_a,GLY_a) = k_L2_GS_a*GPa_a*GLY_a/(km_L2_GS_a+GLY_a)
    
    psiPGLM_a(G1P_a,G6P_a) = (Vmaxfpglm_a*G1P_a/KmG1PPGLM_a - ((Vmaxfpglm_a*KmG6PPGLM_a) /(KmG1PPGLM_a*KeqPGLM_a))*G6P_a/KmG6PPGLM_a) / (1.0 + G1P_a/KmG1PPGLM_a + G6P_a/KmG6PPGLM_a)
    PDE_a(cAMP_a) = VmaxPDE_a*cAMP_a/(Kmcamppde_a + cAMP_a)  
    
    # UDP-glucose pyrophosphorylase; G1P + UTP ⇒ udp-gluco +PPi  # rates adapted from Hester and Raushel 1987
    psiUDPGP_a(UTP_a,G1P_a,PPi_a,UDPgluco_a) = (VmaxfUDPGP*UTP_a*G1P_a/(KutpUDPGP*Kg1pUDPGP) - VmaxrUDPGP*PPi_a*UDPgluco_a/(KpiUDPGP*KUDPglucoUDPGP_a)) / (1.0 + G1P_a/Kg1pUDPGP + UTP_a/KutpUDPGP + (G1P_a*UTP_a)/(Kg1pUDPGP*KutpUDPGP) + UDPgluco_a/KUDPglucoUDPGP_a + PPi_a/KpiUDPGP + (PPi_a*UDPgluco_a)/(KpiUDPGP*KUDPglucoUDPGP_a))           

    psiGSAJay(GS_a,UDPgluco_a,PKAa_a,PHKa_a) = ((kg8_GSAJay*PP1_a0*(st_GSAJay-GS_a))/((kmg8_GSAJay/(1.0 + s1_GSAJay*UDPgluco_a/kg2_GSAJay)) + (st_GSAJay-GS_a))) - ((kg7_GSAJay*(PHKa_a+PKAa_a)*GS_a) /(kmg7_GSAJay*(1+s1_GSAJay*UDPgluco_a/kg2_GSAJay)+GS_a))      

    # Phosphorylase kinase act  0 ⇒ PHKa 
    #psiPHKact(Ca_a,PKAa_a,PHKa_a) = (Ca_a/cai0_ca_ion)*(((kg3_PHKact*PKAa_a*(kt_PHKact-PHKa_a))/(kmg3_PHKact+kt_PHKact-PHKa_a)) - (((kg4_PHKact*PP1_a0*PHKa_a))/(kmg4_PHKact+PHKa_a)))
    # PHK GPb->GPa #  GPb + 2*ATP ⇒ GPa + 2*ADP
    psiPHK(PHKa_a,GPa_a,GLY_a,G6P_a,Glc_a,Ca_a ) = (Ca_a/cai0_ca_ion)*(((kg5_PHK*PHKa_a*(pt_PHK-GPa_a))/(kmg5_PHK*(1.0 +s1_PHK*G6P_a/kg2_PHK) + (pt_PHK-GPa_a))) - ((kg6_PHK*PP1_a0*GPa_a)/(kmg6_PHK/(1+s2_PHK*Glc_a/kgi_PHK)+GPa_a)) - ((0.003198/(1+ GLY_a) + kmind_PHK)*PP1_a0*GPa_a)) 

    # Protein kinase A (=cAMP-dependent protein kinase); VPKA(), PKAb + 4*cAMP -> R2CcAMP4 + 2*PKAa by Jay, Xu-Gomez # or it was PKAb + ATP ⇒ PKAa + ADP
    # psiPKA1(cAMP_a,PKAa_a,PKAb_a,R2CcAMP2_a)  = kgc1_PKA12*PKAb_a*(cAMP_a^2) - k_gc1_PKA12*R2CcAMP2_a*PKAa_a 
    # psiPKA2(cAMP_a,PKAa_a,R2CcAMP2_a,R2CcAMP4_a)  = kgc2_PKA12*R2CcAMP2_a*(cAMP_a^2) - k_gc2_PKA12*R2CcAMP4_a*PKAa_a  

    
    
    
    
    
    # combo of Jolivet2015 and DiNuzzo2010(glycogen)
    psiHK_n(Glc_n,ATP_n,G6P_n) = (VmaxHK_n*Glc_n/(Glc_n + KmHK_n))*(ATP_n/(1+(ATP_n/KIATPhex_n)^nHhexn))*(1/(1 + G6P_n/KiHKG6P_n))
    psiHK_a(Glc_a,ATP_a,G6P_a) = (VmaxHK_a*Glc_a/(Glc_a + KmHK_a))*(ATP_a/(1+(ATP_a/KIATPhex_a)^nHhexa))*(1/(1 + G6P_a/KiHKG6P_a))
    
    # psiPGI_n(G6P_n,F6P_n) = (Vmax_fPGI_n*G6P_n/Km_G6P_fPGI_n - Vmax_rPGI_n*F6P_n/Km_F6P_rPGI_n )/ (1.0+G6P_n/Km_G6P_fPGI_n+ F6P_n/Km_F6P_rPGI_n)       
    # psiPGI_a(G6P_a,F6P_a) = (Vmax_fPGI_a*G6P_a/Km_G6P_fPGI_a - Vmax_rPGI_a*F6P_a/Km_F6P_rPGI_a )/ (1.0+G6P_a/Km_G6P_fPGI_a+ F6P_a/Km_F6P_rPGI_a)       

    psiPGI_n(G6P_n,F6P_n) = (Vmax_fPGI_n*(G6P_n/Km_G6P_fPGI_n - 0.9*F6P_n/Km_F6P_rPGI_n) )/ (1.0+G6P_n/Km_G6P_fPGI_n+ F6P_n/Km_F6P_rPGI_n)       
    psiPGI_a(G6P_a,F6P_a) = (Vmax_fPGI_a*(G6P_a/Km_G6P_fPGI_a - 0.9*F6P_a/Km_F6P_rPGI_a) )/ (1.0+G6P_a/Km_G6P_fPGI_a+ F6P_a/Km_F6P_rPGI_a)       

    psiPFK_n(ATP_n,F6P_n) = VmaxPFK_n*(ATP_n/(1+(ATP_n/KiPFK_ATP_na)^nPFKn))*(F6P_n/(F6P_n + KmPFKF6P_n))
#     psiPFK_a(ATP_a,F6P_a,f26bp_a) = VmaxPFK_a*(ATP_a/(1+(ATP_a/KiPFK_ATP_na)^nPFKa))*(F6P_a/(F6P_a + KmPFKF6P_a*(1 - KoPFK_f26bp_a*((f26bp_a^nPFKf26bp_a)/(KmF26BP_PFK_a^nPFKf26bp_a + f26bp_a^nPFKf26bp_a)))))*(f26bp_a/(KmF26BP_PFK_a + f26bp_a))
    psiPFK_a(ATP_a,F6P_a,f26bp_a) = VmaxPFK_a*(ATP_a/(1+(ATP_a/KiPFK_ATP_a)^nPFKa))*(F6P_a/(F6P_a + KmPFKF6P_a*(1 - KoPFK_f26bp_a*((f26bp_a^nPFKf26bp_a)/(KmF26BP_PFK_a^nPFKf26bp_a + f26bp_a^nPFKf26bp_a)))))*(f26bp_a/(KmF26BP_PFK_a + f26bp_a))
    
    #psiPFK2_a(F6P_a,ATP_a,ADP_a,f26bp_a) = Vmax_PFKII_g*F6P_a*ATP_a*ADP_a / ((F6P_a + Kmf6pPFKII_g)*(ATP_a + KmatpPFKII_g)*(ADP_a + Km_act_adpPFKII_g) )  -  (Vmax_f26pase_g*f26bp_a/(f26bp_a + Km_f26bp_f_26pase_g*(1+F6P_a/Ki_f6p_f_26_pase_g)))  
    psiPFK2_a(F6P_a,ATP_a,ADP_a,f26bp_a) = Vmax_PFKII_g*F6P_a*ATP_a*ADP_a / ((F6P_a + Kmf6pPFKII_g)*(ATP_a + KmatpPFKII_g)*(ADP_a + Km_act_adpPFKII_g) )  -  (Vmax_PFKII_g*f26bp_a/(f26bp_a + Km_f26bp_f_26pase_g*(1+F6P_a/Ki_f6p_f_26_pase_g)))  
    
    psiALD_n(FBP_n,GAP_n,DHAP_n) = Vmaxald_n*(FBP_n - GAP_n*DHAP_n/Keqald_n)/ ((1 + FBP_n/KmfbpAld_n) + (1 + GAP_n/KmgapAld_n)*(1 + DHAP_n/KmdhapAld_n) -1)
    psiALD_a(FBP_a,GAP_a,DHAP_a) = Vmaxald_a*(FBP_a - GAP_a*DHAP_a/Keqald_a)/ ((1 + FBP_a/KmfbpAld_a) + (1 + GAP_a/KmgapAld_a)*(1 + DHAP_a/KmdhapAld_a) -1)

    psiTPI_n(DHAP_n,GAP_n) = Vmaxtpi_n*(DHAP_n-GAP_n/Keqtpi_n)/(1+DHAP_n/KmdhapTPI_n+GAP_n/KmgapTPI_n)
    psiTPI_a(DHAP_a,GAP_a) = Vmaxtpi_a*(DHAP_a-GAP_a/Keqtpi_a)/(1+DHAP_a/KmdhapTPI_a+GAP_a/KmgapTPI_a)

    psiGAPDH_n(NAD_n,GAP_n,Pi_n,BPG13_n,NADH_n) = Vmaxgapdh_n*(NAD_n*GAP_n*Pi_n-BPG13_n*NADH_n/Keqgapdh_na) / ((1+NAD_n/KmnadGpdh_n)*(1+GAP_n/KmGapGapdh_n)*(1+Pi_n/KmpiGpdh_n) + (1+NADH_n/KmnadhGapdh_n)*(1+ BPG13_n/KmBPG13Gapdh_n) -1 ) 
    psiGAPDH_a(NAD_a,GAP_a,Pi_a,BPG13_a,NADH_a) = Vmaxgapdh_a*(NAD_a*GAP_a*Pi_a-BPG13_a*NADH_a/Keqgapdh_na) / ((1+NAD_a/KmnadGpdh_a)*(1+GAP_a/KmGapGapdh_a)*(1+Pi_a/KmpiGpdh_a) + (1+NADH_a/KmnadhGapdh_a)*(1+ BPG13_a/KmBPG13Gapdh_a) -1 ) 

    psiPGK_n(BPG13_n,ADP_n,PG3_n,ATP_n) = Vmaxpgk_n*(BPG13_n*ADP_n - PG3_n*ATP_n/Keqpgk_na) / ((1+BPG13_n/Kmbpg13pgk_n)*(1+ADP_n/Kmadppgk_n) + (1+PG3_n/Kmpg3pgk_n)*(1+ATP_n/Kmatppgk_n)-1)
    psiPGK_a(BPG13_a,ADP_a,PG3_a,ATP_a) = Vmaxpgk_a*(BPG13_a*ADP_a - PG3_a*ATP_a/Keqpgk_na) / ((1+BPG13_a/Kmbpg13pgk_a)*(1+ADP_a/Kmadppgk_a) + (1+PG3_a/Kmpg3pgk_a)*(1+ATP_a/Kmatppgk_a)-1)

    psiPGM_n(PG3_n,PG2_n) = Vmaxpgm_n*(PG3_n-PG2_n/Keqpgm_n) / ((1+PG3_n/Kmpg3pgm_n)+(1+PG2_n/Kmpg2pgm_n)-1)
    psiPGM_a(PG3_a,PG2_a) = Vmaxpgm_a*(PG3_a-PG2_a/Keqpgm_a) / ((1+PG3_a/Kmpg3pgm_a)+(1+PG2_a/Kmpg2pgm_a)-1)

    psiENOL_n(PG2_n,PEP_n) = Vmaxenol_n*(PG2_n-PEP_n/Keqenol_n) / ((1+PG2_n/Kmpg2enol_n)+(1+PEP_n/Km_pep_enol_n)-1)
    psiENOL_a(PG2_a,PEP_a) = Vmaxenol_a*(PG2_a-PEP_a/Keqenol_a) / ((1+PG2_a/Kmpg2enol_a)+(1+PEP_a/Km_pep_enol_a)-1)

    psiPK_n(PEP_n,ADP_n,ATP_n) = Vmaxpk_n*PEP_n*ADP_n/((PEP_n + Km_pep_pk_n)*(ADP_n + Km_adp_pk_n*(1+ATP_n/Ki_ATP_pk_n)))
    psiPK_a(PEP_a,ADP_a,ATP_a) = Vmaxpk_a*PEP_a*ADP_a/((PEP_a + Km_pep_pk_a)*(ADP_a + Km_adp_pk_a*(1+ATP_a/Ki_ATP_pk_a)))
    
    JLacTr(t,Lac_b) = (2*(C_Lac_a - Lac_b)/eto_b)*(global_par_F_0 * (1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))) 
    MCT1_LAC_b(Lac_b,Lac_ecs) = TbLac*( Lac_b/(Lac_b + KbLac) - Lac_ecs/(Lac_ecs + KbLac) )    
    jLac_a(Lac_ecs,Lac_a) = TaLac*( Lac_ecs/(Lac_ecs + Km_Lac_a) - Lac_a/(Lac_a + Km_Lac_a) )    
    jLac_n(Lac_ecs,Lac_n) = TnLac*( Lac_ecs/(Lac_ecs + Km_LacTr_n) - Lac_n/(Lac_n + Km_LacTr_n) )
    jLacDiff_e(Lac_ecs) = betaLacDiff*(Lac_ecs0 - Lac_ecs)

    vLACgc(Lac_b,Lac_a)  =  TMaxLACgc*( Lac_b/(Lac_b+KtLACgc) - Lac_a/(Lac_a+KtLACgc)) # jlv
    
#     psiLDH_a(NADH_a,Pyr_a,NAD_a,Lac_a) =  VmfLDH_a*Pyr_a*NADH_a - VmrLDH_a*Lac_a*NAD_a
#     psiLDH_n(NADH_n,Pyr_n,NAD_n,Lac_n) =  VmfLDH_n*Pyr_n*NADH_n - VmrLDH_n*Lac_n*NAD_n
    psiLDH_a(NADH_a,Pyr_a,NAD_a,Lac_a) =  VmfLDH_a*Pyr_a*NADH_a - KeLDH_a*VmfLDH_a*Lac_a*NAD_a
    psiLDH_n(NADH_n,Pyr_n,NAD_n,Lac_n) =  VmfLDH_n*Pyr_n*NADH_n - KeLDH_n*VmfLDH_n*Lac_n*NAD_n
    
#     psiLDH_a(NADH_a,Pyr_a,NAD_a,Lac_a) =  VmfLDH1_a*( (NADH_a/NAD_a) / (nu_LDH1f_a + (NADH_a/NAD_a)) )*(Pyr_a/(Pyr_a + KmLDH1pyr_a) ) - (VmrLDH1_phase1_a*( (NAD_a/NADH_a) / (nu_LDH1r_a + NAD_a/NADH_a) )*(Lac_a/(Lac_a + KmLDH1lac_phase1_a)) + VmrLDH1_phase2_a*( (NAD_a/NADH_a) / (nu_LDH1r_a + NAD_a/NADH_a) )*(Lac_a/(Lac_a + KmLDH1lac_phase2_a)))

#     #psiLDH_n(NADH_n,Pyr_n,NAD_n,Lac_n) = (VmfLDH5_n*( (NADH_n/NAD_n) / (nu_LDH5f_n + (NADH_n/NAD_n)) )*(Pyr_n/(Pyr_n + KmLDH5pyr_n) ) - (VmrLDH5_phase1_n*( (NAD_n/NADH_n) / (nu_LDH5r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH5lac_phase1_n)) + VmrLDH5_phase2_n*( (NAD_n/NADH_n) / (nu_LDH5r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH5lac_phase2_n)))) + (VmfLDH1_a*( (NADH_n/NAD_n) / (nu_LDH1f_a + (NADH_n/NAD_n)) )*(Pyr_n/(Pyr_n + KmLDH1pyr_a) ) - (VmrLDH1_phase1_a*( (NAD_n/NADH_n) / (nu_LDH1r_a + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH1lac_phase1_a)) + VmrLDH1_phase2_a*( (NAD_n/NADH_n) / (nu_LDH1r_a + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH1lac_phase2_a)))) 
#     #psiLDH_cyto_n(NADH_n,Pyr_n,NAD_n,Lac_n) = (VmfLDH1_a*( (NADH_n/NAD_n) / (nu_LDH1f_a + (NADH_n/NAD_n)) )*(Pyr_n/(Pyr_n + KmLDH1pyr_a) ) - (VmrLDH1_phase1_a*( (NAD_n/NADH_n) / (nu_LDH1r_a + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH1lac_phase1_a)) + VmrLDH1_phase2_a*( (NAD_n/NADH_n) / (nu_LDH1r_a + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH1lac_phase2_a)))) 
#     #psiLDH_syn_n(NADH_n,Pyr_n,NAD_n,Lac_n) =  (VmfLDH5_n*( (NADH_n/NAD_n) / (nu_LDH5f_n + (NADH_n/NAD_n)) )*(Pyr_n/(Pyr_n + KmLDH5pyr_n) ) - (VmrLDH5_phase1_n*( (NAD_n/NADH_n) / (nu_LDH5r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH5lac_phase1_n)) + VmrLDH5_phase2_n*( (NAD_n/NADH_n) / (nu_LDH5r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH5lac_phase2_n)))) 

#     psiLDH_cyto_n(NADH_n,Pyr_n,NAD_n,Lac_n) = (VmfLDH1_n*( (NADH_n/NAD_n) / (nu_LDH1f_n + (NADH_n/NAD_n)) )*(Pyr_n/(Pyr_n + KmLDH1pyr_n) ) - (VmrLDH1_phase1_n*( (NAD_n/NADH_n) / (nu_LDH1r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH1lac_phase1_n)) + VmrLDH1_phase2_n*( (NAD_n/NADH_n) / (nu_LDH1r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH1lac_phase2_n)))) 
#     psiLDH_syn_n(NADH_n,Pyr_n,NAD_n,Lac_n) = (VmfLDH5_n*( (NADH_n/NAD_n) / (nu_LDH5f_n + (NADH_n/NAD_n)) )*(Pyr_n/(Pyr_n + KmLDH5pyr_n) ) - (VmrLDH5_phase1_n*( (NAD_n/NADH_n) / (nu_LDH5r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH5lac_phase1_n)) + VmrLDH5_phase2_n*( (NAD_n/NADH_n) / (nu_LDH5r_n + NAD_n/NADH_n) )*(Lac_n/(Lac_n + KmLDH5lac_phase2_n)))) 

    #############
    # PPP
    #############
    # r01 G6PDH: G6P + NADP ⇒ NADPH + GL6P
    psiG6PDH_n(NADP_n,G6P_n,NADPH_n,GL6P_n) = VmaxG6PDH_n * (1/(K_G6P_G6PDH_n * K_NADP_G6PDH_n)) * ( (G6P_n*NADP_n - GL6P_n*NADPH_n/KeqG6PDH_n) / ( (1 + G6P_n/K_G6P_G6PDH_n)*(1 + NADP_n/K_NADP_G6PDH_n) + (1 + GL6P_n/K_GL6P_G6PDH_n)*(1 + NADPH_n/K_NADPH_G6PDH_n) - 1 ) )           
    psiG6PDH_a(NADP_a,G6P_a,NADPH_a,GL6P_a) = VmaxG6PDH_a * (1/(K_G6P_G6PDH_a * K_NADP_G6PDH_a)) * ( (G6P_a*NADP_a - GL6P_a*NADPH_a/KeqG6PDH_a ) / (  (1 + G6P_a/K_G6P_G6PDH_a) * ( 1 + NADP_a/K_NADP_G6PDH_a ) + (1 + GL6P_a/K_GL6P_G6PDH_a) * (1 + NADPH_a/K_NADPH_G6PDH_a) - 1   ) )         

    # r02 6PGL: GL6P ⇒ GO6P
    psi6PGL_n(GL6P_n,GO6P_n) = Vmax6PGL_n*(1/K_GL6P_6PGL_n)*((GL6P_n - GO6P_n/Keq6PGL_n) / ( (1 + GL6P_n/K_GL6P_6PGL_n) + (1 + GO6P_n/K_GO6P_6PGL_n) - 1 ))
    psi6PGL_a(GL6P_a,GO6P_a) = Vmax6PGL_a*(1/K_GL6P_6PGL_a)*((GL6P_a - GO6P_a/Keq6PGL_a) / ( (1 + GL6P_a/K_GL6P_6PGL_a) + (1 + GO6P_a/K_GO6P_6PGL_a) - 1 )) 

    # r03 6PGDH: NADP_n + GO6P_n  ⇒ RU5P_n + NADPH_n
    psi6PGDH_n(NADP_n,GO6P_n,RU5P_n,NADPH_n) = Vmax6PGDH_n * (1 / (K_GO6P_6PGDH_n * K_NADP_6PGDH_n)) * (GO6P_n*NADP_n - RU5P_n*NADPH_n/Keq6PGDH_n ) / ( (1 + GO6P_n/K_GO6P_6PGDH_n)*(1 + NADP_n/K_NADP_6PGDH_n) + (1 + RU5P_n/K_RU5P_6PGDH_n)*(1 + NADPH_n/K_NADPH_6PGDH_n) - 1 )
    psi6PGDH_a(NADP_a,GO6P_a,RU5P_a,NADPH_a) = Vmax6PGDH_a * (1 / (K_GO6P_6PGDH_a * K_NADP_6PGDH_a)) * (GO6P_a*NADP_a - RU5P_a*NADPH_a/Keq6PGDH_a ) / ( (1 + GO6P_a/K_GO6P_6PGDH_a)*(1 + NADP_a/K_NADP_6PGDH_a) + (1 + RU5P_a/K_RU5P_6PGDH_a)*(1 + NADPH_a/K_NADPH_6PGDH_a) - 1 )

    # r04 RU5P_n ⇒ R5P_n 
    psiRPI_n(RU5P_n,R5P_n) = VmaxRPI_n * (1/K_RU5P_RPI_n) * (RU5P_n - R5P_n/KeqRPI_n) / ( (1 + RU5P_n/K_RU5P_RPI_n) + (1 + R5P_n/K_R5P_RPI_n) - 1)
    psiRPI_a(RU5P_a,R5P_a) = VmaxRPI_a * (1/K_RU5P_RPI_a) * (RU5P_a - R5P_a/KeqRPI_a) / ( (1 + RU5P_a/K_RU5P_RPI_a) + (1 + R5P_a/K_R5P_RPI_a) - 1)

    # r05 RU5P_n ⇒ X5P_n
    psiRPEppp_n(RU5P_n,X5P_n) = VmaxRPE_n * (1/K_RU5P_RPE_n) * (RU5P_n - X5P_n/KeqRPE_n) / ((1 + RU5P_n/K_RU5P_RPE_n) + (1 + X5P_n/K_X5P_RPE_n) - 1)
    psiRPEppp_a(RU5P_a,X5P_a) = VmaxRPE_a * (1/K_RU5P_RPE_a) * (RU5P_a - X5P_a/KeqRPE_a) / ((1 + RU5P_a/K_RU5P_RPE_a) + (1 + X5P_a/K_X5P_RPE_a) - 1)

    # r06 R5P_n + X5P_n  ⇒ S7P_n + GAP_n
    psiTKL1_n(R5P_n,X5P_n,S7P_n,GAP_n) = VmaxTKL1_n * (1/(K_X5P_TKL1_n*K_R5P_TKL1_n)) * (X5P_n*R5P_n - GAP_n*S7P_n/KeqTKL1_n) / ( (1 + X5P_n/K_X5P_TKL1_n)*(1 + R5P_n/K_R5P_TKL1_n) + (1 + GAP_n/K_GAP_TKL1_n)*(1 + S7P_n/K_S7P_TKL1_n) - 1 )
    psiTKL1_a(R5P_a,X5P_a,S7P_a,GAP_a) = VmaxTKL1_a * (1/(K_X5P_TKL1_a*K_R5P_TKL1_a)) * (X5P_a*R5P_a - GAP_a*S7P_a/KeqTKL1_a) / ( (1 + X5P_a/K_X5P_TKL1_a)*(1 + R5P_a/K_R5P_TKL1_a) + (1 + GAP_a/K_GAP_TKL1_a)*(1 + S7P_a/K_S7P_TKL1_a) - 1 )

    # r07 F6P_n + GAP_n ⇒ X5P_n + E4P_n
    psiTKL2_n(X5P_n,E4P_n,F6P_n,GAP_n) = VmaxTKL2_n * (1/(K_F6P_TKL2_n*K_GAP_TKL2_n)) * (F6P_n*GAP_n - X5P_n*E4P_n/KeqTKL2_n) / ( (1 + F6P_n/K_F6P_TKL2_n)*(1 + GAP_n/K_GAP_TKL2_n) + (1 + X5P_n/K_X5P_TKL2_n)*(1 + E4P_n/K_E4P_TKL2_n) - 1)
    psiTKL2_a(X5P_a,E4P_a,F6P_a,GAP_a) = VmaxTKL2_a * (1/(K_F6P_TKL2_a*K_GAP_TKL2_a)) * (F6P_a*GAP_a - X5P_a*E4P_a/KeqTKL2_a) / ( (1 + F6P_a/K_F6P_TKL2_a)*(1 + GAP_a/K_GAP_TKL2_a) + (1 + X5P_a/K_X5P_TKL2_a)*(1 + E4P_a/K_E4P_TKL2_a) - 1)

    # r08 S7P_n + GAP_n ⇒ E4P_n + F6P_n
    psiTALppp_n(S7P_n,GAP_n,E4P_n,F6P_n) = VmaxTAL_n*(1/(K_GAP_TAL_n*K_S7P_TAL_n)) * (GAP_n*S7P_n - F6P_n*E4P_n/KeqTAL_n) / ( (1 + GAP_n/K_GAP_TAL_n)*(1 + S7P_n/K_S7P_TAL_n) + (1 + F6P_n/K_F6P_TAL_n)*(1 + E4P_n/K_E4P_TAL_n) - 1)
    psiTALppp_a(S7P_a,GAP_a,E4P_a,F6P_a) = VmaxTAL_a*(1/(K_GAP_TAL_a*K_S7P_TAL_a)) * (GAP_a*S7P_a - F6P_a*E4P_a/KeqTAL_a) / ( (1 + GAP_a/K_GAP_TAL_a)*(1 + S7P_a/K_S7P_TAL_a) + (1 + F6P_a/K_F6P_TAL_a)*(1 + E4P_a/K_E4P_TAL_a) - 1)

    psiNADPHox_n(NADPH_n) = k1NADPHox_n*NADPH_n
    psiNADPHox_a(NADPH_a) = k1NADPHox_a*NADPH_a
    
    ###########
    # GSH
    ###########
    psiGSSGR_n(GSSG_n,NADPH_n) = (Vmf_GSSGR_n*GSSG_n*NADPH_n) / ( ( KmGSSGRGSSG_n + GSSG_n )*( KmGSSGRNADPH_n + NADPH_n )  )  # GSSG_n + NADPH_n ⇒ 2GSH_n + NADP_n
    psiGSSGR_a(GSSG_a,NADPH_a) = (Vmf_GSSGR_a*GSSG_a*NADPH_a ) / ( ( KmGSSGRGSSG_a + GSSG_a )*( KmGSSGRNADPH_a + NADPH_a )  )   # GSSG_a + NADPH_a ⇒  2GSH_a + NADP_a
    psiGPX_n(GSH_n) = V_GPX_n * GSH_n / (GSH_n + KmGPXGSH_n)
    psiGPX_a(GSH_a) = V_GPX_a * GSH_a / (GSH_a + KmGPXGSH_a)

    glutathioneSyntase_n(GSH_n) = VmaxGSHsyn_n*GSH_n/(GSH_n + KmGSHsyn_n)
    glutathioneSyntase_a(GSH_a) = VmaxGSHsyn_a*GSH_a/(GSH_a + KmGSHsyn_a)
    
    ############
    # ATDMP
#     psiADK_n(ATP_n,AMP_n,ADP_n) = ((Vmaxfadk_n*(ATP_n*AMP_n)/(KmATPADK_n*KmAMPADK_n) - (((Vmaxfadk_n*(KmADPADK_n^2)) / (KmATPADK_n*KmAMPADK_n*KeqADK_n))*(((ADP_n)^2)/(KmADPADK_n^2)))) / (1 + ATP_n/KmATPADK_n + AMP_n/KmAMPADK_n + (ATP_n*AMP_n)/(KmATPADK_n*KmAMPADK_n) + (2*ADP_n)/KmADPADK_n + ((ADP_n)^2)/(KmADPADK_n^2)))   
#     psiADK_a(ATP_a,AMP_a,ADP_a) = ((Vmaxfadk_a*(ATP_a*AMP_a)/(KmATPADK_a*KmAMPADK_a) - (((Vmaxfadk_a*(KmADPADK_a^2)) / (KmATPADK_a*KmAMPADK_a*KeqADK_a))*(((ADP_a)^2)/(KmADPADK_a^2)))) / (1 + ATP_a/KmATPADK_a + AMP_a/KmAMPADK_a + (ATP_a*AMP_a)/(KmATPADK_a*KmAMPADK_a) + (2*ADP_a)/KmADPADK_a + ((ADP_a)^2)/(KmADPADK_a^2)))    
    # psiAC_a(ATP_a,cAMP_a,PPi_a) = ((VmaxfAC_a*ATP_a/(KmACATP_a*(1+cAMP_a/KicAMPAC_a)) - VmaxrAC_a*cAMP_a*PPi_a/(KmpiAC_a*KmcAMPAC_a))/(1 + ATP_a/(KmACATP_a*(1+cAMP_a/KicAMPAC_a) )+ cAMP_a/KmcAMPAC_a + (cAMP_a*PPi_a)/(KmcAMPAC_a*KmpiAC_a) + PPi_a/KmpiAC_a))
    # psiAC_a(ATP_a,cAMP_a,Pi_a) = ((VmaxfAC_a*ATP_a/(KmACATP_a*(1+cAMP_a/KicAMPAC_a)) - VmaxrAC_a*cAMP_a*Pi_a/(KmpiAC_a*KmcAMPAC_a))/(1 + ATP_a/(KmACATP_a*(1+cAMP_a/KicAMPAC_a) )+ cAMP_a/KmcAMPAC_a + (cAMP_a*Pi_a)/(KmcAMPAC_a*KmpiAC_a) + Pi_a/KmpiAC_a))  # Pi_a instead of PPi_a as approx
    psiAC_a(ATP_a,cAMP_a) = ((VmaxfAC_a*ATP_a/(KmACATP_a*(1+cAMP_a/KicAMPAC_a)) - VmaxrAC_a*cAMP_a/(KmpiAC_a*KmcAMPAC_a))/(1 + ATP_a/(KmACATP_a*(1+cAMP_a/KicAMPAC_a) )+ cAMP_a/KmcAMPAC_a ))  # Pi_a instead of PPi_a as approx

    
    # Calv
    # psiCrKinase_n(Cr_n,ATP_n,ADP_n,PCr_n) = V_Cr_n  * (Cr_n / (Cr_n + Km_Cr_n)) * ( (ATP_n/ADP_n)/ ( mu_Cr_n + ATP_n/ADP_n )) - ( V_PCr_n * (PCr_n / (PCr_n + Km_PCr_n)) * ( ADP_n/ATP_n ) / ( mu_PCr_n + ADP_n/ATP_n ) )
    # psiCrKinase_a(Cr_a,ATP_a,ADP_a,PCr_a) = V_Cr_a * (Cr_a / (Cr_a + Km_Cr_a)) * ( (ATP_a/ADP_a) / ( mu_Cr_a + ATP_a/ADP_a ))  -  (V_PCr_a * (PCr_a / (PCr_a + Km_PCr_a)) * ( ADP_a/ATP_a ) / ( mu_PCr_a + ADP_a/ATP_a ))    
    #Jlv
    psiCrKinase_n(PCr_n,ATP_n,ADP_n)  =  kCKnps*PCr_n*ADP_n - KeqCKnpms*kCKnps*(Crtot - PCr_n)*ATP_n # kCKnps*PCr_n*ADP_n - kCKnms*(Crtot - PCr_n)*ATP_n
    psiCrKinase_a(PCr_a,ATP_a,ADP_a)  =  kCKgps*PCr_a*ADP_a - KeqCKgpms*kCKgps*(Crtot - PCr_a)*ATP_a # kCKgps*PCr_a*ADP_a - kCKgms*(Crtot - PCr_a)*ATP_a
    
    ############
    # PyrTr
    psiPYRtrcyt2mito_n(Pyr_n,PYRmito_n,C_H_mitomatr_n) = Vmax_PYRtrcyt2mito_nH*(Pyr_n*C_H_cyt_n - PYRmito_n*C_H_mitomatr_n)/((1.0+Pyr_n/KmPyrCytTr_n)*(1.0+PYRmito_n/KmPyrMitoTr_n))  
    psiPYRtrcyt2mito_a(Pyr_a,PYRmito_a,C_H_mitomatr_a) = Vmax_PYRtrcyt2mito_aH*(Pyr_a*C_H_cyt_a - PYRmito_a*C_H_mitomatr_a)/((1.0+Pyr_a/KmPyrCytTr_a)*(1.0+PYRmito_a/KmPyrMitoTr_a))    
    
    #########
    # ephys
    V = VNeu
    rTRPVsinf = vV
    Glutamate_syn = GLUT_syn # = Glutamate_syn #u[12] == u[144] #################
    
    alpham = -0.1*(V+33)/(exp(-0.1*(V+33))-1) #-0.32*(V - V_T - 13.0)/(exp(-(V - V_T -13.0)/4.0) - 1.0) #0.1*(V+30.0)/(1.0-exp(-0.1*(V+30.0)))
    betam = 4*exp(-(V+58)/12) #0.28*(V - V_T - 40.0)/( exp((V - V_T - 40.0)/5.0) -1.0) #4.0*exp(-(V+55.0)/18.0)
    
    alphah = 0.07*exp(-(V+50)/10) #0.128*exp(-(V - V_T - 17.0)/18.0) #0.07*exp(-(V+44.0)/20.0)
    betah = 1/(exp(-0.1*(V+20))+1) #4.0/(1.0 + exp(-(V - V_T - 40.0)/5.0)) #1.0/(1.0+exp(-0.1*(V+14.0)))
    
    alphan = -0.01*(V+34)/(exp(-0.1*(V+34))-1) #-0.032*(V - V_T - 15.0)/(exp(-(V - V_T - 15.0)/5.0) -1.0)  #0.01*(V+34.0)/(1.0-exp(-0.1*(V+34.0)))
    betan = 0.125*exp(-(V+44)/25) #0.5*exp(- (V - V_T - 10.0)/40.0 )  #0.125*exp(-(V+44.0)/80.0)
    
    minf  =   alpham/(alpham+betam);    ninf  =   alphan/(alphan+betan);    hinf =   alphah/(alphah+betah);    taun =   1/(alphan+betan)*1e-03;    tauh  =   1/(alphah+betah)*1e-03;
    p_inf = 1.0/(1.0 + exp(-(V + 35.0)/10.0));  tau_p = tau_max/ ( 3.3*exp((V+35.0)/20.0) + exp(-(V+35.0)/20.0) )
    
    K_n = K_n_Rest + (Na_n_Rest - Na_n)     
    EK = RTF*log(K_out/K_n)    
    EL          =   gKpas*EK/(gKpas+gNan) + gNan/(gKpas+gNan)*RTF*log(Na_out/Na_n);
    IL          =   gL*(V-EL);
    INa         =   gNa*minf^3*h*(V-RTF*log(Na_out/Na_n));
    IK          =   gK*n^4*(V-EK);
    mCa         =   1/(1+exp(-(V+20)/9));
    ICa         =   gCa*mCa^2*(V-ECa);
    ImAHP       =   gmAHP*Ca_n/(Ca_n+KD)*(V-EK);
    IM = g_M * pgate * (V - EK) # not Jlv, enzymes/enzymes_preBigg/gen_mix.ipynb
    
    dIPump      =   F*kPumpn*ATP_n*(Na_n-Na_n0)/(1+ATP_n/KmPump);
    dIPump_a      =   F*kPumpg*ATP_a*(Na_a-Na_a0)/(1+ATP_a/KmPump)
    
    Isyne       =   -synInput*(V-Ee);
    Isyni       = 0 #  -InhibitoryConductance(t+TIME,t1,tstim)*(V-Ei);

    vnstim  =   SmVn/F*(2/3*Isyne-INa);
    vgstim      = SmVg/F*2/3*glia*synInput;
    vLeakNan    =   SmVn*gNan/F*(RTF*log(Na_out/Na_n)-V);
    vLeakNag    =   SmVg*gNag/F*(RTF*log(Na_out/Na_a)-V);
    vPumpn      =   SmVn*kPumpn*ATP_n*Na_n/(1+ATP_n/KmPump);   # #JpumpNa = ((ATP_n/ADP_n)/(mu_pump_ephys + (ATP_n/ADP_n))) * ImaxNKApump * (( 1/(1 + Kout_alpha/K_out))^2) * (( 1/(1 + Na_in_alpha/Na_n))^3)  # doi:10.1152/jn.00460.2014
    vPumpg      =   SmVg*kPumpg*ATP_a*Na_a/(1+ATP_a/KmPump);
    JgliaK = ((ATP_a/ADP_a)/(mu_glia_ephys + (ATP_a/ADP_a))) * (glia_c/(1+exp((Na_n2_baseNKA - K_out)/2.5)))
    JdiffK = epsilon*(K_out - kbath)
    
    nBKinf = 0.5*( 1 + tanh( (Va + EETshift*EET_a - (-0.5*v5BK*tanh((Ca_a - Ca3BK)/Ca4BK )  + v6BK) )/v4BK )  )  
    IBK = gBK*nBK_a*(Va - EBK)
    
    #INaK_a = -0.5*(ImaxNaKa*( K_out/(K_out + INaKaKThr)  )*( (Na_a^1.5)/(Na_a^1.5 + INaKaNaThr^1.5) ) )
    JNaK_a = (ImaxNaKa*( K_out/(K_out + INaKaKThr)  )*( (Na_a^1.5)/(Na_a^1.5 + INaKaNaThr^1.5) ) )
    
    IKirAS = gKirS*(K_out^0.5)*(Va - VKirS*log(K_out/K_a))
    IKirAV = gKirV*(K_out^0.5)*(Va - VKirAV*log(K_out/K_a))
    IleakA = gleakA*(Va - VleakA)
    
    Ileak_CaER_a = Pleak_CaER_a*(1. - Ca_a/Ca_r_a)
    ICa_pump_a = VCa_pump_a*((Ca_a^2)/(Ca_a^2 + KpCa_pump_a^2 ))
    IIP3_a = ImaxIP3_a*( ( ( IP3_a/(IP3_a + KIIP3_a) )*( Ca_a/(Ca_a + KCaactIP3_a) )*hIP3Ca_a  )^3 )*(1. - Ca_a/Ca_r_a)
    ITRP_a = gTRP*(Va - VTRP)*sTRP_a  
    sinfTRPV = (1/(1 + exp( -(((rTRPVsinf^(1/3)-r0TRPVsinf^(1/3))/r0TRPVsinf^(1/3)) - e2TRPVsinf^(1/3))/kTRPVsinf )))*( (1/(1+ Ca_a/gammaCaaTRPVsinf + Ca_perivasc/gammaCapTRPVsinf ))*( Ca_a/gammaCaaTRPVsinf + Ca_perivasc/gammaCapTRPVsinf  + tanh((Va - v1TRPsinf_a)/v2TRPsinf_a )   ) )     
    
    # Jolivet2015 NADH shuttles cyto-mito
    vShuttlen(NADH_n,NADHmito_n)   =   TnNADH_jlv*(NADH_n/(0.212-NADH_n))/(MnCyto_jlv+(NADH_n/(0.212-NADH_n)))*((1000*NADtot-NADHmito_n)/NADHmito_n)/(MnMito_jlv+((1000*NADtot-NADHmito_n)/NADHmito_n))
    vShuttleg(NADH_a,NADHmito_a)   =   TgNADH_jlv*(NADH_a/(0.212-NADH_a))/(MgCyto_jlv+(NADH_a/(0.212-NADH_a)))*((1000*NADtot-NADHmito_a)/NADHmito_a )/(MgMito_jlv+((1000*NADtot-NADHmito_a)/NADHmito_a ))

    # Jolivet2015 mito
#     vMitooutn(O2_n,ADPmito_n,NADHmito_n)   =   VrespMitoout_n*O2_n/(O2_n+KO2MitoResp)*ADPmito_n/(ADPmito_n+KRespADP_n)*NADHmito_n/(NADHmito_n + KRespNADH_n)
#     vMitooutg(O2_a,ADPmito_a,NADHmito_a)   =   VrespMitoout_a*O2_a/(O2_a+KO2MitoResp)*ADPmito_a/(ADPmito_a+KRespADP_a)*NADHmito_a/(NADHmito_a + KRespNADH_a)
 
    # vMitooutn(O2_n,ADP_i_n,NADHmito_n)   =   VrespMitoout_n*O2_n/(O2_n+KO2MitoResp)*ADP_i_n/(ADP_i_n+KRespADP_n)*NADHmito_n/(NADHmito_n + KRespNADH_n)
    # vMitooutg(O2_a,ADP_i_a,NADHmito_a)   =   VrespMitoout_a*O2_a/(O2_a+KO2MitoResp)*ADP_i_a/(ADP_i_a+KRespADP_a)*NADHmito_a/(NADHmito_a + KRespNADH_a)
 
    
    
    # Calvetti
    vMitooutn(O2_n,ATP_i_n,ADP_i_n,NADHmito_n,NADmito_n) = V_oxphos_n * ((1/(ATP_i_n/ADP_i_n)) / ( mu_oxphos_n + (1/(ATP_i_n/ADP_i_n)) )) * ((NADHmito_n/NADmito_n) / (nu_oxphos_n + (NADHmito_n/NADmito_n)) ) * (O2_n / (O2_n + K_oxphos_n))
    vMitooutg(O2_a,ATP_i_a,ADP_i_a,NADHmito_a,NADmito_a) = V_oxphos_a * ((1/(ATP_i_a/ADP_i_a)) / ( mu_oxphos_a + (1/(ATP_i_a/ADP_i_a)) )) * ((NADHmito_a/NADmito_a) / (nu_oxphos_a + (NADHmito_a/NADmito_a)) ) * (O2_a / (O2_a + K_oxphos_a))

    
#     vMitoinn(Pyr_n,NADHmito_n)    =   VMaxMitoinn*Pyr_n/(Pyr_n+KmMito)*(0.212-NADHmito_n)/(0.212-NADHmito_n+KmNADn_jlv);
#     vMitoing(Pyr_a,NADHmito_a)    =   VMaxMitoing*Pyr_a/(Pyr_a+KmMito)*(0.212-NADHmito_a)/(0.212-NADHmito_a+KmNADg_jlv);
    
    vMitoinn(PYRmito_n,NADHmito_n)    =   VMaxMitoinn*PYRmito_n/(PYRmito_n+KmMito)*(1000*NADtot-NADHmito_n)/(1000*NADtot-NADHmito_n+KmNADn_jlv);
    vMitoing(PYRmito_a,NADHmito_a)    =   VMaxMitoing*PYRmito_a/(PYRmito_a+KmMito_a)*(1000*NADtot-NADHmito_a)/(1000*NADtot-NADHmito_a+KmNADg_jlv);
    
    du[1]  = 0 #0.5*T2Jcorrection*(1000*x_buff*C_H_mitomatr_nM*( +1*J_DH_n - (4+1)*J_C1_n - (4-2)*J_C3_n - (2+2)*J_C4_n + (n_A-1)*J_F1_n + 2*J_Pi1_n + J_Hle_n - J_KH_n )/W_x) # C_H_mitomatr_nM
    du[2]  = 0.5*T2Jcorrection*(1000*(J_KH_n + J_K_n)/W_x) # K_x_nM
    du[3]  = 0.5*T2Jcorrection*(1000*(-J_MgATPx_n - J_MgADPx_n)/W_x) # Mg_x_nM

#     du[4]  = 6.96*(vMitoinn(PYRmito_n,NADHmito_n) + vShuttlen(NADH_n,NADHmito_n) - vMitooutn(O2_n,ADPmito_n,NADHmito_n)) #0.5*T2Jcorrection*(1000*(+J_DH_n - J_C1_n)/W_x) # #  NADHmito_nM = 1e-3*u[27]
    # du[4]  = 6.96*(vMitoinn(PYRmito_n,NADHmito_n) + vShuttlen(NADH_n,NADHmito_n) - vMitooutn(O2_n,ADP_i_n,NADHmito_n)) #0.5*T2Jcorrection*(1000*(+J_DH_n - J_C1_n)/W_x) # #  NADHmito_nM = 1e-3*u[27]
    du[4]  = 6.96*(vMitoinn(PYRmito_n,NADHmito_n) + vShuttlen(NADH_n,NADHmito_n) - vMitooutn(O2_n,ATP_i_n,ADP_i_n,NADHmito_n,NADmito_n)) #0.5*T2Jcorrection*(1000*(+J_DH_n - J_C1_n)/W_x) # #  NADHmito_nM = 1e-3*u[27]
    
    
    du[5]  = 0.5*T2Jcorrection*(1000*(+J_C1_n - J_C3_n)/W_x) # QH2mito_nM  # Reduced ubiquinol in matrix
    du[6]  = 0.5*T2Jcorrection*(1000*(+2*J_C3_n - 2*J_C4_n)/W_i) # CytCredmito_nM

    # du[7]  = JO2fromCap2n(O2cap,O2_n) - 0.6*vMitooutn(O2_n,ADPmito_n,NADHmito_n) #- 0.5*T2Jcorrection*1000*J_C4_n/W_x #- 0.6*vMitooutn(O2_n,ADPmito_n,NADHmito_n) #- 0.4*1000*J_C4_n # 1000*J_C4_n because J_C4_n in M/s #- 0.6*vMitooutn(O2_n,ADP_n,NADHmito_n) # O2_n # O2 stays constant in Theurey2019
    # du[7]  = JO2fromCap2n(O2cap,O2_n) - 0.6*vMitooutn(O2_n,ADP_i_n,NADHmito_n) #- 0.5*T2Jcorrection*1000*J_C4_n/W_x #- 0.6*vMitooutn(O2_n,ADPmito_n,NADHmito_n) #- 0.4*1000*J_C4_n # 1000*J_C4_n because J_C4_n in M/s #- 0.6*vMitooutn(O2_n,ADP_n,NADHmito_n) # O2_n # O2 stays constant in Theurey2019
    du[7]  = JO2fromCap2n(O2cap,O2_n) - 0.6*vMitooutn(O2_n,ATP_i_n,ADP_i_n,NADHmito_n,NADmito_n) #- 0.5*T2Jcorrection*1000*J_C4_n/W_x #- 0.6*vMitooutn(O2_n,ADPmito_n,NADHmito_n) #- 0.4*1000*J_C4_n # 1000*J_C4_n because J_C4_n in M/s #- 0.6*vMitooutn(O2_n,ADP_n,NADHmito_n) # O2_n # O2 stays constant in Theurey2019

    
    du[8]  = 0.5*T2Jcorrection*(1000*(+J_F1_n - J_ANT_n)/W_x) # ATPmito_nM
    du[9]  = 0.5*T2Jcorrection*(1000*(-J_F1_n + J_ANT_n)/W_x) # ADPmito_nM

    du[10] = 0.5*T2Jcorrection*(1000*(J_MgATPx_n)/W_x) # ATP_mx_nM
    du[11] = 0.5*T2Jcorrection*(1000*(J_MgADPx_n)/W_x) # ADP_mx_nM

    du[12] = 0.5*T2Jcorrection*(1000*(-J_F1_n + J_Pi1_n )/W_x)  # Pimito_nM

    du[13] = 0.5*T2Jcorrection*(1000*(+J_ATP_n + J_ANT_n + J_AKi_n )/W_i) # ATP_i_nM
    du[14] = 0.5*T2Jcorrection*(1000*(+J_ADP_n - J_ANT_n - 2*J_AKi_n )/W_i) # ADP_i_nM
    du[15] = 0.5*T2Jcorrection*(1000*(+J_AMP_n + J_AKi_n)/W_i) # AMP_i_nM

    du[16] = 0.5*T2Jcorrection*(1000*(J_MgATPi_n)/W_i) # ATP_mi_nM
    du[17] = 0.5*T2Jcorrection*(1000*(J_MgADPi_n)/W_i) # ADP_mi_nM
    du[18] = 0.5*T2Jcorrection*(1000*(-J_Pi1_n + J_Pi2_n )/W_i) # Pi_i_nM

    du[19] = 0.5*T2Jcorrection*( 4*J_C1_n + 2*J_C3_n + 4*J_C4_n - n_A*J_F1_n - J_ANT_n - J_Hle_n  - J_K_n )/CIM #MOD MitoMembrPotent_n

    du[20] = 0 # Cytochrome-c loss with half time t12cyto (used in Huber 2011)
    du[21] = 0 # Cleavage of complex I/II total (used in Huber 2011)

    du[22] = 0 #0.5*T2Jcorrection*(1000*(- J_DH_n +(4+1)*J_C1_n + (4-2)*J_C3_n + (2+2)*J_C4_n - (n_A-1)*J_F1_n - 2*J_Pi1_n - J_Hle_n + J_KH_n + J_Ht_n)/W_i)  # C_H_ims_nM
        
    du[23] =  (psiCrKinase_n(PCr_n,ATP_n,ADP_n)  + 0.5*(1/6.96)*1000*(-J_ATP_n)  - psiHK_n(Glc_n,ATP_n,G6P_n) - psiPFK_n(ATP_n,F6P_n) + psiPGK_n(BPG13_n,ADP_n,PG3_n,ATP_n) + psiPK_n(PEP_n,ADP_n,ATP_n) - 0.15*vPumpn - vATPasesn )/(1-dAMPdATPn)
        
    du[24] = 0 # see above calc from ATP as in Jlv #0.5*T2Jcorrection*(1000*(-J_ADP_n + J_ATPK_n)/W_c) # - psiCrKinase_n(PCr_n,ATP_n,ADP_n)  + 1000*(-J_ADP_n)/W_c + psiHK_n(Glc_n,ATP_n,G6P_n) + psiPFK_n(ATP_n,F6P_n) - psiPGK_n(BPG13_n,ADP_n,PG3_n,ATP_n) - psiPK_n(PEP_n,ADP_n,Pyr_n,ATP_n) + psiADK_n(ATP_n,AMP_n,ADP_n)  + vATPasesn + vPumpn     # ADP_n
    
    du[25] = T2Jcorrection*(0.5*1000*J_DH_n/W_x - psiFUM_n(FUMmito_n,MALmito_n)) # FUMmito_n
    
    du[26] = T2Jcorrection*(psiFUM_n(FUMmito_n,MALmito_n) - psiMDH_n(MALmito_n,NADmito_n,OXAmito_n,NADHmito_n) ) + 6.96*psiMAKGC_n(MAL_n,AKGmito_n,MALmito_n,AKG_n)  # MALmito_n
    
    du[27] = T2Jcorrection*(psiMDH_n(MALmito_n,NADmito_n,OXAmito_n,NADHmito_n) - psiCS_n(OXAmito_n,CITmito_n,AcCoAmito_n,CoAmito_n) ) + psiAAT_n(ASPmito_n,AKGmito_n,OXAmito_n,GLUmito_n)  # OXAmito_n
    
    du[28] = T2Jcorrection*(0.5*psiSCS_n(Pimito_n,SUCCOAmito_n,ADPmito_n,SUCmito_n,CoAmito_n,ATPmito_n) - 0.5*1000*J_DH_n/W_x ) + 0.5*T2Jcorrection*SCOT_n(SUCCOAmito_n,AcAc_n,AcAcCoA_n,SUCmito_n) # SUCmito_n
    du[29] = 0.5*T2Jcorrection*(psiKGDH_n(ADPmito_n,ATPmito_n,CaMito_n,AKGmito_n,NADHmito_n,NADmito_n,CoAmito_n,SUCCOAmito_n) - psiSCS_n(Pimito_n,SUCCOAmito_n,ADPmito_n,SUCmito_n,CoAmito_n,ATPmito_n) ) - 0.5*T2Jcorrection*SCOT_n(SUCCOAmito_n,AcAc_n,AcAcCoA_n,SUCmito_n) # SUCCOAmito_n
    
    # CoAmito_n
    du[30] = T2Jcorrection*(psiCS_n(OXAmito_n,CITmito_n,AcCoAmito_n,CoAmito_n) - psiPDH_n(PYRmito_n,NADmito_n,CoAmito_n) - 0.5*psiKGDH_n(ADPmito_n,ATPmito_n,CaMito_n,AKGmito_n,NADHmito_n,NADmito_n,CoAmito_n,SUCCOAmito_n) + 0.5*psiSCS_n(Pimito_n,SUCCOAmito_n,ADPmito_n,SUCmito_n,CoAmito_n,ATPmito_n) - 0.5*thiolase_n(CoAmito_n,AcAcCoA_n)) # CoAmito_n
    
    du[31] =  0.5*T2Jcorrection*(psiIDH_n(ISOCITmito_n,NADmito_n,NADHmito_n) - psiKGDH_n(ADPmito_n,ATPmito_n,CaMito_n,AKGmito_n,NADHmito_n,NADmito_n,CoAmito_n,SUCCOAmito_n)) - psiAAT_n(ASPmito_n,AKGmito_n,OXAmito_n,GLUmito_n) - psiMAKGC_n(MAL_n,AKGmito_n,MALmito_n,AKG_n) # AKGmito_n
    
    du[32] = 0 # CaMito_n
    du[33] = 0.5*T2Jcorrection*(psiACO_n(CITmito_n,ISOCITmito_n) - psiIDH_n(ISOCITmito_n,NADmito_n,NADHmito_n)) # ISOCITmito_n
    du[34] = T2Jcorrection*(psiCS_n(OXAmito_n,CITmito_n,AcCoAmito_n,CoAmito_n) - 0.5*psiACO_n(CITmito_n,ISOCITmito_n)) # CITmito_n
    
    #du[35] = T2Jcorrection*(psiPDH_n(PYRmito_n,NADmito_n,CoAmito_n) - psiCS_n(OXAmito_n,CITmito_n,AcCoAmito_n,CoAmito_n) ) + 0.5*T2Jcorrection*thiolase_n(CoAmito_n,AcAcCoA_n) # AcCoAmito_n
    du[35] = T2Jcorrection*(psiPDH_n(PYRmito_n,NADmito_n,CoAmito_n) - psiCS_n(OXAmito_n,CITmito_n,AcCoAmito_n,CoAmito_n) ) + T2Jcorrection*thiolase_n(CoAmito_n,AcAcCoA_n) # AcCoAmito_n
    
    du[36] = 0.5*T2Jcorrection*(bHBDH_n(NADmito_n,NADHmito_n,bHB_n,AcAc_n) - SCOT_n(SUCCOAmito_n,AcAc_n,AcAcCoA_n,SUCmito_n)) # AcAc_n
    du[37] = 0.5*T2Jcorrection*(SCOT_n(SUCCOAmito_n,AcAc_n,AcAcCoA_n,SUCmito_n) - thiolase_n(CoAmito_n,AcAcCoA_n))  # AcAcCoA_n
    
    # PYRmito_n
    du[38] = 6.96*psiPYRtrcyt2mito_n(Pyr_n,PYRmito_n,C_H_mitomatr_n) - T2Jcorrection*psiPDH_n(PYRmito_n,NADmito_n,CoAmito_n) # PYRmito_n
    
    du[39] = 0.44*MCT2_bHB_n(bHB_ecs,bHB_n) - bHBDH_n(NADmito_n,NADHmito_n,bHB_n,AcAc_n)  # bHB_n
    du[40] = 0.0275*MCT1_bHB_b(bHB_b,bHB_ecs) - MCT2_bHB_n(bHB_ecs,bHB_n) - MCT1_bHB_a(bHB_ecs,bHB_a) # bHB_ecs
    du[41] = 0 # bHB_a fixed   
    du[42] = JbHBTrArtCap(t,bHB_b) - MCT1_bHB_b(bHB_b,bHB_ecs) # bHB_b
    
    
    
    du[43] =  0 #0.5*T2Jcorrection*(- psiAAT_n(ASPmito_n,AKGmito_n,OXAmito_n,GLUmito_n) - psiAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n,MitoMembrPotent_n) ) # ASPmito_n
    du[44] =  0 #0.5*T2Jcorrection*(- psiCAAT_n(ASP_n,AKG_n,OXA_n,GLU_n) + 0.14375*psiAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n,MitoMembrPotent_n)) # ASP_n  # 0.14375 is volumes scaling mito/cyto
    
    du[45] =  0.5*T2Jcorrection*(psiAAT_n(ASPmito_n,AKGmito_n,OXAmito_n,GLUmito_n) + 6.96*psiAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n,MitoMembrPotent_n) + psiGLS_n(GLN_n,GLUmito_n)) # GLUmito_n
    
    du[46] =  0 #0.5*T2Jcorrection*(- psicMDH_n(MAL_n,NAD_n,OXA_n,NADH_n) - psiMAKGC_n(MAL_n,AKGmito_n,MALmito_n,AKG_n)) # MAL_n
    du[47] =  0 #0.5*T2Jcorrection*( psicMDH_n(MAL_n,NAD_n,OXA_n,NADH_n) + psiCAAT_n(ASP_n,AKG_n,OXA_n,GLU_n)) # OXA_n
    du[48] =  0 #0.5*T2Jcorrection*(- psiCAAT_n(ASP_n,AKG_n,OXA_n,GLU_n) + 0.14375*psiMAKGC_n(MAL_n,AKGmito_n,MALmito_n,AKG_n)) # AKG_n   # 0.14375 is volumes scaling mito/cyto
    
    # GLU_n
    du[49] = psiCAAT_n(ASP_n,AKG_n,OXA_n,GLU_n) - psiAGC_n(ASPmito_n,GLU_n,ASP_n,GLUmito_n,MitoMembrPotent_n) # -synRelease # set in CB for syn release      # GLU_n
    
    #NADH_n
    du[50] = psiGAPDH_n(NAD_n,GAP_n,Pi_n,BPG13_n,NADH_n) - psiLDH_n(NADH_n,Pyr_n,NAD_n,Lac_n) - vShuttlen(NADH_n,NADHmito_n)  #psiGAPDH_n(NAD_n,GAP_n,Pi_n,BPG13_n,NADH_n) - 0.1*psiLDH_syn_n(NADH_n,Pyr_n,NAD_n,Lac_n) - 0.9*psiLDH_cyto_n(NADH_n,Pyr_n,NAD_n,Lac_n) - vShuttlen(NADH_n,NADHmito_n)  #+ psicMDH_n(MAL_n,NAD_n,OXA_n,NADH_n)  # NADH_n
    
    ##########
    # a
    du[51]  = 0 #0.5*T2Jcorrection*(1000*x_buff*C_H_mitomatr_aM*( +1*J_DH_a - (4+1)*J_C1_a - (4-2)*J_C3_a - (2+2)*J_C4_a + (n_A-1)*J_F1_a + 2*J_Pi1_a + J_Hle_a - J_KH_a )/W_x) # C_H_mitomatr_aM
    du[52]  = 0.5*T2Jcorrection*(1000*(J_KH_a + J_K_a)/W_x) # K_x_aM
    du[53]  = 0.5*T2Jcorrection*(1000*(-J_MgATPx_a - J_MgADPx_a)/W_x) # Mg_x_aM

    # NADHmito_a
#     du[54]  = 6.96*(vMitoing(PYRmito_a,NADHmito_a) + vShuttleg(NADH_a,NADHmito_a) - vMitooutg(O2_a,ADPmito_a,NADHmito_a))  #0.5*T2Jcorrection*(1000*(+J_DH_a - J_C1_a)/W_x) # NADH   NADHmito_aM = 1e-3*u[27]
    # du[54]  = 6.96*(vMitoing(PYRmito_a,NADHmito_a) + vShuttleg(NADH_a,NADHmito_a) - vMitooutg(O2_a,ADP_i_a,NADHmito_a))  #0.5*T2Jcorrection*(1000*(+J_DH_a - J_C1_a)/W_x) # NADH   NADHmito_aM = 1e-3*u[27]
    du[54]  = 6.96*(vMitoing(PYRmito_a,NADHmito_a) + vShuttleg(NADH_a,NADHmito_a) - vMitooutg(O2_a,ATP_i_a,ADP_i_a,NADHmito_a,NADmito_a))  #0.5*T2Jcorrection*(1000*(+J_DH_a - J_C1_a)/W_x) # NADH   NADHmito_aM = 1e-3*u[27]
    
    du[55]  = 0.5*T2Jcorrection*(1000*(+J_C1_a - J_C3_a)/W_x) # QH2mito_aM  # Reduced ubiquinol in matrix

    du[56]  = 0.5*T2Jcorrection*(1000*(+2*J_C3_a - 2*J_C4_a)/W_i) # CytCredmito_aM

    # du[57]  = JO2fromCap2a(O2cap,O2_a) - 0.6*vMitooutg(O2_a,ADPmito_a,NADHmito_a) #JO2fromCap2a(O2cap,O2_a) - 0.5*T2Jcorrection*1000*J_C4_a/W_x  # - 0.6*vMitooutg(O2_a,ADP_a,NADHmito_a) #- 0.3*1000*J_C4_a # 1000*J_C4_a because J_C4_a in M/s  #- 0.6*vMitooutg(O2_a,ADP_a,NADHmito_a) # O2_a # O2 stays constant in Theurey2019
    # du[57]  = JO2fromCap2a(O2cap,O2_a) - 0.6*vMitooutg(O2_a,ADP_i_a,NADHmito_a) #JO2fromCap2a(O2cap,O2_a) - 0.5*T2Jcorrection*1000*J_C4_a/W_x  # - 0.6*vMitooutg(O2_a,ADP_a,NADHmito_a) #- 0.3*1000*J_C4_a # 1000*J_C4_a because J_C4_a in M/s  #- 0.6*vMitooutg(O2_a,ADP_a,NADHmito_a) # O2_a # O2 stays constant in Theurey2019
    du[57]  = JO2fromCap2a(O2cap,O2_a) - 0.6*vMitooutg(O2_a,ATP_i_a,ADP_i_a,NADHmito_a,NADmito_a) #JO2fromCap2a(O2cap,O2_a) - 0.5*T2Jcorrection*1000*J_C4_a/W_x  # - 0.6*vMitooutg(O2_a,ADP_a,NADHmito_a) #- 0.3*1000*J_C4_a # 1000*J_C4_a because J_C4_a in M/s  #- 0.6*vMitooutg(O2_a,ADP_a,NADHmito_a) # O2_a # O2 stays constant in Theurey2019

    
    
    du[58]  = 0.5*T2Jcorrection*(1000*(+J_F1_a - J_ANT_a)/W_x) # ATPmito_aM
    du[59]  = 0.5*T2Jcorrection*(1000*(-J_F1_a + J_ANT_a)/W_x) # ADPmito_aM

    du[60] = 0.5*T2Jcorrection*(1000*(J_MgATPx_a)/W_x) # ATP_mx_aM
    du[61] = 0.5*T2Jcorrection*(1000*(J_MgADPx_a)/W_x) # ADP_mx_aM

    du[62] = 0.5*T2Jcorrection*(1000*(-J_F1_a + J_Pi1_a )/W_x)  # Pimito_aM

    du[63] = 0.5*T2Jcorrection*(1000*(+J_ATP_a + J_ANT_a + J_AKi_a )/W_i) # ATP_i_aM
    du[64] = 0.5*T2Jcorrection*(1000*(+J_ADP_a - J_ANT_a - 2*J_AKi_a )/W_i) # ADP_i_aM
    du[65] = 0.5*T2Jcorrection*(1000*(+J_AMP_a + J_AKi_a)/W_i) # AMP_i_aM

    du[66] = 0.5*T2Jcorrection*(1000*(J_MgATPi_a)/W_i) # ATP_mi_aM
    du[67] = 0.5*T2Jcorrection*(1000*(J_MgADPi_a)/W_i) # ADP_mi_aM
    du[68] = 0.5*T2Jcorrection*(1000*(-J_Pi1_a + J_Pi2_a )/W_i) # Pi_i_aM

    du[69] = 0.5*T2Jcorrection*(( 4*J_C1_a + 2*J_C3_a + 4*J_C4_a - n_A*J_F1_a - J_ANT_a - J_Hle_a  - J_K_a )/CIM) #MOD MitoMembrPotent_a

    du[70] = 0 # Cytochrome-c loss with half time t12cyto (used in Huber 2011)
    du[71] = 0 # Cleavage of complex I/II total (used in Huber 2011)

    du[72] = 0 # 0.5*T2Jcorrection*(1000*(- J_DH_a +(4+1)*J_C1_a + (4-2)*J_C3_a + (2+2)*J_C4_a - (n_A-1)*J_F1_a - 2*J_Pi1_a - J_Hle_a + J_KH_a + J_Ht_a)/W_i)  # C_H_ims_aM
    
    #du[73] = ( psiCrKinase_a(PCr_a,ATP_a,ADP_a) + 0.5*(1/6.96)*1000*(-J_ATP_a) - psiHK_a(Glc_a,ATP_a,G6P_a) - psiPFK_a(ATP_a,F6P_a,f26bp_a) - psiPFK2_a(F6P_a,ATP_a,ADP_a,f26bp_a) + psiPGK_a(BPG13_a,ADP_a,PG3_a,ATP_a) + psiPK_a(PEP_a,ADP_a,ATP_a) -0.15*(7/4)*vPumpg - vATPasesg)/(1-dAMPdATPg)    
    du[73] = (-(Ca_a/cai0_ca_ion)*(1 + xNEmod*(u[178]/(KdNEmod + u[178])))*psiAC_a(ATP_a,cAMP_a) + psiCrKinase_a(PCr_a,ATP_a,ADP_a) + 0.5*(1/6.96)*1000*(-J_ATP_a) - psiHK_a(Glc_a,ATP_a,G6P_a) - psiPFK_a(ATP_a,F6P_a,f26bp_a) - psiPFK2_a(F6P_a,ATP_a,ADP_a,f26bp_a) + psiPGK_a(BPG13_a,ADP_a,PG3_a,ATP_a) + psiPK_a(PEP_a,ADP_a,ATP_a) -0.15*(7/4)*vPumpg - vATPasesg)/(1-dAMPdATPg)    
    
    du[74] = 0 # see above calc from ATP as in Jlv # 0.5*T2Jcorrection*(1000*(-J_ADP_a + J_ATPK_a)/W_c) # -psiCrKinase_a(PCr_a,ATP_a,ADP_a) +  1000*(-J_ADP_a)/W_c + psiHK_a(Glc_a,ATP_a,G6P_a) + psiPFK_a(ATP_a,F6P_a,f26bp_a) + psiPFK2_a(F6P_a,ATP_a,ADP_a,f26bp_a) - psiPGK_a(BPG13_a,ADP_a,PG3_a,ATP_a) -psiPK_a(PEP_a,ADP_a,ATP_a)  + psiADK_a(ATP_a,AMP_a,ADP_a) + vATPasesg + vPumpg   # ADP_a

    du[75] = 0.5*T2Jcorrection*(1000*J_DH_a/W_x - psiFUM_a(FUMmito_a,MALmito_a)) # FUMmito_a
    du[76] = 0.5*T2Jcorrection*(psiFUM_a(FUMmito_a,MALmito_a) - psiMDH_a(MALmito_a,NADmito_a,OXAmito_a,NADHmito_a))  # MALmito_a
    du[77] = 0.5*T2Jcorrection*(psiMDH_a(MALmito_a,NADmito_a,OXAmito_a,NADHmito_a) - psiCS_a(OXAmito_a,CITmito_a,AcCoAmito_a,CoAmito_a) + psiPYRCARB_a(ATPmito_a,ADPmito_a,PYRmito_a,OXAmito_a) ) # OXAmito_a
    
    du[78] = 0.5*T2Jcorrection*(psiSCS_a(Pimito_a,SUCCOAmito_a,ADPmito_a,SUCmito_a,CoAmito_a,ATPmito_a) - 1000*J_DH_a/W_x) # SUCmito_a

    du[79] = 0.5*T2Jcorrection*(psiKGDH_a(ADPmito_a,ATPmito_a,CaMito_a,AKGmito_a,NADHmito_a,NADmito_a,CoAmito_a,SUCCOAmito_a) - psiSCS_a(Pimito_a,SUCCOAmito_a,ADPmito_a,SUCmito_a,CoAmito_a,ATPmito_a)) # SUCCOAmito_a
    
    # CoAmito_a
    du[80] = 0.5*T2Jcorrection*(psiCS_a(OXAmito_a,CITmito_a,AcCoAmito_a,CoAmito_a) - psiPDH_a(PYRmito_a,NADmito_a,CoAmito_a) - psiKGDH_a(ADPmito_a,ATPmito_a,CaMito_a,AKGmito_a,NADHmito_a,NADmito_a,CoAmito_a,SUCCOAmito_a) + psiSCS_a(Pimito_a,SUCCOAmito_a,ADPmito_a,SUCmito_a,CoAmito_a,ATPmito_a))  # CoAmito_a
    
    du[81] = 0.5*T2Jcorrection*(psiIDH_a(ISOCITmito_a,NADmito_a,NADHmito_a) - psiKGDH_a(ADPmito_a,ATPmito_a,CaMito_a,AKGmito_a,NADHmito_a,NADmito_a,CoAmito_a,SUCCOAmito_a) + psiGDH_simplif_a(NADmito_a,GLUT_a,NADHmito_a,AKGmito_a)) # AKGmito_a
    du[82] = 0 # CaMito_a
    du[83] = 0.5*T2Jcorrection*(psiACO_a(CITmito_a,ISOCITmito_a) - psiIDH_a(ISOCITmito_a,NADmito_a,NADHmito_a)) # ISOCITmito_a
    du[84] = 0.5*T2Jcorrection*(psiCS_a(OXAmito_a,CITmito_a,AcCoAmito_a,CoAmito_a) - psiACO_a(CITmito_a,ISOCITmito_a)) # CITmito_a
    du[85] = 0.5*T2Jcorrection*(psiPDH_a(PYRmito_a,NADmito_a,CoAmito_a) - psiCS_a(OXAmito_a,CITmito_a,AcCoAmito_a,CoAmito_a)) # AcCoAmito_a
    
    du[86] = 0 # AcAc_a
    du[87] = 0 # AcAcCoA_a
    
    # PYRmito_a
    du[88] = 6.96*psiPYRtrcyt2mito_a(Pyr_a,PYRmito_a,C_H_mitomatr_a) - T2Jcorrection*(psiPDH_a(PYRmito_a,NADmito_a,CoAmito_a) + psiPYRCARB_a(ATPmito_a,ADPmito_a,PYRmito_a,OXAmito_a)) # PYRmito_a
    
    du[89] = T2Jcorrection*(psiSNAT_GLN_n(GLN_out,GLN_n) - psiGLS_n(GLN_n,GLUmito_n)) # GLN_n
    du[90] = T2Jcorrection*(- psiSNAT_GLN_n(GLN_out,GLN_n) + psiSNAT_GLN_a(GLN_a,GLN_out)) # GLN_out
    du[91] = T2Jcorrection*(- psiSNAT_GLN_a(GLN_a,GLN_out) + psiGLNsynth_a(GLUT_a,ATP_a,ADP_a)) # GLN_a
    
    du[92] = T2Jcorrection*(0.0266*psiEAAT12(Va,Na_a,GLUT_syn,GLUT_a,K_a,K_out) - psiGLNsynth_a(GLUT_a,ATP_a,ADP_a) - psiGDH_simplif_a(NADmito_a,GLUT_a,NADHmito_a,AKGmito_a)) # GLUT_a  # Vol_syn/Vol_astEAAT
    
    ##########
    # ephys
    ##########
    
    du[93] = (1/4e-4)*( -dIPump_a -IBK - IKirAS  - IKirAV - IleakA  - ITRP_a )   # Va = -90
    du[94] = vLeakNag-3*vPumpg +vgstim # Na_a
    du[95] = JNaK_a + ( - IleakA - IBK - IKirAS - IKirAV)/(4e-4*843.0*1000.) - RateDecayK_a*(K_a - K_a0) # W2013  # K_a
    
    du[96] = SmVn/F*(IK + IM)*(eto_n/eto_ecs) - 2*vPumpn*(eto_n/eto_ecs) - 2*(eto_a/eto_ecs)*vPumpg - JdiffK  -( ( - IleakA - IBK - IKirAS - IKirAV)/(4e-4*843.0*1000.))     #  K_out
    du[97] = 0 # - psiEAAT12(Va,Na_a,GLUT_syn,GLUT_a,K_a,K_out) + synGlutRelease(V)   # GLUT_syn == Glutamate_syn set in CB # see u[12] in cyto
    
    du[98] = 1/Cm*(-IL-INa-IK-ICa-ImAHP-dIPump+Isyne+Isyni - IM + Iinj)  # VNeu
    du[99] =  vLeakNan-3*vPumpn +vnstim # Na_n
    
    du[100] = phi*(hinf-h)/tauh # h               
    du[101] = phi*(ninf-n)/taun # n               
    du[102] = -SmVn/F*ICa-(Ca_n-Ca_n0)/tauCa  # Ca_n 2+ 
    du[103] = phi*(p_inf - pgate)/tau_p # pgate    
    du[104] = psiBK*cosh(  (Va - (-0.5*v5BK*tanh((Ca_a - Ca3BK)/Ca4BK )  + v6BK) )  / ( 2*v4BK  ) ) * ( nBKinf - nBK_a)  # nBK_a  # approx use https://www.cell.com/biophysj/pdfExtended/S0006-3495(13)01032-1 Witthoft 2013 with Ca_n istead of Ca_a 
    du[105] = 0 # set in CB # mGluRboundRatio_a
    
    du[106] =  rhIP3a*((mGluRboundRatio_a + deltaGlutSyn) / (KGlutSyn + mGluRboundRatio_a + deltaGlutSyn)) - kdegIP3a*IP3_a   # IP3_a   
    du[107] =  konhIP3Ca_a*( khIP3Ca_aINH - ( Ca_a + khIP3Ca_aINH)*hIP3Ca_a )       # hIP3Ca_a
    du[108] =  beta_Ca_a*( IIP3_a - ICa_pump_a + Ileak_CaER_a ) - 0.5*ITRP_a/(4e-4*843.0*1000.) #   #S12  # Ca_a
    du[109] = 0 # Ca_r_a Ca ER a
    du[110] =  (Ca_perivasc/tauTRPCa_perivasc)*( sinfTRPV  - sTRP_a )   # sTRP_a
    
    
    ###############
    # metab cyto
    ###############
    
    du[111] = FinDyn_W2017(t) - Fout_W2017(vV,t)   #vV
    du[112] = VprodEET_a*(Ca_a - CaMinEET_a) - kdeg_EET_a*EET_a # EET_a
    du[113] = JdHbin(O2cap,t) - JdHbout(vV,t,ddHb) # ddHb
    du[114] = JO2art2cap(O2cap,t) - (eto_n/eto_b)*JO2fromCap2n(O2cap,O2_n) - (eto_a/eto_b)*JO2fromCap2a(O2cap,O2_a) # O2cap
        
    du[115] = trGLC_art_cap(t,Glc_b) - JGlc_b(Glc_b,glc_D_ecsEndothelium) # volArt == volCap == 0.0055 # Glc_b
    du[116] = 0.32*JGlc_b(Glc_b,glc_D_ecsEndothelium) - JGlc_e2ecsBA(glc_D_ecsEndothelium,Glc_ecsBA) # glc_D_ecsEndothelium == Glc_t_t
    du[117] = 1.13*JGlc_e2ecsBA(glc_D_ecsEndothelium,Glc_ecsBA) - JGlc_ecsBA2a(Glc_ecsBA,Glc_a) - JGlc_diffEcs(Glc_ecsBA,Glc_ecsAN) # Glc_ecsBA
    du[118] = 0.06*JGlc_ecsBA2a(Glc_ecsBA,Glc_a) - psiHK_a(Glc_a,ATP_a,G6P_a) - JGlc_a2ecsAN(Glc_a,Glc_ecsAN) # Glc_a
    
    du[119] = 1.35*JGlc_a2ecsAN(Glc_a,Glc_ecsAN) - JGlc_ecsAN2n(Glc_ecsAN,Glc_n) + 0.08*JGlc_diffEcs(Glc_ecsBA,Glc_ecsAN) # Glc_ecsAN
    
    du[120] = 0.41*JGlc_ecsAN2n(Glc_ecsAN,Glc_n) - psiHK_n(Glc_n,ATP_n,G6P_n) # Glc_n
    
    du[121] = psiHK_n(Glc_n,ATP_n,G6P_n) - psiPGI_n(G6P_n,F6P_n) - psiG6PDH_n(NADP_n,G6P_n,NADPH_n,GL6P_n) # G6P_n
    du[122] = psiHK_a(Glc_a,ATP_a,G6P_a) - psiPGI_a(G6P_a,F6P_a) - psiG6PDH_a(NADP_a,G6P_a,NADPH_a,GL6P_a) + psiPGLM_a(G1P_a,G6P_a) # G6P_a 
    
    du[123] = psiPGI_n(G6P_n,F6P_n) - psiPFK_n(ATP_n,F6P_n) - psiTKL2_n(X5P_n,E4P_n,F6P_n,GAP_n) + psiTALppp_n(S7P_n,GAP_n,E4P_n,F6P_n) # F6P_n
    du[124] = psiPGI_a(G6P_a,F6P_a) - psiPFK_a(ATP_a,F6P_a,f26bp_a) - psiPFK2_a(F6P_a,ATP_a,ADP_a,f26bp_a)  - psiTKL2_a(X5P_a,E4P_a,F6P_a,GAP_a) + psiTALppp_a(S7P_a,GAP_a,E4P_a,F6P_a) # F6P_a
    
    du[125] = psiPFK_n(ATP_n,F6P_n) - psiALD_n(FBP_n,GAP_n,DHAP_n) # FBP_n
    du[126] = psiPFK_a(ATP_a,F6P_a,f26bp_a) - psiALD_a(FBP_a,GAP_a,DHAP_a) # FBP_a
    
    du[127] = psiPFK2_a(F6P_a,ATP_a,ADP_a,f26bp_a) # f26bp_a
    
    du[128] = psiGS_a(GS_a,UDPgluco_a) - 0.1*psiGPa_a(GPa_a,GLY_a) # GLY_a
    
    du[129] = 0 # accounted for it in /(1-dAMPdATPn) and ADp calc as in Jlv # - psiADK_n(ATP_n,AMP_n,ADP_n) # AMP_n
    du[130] = 0 # accounted for it in /(1-dAMPdATPn) and ADp calc as in Jlv # PDE_a(cAMP_a) - psiADK_a(ATP_a,AMP_a,ADP_a) # AMP_a
    
    du[131] = psiGPa_a(GPa_a,GLY_a) - psiPGLM_a(G1P_a,G6P_a) - psiUDPGP_a(UTP_a,G1P_a,PPi_a0,UDPgluco_a) # G1P_a
    
    du[132] = psiALD_n(FBP_n,GAP_n,DHAP_n) - psiGAPDH_n(NAD_n,GAP_n,Pi_n,BPG13_n,NADH_n) + psiTPI_n(DHAP_n,GAP_n) - psiTKL2_n(X5P_n,E4P_n,F6P_n,GAP_n) - psiTALppp_n(S7P_n,GAP_n,E4P_n,F6P_n) + psiTKL1_n(R5P_n,X5P_n,S7P_n,GAP_n)   # GAP_n
    du[133] = psiALD_a(FBP_a,GAP_a,DHAP_a) - psiGAPDH_a(NAD_a,GAP_a,Pi_a,BPG13_a,NADH_a) + psiTPI_a(DHAP_a,GAP_a) - psiTKL2_a(X5P_a,E4P_a,F6P_a,GAP_a) - psiTALppp_a(S7P_a,GAP_a,E4P_a,F6P_a) + psiTKL1_a(R5P_a,X5P_a,S7P_a,GAP_a)  #GAP_a
    
    #
    du[134] = psiALD_n(FBP_n,GAP_n,DHAP_n) - psiTPI_n(DHAP_n,GAP_n) # DHAP_n
    du[135] = psiALD_a(FBP_a,GAP_a,DHAP_a) - psiTPI_a(DHAP_a,GAP_a) # DHAP_a
    
    du[136] = psiGAPDH_n(NAD_n,GAP_n,Pi_n,BPG13_n,NADH_n) - psiPGK_n(BPG13_n,ADP_n,PG3_n,ATP_n) # BPG13_n
    du[137] = psiGAPDH_a(NAD_a,GAP_a,Pi_a,BPG13_a,NADH_a) - psiPGK_a(BPG13_a,ADP_a,PG3_a,ATP_a) # BPG13_a
    
    # NADH_a
    du[138] = psiGAPDH_a(NAD_a,GAP_a,Pi_a,BPG13_a,NADH_a) - psiLDH_a(NADH_a,Pyr_a,NAD_a,Lac_a) - vShuttleg(NADH_a,NADHmito_a) # NADH_a
    
    du[139] = 0 # Pi_n
    du[140] = 0 # Pi_a
    
    du[141] = psiPGK_n(BPG13_n,ADP_n,PG3_n,ATP_n) - psiPGM_n(PG3_n,PG2_n) # PG3_n
    du[142] = psiPGK_a(BPG13_a,ADP_a,PG3_a,ATP_a) - psiPGM_a(PG3_a,PG2_a) # PG3_a
    
    #
    du[143] = psiPGM_n(PG3_n,PG2_n) - psiENOL_n(PG2_n,PEP_n) # PG2_n
    du[144] = psiPGM_a(PG3_a,PG2_a) - psiENOL_a(PG2_a,PEP_a) # PG2_a
    
    du[145] = psiENOL_n(PG2_n,PEP_n) - psiPK_n(PEP_n,ADP_n,ATP_n)  # PEP_n
    du[146] = psiENOL_a(PG2_a,PEP_a) - psiPK_a(PEP_a,ADP_a,ATP_a)  # PEP_a
    
    du[147] = psiPK_n(PEP_n,ADP_n,ATP_n) - psiPYRtrcyt2mito_n(Pyr_n,PYRmito_n,C_H_mitomatr_n) - psiLDH_n(NADH_n,Pyr_n,NAD_n,Lac_n)  #Pyr_n   
    du[148] = psiPK_a(PEP_a,ADP_a,ATP_a) - psiPYRtrcyt2mito_a(Pyr_a,PYRmito_a,C_H_mitomatr_a) - psiLDH_a(NADH_a,Pyr_a,NAD_a,Lac_a)  #Pyr_a    
    
    du[149] = JLacTr(t,Lac_b) - MCT1_LAC_b(Lac_b,Lac_ecs) - vLACgc(Lac_b,Lac_a) # Lac_b
    du[150] = 0.0275*MCT1_LAC_b(Lac_b,Lac_ecs) - jLac_a(Lac_ecs,Lac_a) - jLac_n(Lac_ecs,Lac_n) + jLacDiff_e(Lac_ecs)      # Lac_ecs # lac diff www.pnas.org􏱵cgi􏱵doi􏱵10.1073􏱵pnas.0605864104    #  0.275 because common ecs (no sep into BA and AN, no endoth for lac model) 
    du[151] = 0.8*jLac_a(Lac_ecs,Lac_a) + 0.022*vLACgc(Lac_b,Lac_a) +  psiLDH_a(NADH_a,Pyr_a,NAD_a,Lac_a)     # Lac_a                         #0.8 = 0.2/0.25 volumes scaling common ecs n endoth
    du[152] = 0.44*jLac_n(Lac_ecs,Lac_n) + psiLDH_n(NADH_n,Pyr_n,NAD_n,Lac_n) # Lac_n                         #0.44 = 0.2/0.45 volumes scaling common ecs n endoth   # assume syn vol 10% -> 0.1*0.45 = 0.045 -> from/to 0.045/0.45 = 0.1
    
    
    du[153] =  psiG6PDH_n(NADP_n,G6P_n,NADPH_n,GL6P_n) + psi6PGDH_n(NADP_n,GO6P_n,RU5P_n,NADPH_n) - psiNADPHox_n(NADPH_n)  - psiGSSGR_n(GSSG_n,NADPH_n) # NADPH_n
    du[154] =  psiG6PDH_a(NADP_a,G6P_a,NADPH_a,GL6P_a) + psi6PGDH_a(NADP_a,GO6P_a,RU5P_a,NADPH_a) - psiNADPHox_a(NADPH_a)  - psiGSSGR_a(GSSG_a,NADPH_a) # NADPH_a
    
    du[155] =  psiG6PDH_n(NADP_n,G6P_n,NADPH_n,GL6P_n) - psi6PGL_n(GL6P_n,GO6P_n) # GL6P_n
    du[156] =  psiG6PDH_a(NADP_a,G6P_a,NADPH_a,GL6P_a) - psi6PGL_a(GL6P_a,GO6P_a) # GL6P_a
    
    du[157] =  psi6PGL_n(GL6P_n,GO6P_n) - psi6PGDH_n(NADP_n,GO6P_n,RU5P_n,NADPH_n) # GO6P_n
    du[158] =  psi6PGL_a(GL6P_a,GO6P_a) - psi6PGDH_a(NADP_a,GO6P_a,RU5P_a,NADPH_a) # GO6P_a
    
    du[159] = psi6PGDH_n(NADP_n,GO6P_n,RU5P_n,NADPH_n) - psiRPEppp_n(RU5P_n,X5P_n) - psiRPI_n(RU5P_n,R5P_n) # RU5P_n
    du[160] = psi6PGDH_a(NADP_a,GO6P_a,RU5P_a,NADPH_a) - psiRPEppp_a(RU5P_a,X5P_a) - psiRPI_a(RU5P_a,R5P_a)  # RU5P_a
    
    du[161] = psiRPI_n(RU5P_n,R5P_n) - psiTKL1_n(R5P_n,X5P_n,S7P_n,GAP_n) # R5P_n
    du[162] = psiRPI_a(RU5P_a,R5P_a) - psiTKL1_a(R5P_a,X5P_a,S7P_a,GAP_a) # R5P_a
    
    du[163] = psiRPEppp_n(RU5P_n,X5P_n)  - psiTKL1_n(R5P_n,X5P_n,S7P_n,GAP_n)  + psiTKL2_n(X5P_n,E4P_n,F6P_n,GAP_n) # X5P_n
    du[164] = psiRPEppp_a(RU5P_a,X5P_a)  - psiTKL1_a(R5P_a,X5P_a,S7P_a,GAP_a)  + psiTKL2_a(X5P_a,E4P_a,F6P_a,GAP_a) # X5P_a
    
    du[165] = psiTKL1_n(R5P_n,X5P_n,S7P_n,GAP_n) - psiTALppp_n(S7P_n,GAP_n,E4P_n,F6P_n) # S7P_n
    du[166] = psiTKL1_a(R5P_a,X5P_a,S7P_a,GAP_a) - psiTALppp_a(S7P_a,GAP_a,E4P_a,F6P_a) # S7P_a
    
    du[167] = psiTKL2_n(X5P_n,E4P_n,F6P_n,GAP_n) + psiTALppp_n(S7P_n,GAP_n,E4P_n,F6P_n) # E4P_n
    du[168] = psiTKL2_a(X5P_a,E4P_a,F6P_a,GAP_a) + psiTALppp_a(S7P_a,GAP_a,E4P_a,F6P_a) # E4P_a
            
    du[169] = 2*(psiGSSGR_n(GSSG_n,NADPH_n) - psiGPX_n(GSH_n)) #+ glutathioneSyntase_n(GSH_n) # GSH_n
    du[170] = 2*( psiGSSGR_a(GSSG_a,NADPH_a) - psiGPX_a(GSH_a)) #+ glutathioneSyntase_a(GSH_a) # GSH_a
    du[171] = - psiGSSGR_n(GSSG_n,NADPH_n) + psiGPX_n(GSH_n) # GSSG_n
    du[172] = - psiGSSGR_a(GSSG_a,NADPH_a) + psiGPX_a(GSH_a) # GSSG_a
    
    du[173] = 0 # calc above Cr_n = Crtot - PCr_n psiCrKinase_n(PCr_n,ATP_n,ADP_n)  # Cr_n
    du[174] = - psiCrKinase_n(PCr_n,ATP_n,ADP_n) # PCr_n
    
    du[175] = 0 # calc above Cr_a = Crtot - PCr_a # psiCrKinase_a(PCr_a,ATP_a,ADP_a) # Cr_a
    du[176] = - psiCrKinase_a(PCr_a,ATP_a,ADP_a) # PCr_a
    
    du[177] = (Ca_a/cai0_ca_ion)*(1 + xNEmod*(u[178]/(KdNEmod + u[178])))*psiAC_a(ATP_a,cAMP_a) - PDE_a(cAMP_a) # cAMP_a  
    
    du[178] = 0 # NE_neuromod defined in CB
    
    du[179] = 0 # psiUDPGP_a(UTP_a,G1P_a,PPi_a0,UDPgluco_a) - psiGS_a(GS_a,UDPgluco_a) - 1e-6*u[179]*(u[179] + 0.109)   #  UDPgluco_a
    du[180] = 0 #0.000117-psiUDPGP_a(UTP_a,G1P_a,PPi_a0,UDPgluco_a) # UTP_a
    
    
    du[181] = 0 #psiGSAJay(GS_a,UDPgluco_a,PKAa_a0,PHKa_a0)  # GS_a
    du[182] = psiPHK(PHKa_a0,GPa_a,GLY_a,G1P_a,UDPgluco_a,Ca_a) # GPa_a
    du[183] = -psiPHK(PHKa_a0,GPa_a,GLY_a,G1P_a,UDPgluco_a,Ca_a) # GPb_a
    
    
end