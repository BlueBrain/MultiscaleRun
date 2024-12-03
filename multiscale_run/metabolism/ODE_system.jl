function metabolism!(du,u,p,t)
    
    """
        This function defines the metabolism model in a system of ordinary differential equations. It is called from main.py in multiscale_run software. It can also be used independently of main.py from pure Julia programming language code script/notebook. For the details on this function/metabolism model itself please refer to [aging metabolism paper link and corresponding github repo link, paper is currently in preparation, this will be updated once the paper will be accepted for publication].

    For the details on Julia Differential Equations please see https://docs.sciml.ai/DiffEqDocs/stable/ (this webpage also provides the tutorial to help you run simulation with this function from pure Julia without multiscale_run if need be).



    Args: 

                      Formally, the args are du,u,p,t. In fact, du is a placeholder to keep the solution of the system. It should be the same size as u, which is an initial values list. Next, p is a list of parameters and t is a time span. Please see the details on arguments in https://docs.sciml.ai/DiffEqDocs/stable/  In diffeqpy interface used in multiscale run we just define u,t,p as input args.





    Some files with annotations to help you understanding the u (variables) of the function:

    variable idx to variable name:  https://bbpgitlab.epfl.ch/molsys/metabolismndam/-/blob/main/sim/metabolism_unit_model[…]igg/COMBO/parameters_GLYCOGEN_cleaned4bigg/variables_idxs.jl
    descriptive names of variables (it’s quite old file already and I’ll update it as some time in the future; also please ignore the  initial value in that file):   https://bbpgitlab.epfl.ch/molsys/metabolismndam/-/blob/main/sim/metabolism_unit_model[…]values_descriptive_cleaned_u0_db_refined_selected_oct2021.jl 


    Parameters list (p) depends on the version of multiscale_run you are using, because it lists the parameters that are neuron specific inputs from other simulators (Neurodamus, STEPS, BloodFlow) and the dictionary of layer-specific properties (mito_scale - scaling factor of mitochondrial density, if given in particular version of multiscale) and EXC - INH specific properties scaling factor (glutamatergic_gaba_scaling).

    In current main.py, the parameters are:

    ina_density,ik_density,mito_scale,glutamatergic_gaba_scaling,outs_r_to_met  = @. p

    where

    ina_density,ik_density - ion currents from Neurodamus

    glutamatergic_gaba_scaling -  EXC - INH specific properties scaling factor

    outs_r_to_met - number of glutamate or GABA release events (depending on whether this particular neuron is EXC or INH).



    For the other versions:

    When including coupling to BF, these parameters need to be added: notBigg_FinDyn_W2017, notBigg_Fout_W2017, u_notBigg_vV_b_b (current version of BF has Fin and Fout set to the same value)



    When including coupling to STEPS: Na_out needs to be added to the parameters list.



    Returns:

                      Formally, there is no "return" of this function, but the function itself is being used for metabolism dynamics simulation from main.py as shown below:
                      prob_metabo = de.ODEProblem(metabolism, vm, tspan_m, param)
    In a way the function used with de.ODEProblem it defines an ODE system problem to be solved with
    sol = de.solve( prob_metabo, de.Rosenbrock23(autodiff=False), reltol=1e-8, abstol=1e-8, saveat=1, maxiters=1e6, )

    """
    
    ina_density,ik_density,mito_scale, notBigg_FinDyn_W2017, notBigg_Fout_W2017, u_notBigg_vV_b_b  = @. p
    
    #NOTE: u_notBigg_vV_b_b is now a parameter which is given from BF, but in the main metabolism unit model we have it as a variable, because the BF is simulated within metab model and not given from outside as in multiscale run
    
    
    
    #bkp: ina_density,ik_density,mito_scale,glutamatergic_gaba_scaling,outs_r_to_met, global_par_t_0,global_par_t_fin,  notBigg_FinDyn_W2017, notBigg_Fout_W2017  = @. p
    
    # notBigg_FinDyn_W2017 notBigg_Fout_W2017
    notBigg_Fout_W2017 = notBigg_FinDyn_W2017
    
    # Before coupling with SB-BF: notBigg_FinDyn_W2017 in W2017 = (global_par_F_0*(1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3)))))))
    
    #notBigg_FinDyn_W2017(t) = 1*(global_par_F_0*(1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3)))))))
    #notBigg_Fout_W2017(t,u_notBigg_vV_b_b) = 1*(global_par_F_0*((u_notBigg_vV_b_b/u0_ss[111])^2+(u_notBigg_vV_b_b/u0_ss[111])^(-0.5)*global_par_tau_v/u0_ss[111]*(global_par_F_0*(1+global_par_delta_F*(1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1-3))))-1/(1+exp((-4.59186)*(t-(global_par_t_0+global_par_t_1+global_par_t_fin+3))))))))/(1+global_par_F_0*(u_notBigg_vV_b_b/u0_ss[111])^(-0.5)*global_par_tau_v/u0_ss[111]))

    
    NADtot_n = NADtot
    NADtot_a = NADtot

    NAD_aging_coeff_n = 1  
    NAD_aging_coeff_a = 1  

    NAD_aging_coeff_mn = 1  
    NAD_aging_coeff_ma = 1 

    NADHshuttle_aging_n = 1
    NADHshuttle_aging_a = 1

    syn_aging_coeff = 1 

u_h_m_n = u[1]
u_k_m_n = u[2]
u_mg2_m_n = u[3]
u_nadh_m_n = u[4]
u_q10h2_m_n = u[5]
u_focytC_m_n = u[6]
u_o2_c_n = u[7]
u_atp_m_n = u[8]
u_adp_m_n = u[9]
u_notBigg_ATP_mx_m_n = u[10]
u_notBigg_ADP_mx_m_n = u[11]
u_pi_m_n = u[12]
u_atp_i_n = u[13]
u_adp_i_n = u[14]
u_amp_i_n = u[15]
u_notBigg_ATP_mi_i_n = u[16]
u_notBigg_ADP_mi_i_n = u[17]
u_pi_i_n = u[18]
u_notBigg_MitoMembrPotent_m_n = u[19]
u_notBigg_Ctot_m_n = u[20]
u_notBigg_Qtot_m_n = u[21]
u_h_i_n = u[22]
u_atp_c_n = u[23]
u_adp_c_n = u[23]/2*(-qAK+sqrt(qAK*qAK+4*qAK*(ATDPtot_n/u[23]-1)))
u_fum_m_n = u[25]
u_mal_L_m_n = u[26]
u_oaa_m_n = u[27]
u_succ_m_n = u[28]
u_succoa_m_n = u[29]
u_coa_m_n = u[30]
u_akg_m_n = u[31]
u_ca2_m_n = u[32]
u_icit_m_n = u[33]
u_cit_m_n = u[34]
u_accoa_m_n = u[35]
u_acac_c_n = u[36]
u_aacoa_m_n = u[37]
u_pyr_m_n = u[38]
u_bhb_c_n = u[39]
u_bhb_e_e = u[40]
u_bhb_c_a = u[41]
u_bhb_b_b = u[42]
u_asp_L_m_n = u[43]
u_asp_L_c_n = u[44]
u_glu_L_m_n = u[45]
u_mal_L_c_n = u[46]
u_oaa_c_n = u[47]
u_akg_c_n = u[48]
u_glu_L_c_n = u[49]
u_nadh_c_n = u[50]
u_h_m_a = u[51]
u_k_m_a = u[52]
u_mg2_m_a = u[53]
u_nadh_m_a = u[54]
u_q10h2_m_a = u[55]
u_focytC_m_a = u[56]
u_o2_c_a = u[57]
u_atp_m_a = u[58]
u_adp_m_a = u[59]
u_notBigg_ATP_mx_m_a = u[60]
u_notBigg_ADP_mx_m_a = u[61]
u_pi_m_a = u[62]
u_atp_i_a = u[63]
u_adp_i_a = u[64]
u_amp_i_a = u[65]
u_notBigg_ATP_mi_i_a = u[66]
u_notBigg_ADP_mi_i_a = u[67]
u_pi_i_a = u[68]
u_notBigg_MitoMembrPotent_m_a = u[69]
u_notBigg_Ctot_m_a = u[70]
u_notBigg_Qtot_m_a = u[71]
u_h_i_a = u[72]
u_atp_c_a = u[73]
u_adp_c_a = u[73]/2*(-qAK+sqrt(qAK*qAK+4*qAK*(ATDPtot_a/u[73]-1)))
u_fum_m_a = u[75]
u_mal_L_m_a = u[76]
u_oaa_m_a = u[77]
u_succ_m_a = u[78]
u_succoa_m_a = u[79]
u_coa_m_a = u[80]
u_akg_m_a = u[81]
u_ca2_m_a = u[82]
u_icit_m_a = u[83]
u_cit_m_a = u[84]
u_accoa_m_a = u[85]
u_acac_c_a = u[86]
u_aacoa_m_a = u[87]
u_pyr_m_a = u[88]
u_gln_L_c_n = u[89]
u_gln_L_e_e = u[90]
u_gln_L_c_a = u[91]
u_glu_L_c_a = u[92]
u_notBigg_Va_c_a = u[93]
u_na1_c_a = u[94]
u_k_c_a = u[95]
u_k_e_e = u[96]
u_glu_L_syn_syn = u[97]
u_notBigg_VNeu_c_n = u[98]
u_na1_c_n = u[99]
u_notBigg_hgate_c_n = u[100]
u_notBigg_ngate_c_n = u[101]
u_ca2_c_n = u[102]
u_notBigg_pgate_c_n = u[103]
u_notBigg_nBK_c_a = u[104]
u_notBigg_mGluRboundRatio_c_a = u[105]
u_notBigg_IP3_c_a = u[106]
u_notBigg_hIP3Ca_c_a = u[107]
u_ca2_c_a = u[108]
u_ca2_r_a = u[109]
u_notBigg_sTRP_c_a = u[110]
# u_notBigg_vV_b_b = u[111]
u_notBigg_EET_c_a = u[112]
u_notBigg_ddHb_b_b = u[113]
u_o2_b_b = u[114]
u_glc_D_b_b = u[115]
u_glc_D_ecsEndothelium_ecsEndothelium = u[116]
u_glc_D_ecsBA_ecsBA = u[117]
u_glc_D_c_a = u[118]
u_glc_D_ecsAN_ecsAN = u[119]
u_glc_D_c_n = u[120]
u_g6p_c_n = u[121]
u_g6p_c_a = u[122]
u_f6p_c_n = u[123]
u_f6p_c_a = u[124]
u_fdp_c_n = u[125]
u_fdp_c_a = u[126]
u_f26bp_c_a = u[127]
u_glycogen_c_a = u[128]
u_amp_c_n = u[129]
u_amp_c_a = u[130]
u_g1p_c_a = u[131]
u_g3p_c_n = u[132]
u_g3p_c_a = u[133]
u_dhap_c_n = u[134]
u_dhap_c_a = u[135]
u_13dpg_c_n = u[136]
u_13dpg_c_a = u[137]
u_nadh_c_a = u[138]
u_pi_c_n = u[139]
u_pi_c_a = u[140]
u_3pg_c_n = u[141]
u_3pg_c_a = u[142]
u_2pg_c_n = u[143]
u_2pg_c_a = u[144]
u_pep_c_n = u[145]
u_pep_c_a = u[146]
u_pyr_c_n = u[147]
u_pyr_c_a = u[148]
u_lac_L_b_b = u[149]
u_lac_L_e_e = u[150]
u_lac_L_c_a = u[151]
u_lac_L_c_n = u[152]
u_nadph_c_n = u[153]
u_nadph_c_a = u[154]
u_6pgl_c_n = u[155]
u_6pgl_c_a = u[156]
u_6pgc_c_n = u[157]
u_6pgc_c_a = u[158]
u_ru5p_D_c_n = u[159]
u_ru5p_D_c_a = u[160]
u_r5p_c_n = u[161]
u_r5p_c_a = u[162]
u_xu5p_D_c_n = u[163]
u_xu5p_D_c_a = u[164]
u_s7p_c_n = u[165]
u_s7p_c_a = u[166]
u_e4p_c_n = u[167]
u_e4p_c_a = u[168]
u_gthrd_c_n = u[169]
u_gthrd_c_a = u[170]
u_gthox_c_n = u[171]
u_gthox_c_a = u[172]
u_creat_c_n = u[173]
u_pcreat_c_n = u[174]
u_creat_c_a = u[175]
u_pcreat_c_a = u[176]
u_camp_c_a = u[177]
u_nrpphr_e_e = u[178]
u_udpg_c_a = u[179]
u_utp_c_a = u[180]
u_notBigg_GS_c_a = u[181]
u_notBigg_GPa_c_a = u[182]
u_notBigg_GPb_c_a = u[183]
u_h_c_n = C_H_cyt_n
u_h_c_a = C_H_cyt_a
u_co2_m_n = CO2_mito_n
u_co2_m_a = CO2_mito_a
u_ppi_c_a = PPi_a0
u_notBigg_PHKa_c_a = PHKa_a0
u_notBigg_PP1_c_a_initialValue = PP1_a0
H2PIi_n = (1e-3*u_pi_i_n)*(1e-3*u_h_i_n)/((1e-3*u_h_i_n)+k_dHPi)
H2PIx_n = (1e-3*u_pi_m_n)*(1e-3*u_h_m_n)/((1e-3*u_h_m_n)+k_dHPi)
H2PIi_a = (1e-3*u_pi_i_a)*(1e-3*u_h_i_a)/((1e-3*u_h_i_a)+k_dHPi)
H2PIx_a = (1e-3*u_pi_m_a)*(1e-3*u_h_m_a)/((1e-3*u_h_m_a)+k_dHPi)

    

Pi_n=u[139]
Pi_a=u[140]
C_H_mitomatr_nM=1e-3*u[1];
K_x_nM=1e-3*u[2];
Mg_x_nM=1e-3*u[3];
NADHmito_nM=1e-3*u[4];
QH2mito_nM=1e-3*u[5];
CytCredmito_nM=1e-3*u[6];
O2_nM=1e-3*u[7];
ATPmito_nM=1e-3*u[8];
ADPmito_nM=1e-3*u[9]
Cr_n=Crtot-u[174];
Cr_a=Crtot-u[176]
ADP_n=u[23]/2*(-qAK+sqrt(qAK*qAK+4*qAK*(ATDPtot_n/u[23]-1)))
ADP_a=u[73]/2*(-qAK+sqrt(qAK*qAK+4*qAK*(ATDPtot_a/u[73]-1)))
j_un=qAK*qAK+4*qAK*(ATDPtot_n/u[23]-1)
j_ug=qAK*qAK+4*qAK*(ATDPtot_a/u[73]-1)
dAMPdATPn=-1+qAK/2-0.5*sqrt(j_un)+qAK*ATDPtot_n/(u[23]*sqrt(j_un))
dAMPdATPg=-1+qAK/2-0.5*sqrt(j_ug)+qAK*ATDPtot_a/(u[73]*sqrt(j_ug))
ATP_nM=1e-3*u[23];
ADP_nM=1e-3*ADP_n
ATP_aM=1e-3*u[73];
ADP_aM=1e-3*ADP_a
ATP_mx_nM=1e-3*u[10];
ADP_mx_nM=1e-3*u[11]
Pimito_nM=1e-3*u[12]
ATP_i_nM=1e-3*u[13];
ADP_i_nM=1e-3*u[14];
AMP_i_nM=1e-3*u[15]
ATP_mi_nM=1e-3*u[16];
ADP_mi_nM=1e-3*u[17]
Pi_i_nM=1e-3*u[18]
Ctot_nM=1e-3*u[20]
Qtot_nM=1e-3*u[21]
C_H_ims_nM=1e-3*u[22]
AMP_nM=0
NAD_x_n=NADtot_n-NADHmito_nM;
u_nad_m_n=1000*NAD_x_n
Q_n=Qtot_nM-QH2mito_nM;
Qmito_n=1000*Q_n
Cox_n=Ctot_nM-CytCredmito_nM;
ATP_fx_n=ATPmito_nM-ATP_mx_nM
ADP_fx_n=ADPmito_nM-ADP_mx_nM
ATP_fi_n=ATP_i_nM-ATP_mi_nM
ADP_fi_n=ADP_i_nM-ADP_mi_nM
ADP_me_n=((K_DD+ADP_nM+Mg_tot)-sqrt((K_DD+ADP_nM+Mg_tot)^2-4*(Mg_tot*ADP_nM)))/2;
Mg_i_n=Mg_tot-ADP_me_n;
dG_H_n=etcF*u[19]+1*etcRT*log(C_H_ims_nM/C_H_mitomatr_nM);
dG_C1op_n=dG_C1o-1*etcRT*log(C_H_mitomatr_nM/1e-7);
dG_C3op_n=dG_C3o+2*etcRT*log(C_H_mitomatr_nM/1e-7);
dG_C4op_n=dG_C4o-2*etcRT*log(C_H_mitomatr_nM/1e-7);
dG_F1op_n=dG_F1o-1*etcRT*log(C_H_mitomatr_nM/1e-7);
C_H_mitomatr_a=u[51];
C_H_mitomatr_aM=1e-3*u[51]
K_x_aM=1e-3*u[52]
Mg_x_aM=1e-3*u[53]
NADHmito_aM=1e-3*u[54]
QH2mito_aM=1e-3*u[55]
CytCredmito_aM=1e-3*u[56]
O2_aM=1e-3*u[57]
ATPmito_aM=1e-3*u[58]
ADPmito_aM=1e-3*u[59]
ATP_mx_aM=1e-3*u[60]
ADP_mx_aM=1e-3*u[61]
Pimito_aM=1e-3*u[62]
ATP_i_aM=1e-3*u[63]
ADP_i_aM=1e-3*u[64];
AMP_i_aM=1e-3*u[65]
ATP_mi_aM=1e-3*u[66]
ADP_mi_aM=1e-3*u[67]
Pi_i_aM=1e-3*u[68]
Ctot_aM=1e-3*u[70]
Qtot_aM=1e-3*u[71]
C_H_ims_aM=1e-3*u[72]
AMP_aM=0
NAD_x_a=NADtot_a-NADHmito_aM;
u_nad_m_a=1000*NAD_x_a
Q_a=Qtot_aM-QH2mito_aM;
Qmito_a=1000*Q_a
Cox_a=Ctot_aM-CytCredmito_aM;
ATP_fx_a=ATPmito_aM-ATP_mx_aM
ADP_fx_a=ADPmito_aM-ADP_mx_aM
ATP_fi_a=ATP_i_aM-ATP_mi_aM
ADP_fi_a=ADP_i_aM-ADP_mi_aM
ADP_me_a=((K_DD_a+ADP_aM+Mg_tot)-sqrt((K_DD_a+ADP_aM+Mg_tot)^2-4*(Mg_tot*ADP_aM)))/2;
Mg_i_a=Mg_tot-ADP_me_a;
dG_H_a=etcF*u[69]+1*etcRT*log(C_H_ims_aM/C_H_mitomatr_aM);
dG_C1op_a=dG_C1o-1*etcRT*log(C_H_mitomatr_aM/1e-7);
dG_C3op_a=dG_C3o+2*etcRT*log(C_H_mitomatr_aM/1e-7);
dG_C4op_a=dG_C4o-2*etcRT*log(C_H_mitomatr_aM/1e-7);
dG_F1op_a=dG_F1o-1*etcRT*log(C_H_mitomatr_aM/1e-7);
u_nadp_c_n=0.0303-u[153];
u_nadp_c_a=0.0303-u[154]
u_nad_c_n=0.212-u[50];
u_nad_c_a=0.212-u[138]

# V=u[98]
# rTRPVsinf=u[111]
    
Glutamate_syn=u[97]
    
# alpham=-0.1*(V+33)/(exp(-0.1*(V+33))-1)
# betam=4*exp(-(V+58)/12)
# alphah=0.07*exp(-(V+50)/10)
# betah=1/(exp(-0.1*(V+20))+1)
# alphan=-0.01*(V+34)/(exp(-0.1*(V+34))-1)
# betan=0.125*exp(-(V+44)/25)
# minf=alpham/(alpham+betam);
# ninf=alphan/(alphan+betan);
# hinf=alphah/(alphah+betah);
# taun=1/(alphan+betan)*1e-03;
# tauh=1/(alphah+betah)*1e-03;
# p_inf=1.0/(1.0+exp(-(V+35.0)/10.0));
# tau_p=tau_max/(3.3*exp((V+35.0)/20.0)+exp(-(V+35.0)/20.0))
# K_n=K_n_Rest+(Na_n_Rest-u[99])
# EK=RTF*log(u[96]/K_n)
# EL=gKpas*EK/(gKpas+gNan)+gNan/(gKpas+gNan)*RTF*log(Na_out/u[99]);
# IL=gL*(V-EL);
# INa=ina_density #gNa*minf^3*u[100]*(V-RTF*log(Na_out/u[99]));
# IK=ik_density #gK*u[101]^4*(V-EK);
# mCa=1/(1+exp(-(V+20)/9));
# ICa=gCa*mCa^2*(V-ECa);
# ImAHP=gmAHP*u[102]/(u[102]+KD)*(V-EK);
# IM=g_M*u[103]*(V-EK)
    
# dIPump=F*kPumpn*u[23]*(u[99]-u0_ss[99])/(1+u[23]/KmPump);
# dIPump_a=F*kPumpg*u[73]*(u[94]-u0_ss[94])/(1+u[73]/KmPump)
#Isyne=-synInput*(V-Ee);
#Isyni=0
    
# vnstim=SmVn/F*ina_density #SmVn/F*(2/3*Isyne-INa);
# vgstim=SmVg/F*glia*ina_density #SmVg/F*2/3*glia*synInput;
# vLeakNan=SmVn*gNan/F*(RTF*log(Na_out/u[99])-V);
# vLeakNag=SmVg*gNag/F*(RTF*log(Na_out/u[94])-V);
# vPumpn=SmVn*kPumpn*u[23]*u[99]/(1+u[23]/KmPump);
vPumpg=SmVg*kPumpg*u[73]*u[94]/(1+u[73]/KmPump);
# JgliaK=((u[73]/ADP_a)/(mu_glia_ephys+(u[73]/ADP_a)))*(glia_c/(1+exp((Na_n2_baseNKA-u[96])/2.5)))
# JdiffK=epsilon*(u[96]-kbath)
# nBKinf=0.5*(1+tanh((u[93]+EETshift*u[112]-(-0.5*v5BK*tanh((u[108]-Ca3BK)/Ca4BK)+v6BK))/v4BK))
# IBK=gBK*u[104]*(u[93]-EBK)
# JNaK_a=(ImaxNaKa*(u[96]/(u[96]+INaKaKThr))*((u[94]^1.5)/(u[94]^1.5+INaKaNaThr^1.5)))
# IKirAS=gKirS*(u[96]^0.5)*(u[93]-VKirS*log(u[96]/u[95]))
# IKirAV=gKirV*(u[96]^0.5)*(u[93]-VKirAV*log(u[96]/u[95]))
# IleakA=gleakA*(u[93]-VleakA)
# Ileak_CaER_a=Pleak_CaER_a*(1.0-u[108]/u[109])
# ICa_pump_a=VCa_pump_a*((u[108]^2)/(u[108]^2+KpCa_pump_a^2))
# IIP3_a=ImaxIP3_a*(((u[106]/(u[106]+KIIP3_a))*(u[108]/(u[108]+KCaactIP3_a))*u[107])^3)*(1.0-u[108]/u[109])
# ITRP_a=gTRP*(u[93]-VTRP)*u[110]
    
# sinfTRPV=(1/(1+exp(-(((rTRPVsinf^(1/3)-r0TRPVsinf^(1/3))/r0TRPVsinf^(1/3))-e2TRPVsinf^(1/3))/kTRPVsinf)))*((1/(1+u[108]/gammaCaaTRPVsinf+Ca_perivasc/gammaCapTRPVsinf))*(u[108]/gammaCaaTRPVsinf+Ca_perivasc/gammaCapTRPVsinf+tanh((u[93]-v1TRPsinf_a)/v2TRPsinf_a)))
    
r0509_n(u_nadh_m_n,u_pi_m_n) = 1*(x_DH*(r_DH*NAD_x_n-(1e-3*u_nadh_m_n))*((1+(1e-3*u_pi_m_n)/k_Pi1)/(1+(1e-3*u_pi_m_n)/k_Pi2)))
NADH2_u10mi_n(u_nadh_m_n,u_q10h2_m_n) = 1*(x_C1*(exp(-(dG_C1op_n+4*dG_H_n)/etcRT)*(1e-3*u_nadh_m_n)*Q_n-NAD_x_n*(1e-3*u_q10h2_m_n)))
CYOR_u10mi_n(u_focytC_m_n,u_notBigg_MitoMembrPotent_m_n,u_q10h2_m_n,u_pi_m_n) = 1*(x_C3*((1+(1e-3*u_pi_m_n)/k_Pi3)/(1+(1e-3*u_pi_m_n)/k_Pi4))*(exp(-(dG_C3op_n+4*dG_H_n-2*etcF*u_notBigg_MitoMembrPotent_m_n)/(2*etcRT))*Cox_n*(1e-3*u_q10h2_m_n)^0.5-(1e-3*u_focytC_m_n)*Q_n^0.5))
CYOOm2i_n(u_focytC_m_n,u_notBigg_MitoMembrPotent_m_n,u_notBigg_Ctot_m_n,u_o2_c_n) = 1*(x_C4*((1e-3*u_o2_c_n)/((1e-3*u_o2_c_n)+k_O2))*((1e-3*u_focytC_m_n)/(1e-3*u_notBigg_Ctot_m_n))*(exp(-(dG_C4op_n+2*dG_H_n)/(2*etcRT))*(1e-3*u_focytC_m_n)*((1e-3*u_o2_c_n)^0.25)-Cox_n*exp(etcF*u_notBigg_MitoMembrPotent_m_n/etcRT)))
ATPS4mi_n(u_notBigg_ATP_mx_m_n,u_notBigg_ADP_mx_m_n,u_pi_m_n) = 1*(x_F1*(exp(-(dG_F1op_n-n_A*dG_H_n)/etcRT)*(K_DD/K_DT)*(1e-3*u_notBigg_ADP_mx_m_n)*(1e-3*u_pi_m_n)-(1e-3*u_notBigg_ATP_mx_m_n)))
ATPtm_n(u_notBigg_MitoMembrPotent_m_n) = 1*(x_ANT*(ADP_fi_n/(ADP_fi_n+ATP_fi_n*exp(-etcF*(0.35*u_notBigg_MitoMembrPotent_m_n)/etcRT))-ADP_fx_n/(ADP_fx_n+ATP_fx_n*exp(-etcF*(-0.65*u_notBigg_MitoMembrPotent_m_n)/etcRT)))*(ADP_fi_n/(ADP_fi_n+k_mADP)))
notBigg_J_Pi1_n(u_h_m_n,u_h_i_n) = 1*(x_Pi1*((1e-3*u_h_m_n)*H2PIi_n-(1e-3*u_h_i_n)*H2PIx_n)/(H2PIi_n+k_PiH))
notBigg_J_Hle_n(u_h_m_n,u_h_i_n,u_notBigg_MitoMembrPotent_m_n) = 1*(x_Hle*u_notBigg_MitoMembrPotent_m_n*((1e-3*u_h_i_n)*exp(etcF*u_notBigg_MitoMembrPotent_m_n/etcRT)-(1e-3*u_h_m_n))/(exp(etcF*u_notBigg_MitoMembrPotent_m_n/etcRT)-1))
notBigg_J_KH_n(u_h_m_n,u_h_i_n,u_k_m_n) = 1*(x_KH*(K_i*(1e-3*u_h_m_n)-(1e-3*u_k_m_n)*(1e-3*u_h_i_n)))
notBigg_J_K_n(u_k_m_n,u_notBigg_MitoMembrPotent_m_n) = 1*(x_K*u_notBigg_MitoMembrPotent_m_n*(K_i*exp(etcF*u_notBigg_MitoMembrPotent_m_n/etcRT)-(1e-3*u_k_m_n))/(exp(etcF*u_notBigg_MitoMembrPotent_m_n/etcRT)-1))
ADK1m_n(u_amp_i_n,u_atp_i_n,u_adp_i_n) = 1*(x_AK*(K_AK*(1e-3*u_adp_i_n)*(1e-3*u_adp_i_n)-(1e-3*u_amp_i_n)*(1e-3*u_atp_i_n)))
notBigg_J_AMP_n(u_amp_i_n,u_amp_c_n) = 1*(gamma*x_A*((1e-3*u_amp_c_n)-(1e-3*u_amp_i_n)))
notBigg_J_ADP_n(u_adp_i_n,u_adp_c_n) = 1*(gamma*x_A*((1e-3*u_adp_c_n)-(1e-3*u_adp_i_n)))
notBigg_J_ATP_n(u_atp_c_n,u_atp_i_n) = 1*(gamma*x_A*((1e-3*u_atp_c_n)-(1e-3*u_atp_i_n)))
notBigg_J_Pi2_n(u_pi_i_n,u_pi_c_n) = 1*(gamma*x_Pi2*(1e-3*u_pi_c_n-(1e-3*u_pi_i_n)))
notBigg_J_Ht_n(u_h_i_n,u_h_c_n) = 1*(gamma*x_Ht*(1e-3*u_h_c_n-(1e-3*u_h_i_n)))
notBigg_J_MgATPx_n(u_notBigg_ATP_mx_m_n,u_mg2_m_n) = 1*(x_MgA*(ATP_fx_n*(1e-3*u_mg2_m_n)-K_DT*(1e-3*u_notBigg_ATP_mx_m_n)))
notBigg_J_MgADPx_n(u_mg2_m_n,u_notBigg_ADP_mx_m_n) = 1*(x_MgA*(ADP_fx_n*(1e-3*u_mg2_m_n)-K_DD*(1e-3*u_notBigg_ADP_mx_m_n)))
notBigg_J_MgATPi_n(u_notBigg_ATP_mi_i_n) = 1*(x_MgA*(ATP_fi_n*Mg_i_n-K_DT*(1e-3*u_notBigg_ATP_mi_i_n)))
notBigg_J_MgADPi_n(u_notBigg_ADP_mi_i_n) = 1*(x_MgA*(ADP_fi_n*Mg_i_n-K_DD*(1e-3*u_notBigg_ADP_mi_i_n)))
PDHm_n(u_pyr_m_n,u_coa_m_n,u_nad_m_n) = 1*(VmaxPDHCmito_n*(u_pyr_m_n/(u_pyr_m_n+KmPyrMitoPDH_n))*(u_nad_m_n/(u_nad_m_n+KmNADmitoPDH_na))*(u_coa_m_n/(u_coa_m_n+KmCoAmitoPDH_n)))
CSm_n(u_accoa_m_n,u_oaa_m_n,u_cit_m_n,u_coa_m_n) = 1*(VmaxCSmito_n*(u_oaa_m_n/(u_oaa_m_n+KmOxaMito_n*(1.0+u_cit_m_n/KiCitMito_n)))*(u_accoa_m_n/(u_accoa_m_n+KmAcCoAmito_n*(1.0+u_coa_m_n/KiCoA_n))))
ACONTm_n(u_icit_m_n,u_cit_m_n) = 1*(VmaxAco_n*(u_cit_m_n-u_icit_m_n/KeqAco_na)/(1.0+u_cit_m_n/KmCitAco_n+u_icit_m_n/KmIsoCitAco_n))
ICDHxm_n(u_nadh_m_n,u_icit_m_n,u_nad_m_n) = 1*(VmaxIDH_n*(u_nad_m_n/KiNADmito_na)*((u_icit_m_n/KmIsoCitIDHm_n)^nIDH)/(1.0+u_nad_m_n/KiNADmito_na+(KmNADmito_na/KiNADmito_na)*((u_icit_m_n/KmIsoCitIDHm_n)^nIDH)+u_nadh_m_n/KiNADHmito_na+(u_nad_m_n/KiNADmito_na)*((u_icit_m_n/KmIsoCitIDHm_n)^nIDH)+((KmNADmito_na*u_nadh_m_n)/(KiNADmito_na*KiNADHmito_na))*((u_icit_m_n/KmIsoCitIDHm_n)^nIDH)))
AKGDm_n(u_coa_m_n,u_nadh_m_n,u_ca2_m_n,u_adp_m_n,u_succoa_m_n,u_nad_m_n,u_atp_m_n,u_akg_m_n) = 1*((VmaxKGDH_n*(1+u_adp_m_n/KiADPmito_KGDH_n)*(u_akg_m_n/Km1KGDHKGDH_n)*(u_coa_m_n/Km_CoA_kgdhKGDH_n)*(u_nad_m_n/KmNADkgdhKGDH_na))/(((u_coa_m_n/Km_CoA_kgdhKGDH_n)*(u_nad_m_n/KmNADkgdhKGDH_na)*(u_akg_m_n/Km1KGDHKGDH_n+(1+u_atp_m_n/KiATPmito_KGDH_n)/(1+u_ca2_m_n/KiCa2KGDH_n)))+((u_akg_m_n/Km1KGDHKGDH_n)*(u_coa_m_n/Km_CoA_kgdhKGDH_n+u_nad_m_n/KmNADkgdhKGDH_na)*(1+u_nadh_m_n/KiNADHKGDHKGDH_na+u_succoa_m_n/Ki_SucCoA_kgdhKGDH_n))))
SUCOASm_n(u_coa_m_n,u_succ_m_n,u_pi_m_n,u_adp_m_n,u_succoa_m_n,u_atp_m_n) = 1*(VmaxSuccoaATPscs_n*(1+AmaxPscs_n*((u_pi_m_n^npscs_n)/((u_pi_m_n^npscs_n)+(Km_pi_scs_na^npscs_n))))*(u_succoa_m_n*u_adp_m_n*u_pi_m_n-u_succ_m_n*u_coa_m_n*u_atp_m_n/Keqsuccoascs_na)/((1+u_succoa_m_n/Km_succoa_scs_n)*(1+u_adp_m_n/Km_ADPmito_scs_n)*(1+u_pi_m_n/Km_pi_scs_na)+(1+u_succ_m_n/Km_succ_scs_n)*(1+u_coa_m_n/Km_coa_scs_n)*(1+u_atp_m_n/Km_atpmito_scs_n)))
FUMm_n(u_fum_m_n,u_mal_L_m_n) = 1*(Vmaxfum_n*(u_fum_m_n-u_mal_L_m_n/Keqfummito_na)/(1.0+u_fum_m_n/Km_fummito_n+u_mal_L_m_n/Km_malmito_n))
MDHm_n(u_nadh_m_n,u_oaa_m_n,u_nad_m_n,u_mal_L_m_n) = 1*(VmaxMDHmito_n*(u_mal_L_m_n*u_nad_m_n-u_oaa_m_n*u_nadh_m_n/Keqmdhmito_na)/((1.0+u_mal_L_m_n/Km_mal_mdh_n)*(1.0+u_nad_m_n/Km_nad_mdh_na)+(1.0+u_oaa_m_n/Km_oxa_mdh_n)*(1.0+u_nadh_m_n/Km_nadh_mdh_na)))
OCOAT1m_n(u_acac_c_n,u_aacoa_m_n,u_succoa_m_n,u_succ_m_n) = 1*((VmaxfSCOT_n*u_succoa_m_n*u_acac_c_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n*(u_succoa_m_n/Ki_SucCoA_SCOT_n+Km_SucCoA_SCOT_n*u_acac_c_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n)+u_aacoa_m_n/Ki_AcAcCoA_SCOT_n+Km_AcAcCoA_SCOT_n*u_succ_m_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n)+u_succoa_m_n*u_acac_c_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n)+u_succoa_m_n*u_aacoa_m_n/(Ki_SucCoA_SCOT_n*Ki_AcAcCoA_SCOT_n)+Km_SucCoA_SCOT_n*u_acac_c_n*u_succ_m_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n*Ki_SUC_SCOT_n)+u_aacoa_m_n*u_succ_m_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n))))-(VmaxrSCOT_n*u_aacoa_m_n*u_succ_m_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n*(u_succoa_m_n/Ki_SucCoA_SCOT_n+Km_SucCoA_SCOT_n*u_acac_c_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n)+u_aacoa_m_n/Ki_AcAcCoA_SCOT_n+Km_AcAcCoA_SCOT_n*u_succ_m_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n)+u_succoa_m_n*u_acac_c_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n)+u_succoa_m_n*u_aacoa_m_n/(Ki_SucCoA_SCOT_n*Ki_AcAcCoA_SCOT_n)+Km_SucCoA_SCOT_n*u_acac_c_n*u_succ_m_n/(Ki_SucCoA_SCOT_n*Km_AcAc_SCOT_n*Ki_SUC_SCOT_n)+u_aacoa_m_n*u_succ_m_n/(Ki_AcAcCoA_SCOT_n*Km_SUC_SCOT_n)))))
ACACT1rm_n(u_aacoa_m_n,u_coa_m_n) = 1*(Vmax_thiolase_f_n*u_coa_m_n*u_aacoa_m_n/(Ki_CoA_thiolase_f_n*Km_AcAcCoA_thiolase_f_n+Km_AcAcCoA_thiolase_f_n*u_coa_m_n+Km_CoA_thiolase_f_n*u_aacoa_m_n+u_coa_m_n*u_aacoa_m_n))

notBigg_JbHBTrArtCap(t,u_bhb_b_b) = (2*(C_bHB_a-u_bhb_b_b)/eto_b)*notBigg_FinDyn_W2017

notBigg_MCT1_bHB_b(u_bhb_e_e,u_bhb_b_b) = 1*(VmaxMCTbhb_b*(u_bhb_b_b/(u_bhb_b_b+KmMCT1_bHB_b)-u_bhb_e_e/(u_bhb_e_e+KmMCT1_bHB_b)))
BHBt_n(u_bhb_c_n,u_bhb_e_e) = 1*(VmaxMCTbhb_n*(u_bhb_e_e/(u_bhb_e_e+KmMCT2_bHB_n)-u_bhb_c_n/(u_bhb_c_n+KmMCT2_bHB_n)))
BHBt_a(u_bhb_c_a,u_bhb_e_e) = 1*(VmaxMCTbhb_a*(u_bhb_e_e/(u_bhb_e_e+KmMCT1_bHB_a)-u_bhb_c_a/(u_bhb_c_a+KmMCT1_bHB_a)))
BDHm_n(u_acac_c_n,u_bhb_c_n,u_nadh_m_n,u_nad_m_n) = 1*(Vmax_bHBDH_f_n*u_nad_m_n*u_bhb_c_n/(Ki_NAD_B_HBD_f_n*Km_betaHB_BHBD_n+Km_betaHB_BHBD_n*u_nad_m_n+Km_NAD_B_HBD_n*u_bhb_c_n+u_nad_m_n*u_bhb_c_n)-(Vmax_bHBDH_r_n*u_nadh_m_n*u_acac_c_n/(Ki_NADH_BHBD_r_n*Km_AcAc_BHBD_n+Km_AcAc_BHBD_n*u_nadh_m_n+Km_NADH_BHBD_n*u_acac_c_n+u_nadh_m_n*u_acac_c_n)))
    
# BDHm_a(u_bhb_c_a,u_acac_c_a,u_nad_m_a,u_nadh_m_a) = 1*(Vmax_bHBDH_f_n*u_nad_m_a*u_bhb_c_a/(Ki_NAD_B_HBD_f_n*Km_betaHB_BHBD_n+Km_betaHB_BHBD_n*u_nad_m_a+Km_NAD_B_HBD_n*u_bhb_c_a+u_nad_m_a*u_bhb_c_a)-(Vmax_bHBDH_r_n*u_nadh_m_a*u_acac_c_a/(Ki_NADH_BHBD_r_n*Km_AcAc_BHBD_n+Km_AcAc_BHBD_n*u_nadh_m_a+Km_NADH_BHBD_n*u_acac_c_a+u_nadh_m_a*u_acac_c_a)))


ASPTAm_n(u_oaa_m_n,u_akg_m_n,u_glu_L_m_n,u_asp_L_m_n) = 1*(VfAAT_n*(u_asp_L_m_n*u_akg_m_n-u_oaa_m_n*u_glu_L_m_n/KeqAAT_n)/(KmAKG_AAT_n*u_asp_L_m_n+KmASP_AAT_n*(1.0+u_akg_m_n/KiAKG_AAT_n)*u_akg_m_n+u_asp_L_m_n*u_akg_m_n+KmASP_AAT_n*u_akg_m_n*u_glu_L_m_n/KiGLU_AAT_n+(KiASP_AAT_n*KmAKG_AAT_n/(KmOXA_AAT_n*KiGLU_AAT_n))*(KmGLU_AAT_n*u_asp_L_m_n*u_oaa_m_n/KiASP_AAT_n+u_oaa_m_n*u_glu_L_m_n+KmGLU_AAT_n*(1.0+u_akg_m_n/KiAKG_AAT_n)*u_oaa_m_n+KmOXA_AAT_n*u_glu_L_m_n)))
MDH_n(u_nadh_c_n,u_mal_L_c_n,u_oaa_c_n,u_nad_c_n) = 1*(VmaxcMDH_n*(u_mal_L_c_n*u_nad_c_n-u_oaa_c_n*u_nadh_c_n/Keqcmdh_n)/((1+u_mal_L_c_n/Kmmalcmdh_n)*(1+u_nad_c_n/Kmnadcmdh_n)+(1+u_oaa_c_n/Kmoxacmdh_n)*(1+u_nadh_c_n/Kmnadhcmdh_n)-1))
ASPTA_n(u_glu_L_c_n,u_asp_L_c_n,u_oaa_c_n,u_akg_c_n) = 1*(VfCAAT_n*(u_asp_L_c_n*u_akg_c_n-u_oaa_c_n*u_glu_L_c_n/KeqCAAT_n)/(KmAKG_CAAT_n*u_asp_L_c_n+KmASP_CAAT_n*(1.0+u_akg_c_n/KiAKG_CAAT_n)*u_akg_c_n+u_asp_L_c_n*u_akg_c_n+KmASP_CAAT_n*u_akg_c_n*u_glu_L_c_n/KiGLU_CAAT_n+(KiASP_CAAT_n*KmAKG_CAAT_n/(KmOXA_CAAT_n*KiGLU_CAAT_n))*(KmGLU_CAAT_n*u_asp_L_c_n*u_oaa_c_n/KiASP_CAAT_n+u_oaa_c_n*u_glu_L_c_n+KmGLU_CAAT_n*(1.0+u_akg_c_n/KiAKG_CAAT_n)*u_oaa_c_n+KmOXA_CAAT_n*u_glu_L_c_n)))
ASPGLUm_n(u_h_c_n,u_glu_L_c_n,u_asp_L_m_n,u_asp_L_c_n,u_notBigg_MitoMembrPotent_m_n,u_glu_L_m_n,u_h_m_n) = 1*(Vmaxagc_n*(u_asp_L_m_n*u_glu_L_c_n-u_asp_L_c_n*u_glu_L_m_n/((exp(u_notBigg_MitoMembrPotent_m_n)^(F/(R*T)))*(u_h_c_n/u_h_m_n)))/((u_asp_L_m_n+Km_aspmito_agc_n)*(u_glu_L_c_n+Km_glu_agc_n)+(u_asp_L_c_n+Km_asp_agc_n)*(u_glu_L_m_n+Km_glumito_agc_n)))
AKGMALtm_n(u_mal_L_c_n,u_akg_m_n,u_akg_c_n,u_mal_L_m_n) = 1*(Vmaxmakgc_n*(u_mal_L_c_n*u_akg_m_n-u_mal_L_m_n*u_akg_c_n)/((u_mal_L_c_n+Km_mal_mkgc_n)*(u_akg_m_n+Km_akgmito_mkgc_n)+(u_mal_L_m_n+Km_malmito_mkgc_n)*(u_akg_c_n+Km_akg_mkgc_n)))
r0509_a(u_nadh_m_a,u_pi_m_a) = 1*(x_DH_a*(r_DH_a*NAD_x_a-(1e-3*u_nadh_m_a))*((1+(1e-3*u_pi_m_a)/k_Pi1)/(1+(1e-3*u_pi_m_a)/k_Pi2)))
NADH2_u10mi_a(u_nadh_m_a,u_q10h2_m_a) = 1*(x_C1*(exp(-(dG_C1op_a+4*dG_H_a)/etcRT)*(1e-3*u_nadh_m_a)*Q_a-NAD_x_a*(1e-3*u_q10h2_m_a)))
CYOR_u10mi_a(u_focytC_m_a,u_q10h2_m_a,u_pi_m_a,u_notBigg_MitoMembrPotent_m_a) = 1*(x_C3*((1+(1e-3*u_pi_m_a)/k_Pi3)/(1+(1e-3*u_pi_m_a)/k_Pi4))*(exp(-(dG_C3op_a+4*dG_H_a-2*etcF*u_notBigg_MitoMembrPotent_m_a)/(2*etcRT))*Cox_a*(1e-3*u_q10h2_m_a)^0.5-(1e-3*u_focytC_m_a)*Q_a^0.5))
CYOOm2i_a(u_notBigg_Ctot_m_a,u_focytC_m_a,u_notBigg_MitoMembrPotent_m_a,u_o2_c_a) = 1*(x_C4*((1e-3*u_o2_c_a)/((1e-3*u_o2_c_a)+k_O2))*((1e-3*u_focytC_m_a)/(1e-3*u_notBigg_Ctot_m_a))*(exp(-(dG_C4op_a+2*dG_H_a)/(2*etcRT))*(1e-3*u_focytC_m_a)*((1e-3*u_o2_c_a)^0.25)-Cox_a*exp(etcF*u_notBigg_MitoMembrPotent_m_a/etcRT)))
ATPS4mi_a(u_notBigg_ATP_mx_m_a,u_pi_m_a,u_notBigg_ADP_mx_m_a) = 1*(x_F1_a*(exp(-(dG_F1op_a-n_A*dG_H_a)/etcRT)*(K_DD_a/K_DT_a)*(1e-3*u_notBigg_ADP_mx_m_a)*(1e-3*u_pi_m_a)-(1e-3*u_notBigg_ATP_mx_m_a)))
ATPtm_a(u_notBigg_MitoMembrPotent_m_a) = 1*(x_ANT_a*(ADP_fi_a/(ADP_fi_a+ATP_fi_a*exp(-etcF*(0.35*u_notBigg_MitoMembrPotent_m_a)/etcRT))-ADP_fx_a/(ADP_fx_a+ATP_fx_a*exp(-etcF*(-0.65*u_notBigg_MitoMembrPotent_m_a)/etcRT)))*(ADP_fi_a/(ADP_fi_a+k_mADP_a)))
notBigg_J_Pi1_a(u_h_i_a,u_h_m_a) = 1*(x_Pi1*((1e-3*u_h_m_a)*H2PIi_a-(1e-3*u_h_i_a)*H2PIx_a)/(H2PIi_a+k_PiH))
notBigg_J_Hle_a(u_h_i_a,u_h_m_a,u_notBigg_MitoMembrPotent_m_a) = 1*(x_Hle*u_notBigg_MitoMembrPotent_m_a*((1e-3*u_h_i_a)*exp(etcF*u_notBigg_MitoMembrPotent_m_a/etcRT)-(1e-3*u_h_m_a))/(exp(etcF*u_notBigg_MitoMembrPotent_m_a/etcRT)-1))
notBigg_J_KH_a(u_h_i_a,u_h_m_a,u_k_m_a) = 1*(x_KH*(K_i*(1e-3*u_h_m_a)-(1e-3*u_k_m_a)*(1e-3*u_h_i_a)))
notBigg_J_K_a(u_k_m_a,u_notBigg_MitoMembrPotent_m_a) = 1*(x_K*u_notBigg_MitoMembrPotent_m_a*(K_i*exp(etcF*u_notBigg_MitoMembrPotent_m_a/etcRT)-(1e-3*u_k_m_a))/(exp(etcF*u_notBigg_MitoMembrPotent_m_a/etcRT)-1))
ADK1m_a(u_atp_i_a,u_adp_i_a,u_amp_i_a) = 1*(x_AK*(K_AK*(1e-3*u_adp_i_a)*(1e-3*u_adp_i_a)-(1e-3*u_amp_i_a)*(1e-3*u_atp_i_a)))
notBigg_J_AMP_a(u_amp_i_a,u_amp_c_a) = 1*(gamma*x_A*((1e-3*u_amp_c_a)-(1e-3*u_amp_i_a)))
notBigg_J_ADP_a(u_adp_c_a,u_adp_i_a) = 1*(gamma*x_A*((1e-3*u_adp_c_a)-(1e-3*u_adp_i_a)))
notBigg_J_ATP_a(u_atp_i_a,u_atp_c_a) = 1*(gamma*x_A*((1e-3*u_atp_c_a)-(1e-3*u_atp_i_a)))
notBigg_J_Pi2_a(u_pi_i_a,u_pi_c_a) = 1*(gamma*x_Pi2*(1e-3*u_pi_c_a-(1e-3*u_pi_i_a)))
notBigg_J_Ht_a(u_h_i_a,u_h_c_a) = 1*(gamma*x_Ht*(1e-3*u_h_c_a-(1e-3*u_h_i_a)))
notBigg_J_MgATPx_a(u_notBigg_ATP_mx_m_a,u_mg2_m_a) = 1*(x_MgA*(ATP_fx_a*(1e-3*u_mg2_m_a)-K_DT_a*(1e-3*u_notBigg_ATP_mx_m_a)))
notBigg_J_MgADPx_a(u_notBigg_ADP_mx_m_a,u_mg2_m_a) = 1*(x_MgA*(ADP_fx_a*(1e-3*u_mg2_m_a)-K_DD_a*(1e-3*u_notBigg_ADP_mx_m_a)))
notBigg_J_MgATPi_a(u_notBigg_ATP_mi_i_a) = 1*(x_MgA*(ATP_fi_a*Mg_i_a-K_DT_a*(1e-3*u_notBigg_ATP_mi_i_a)))
notBigg_J_MgADPi_a(u_notBigg_ADP_mi_i_a) = 1*(x_MgA*(ADP_fi_a*Mg_i_a-K_DD_a*(1e-3*u_notBigg_ADP_mi_i_a)))
PDHm_a(u_nad_m_a,u_pyr_m_a,u_coa_m_a) = 1*(VmaxPDHCmito_a*(u_pyr_m_a/(u_pyr_m_a+KmPyrMitoPDH_a))*(u_nad_m_a/(u_nad_m_a+KmNADmitoPDH_na))*(u_coa_m_a/(u_coa_m_a+KmCoAmitoPDH_a)))
CSm_a(u_oaa_m_a,u_cit_m_a,u_accoa_m_a,u_coa_m_a) = 1*(VmaxCSmito_a*(u_oaa_m_a/(u_oaa_m_a+KmOxaMito_a*(1.0+u_cit_m_a/KiCitMito_a)))*(u_accoa_m_a/(u_accoa_m_a+KmAcCoAmito_a*(1.0+u_coa_m_a/KiCoA_a))))
ACONTm_a(u_cit_m_a,u_icit_m_a) = 1*(VmaxAco_a*(u_cit_m_a-u_icit_m_a/KeqAco_na)/(1.0+u_cit_m_a/KmCitAco_a+u_icit_m_a/KmIsoCitAco_a))
ICDHxm_a(u_nad_m_a,u_nadh_m_a,u_icit_m_a) = 1*(VmaxIDH_a*(u_nad_m_a/KiNADmito_na)*((u_icit_m_a/KmIsoCitIDHm_a)^nIDH)/(1.0+u_nad_m_a/KiNADmito_na+(KmNADmito_na/KiNADmito_na)*((u_icit_m_a/KmIsoCitIDHm_a)^nIDH)+u_nadh_m_a/KiNADHmito_na+(u_nad_m_a/KiNADmito_na)*((u_icit_m_a/KmIsoCitIDHm_a)^nIDH)+((KmNADmito_na*u_nadh_m_a)/(KiNADmito_na*KiNADHmito_na))*((u_icit_m_a/KmIsoCitIDHm_a)^nIDH)))
AKGDm_a(u_coa_m_a,u_nadh_m_a,u_akg_m_a,u_ca2_m_a,u_nad_m_a,u_succoa_m_a,u_atp_m_a,u_adp_m_a) = 1*((VmaxKGDH_a*(1+u_adp_m_a/KiADPmito_KGDH_a)*(u_akg_m_a/Km1KGDHKGDH_a)*(u_coa_m_a/Km_CoA_kgdhKGDH_a)*(u_nad_m_a/KmNADkgdhKGDH_na))/(((u_coa_m_a/Km_CoA_kgdhKGDH_a)*(u_nad_m_a/KmNADkgdhKGDH_na)*(u_akg_m_a/Km1KGDHKGDH_a+(1+u_atp_m_a/KiATPmito_KGDH_a)/(1+u_ca2_m_a/KiCa2KGDH_a)))+((u_akg_m_a/Km1KGDHKGDH_a)*(u_coa_m_a/Km_CoA_kgdhKGDH_a+u_nad_m_a/KmNADkgdhKGDH_na)*(1+u_nadh_m_a/KiNADHKGDHKGDH_na+u_succoa_m_a/Ki_SucCoA_kgdhKGDH_a))))
SUCOASm_a(u_coa_m_a,u_succoa_m_a,u_atp_m_a,u_succ_m_a,u_pi_m_a,u_adp_m_a) = 1*(VmaxSuccoaATPscs_a*(1+AmaxPscs_a*((u_pi_m_a^npscs_a)/((u_pi_m_a^npscs_a)+(Km_pi_scs_na^npscs_a))))*(u_succoa_m_a*u_adp_m_a*u_pi_m_a-u_succ_m_a*u_coa_m_a*u_atp_m_a/Keqsuccoascs_na)/((1+u_succoa_m_a/Km_succoa_scs_a)*(1+u_adp_m_a/Km_ADPmito_scs_a)*(1+u_pi_m_a/Km_pi_scs_na)+(1+u_succ_m_a/Km_succ_scs_a)*(1+u_coa_m_a/Km_coa_scs_a)*(1+u_atp_m_a/Km_atpmito_scs_a)))
FUMm_a(u_mal_L_m_a,u_fum_m_a) = 1*(Vmaxfum_a*(u_fum_m_a-u_mal_L_m_a/Keqfummito_na)/(1.0+u_fum_m_a/Km_fummito_a+u_mal_L_m_a/Km_malmito_a))
MDHm_a(u_nad_m_a,u_oaa_m_a,u_mal_L_m_a,u_nadh_m_a) = 1*(VmaxMDHmito_a*(u_mal_L_m_a*u_nad_m_a-u_oaa_m_a*u_nadh_m_a/Keqmdhmito_na)/((1.0+u_mal_L_m_a/Km_mal_mdh_a)*(1.0+u_nad_m_a/Km_nad_mdh_na)+(1.0+u_oaa_m_a/Km_oxa_mdh_a)*(1.0+u_nadh_m_a/Km_nadh_mdh_na)))
PCm_a(u_co2_m_a,u_oaa_m_a,u_pyr_m_a,u_atp_m_a,u_adp_m_a) = 1*(((u_atp_m_a/u_adp_m_a)/(muPYRCARB_a+(u_atp_m_a/u_adp_m_a)))*VmPYRCARB_a*(u_pyr_m_a*u_co2_m_a-u_oaa_m_a/KeqPYRCARB_a)/(KmPYR_PYRCARB_a*KmCO2_PYRCARB_a+KmPYR_PYRCARB_a*u_co2_m_a+KmCO2_PYRCARB_a*u_pyr_m_a+u_co2_m_a*u_pyr_m_a))
GLNt4_n(u_gln_L_c_n,u_gln_L_e_e) = 1*(TmaxSNAT_GLN_n*(u_gln_L_e_e-u_gln_L_c_n/coeff_gln_ratio_n_ecs)/(KmSNAT_GLN_n+u_gln_L_c_n))
GLUNm_n(u_gln_L_c_n,u_glu_L_m_n) = 1*(VmGLS_n*(u_gln_L_c_n-u_glu_L_m_n/KeqGLS_n)/(KmGLNGLS_n*(1.0+u_glu_L_m_n/KiGLUGLS_n)+u_gln_L_c_n))
GLUt6_a(u_na1_c_a,u_notBigg_Va_c_a,u_k_e_e,u_glu_L_syn_syn,u_k_c_a,u_glu_L_c_a) = 1*(-((1/(2*F*1e-3))*(-alpha_EAAT*exp(-beta_EAAT*(u_notBigg_Va_c_a-((R*T/(2*F*1e-3))*log(((Na_syn_EAAT/u_na1_c_a)^3)*(H_syn_EAAT/H_ast_EAAT)*(u_glu_L_syn_syn/u_glu_L_c_a)*(u_k_c_a/u_k_e_e))))))))
GLUDxm_a(u_nad_m_a,u_nadh_m_a,u_glu_L_c_a,u_akg_m_a) = 1*(VmGDH_a*(u_nad_m_a*u_glu_L_c_a-u_nadh_m_a*u_akg_m_a/KeqGDH_a)/(KiNAD_GDH_a*KmGLU_GDH_a+KmGLU_GDH_a*u_nad_m_a+KiNAD_GDH_a*u_glu_L_c_a+u_glu_L_c_a*u_nad_m_a+u_glu_L_c_a*u_nad_m_a/KiAKG_GDH_a+KiNAD_GDH_a*KmGLU_GDH_a*u_nadh_m_a/KiNADH_GDH_a+KiNAD_GDH_a*KmGLU_GDH_a*KmNADH_GDH_a*u_akg_m_a/(KiAKG_GDH_a*KiNADH_GDH_a)+KmNADH_GDH_a*u_glu_L_c_a*u_nadh_m_a/KiNADH_GDH_a+KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*u_nadh_m_a/(KiAKG_GDH_a*KiNADH_GDH_a)+KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a*u_akg_m_a*u_nadh_m_a/(KiAKG_GDH_a*KiNADH_GDH_a)+u_glu_L_c_a*u_nad_m_a*u_akg_m_a/KiAKG_GDH_a+KiNAD_GDH_a*KmGLU_GDH_a*KmAKG_GDH_a/KiAKG_GDH_a+KiNAD_GDH_a*KmGLU_GDH_a*u_glu_L_c_a*u_nadh_m_a*u_akg_m_a/(KiGLU_GDH_a*KiAKG_GDH_a*KiNADH_GDH_a)+KiNAD_GDH_a*KmGLU_GDH_a*u_akg_m_a*u_nadh_m_a/(KiAKG_GDH_a*KiNADH_GDH_a)+KmNADH_GDH_a*KmGLU_GDH_a*u_akg_m_a*u_nad_m_a/(KiAKG_GDH_a*KiNADH_GDH_a)))
GLNS_a(u_adp_c_a,u_atp_c_a,u_glu_L_c_a) = 1*(VmaxGLNsynth_a*(u_glu_L_c_a/(KmGLNsynth_a+u_glu_L_c_a))*((1/(u_atp_c_a/u_adp_c_a))/(muGLNsynth_a+(1/(u_atp_c_a/u_adp_c_a)))))
GLNt4_a(u_gln_L_c_a,u_gln_L_e_e) = 1*(TmaxSNAT_GLN_a*(u_gln_L_c_a-u_gln_L_e_e)/(KmSNAT_GLN_a+u_gln_L_c_a))

    

    
notBigg_JdHbin(t,u_o2_b_b) = 2*(C_O_a-u_o2_b_b)*notBigg_FinDyn_W2017

# notBigg_JdHbout(t,u_notBigg_ddHb_b_b,u_notBigg_vV_b_b) = (u_notBigg_ddHb_b_b/u_notBigg_vV_b_b)*(global_par_F_0*((u_notBigg_vV_b_b/u0_ss[111])^2+(u_notBigg_vV_b_b/u0_ss[111])^(-0.5)*global_par_tau_v/u0_ss[111]*notBigg_FinDyn_W2017)/(1+global_par_F_0*(u_notBigg_vV_b_b/u0_ss[111])^(-0.5)*global_par_tau_v/u0_ss[111]))

notBigg_JdHbout(t,u_notBigg_ddHb_b_b) = (u_notBigg_ddHb_b_b/u_notBigg_vV_b_b)*notBigg_Fout_W2017

    
    
notBigg_JO2art2cap(t,u_o2_b_b) = (1/eto_b)*2*(C_O_a-u_o2_b_b)*notBigg_FinDyn_W2017
    
notBigg_JO2fromCap2a(u_o2_b_b,u_o2_c_a) = 1*((PScapA*(KO2b*(HbOP/u_o2_b_b-1.)^(-1/param_degree_nh)-u_o2_c_a)))
notBigg_JO2fromCap2n(u_o2_b_b,u_o2_c_n) = 1*((PScapNAratio*PScapA*(KO2b*(HbOP/u_o2_b_b-1.)^(-1/param_degree_nh)-u_o2_c_n)))
    
notBigg_trGLC_art_cap(t,u_glc_D_b_b) = (1/eto_b)*(2*(C_Glc_a-u_glc_D_b_b))*notBigg_FinDyn_W2017


    
notBigg_JGlc_be(u_glc_D_ecsEndothelium_ecsEndothelium,u_glc_D_b_b) = 1*(TmaxGLCce*(u_glc_D_b_b*(KeG+u_glc_D_ecsEndothelium_ecsEndothelium)-u_glc_D_ecsEndothelium_ecsEndothelium*(KeG+u_glc_D_b_b))/(KeG^2+KeG*ReGoi*u_glc_D_b_b+KeG*ReGio*u_glc_D_ecsEndothelium_ecsEndothelium+ReGee*u_glc_D_b_b*u_glc_D_ecsEndothelium_ecsEndothelium))
notBigg_JGlc_e2ecsBA(u_glc_D_ecsBA_ecsBA,u_glc_D_ecsEndothelium_ecsEndothelium) = 1*(TmaxGLCeb*(u_glc_D_ecsEndothelium_ecsEndothelium*(KeG2+u_glc_D_ecsBA_ecsBA)-u_glc_D_ecsBA_ecsBA*(KeG2+u_glc_D_ecsEndothelium_ecsEndothelium))/(KeG2^2+KeG2*ReGoi2*u_glc_D_ecsEndothelium_ecsEndothelium+KeG2*ReGio2*u_glc_D_ecsBA_ecsBA+ReGee2*u_glc_D_ecsEndothelium_ecsEndothelium*u_glc_D_ecsBA_ecsBA))
notBigg_JGlc_ecsBA2a(u_glc_D_ecsBA_ecsBA,u_glc_D_c_a) = 1*(TmaxGLCba*(u_glc_D_ecsBA_ecsBA*(KeG3+u_glc_D_c_a)-u_glc_D_c_a*(KeG3+u_glc_D_ecsBA_ecsBA))/(KeG3^2+KeG3*ReGoi3*u_glc_D_ecsBA_ecsBA+KeG3*ReGio3*u_glc_D_c_a+ReGee3*u_glc_D_ecsBA_ecsBA*u_glc_D_c_a))
notBigg_JGlc_a2ecsAN(u_glc_D_ecsAN_ecsAN,u_glc_D_c_a) = 1*(TmaxGLCai*(u_glc_D_c_a*(KeG4+u_glc_D_ecsAN_ecsAN)-u_glc_D_ecsAN_ecsAN*(KeG4+u_glc_D_c_a))/(KeG4^2+KeG4*ReGoi4*u_glc_D_c_a+KeG4*ReGio4*u_glc_D_ecsAN_ecsAN+ReGee4*u_glc_D_c_a*u_glc_D_ecsAN_ecsAN))
notBigg_JGlc_ecsAN2n(u_glc_D_c_n,u_glc_D_ecsAN_ecsAN) = 1*(TmaxGLCin*(u_glc_D_ecsAN_ecsAN*(KeG5+u_glc_D_c_n)-u_glc_D_c_n*(KeG5+u_glc_D_ecsAN_ecsAN))/(KeG5^2+KeG5*ReGoi5*u_glc_D_ecsAN_ecsAN+KeG5*ReGio5*u_glc_D_c_n+ReGee5*u_glc_D_ecsAN_ecsAN*u_glc_D_c_n))
notBigg_JGlc_diffEcs(u_glc_D_ecsBA_ecsBA,u_glc_D_ecsAN_ecsAN) = 1*(kGLCdiff*(u_glc_D_ecsBA_ecsBA-u_glc_D_ecsAN_ecsAN))
GLCS2_a(u_notBigg_GS_c_a,u_udpg_c_a) = 1*(kL2_GS_a*u_notBigg_GS_c_a*u_udpg_c_a/(kmL2_GS_a+u_udpg_c_a))
GLCP_a(u_glycogen_c_a,u_notBigg_GPa_c_a,u_camp_c_a,u_notBigg_GPb_c_a) = 1*((u_notBigg_GPa_c_a/(u_notBigg_GPa_c_a+u_notBigg_GPb_c_a))*VmaxGP_a*u_glycogen_c_a*(1/(1+(KmGP_AMP_a^hGPa)/(u_camp_c_a^hGPa))))
PGMT_a(u_g1p_c_a,u_g6p_c_a) = 1*((Vmaxfpglm_a*u_g1p_c_a/KmG1PPGLM_a-((Vmaxfpglm_a*KmG6PPGLM_a)/(KmG1PPGLM_a*KeqPGLM_a))*u_g6p_c_a/KmG6PPGLM_a)/(1.0+u_g1p_c_a/KmG1PPGLM_a+u_g6p_c_a/KmG6PPGLM_a))
PDE1_a(u_camp_c_a) = 1*(VmaxPDE_a*u_camp_c_a/(Kmcamppde_a+u_camp_c_a))
GALUi_a(u_ppi_c_a,u_g1p_c_a,u_udpg_c_a,u_utp_c_a) = 1*((VmaxfUDPGP*u_utp_c_a*u_g1p_c_a/(KutpUDPGP*Kg1pUDPGP)-VmaxrUDPGP*u_ppi_c_a*u_udpg_c_a/(KpiUDPGP*KUDPglucoUDPGP_a))/(1.0+u_g1p_c_a/Kg1pUDPGP+u_utp_c_a/KutpUDPGP+(u_g1p_c_a*u_utp_c_a)/(Kg1pUDPGP*KutpUDPGP)+u_udpg_c_a/KUDPglucoUDPGP_a+u_ppi_c_a/KpiUDPGP+(u_ppi_c_a*u_udpg_c_a)/(KpiUDPGP*KUDPglucoUDPGP_a)))
notBigg_psiGSAJay_a(u_notBigg_GS_c_a,u_udpg_c_a,u_notBigg_PHKa_c_a,u_notBigg_PKAa_c_a) = 1*(((kg8_GSAJay*PP1_a0*(st_GSAJay-u_notBigg_GS_c_a))/((kmg8_GSAJay/(1.0+s1_GSAJay*u_udpg_c_a/kg2_GSAJay))+(st_GSAJay-u_notBigg_GS_c_a)))-((kg7_GSAJay*(u_notBigg_PHKa_c_a+u_notBigg_PKAa_c_a)*u_notBigg_GS_c_a)/(kmg7_GSAJay*(1+s1_GSAJay*u_udpg_c_a/kg2_GSAJay)+u_notBigg_GS_c_a)))
notBigg_psiPHK_a(u_ca2_c_a,u_glycogen_c_a,u_g1p_c_a,u_notBigg_PHKa_c_a,u_notBigg_GPa_c_a,u_udpg_c_a) = 1*((u_ca2_c_a/cai0_ca_ion)*(((kg5_PHK*u_notBigg_PHKa_c_a*(pt_PHK-u_notBigg_GPa_c_a))/(kmg5_PHK*(1.0+s1_PHK*u_g1p_c_a/kg2_PHK)+(pt_PHK-u_notBigg_GPa_c_a)))-((kg6_PHK*PP1_a0*u_notBigg_GPa_c_a)/(kmg6_PHK/(1+s2_PHK*u_udpg_c_a/kgi_PHK)+u_notBigg_GPa_c_a))-((0.003198/(1+u_glycogen_c_a)+kmind_PHK)*PP1_a0*u_notBigg_GPa_c_a)))
HEX1_n(u_glc_D_c_n,u_atp_c_n,u_g6p_c_n) = 1*((VmaxHK_n*u_glc_D_c_n/(u_glc_D_c_n+KmHK_n))*(u_atp_c_n/(1+(u_atp_c_n/KIATPhex_n)^nHhexn))*(1/(1+u_g6p_c_n/KiHKG6P_n)))
HEX1_a(u_atp_c_a,u_g6p_c_a,u_glc_D_c_a) = 1*((VmaxHK_a*u_glc_D_c_a/(u_glc_D_c_a+KmHK_a))*(u_atp_c_a/(1+(u_atp_c_a/KIATPhex_a)^nHhexa))*(1/(1+u_g6p_c_a/KiHKG6P_a)))
PGI_n(u_f6p_c_n,u_g6p_c_n) = 1*((Vmax_fPGI_n*(u_g6p_c_n/Km_G6P_fPGI_n-0.9*u_f6p_c_n/Km_F6P_rPGI_n))/(1.0+u_g6p_c_n/Km_G6P_fPGI_n+u_f6p_c_n/Km_F6P_rPGI_n))
PGI_a(u_g6p_c_a,u_f6p_c_a) = 1*((Vmax_fPGI_a*(u_g6p_c_a/Km_G6P_fPGI_a-0.9*u_f6p_c_a/Km_F6P_rPGI_a))/(1.0+u_g6p_c_a/Km_G6P_fPGI_a+u_f6p_c_a/Km_F6P_rPGI_a))
PFK_n(u_f6p_c_n,u_atp_c_n) = 1*(VmaxPFK_n*(u_atp_c_n/(1+(u_atp_c_n/KiPFK_ATP_na)^nPFKn))*(u_f6p_c_n/(u_f6p_c_n+KmPFKF6P_n)))
PFK_a(u_atp_c_a,u_f26bp_c_a,u_f6p_c_a) = 1*(VmaxPFK_a*(u_atp_c_a/(1+(u_atp_c_a/KiPFK_ATP_a)^nPFKa))*(u_f6p_c_a/(u_f6p_c_a+KmPFKF6P_a*(1-KoPFK_f26bp_a*((u_f26bp_c_a^nPFKf26bp_a)/(KmF26BP_PFK_a^nPFKf26bp_a+u_f26bp_c_a^nPFKf26bp_a)))))*(u_f26bp_c_a/(KmF26BP_PFK_a+u_f26bp_c_a)))
PFK26_a(u_adp_c_a,u_atp_c_a,u_f26bp_c_a,u_f6p_c_a) = 1*(Vmax_PFKII_g*u_f6p_c_a*u_atp_c_a*u_adp_c_a/((u_f6p_c_a+Kmf6pPFKII_g)*(u_atp_c_a+KmatpPFKII_g)*(u_adp_c_a+Km_act_adpPFKII_g))-(Vmax_PFKII_g*u_f26bp_c_a/(u_f26bp_c_a+Km_f26bp_f_26pase_g*(1+u_f6p_c_a/Ki_f6p_f_26_pase_g))))
FBA_n(u_fdp_c_n,u_dhap_c_n,u_g3p_c_n) = 1*(Vmaxald_n*(u_fdp_c_n-u_g3p_c_n*u_dhap_c_n/Keqald_n)/((1+u_fdp_c_n/KmfbpAld_n)+(1+u_g3p_c_n/KmgapAld_n)*(1+u_dhap_c_n/KmdhapAld_n)-1))
FBA_a(u_g3p_c_a,u_fdp_c_a,u_dhap_c_a) = 1*(Vmaxald_a*(u_fdp_c_a-u_g3p_c_a*u_dhap_c_a/Keqald_a)/((1+u_fdp_c_a/KmfbpAld_a)+(1+u_g3p_c_a/KmgapAld_a)*(1+u_dhap_c_a/KmdhapAld_a)-1))
TPI_n(u_dhap_c_n,u_g3p_c_n) = 1*(Vmaxtpi_n*(u_dhap_c_n-u_g3p_c_n/Keqtpi_n)/(1+u_dhap_c_n/KmdhapTPI_n+u_g3p_c_n/KmgapTPI_n))
TPI_a(u_g3p_c_a,u_dhap_c_a) = 1*(Vmaxtpi_a*(u_dhap_c_a-u_g3p_c_a/Keqtpi_a)/(1+u_dhap_c_a/KmdhapTPI_a+u_g3p_c_a/KmgapTPI_a))
GAPD_n(u_nadh_c_n,u_nad_c_n,u_g3p_c_n,u_pi_c_n,u_13dpg_c_n) = 1*(Vmaxgapdh_n*(u_nad_c_n*u_g3p_c_n*u_pi_c_n-u_13dpg_c_n*u_nadh_c_n/Keqgapdh_na)/((1+u_nad_c_n/KmnadGpdh_n)*(1+u_g3p_c_n/KmGapGapdh_n)*(1+u_pi_c_n/KmpiGpdh_n)+(1+u_nadh_c_n/KmnadhGapdh_n)*(1+u_13dpg_c_n/KmBPG13Gapdh_n)-1))
GAPD_a(u_nadh_c_a,u_nad_c_a,u_pi_c_a,u_13dpg_c_a,u_g3p_c_a) = 1*(Vmaxgapdh_a*(u_nad_c_a*u_g3p_c_a*u_pi_c_a-u_13dpg_c_a*u_nadh_c_a/Keqgapdh_na)/((1+u_nad_c_a/KmnadGpdh_a)*(1+u_g3p_c_a/KmGapGapdh_a)*(1+u_pi_c_a/KmpiGpdh_a)+(1+u_nadh_c_a/KmnadhGapdh_a)*(1+u_13dpg_c_a/KmBPG13Gapdh_a)-1))
PGK_n(u_13dpg_c_n,u_3pg_c_n,u_atp_c_n,u_adp_c_n) = 1*(Vmaxpgk_n*(u_13dpg_c_n*u_adp_c_n-u_3pg_c_n*u_atp_c_n/Keqpgk_na)/((1+u_13dpg_c_n/Kmbpg13pgk_n)*(1+u_adp_c_n/Kmadppgk_n)+(1+u_3pg_c_n/Kmpg3pgk_n)*(1+u_atp_c_n/Kmatppgk_n)-1))
PGK_a(u_adp_c_a,u_atp_c_a,u_3pg_c_a,u_13dpg_c_a) = 1*(Vmaxpgk_a*(u_13dpg_c_a*u_adp_c_a-u_3pg_c_a*u_atp_c_a/Keqpgk_na)/((1+u_13dpg_c_a/Kmbpg13pgk_a)*(1+u_adp_c_a/Kmadppgk_a)+(1+u_3pg_c_a/Kmpg3pgk_a)*(1+u_atp_c_a/Kmatppgk_a)-1))
PGM_n(u_2pg_c_n,u_3pg_c_n) = 1*(Vmaxpgm_n*(u_3pg_c_n-u_2pg_c_n/Keqpgm_n)/((1+u_3pg_c_n/Kmpg3pgm_n)+(1+u_2pg_c_n/Kmpg2pgm_n)-1))
PGM_a(u_2pg_c_a,u_3pg_c_a) = 1*(Vmaxpgm_a*(u_3pg_c_a-u_2pg_c_a/Keqpgm_a)/((1+u_3pg_c_a/Kmpg3pgm_a)+(1+u_2pg_c_a/Kmpg2pgm_a)-1))
ENO_n(u_pep_c_n,u_2pg_c_n) = 1*(Vmaxenol_n*(u_2pg_c_n-u_pep_c_n/Keqenol_n)/((1+u_2pg_c_n/Kmpg2enol_n)+(1+u_pep_c_n/Km_pep_enol_n)-1))
ENO_a(u_2pg_c_a,u_pep_c_a) = 1*(Vmaxenol_a*(u_2pg_c_a-u_pep_c_a/Keqenol_a)/((1+u_2pg_c_a/Kmpg2enol_a)+(1+u_pep_c_a/Km_pep_enol_a)-1))
PYK_n(u_pep_c_n,u_atp_c_n,u_adp_c_n) = 1*(Vmaxpk_n*u_pep_c_n*u_adp_c_n/((u_pep_c_n+Km_pep_pk_n)*(u_adp_c_n+Km_adp_pk_n*(1+u_atp_c_n/Ki_ATP_pk_n))))
PYK_a(u_adp_c_a,u_atp_c_a,u_pep_c_a) = 1*(Vmaxpk_a*u_pep_c_a*u_adp_c_a/((u_pep_c_a+Km_pep_pk_a)*(u_adp_c_a+Km_adp_pk_a*(1+u_atp_c_a/Ki_ATP_pk_a))))

notBigg_JLacTr_b(u_lac_L_b_b,t) = (2*(C_Lac_a-u_lac_L_b_b)/eto_b)*notBigg_FinDyn_W2017

notBigg_MCT1_LAC_b(u_lac_L_b_b,u_lac_L_e_e) = 1*(TbLac*(u_lac_L_b_b/(u_lac_L_b_b+KbLac)-u_lac_L_e_e/(u_lac_L_e_e+KbLac)))
L_LACt2r_a(u_lac_L_e_e,u_lac_L_c_a) = 1*(TaLac*(u_lac_L_e_e/(u_lac_L_e_e+Km_Lac_a)-u_lac_L_c_a/(u_lac_L_c_a+Km_Lac_a)))
L_LACt2r_n(u_lac_L_e_e,u_lac_L_c_n) = 1*(TnLac*(u_lac_L_e_e/(u_lac_L_e_e+Km_LacTr_n)-u_lac_L_c_n/(u_lac_L_c_n+Km_LacTr_n)))
    
notBigg_jLacDiff_e(u_lac_L_e_e) = 0 # 1*(betaLacDiff*(u0_ss[150]-u_lac_L_e_e))
    
notBigg_vLACgc(u_lac_L_b_b,u_lac_L_c_a) = 1*(TMaxLACgc*(u_lac_L_b_b/(u_lac_L_b_b+KtLACgc)-u_lac_L_c_a/(u_lac_L_c_a+KtLACgc)))
LDH_L_a(u_nad_c_a,u_nadh_c_a,u_lac_L_c_a,u_pyr_c_a) = 1*(VmfLDH_a*u_pyr_c_a*u_nadh_c_a-KeLDH_a*VmfLDH_a*u_lac_L_c_a*u_nad_c_a)
LDH_L_n(u_nad_c_n,u_lac_L_c_n,u_nadh_c_n,u_pyr_c_n) = 1*(VmfLDH_n*u_pyr_c_n*u_nadh_c_n-KeLDH_n*VmfLDH_n*u_lac_L_c_n*u_nad_c_n)
G6PDH2r_n(u_g6p_c_n,u_nadph_c_n,u_nadp_c_n,u_6pgl_c_n) = 1*(VmaxG6PDH_n*(1/(K_G6P_G6PDH_n*K_NADP_G6PDH_n))*((u_g6p_c_n*u_nadp_c_n-u_6pgl_c_n*u_nadph_c_n/KeqG6PDH_n)/((1+u_g6p_c_n/K_G6P_G6PDH_n)*(1+u_nadp_c_n/K_NADP_G6PDH_n)+(1+u_6pgl_c_n/K_GL6P_G6PDH_n)*(1+u_nadph_c_n/K_NADPH_G6PDH_n)-1)))
G6PDH2r_a(u_6pgl_c_a,u_nadp_c_a,u_g6p_c_a,u_nadph_c_a) = 1*(VmaxG6PDH_a*(1/(K_G6P_G6PDH_a*K_NADP_G6PDH_a))*((u_g6p_c_a*u_nadp_c_a-u_6pgl_c_a*u_nadph_c_a/KeqG6PDH_a)/((1+u_g6p_c_a/K_G6P_G6PDH_a)*(1+u_nadp_c_a/K_NADP_G6PDH_a)+(1+u_6pgl_c_a/K_GL6P_G6PDH_a)*(1+u_nadph_c_a/K_NADPH_G6PDH_a)-1)))
PGL_n(u_6pgc_c_n,u_6pgl_c_n) = 1*(Vmax6PGL_n*(1/K_GL6P_6PGL_n)*((u_6pgl_c_n-u_6pgc_c_n/Keq6PGL_n)/((1+u_6pgl_c_n/K_GL6P_6PGL_n)+(1+u_6pgc_c_n/K_GO6P_6PGL_n)-1)))
PGL_a(u_6pgc_c_a,u_6pgl_c_a) = 1*(Vmax6PGL_a*(1/K_GL6P_6PGL_a)*((u_6pgl_c_a-u_6pgc_c_a/Keq6PGL_a)/((1+u_6pgl_c_a/K_GL6P_6PGL_a)+(1+u_6pgc_c_a/K_GO6P_6PGL_a)-1)))
GND_n(u_6pgc_c_n,u_nadph_c_n,u_nadp_c_n,u_ru5p_D_c_n) = 1*(Vmax6PGDH_n*(1/(K_GO6P_6PGDH_n*K_NADP_6PGDH_n))*(u_6pgc_c_n*u_nadp_c_n-u_ru5p_D_c_n*u_nadph_c_n/Keq6PGDH_n)/((1+u_6pgc_c_n/K_GO6P_6PGDH_n)*(1+u_nadp_c_n/K_NADP_6PGDH_n)+(1+u_ru5p_D_c_n/K_RU5P_6PGDH_n)*(1+u_nadph_c_n/K_NADPH_6PGDH_n)-1))
GND_a(u_6pgc_c_a,u_nadp_c_a,u_ru5p_D_c_a,u_nadph_c_a) = 1*(Vmax6PGDH_a*(1/(K_GO6P_6PGDH_a*K_NADP_6PGDH_a))*(u_6pgc_c_a*u_nadp_c_a-u_ru5p_D_c_a*u_nadph_c_a/Keq6PGDH_a)/((1+u_6pgc_c_a/K_GO6P_6PGDH_a)*(1+u_nadp_c_a/K_NADP_6PGDH_a)+(1+u_ru5p_D_c_a/K_RU5P_6PGDH_a)*(1+u_nadph_c_a/K_NADPH_6PGDH_a)-1))
RPI_n(u_r5p_c_n,u_ru5p_D_c_n) = 1*(VmaxRPI_n*(1/K_RU5P_RPI_n)*(u_ru5p_D_c_n-u_r5p_c_n/KeqRPI_n)/((1+u_ru5p_D_c_n/K_RU5P_RPI_n)+(1+u_r5p_c_n/K_R5P_RPI_n)-1))
RPI_a(u_r5p_c_a,u_ru5p_D_c_a) = 1*(VmaxRPI_a*(1/K_RU5P_RPI_a)*(u_ru5p_D_c_a-u_r5p_c_a/KeqRPI_a)/((1+u_ru5p_D_c_a/K_RU5P_RPI_a)+(1+u_r5p_c_a/K_R5P_RPI_a)-1))
RPE_n(u_xu5p_D_c_n,u_ru5p_D_c_n) = 1*(VmaxRPE_n*(1/K_RU5P_RPE_n)*(u_ru5p_D_c_n-u_xu5p_D_c_n/KeqRPE_n)/((1+u_ru5p_D_c_n/K_RU5P_RPE_n)+(1+u_xu5p_D_c_n/K_X5P_RPE_n)-1))
RPE_a(u_ru5p_D_c_a,u_xu5p_D_c_a) = 1*(VmaxRPE_a*(1/K_RU5P_RPE_a)*(u_ru5p_D_c_a-u_xu5p_D_c_a/KeqRPE_a)/((1+u_ru5p_D_c_a/K_RU5P_RPE_a)+(1+u_xu5p_D_c_a/K_X5P_RPE_a)-1))
TKT1_n(u_s7p_c_n,u_r5p_c_n,u_xu5p_D_c_n,u_g3p_c_n) = 1*(VmaxTKL1_n*(1/(K_X5P_TKL1_n*K_R5P_TKL1_n))*(u_xu5p_D_c_n*u_r5p_c_n-u_g3p_c_n*u_s7p_c_n/KeqTKL1_n)/((1+u_xu5p_D_c_n/K_X5P_TKL1_n)*(1+u_r5p_c_n/K_R5P_TKL1_n)+(1+u_g3p_c_n/K_GAP_TKL1_n)*(1+u_s7p_c_n/K_S7P_TKL1_n)-1))
TKT1_a(u_r5p_c_a,u_g3p_c_a,u_s7p_c_a,u_xu5p_D_c_a) = 1*(VmaxTKL1_a*(1/(K_X5P_TKL1_a*K_R5P_TKL1_a))*(u_xu5p_D_c_a*u_r5p_c_a-u_g3p_c_a*u_s7p_c_a/KeqTKL1_a)/((1+u_xu5p_D_c_a/K_X5P_TKL1_a)*(1+u_r5p_c_a/K_R5P_TKL1_a)+(1+u_g3p_c_a/K_GAP_TKL1_a)*(1+u_s7p_c_a/K_S7P_TKL1_a)-1))
TKT2_n(u_f6p_c_n,u_e4p_c_n,u_xu5p_D_c_n,u_g3p_c_n) = 1*(VmaxTKL2_n*(1/(K_F6P_TKL2_n*K_GAP_TKL2_n))*(u_f6p_c_n*u_g3p_c_n-u_xu5p_D_c_n*u_e4p_c_n/KeqTKL2_n)/((1+u_f6p_c_n/K_F6P_TKL2_n)*(1+u_g3p_c_n/K_GAP_TKL2_n)+(1+u_xu5p_D_c_n/K_X5P_TKL2_n)*(1+u_e4p_c_n/K_E4P_TKL2_n)-1))
TKT2_a(u_g3p_c_a,u_e4p_c_a,u_xu5p_D_c_a,u_f6p_c_a) = 1*(VmaxTKL2_a*(1/(K_F6P_TKL2_a*K_GAP_TKL2_a))*(u_f6p_c_a*u_g3p_c_a-u_xu5p_D_c_a*u_e4p_c_a/KeqTKL2_a)/((1+u_f6p_c_a/K_F6P_TKL2_a)*(1+u_g3p_c_a/K_GAP_TKL2_a)+(1+u_xu5p_D_c_a/K_X5P_TKL2_a)*(1+u_e4p_c_a/K_E4P_TKL2_a)-1))
TALA_n(u_f6p_c_n,u_e4p_c_n,u_g3p_c_n,u_s7p_c_n) = 1*(VmaxTAL_n*(1/(K_GAP_TAL_n*K_S7P_TAL_n))*(u_g3p_c_n*u_s7p_c_n-u_f6p_c_n*u_e4p_c_n/KeqTAL_n)/((1+u_g3p_c_n/K_GAP_TAL_n)*(1+u_s7p_c_n/K_S7P_TAL_n)+(1+u_f6p_c_n/K_F6P_TAL_n)*(1+u_e4p_c_n/K_E4P_TAL_n)-1))
TALA_a(u_g3p_c_a,u_e4p_c_a,u_s7p_c_a,u_f6p_c_a) = 1*(VmaxTAL_a*(1/(K_GAP_TAL_a*K_S7P_TAL_a))*(u_g3p_c_a*u_s7p_c_a-u_f6p_c_a*u_e4p_c_a/KeqTAL_a)/((1+u_g3p_c_a/K_GAP_TAL_a)*(1+u_s7p_c_a/K_S7P_TAL_a)+(1+u_f6p_c_a/K_F6P_TAL_a)*(1+u_e4p_c_a/K_E4P_TAL_a)-1))
notBigg_psiNADPHox_n(u_nadph_c_n) = 1*(k1NADPHox_n*u_nadph_c_n)
notBigg_psiNADPHox_a(u_nadph_c_a) = 1*(k1NADPHox_a*u_nadph_c_a)
GTHO_n(u_nadph_c_n,u_gthox_c_n) = 1*((Vmf_GSSGR_n*u_gthox_c_n*u_nadph_c_n)/((KmGSSGRGSSG_n+u_gthox_c_n)*(KmGSSGRNADPH_n+u_nadph_c_n)))
GTHO_a(u_gthox_c_a,u_nadph_c_a) = 1*((Vmf_GSSGR_a*u_gthox_c_a*u_nadph_c_a)/((KmGSSGRGSSG_a+u_gthox_c_a)*(KmGSSGRNADPH_a+u_nadph_c_a)))
GTHP_n(u_gthrd_c_n) = 1*(V_GPX_n*u_gthrd_c_n/(u_gthrd_c_n+KmGPXGSH_n))
GTHP_a(u_gthrd_c_a) = 1*(V_GPX_a*u_gthrd_c_a/(u_gthrd_c_a+KmGPXGSH_a))
GTHS_n(u_gthrd_c_n) = 1*(VmaxGSHsyn_n*(glycine_n*glutamylCys_n-u_gthrd_c_n/KeGSHSyn_n)/(Km_glutamylCys_GSHsyn_n*Km_glycine_GSHsyn_n+glutamylCys_n*Km_glutamylCys_GSHsyn_n+glycine_n*Km_glycine_GSHsyn_n*(1+glutamylCys_n/Km_glutamylCys_GSHsyn_n)+u_gthrd_c_n/KmGSHsyn_n))
GTHS_a(u_gthrd_c_a) = 1*(VmaxGSHsyn_a*(glycine_a*glutamylCys_a-u_gthrd_c_a/KeGSHSyn_a)/(Km_glutamylCys_GSHsyn_a*Km_glycine_GSHsyn_a+glutamylCys_a*Km_glutamylCys_GSHsyn_a+glycine_a*Km_glycine_GSHsyn_a*(1+glutamylCys_a/Km_glutamylCys_GSHsyn_a)+u_gthrd_c_a/KmGSHsyn_a))
ADNCYC_a(u_camp_c_a,u_atp_c_a) = 1*(((VmaxfAC_a*u_atp_c_a/(KmACATP_a*(1+u_camp_c_a/KicAMPAC_a))-VmaxrAC_a*u_camp_c_a/(KmpiAC_a*KmcAMPAC_a))/(1+u_atp_c_a/(KmACATP_a*(1+u_camp_c_a/KicAMPAC_a))+u_camp_c_a/KmcAMPAC_a)))
CKc_n(u_pcreat_c_n,u_atp_c_n,u_adp_c_n) = 1*(kCKnps*u_pcreat_c_n*u_adp_c_n-KeqCKnpms*kCKnps*(Crtot-u_pcreat_c_n)*u_atp_c_n)
CKc_a(u_adp_c_a,u_atp_c_a,u_pcreat_c_a) = 1*(kCKgps*u_pcreat_c_a*u_adp_c_a-KeqCKgpms*kCKgps*(Crtot-u_pcreat_c_a)*u_atp_c_a)
PYRt2m_n(u_h_m_n,u_pyr_m_n,u_h_c_n,u_pyr_c_n) = 1*(Vmax_PYRtrcyt2mito_nH*(u_pyr_c_n*u_h_c_n-u_pyr_m_n*u_h_m_n)/((1.0+u_pyr_c_n/KmPyrCytTr_n)*(1.0+u_pyr_m_n/KmPyrMitoTr_n)))
PYRt2m_a(u_h_m_a,u_pyr_c_a,u_h_c_a,u_pyr_m_a) = 1*(Vmax_PYRtrcyt2mito_aH*(u_pyr_c_a*u_h_c_a-u_pyr_m_a*u_h_m_a)/((1.0+u_pyr_c_a/KmPyrCytTr_a)*(1.0+u_pyr_m_a/KmPyrMitoTr_a)))
notBigg_vShuttlen(u_nadh_m_n,u_nadh_c_n) = 1*(TnNADH_jlv*(u_nadh_c_n/(0.212-u_nadh_c_n))/(MnCyto_jlv+(u_nadh_c_n/(0.212-u_nadh_c_n)))*((1000*NADtot-u_nadh_m_n)/u_nadh_m_n)/(MnMito_jlv+((1000*NADtot-u_nadh_m_n)/u_nadh_m_n)))
notBigg_vShuttleg(u_nadh_c_a,u_nadh_m_a) = 1*(TgNADH_jlv*(u_nadh_c_a/(0.212-u_nadh_c_a))/(MgCyto_jlv+(u_nadh_c_a/(0.212-u_nadh_c_a)))*((1000*NADtot-u_nadh_m_a)/u_nadh_m_a)/(MgMito_jlv+((1000*NADtot-u_nadh_m_a)/u_nadh_m_a)))
notBigg_vMitooutn_n(u_nadh_m_n,u_nad_m_n,u_o2_c_n,u_adp_i_n,u_atp_i_n) = 1*(V_oxphos_n*((1/(u_atp_i_n/u_adp_i_n))/(mu_oxphos_n+(1/(u_atp_i_n/u_adp_i_n))))*((u_nadh_m_n/u_nad_m_n)/(nu_oxphos_n+(u_nadh_m_n/u_nad_m_n)))*(u_o2_c_n/(u_o2_c_n+K_oxphos_n)))
notBigg_vMitooutg_a(u_atp_i_a,u_nadh_m_a,u_nad_m_a,u_o2_c_a,u_adp_i_a) = 1*(V_oxphos_a*((1/(u_atp_i_a/u_adp_i_a))/(mu_oxphos_a+(1/(u_atp_i_a/u_adp_i_a))))*((u_nadh_m_a/u_nad_m_a)/(nu_oxphos_a+(u_nadh_m_a/u_nad_m_a)))*(u_o2_c_a/(u_o2_c_a+K_oxphos_a)))
notBigg_vMitoinn(u_pyr_m_n,u_nadh_m_n) = 1*(VMaxMitoinn*u_pyr_m_n/(u_pyr_m_n+KmMito)*(1000*NADtot-u_nadh_m_n)/(1000*NADtot-u_nadh_m_n+KmNADn_jlv))
notBigg_vMitoing(u_nadh_m_a,u_pyr_m_a) = 1*(VMaxMitoing*u_pyr_m_a/(u_pyr_m_a+KmMito_a)*(1000*NADtot-u_nadh_m_a)/(1000*NADtot-u_nadh_m_a+KmNADg_jlv))

du[1] = 0
du[2] = 0.5*T2Jcorrection*(1000*(notBigg_J_KH_n(u_h_m_n,u_h_i_n,u_k_m_n)+notBigg_J_K_n(u_k_m_n,u_notBigg_MitoMembrPotent_m_n))/W_x)
du[3] = 0.5*T2Jcorrection*(1000*(-notBigg_J_MgATPx_n(u_notBigg_ATP_mx_m_n,u_mg2_m_n)-notBigg_J_MgADPx_n(u_mg2_m_n,u_notBigg_ADP_mx_m_n))/W_x)
du[4] = 6.96*(notBigg_vMitoinn(u_pyr_m_n,u_nadh_m_n)+notBigg_vShuttlen(u_nadh_m_n,u_nadh_c_n)-notBigg_vMitooutn_n(u_nadh_m_n,u_nad_m_n,u_o2_c_n,u_adp_i_n,u_atp_i_n))
du[5] = 0.5*T2Jcorrection*(1000*(+NADH2_u10mi_n(u_nadh_m_n,u_q10h2_m_n)-CYOR_u10mi_n(u_focytC_m_n,u_notBigg_MitoMembrPotent_m_n,u_q10h2_m_n,u_pi_m_n))/W_x)
du[6] = 0.5*T2Jcorrection*(1000*(+2*CYOR_u10mi_n(u_focytC_m_n,u_notBigg_MitoMembrPotent_m_n,u_q10h2_m_n,u_pi_m_n)-2*CYOOm2i_n(u_focytC_m_n,u_notBigg_MitoMembrPotent_m_n,u_notBigg_Ctot_m_n,u_o2_c_n))/W_i)
du[7] = notBigg_JO2fromCap2n(u_o2_b_b,u_o2_c_n)-0.6*notBigg_vMitooutn_n(u_nadh_m_n,u_nad_m_n,u_o2_c_n,u_adp_i_n,u_atp_i_n)
du[8] = 0.5*T2Jcorrection*(1000*(+ATPS4mi_n(u_notBigg_ATP_mx_m_n,u_notBigg_ADP_mx_m_n,u_pi_m_n)-ATPtm_n(u_notBigg_MitoMembrPotent_m_n))/W_x)
du[9] = 0.5*T2Jcorrection*(1000*(-ATPS4mi_n(u_notBigg_ATP_mx_m_n,u_notBigg_ADP_mx_m_n,u_pi_m_n)+ATPtm_n(u_notBigg_MitoMembrPotent_m_n))/W_x)
du[10] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgATPx_n(u_notBigg_ATP_mx_m_n,u_mg2_m_n))/W_x)
du[11] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgADPx_n(u_mg2_m_n,u_notBigg_ADP_mx_m_n))/W_x)
du[12] = 0.5*T2Jcorrection*(1000*(-ATPS4mi_n(u_notBigg_ATP_mx_m_n,u_notBigg_ADP_mx_m_n,u_pi_m_n)+notBigg_J_Pi1_n(u_h_m_n,u_h_i_n))/W_x)
du[13] = 0.5*T2Jcorrection*(1000*(+notBigg_J_ATP_n(u_atp_c_n,u_atp_i_n)+ATPtm_n(u_notBigg_MitoMembrPotent_m_n)+ADK1m_n(u_amp_i_n,u_atp_i_n,u_adp_i_n))/W_i)
du[14] = 0.5*T2Jcorrection*(1000*(+notBigg_J_ADP_n(u_adp_i_n,u_adp_c_n)-ATPtm_n(u_notBigg_MitoMembrPotent_m_n)-2*ADK1m_n(u_amp_i_n,u_atp_i_n,u_adp_i_n))/W_i)
du[15] = 0.5*T2Jcorrection*(1000*(+notBigg_J_AMP_n(u_amp_i_n,u_amp_c_n)+ADK1m_n(u_amp_i_n,u_atp_i_n,u_adp_i_n))/W_i)
du[16] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgATPi_n(u_notBigg_ATP_mi_i_n))/W_i)
du[17] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgADPi_n(u_notBigg_ADP_mi_i_n))/W_i)
du[18] = 0.5*T2Jcorrection*(1000*(-notBigg_J_Pi1_n(u_h_m_n,u_h_i_n)+notBigg_J_Pi2_n(u_pi_i_n,u_pi_c_n))/W_i)
du[19] = 0.5*T2Jcorrection*(4*NADH2_u10mi_n(u_nadh_m_n,u_q10h2_m_n)+2*CYOR_u10mi_n(u_focytC_m_n,u_notBigg_MitoMembrPotent_m_n,u_q10h2_m_n,u_pi_m_n)+4*CYOOm2i_n(u_focytC_m_n,u_notBigg_MitoMembrPotent_m_n,u_notBigg_Ctot_m_n,u_o2_c_n)-n_A*ATPS4mi_n(u_notBigg_ATP_mx_m_n,u_notBigg_ADP_mx_m_n,u_pi_m_n)-ATPtm_n(u_notBigg_MitoMembrPotent_m_n)-notBigg_J_Hle_n(u_h_m_n,u_h_i_n,u_notBigg_MitoMembrPotent_m_n)-notBigg_J_K_n(u_k_m_n,u_notBigg_MitoMembrPotent_m_n))/CIM
du[20] = 0
du[21] = 0
du[22] = 0
du[23] = (CKc_n(u_pcreat_c_n,u_atp_c_n,u_adp_c_n)+0.5*(1/6.96)*1000*(-notBigg_J_ATP_n(u_atp_c_n,u_atp_i_n))-HEX1_n(u_glc_D_c_n,u_atp_c_n,u_g6p_c_n)-PFK_n(u_f6p_c_n,u_atp_c_n)+PGK_n(u_13dpg_c_n,u_3pg_c_n,u_atp_c_n,u_adp_c_n)+PYK_n(u_pep_c_n,u_atp_c_n,u_adp_c_n) )/(1-dAMPdATPn) #(CKc_n(u_pcreat_c_n,u_atp_c_n,u_adp_c_n)+0.5*(1/6.96)*1000*(-notBigg_J_ATP_n(u_atp_c_n,u_atp_i_n))-HEX1_n(u_glc_D_c_n,u_atp_c_n,u_g6p_c_n)-PFK_n(u_f6p_c_n,u_atp_c_n)+PGK_n(u_13dpg_c_n,u_3pg_c_n,u_atp_c_n,u_adp_c_n)+PYK_n(u_pep_c_n,u_atp_c_n,u_adp_c_n)-0.15*vPumpn-vATPasesn)/(1-dAMPdATPn)
du[24] = 0
du[25] = T2Jcorrection*(0.5*1000*r0509_n(u_nadh_m_n,u_pi_m_n)/W_x-FUMm_n(u_fum_m_n,u_mal_L_m_n))
du[26] = T2Jcorrection*(FUMm_n(u_fum_m_n,u_mal_L_m_n)-MDHm_n(u_nadh_m_n,u_oaa_m_n,u_nad_m_n,u_mal_L_m_n))+6.96*AKGMALtm_n(u_mal_L_c_n,u_akg_m_n,u_akg_c_n,u_mal_L_m_n)
du[27] = T2Jcorrection*(MDHm_n(u_nadh_m_n,u_oaa_m_n,u_nad_m_n,u_mal_L_m_n)-CSm_n(u_accoa_m_n,u_oaa_m_n,u_cit_m_n,u_coa_m_n))+ASPTAm_n(u_oaa_m_n,u_akg_m_n,u_glu_L_m_n,u_asp_L_m_n)
du[28] = T2Jcorrection*(0.5*SUCOASm_n(u_coa_m_n,u_succ_m_n,u_pi_m_n,u_adp_m_n,u_succoa_m_n,u_atp_m_n)-0.5*1000*r0509_n(u_nadh_m_n,u_pi_m_n)/W_x)+0.5*T2Jcorrection*OCOAT1m_n(u_acac_c_n,u_aacoa_m_n,u_succoa_m_n,u_succ_m_n)
du[29] = 0.5*T2Jcorrection*(AKGDm_n(u_coa_m_n,u_nadh_m_n,u_ca2_m_n,u_adp_m_n,u_succoa_m_n,u_nad_m_n,u_atp_m_n,u_akg_m_n)-SUCOASm_n(u_coa_m_n,u_succ_m_n,u_pi_m_n,u_adp_m_n,u_succoa_m_n,u_atp_m_n))-0.5*T2Jcorrection*OCOAT1m_n(u_acac_c_n,u_aacoa_m_n,u_succoa_m_n,u_succ_m_n)
du[30] = T2Jcorrection*(CSm_n(u_accoa_m_n,u_oaa_m_n,u_cit_m_n,u_coa_m_n)-PDHm_n(u_pyr_m_n,u_coa_m_n,u_nad_m_n)-0.5*AKGDm_n(u_coa_m_n,u_nadh_m_n,u_ca2_m_n,u_adp_m_n,u_succoa_m_n,u_nad_m_n,u_atp_m_n,u_akg_m_n)+0.5*SUCOASm_n(u_coa_m_n,u_succ_m_n,u_pi_m_n,u_adp_m_n,u_succoa_m_n,u_atp_m_n)-0.5*ACACT1rm_n(u_aacoa_m_n,u_coa_m_n))
du[31] = 0.5*T2Jcorrection*(ICDHxm_n(u_nadh_m_n,u_icit_m_n,u_nad_m_n)-AKGDm_n(u_coa_m_n,u_nadh_m_n,u_ca2_m_n,u_adp_m_n,u_succoa_m_n,u_nad_m_n,u_atp_m_n,u_akg_m_n))-ASPTAm_n(u_oaa_m_n,u_akg_m_n,u_glu_L_m_n,u_asp_L_m_n)-AKGMALtm_n(u_mal_L_c_n,u_akg_m_n,u_akg_c_n,u_mal_L_m_n)
du[32] = 0
du[33] = 0.5*T2Jcorrection*(ACONTm_n(u_icit_m_n,u_cit_m_n)-ICDHxm_n(u_nadh_m_n,u_icit_m_n,u_nad_m_n))
du[34] = T2Jcorrection*(CSm_n(u_accoa_m_n,u_oaa_m_n,u_cit_m_n,u_coa_m_n)-0.5*ACONTm_n(u_icit_m_n,u_cit_m_n))
du[35] = T2Jcorrection*(PDHm_n(u_pyr_m_n,u_coa_m_n,u_nad_m_n)-CSm_n(u_accoa_m_n,u_oaa_m_n,u_cit_m_n,u_coa_m_n))+T2Jcorrection*ACACT1rm_n(u_aacoa_m_n,u_coa_m_n)
du[36] = 0.5*T2Jcorrection*(BDHm_n(u_acac_c_n,u_bhb_c_n,u_nadh_m_n,u_nad_m_n)-OCOAT1m_n(u_acac_c_n,u_aacoa_m_n,u_succoa_m_n,u_succ_m_n))
du[37] = 0.5*T2Jcorrection*(OCOAT1m_n(u_acac_c_n,u_aacoa_m_n,u_succoa_m_n,u_succ_m_n)-ACACT1rm_n(u_aacoa_m_n,u_coa_m_n))
du[38] = 6.96*PYRt2m_n(u_h_m_n,u_pyr_m_n,u_h_c_n,u_pyr_c_n)-T2Jcorrection*PDHm_n(u_pyr_m_n,u_coa_m_n,u_nad_m_n)
du[39] = 0.44*BHBt_n(u_bhb_c_n,u_bhb_e_e)-BDHm_n(u_acac_c_n,u_bhb_c_n,u_nadh_m_n,u_nad_m_n)
du[40] = 0.0275*notBigg_MCT1_bHB_b(u_bhb_e_e,u_bhb_b_b)-BHBt_n(u_bhb_c_n,u_bhb_e_e)-BHBt_a(u_bhb_c_a,u_bhb_e_e)
du[41] = 0 #0.8*BHBt_a(u_bhb_e_e,u_bhb_c_a) - BDHm_a(u_bhb_c_a,u_acac_c_a,u_nad_m_a,u_nadh_m_a) # 0
du[42] = notBigg_JbHBTrArtCap(t,u_bhb_b_b)-notBigg_MCT1_bHB_b(u_bhb_e_e,u_bhb_b_b)
du[43] = 0
du[44] = 0
du[45] = 0.5*T2Jcorrection*(ASPTAm_n(u_oaa_m_n,u_akg_m_n,u_glu_L_m_n,u_asp_L_m_n)+6.96*ASPGLUm_n(u_h_c_n,u_glu_L_c_n,u_asp_L_m_n,u_asp_L_c_n,u_notBigg_MitoMembrPotent_m_n,u_glu_L_m_n,u_h_m_n)+GLUNm_n(u_gln_L_c_n,u_glu_L_m_n))
du[46] = 0
du[47] = 0
du[48] = 0
du[49] = ASPTA_n(u_glu_L_c_n,u_asp_L_c_n,u_oaa_c_n,u_akg_c_n)-ASPGLUm_n(u_h_c_n,u_glu_L_c_n,u_asp_L_m_n,u_asp_L_c_n,u_notBigg_MitoMembrPotent_m_n,u_glu_L_m_n,u_h_m_n)
du[50] = GAPD_n(u_nadh_c_n,u_nad_c_n,u_g3p_c_n,u_pi_c_n,u_13dpg_c_n)-LDH_L_n(u_nad_c_n,u_lac_L_c_n,u_nadh_c_n,u_pyr_c_n)-notBigg_vShuttlen(u_nadh_m_n,u_nadh_c_n)
du[51] = 0
du[52] = 0.5*T2Jcorrection*(1000*(notBigg_J_KH_a(u_h_i_a,u_h_m_a,u_k_m_a)+notBigg_J_K_a(u_k_m_a,u_notBigg_MitoMembrPotent_m_a))/W_x)
du[53] = 0.5*T2Jcorrection*(1000*(-notBigg_J_MgATPx_a(u_notBigg_ATP_mx_m_a,u_mg2_m_a)-notBigg_J_MgADPx_a(u_notBigg_ADP_mx_m_a,u_mg2_m_a))/W_x)
du[54] = 6.96*(notBigg_vMitoing(u_nadh_m_a,u_pyr_m_a)+notBigg_vShuttleg(u_nadh_c_a,u_nadh_m_a)-notBigg_vMitooutg_a(u_atp_i_a,u_nadh_m_a,u_nad_m_a,u_o2_c_a,u_adp_i_a))
du[55] = 0.5*T2Jcorrection*(1000*(+NADH2_u10mi_a(u_nadh_m_a,u_q10h2_m_a)-CYOR_u10mi_a(u_focytC_m_a,u_q10h2_m_a,u_pi_m_a,u_notBigg_MitoMembrPotent_m_a))/W_x)
du[56] = 0.5*T2Jcorrection*(1000*(+2*CYOR_u10mi_a(u_focytC_m_a,u_q10h2_m_a,u_pi_m_a,u_notBigg_MitoMembrPotent_m_a)-2*CYOOm2i_a(u_notBigg_Ctot_m_a,u_focytC_m_a,u_notBigg_MitoMembrPotent_m_a,u_o2_c_a))/W_i)
du[57] = notBigg_JO2fromCap2a(u_o2_b_b,u_o2_c_a)-0.6*notBigg_vMitooutg_a(u_atp_i_a,u_nadh_m_a,u_nad_m_a,u_o2_c_a,u_adp_i_a)
du[58] = 0.5*T2Jcorrection*(1000*(+ATPS4mi_a(u_notBigg_ATP_mx_m_a,u_pi_m_a,u_notBigg_ADP_mx_m_a)-ATPtm_a(u_notBigg_MitoMembrPotent_m_a))/W_x)
du[59] = 0.5*T2Jcorrection*(1000*(-ATPS4mi_a(u_notBigg_ATP_mx_m_a,u_pi_m_a,u_notBigg_ADP_mx_m_a)+ATPtm_a(u_notBigg_MitoMembrPotent_m_a))/W_x)
du[60] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgATPx_a(u_notBigg_ATP_mx_m_a,u_mg2_m_a))/W_x)
du[61] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgADPx_a(u_notBigg_ADP_mx_m_a,u_mg2_m_a))/W_x)
du[62] = 0.5*T2Jcorrection*(1000*(-ATPS4mi_a(u_notBigg_ATP_mx_m_a,u_pi_m_a,u_notBigg_ADP_mx_m_a)+notBigg_J_Pi1_a(u_h_i_a,u_h_m_a))/W_x)
du[63] = 0.5*T2Jcorrection*(1000*(+notBigg_J_ATP_a(u_atp_i_a,u_atp_c_a)+ATPtm_a(u_notBigg_MitoMembrPotent_m_a)+ADK1m_a(u_atp_i_a,u_adp_i_a,u_amp_i_a))/W_i)
du[64] = 0.5*T2Jcorrection*(1000*(+notBigg_J_ADP_a(u_adp_c_a,u_adp_i_a)-ATPtm_a(u_notBigg_MitoMembrPotent_m_a)-2*ADK1m_a(u_atp_i_a,u_adp_i_a,u_amp_i_a))/W_i)
du[65] = 0.5*T2Jcorrection*(1000*(+notBigg_J_AMP_a(u_amp_i_a,u_amp_c_a)+ADK1m_a(u_atp_i_a,u_adp_i_a,u_amp_i_a))/W_i)
du[66] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgATPi_a(u_notBigg_ATP_mi_i_a))/W_i)
du[67] = 0.5*T2Jcorrection*(1000*(notBigg_J_MgADPi_a(u_notBigg_ADP_mi_i_a))/W_i)
du[68] = 0.5*T2Jcorrection*(1000*(-notBigg_J_Pi1_a(u_h_i_a,u_h_m_a)+notBigg_J_Pi2_a(u_pi_i_a,u_pi_c_a))/W_i)
du[69] = 0.5*T2Jcorrection*((4*NADH2_u10mi_a(u_nadh_m_a,u_q10h2_m_a)+2*CYOR_u10mi_a(u_focytC_m_a,u_q10h2_m_a,u_pi_m_a,u_notBigg_MitoMembrPotent_m_a)+4*CYOOm2i_a(u_notBigg_Ctot_m_a,u_focytC_m_a,u_notBigg_MitoMembrPotent_m_a,u_o2_c_a)-n_A*ATPS4mi_a(u_notBigg_ATP_mx_m_a,u_pi_m_a,u_notBigg_ADP_mx_m_a)-ATPtm_a(u_notBigg_MitoMembrPotent_m_a)-notBigg_J_Hle_a(u_h_i_a,u_h_m_a,u_notBigg_MitoMembrPotent_m_a)-notBigg_J_K_a(u_k_m_a,u_notBigg_MitoMembrPotent_m_a))/CIM)
du[70] = 0
du[71] = 0
du[72] = 0
du[73] = (-(u_ca2_c_a/cai0_ca_ion)*(1+xNEmod*(u[178]/(KdNEmod+u[178])))*ADNCYC_a(u_camp_c_a,u_atp_c_a)+CKc_a(u_adp_c_a,u_atp_c_a,u_pcreat_c_a)+0.5*(1/6.96)*1000*(-notBigg_J_ATP_a(u_atp_i_a,u_atp_c_a))-HEX1_a(u_atp_c_a,u_g6p_c_a,u_glc_D_c_a)-PFK_a(u_atp_c_a,u_f26bp_c_a,u_f6p_c_a)-PFK26_a(u_adp_c_a,u_atp_c_a,u_f26bp_c_a,u_f6p_c_a)+PGK_a(u_adp_c_a,u_atp_c_a,u_3pg_c_a,u_13dpg_c_a)+PYK_a(u_adp_c_a,u_atp_c_a,u_pep_c_a)-0.15*(7/4)*vPumpg-vATPasesg)/(1-dAMPdATPg)
du[74] = 0
du[75] = 0.5*T2Jcorrection*(1000*r0509_a(u_nadh_m_a,u_pi_m_a)/W_x-FUMm_a(u_mal_L_m_a,u_fum_m_a))
du[76] = 0.5*T2Jcorrection*(FUMm_a(u_mal_L_m_a,u_fum_m_a)-MDHm_a(u_nad_m_a,u_oaa_m_a,u_mal_L_m_a,u_nadh_m_a))
du[77] = 0.5*T2Jcorrection*(MDHm_a(u_nad_m_a,u_oaa_m_a,u_mal_L_m_a,u_nadh_m_a)-CSm_a(u_oaa_m_a,u_cit_m_a,u_accoa_m_a,u_coa_m_a)+PCm_a(u_co2_m_a,u_oaa_m_a,u_pyr_m_a,u_atp_m_a,u_adp_m_a))
du[78] = 0.5*T2Jcorrection*(SUCOASm_a(u_coa_m_a,u_succoa_m_a,u_atp_m_a,u_succ_m_a,u_pi_m_a,u_adp_m_a)-1000*r0509_a(u_nadh_m_a,u_pi_m_a)/W_x)
du[79] = 0.5*T2Jcorrection*(AKGDm_a(u_coa_m_a,u_nadh_m_a,u_akg_m_a,u_ca2_m_a,u_nad_m_a,u_succoa_m_a,u_atp_m_a,u_adp_m_a)-SUCOASm_a(u_coa_m_a,u_succoa_m_a,u_atp_m_a,u_succ_m_a,u_pi_m_a,u_adp_m_a))
du[80] = 0.5*T2Jcorrection*(CSm_a(u_oaa_m_a,u_cit_m_a,u_accoa_m_a,u_coa_m_a)-PDHm_a(u_nad_m_a,u_pyr_m_a,u_coa_m_a)-AKGDm_a(u_coa_m_a,u_nadh_m_a,u_akg_m_a,u_ca2_m_a,u_nad_m_a,u_succoa_m_a,u_atp_m_a,u_adp_m_a)+SUCOASm_a(u_coa_m_a,u_succoa_m_a,u_atp_m_a,u_succ_m_a,u_pi_m_a,u_adp_m_a))
du[81] = 0.5*T2Jcorrection*(ICDHxm_a(u_nad_m_a,u_nadh_m_a,u_icit_m_a)-AKGDm_a(u_coa_m_a,u_nadh_m_a,u_akg_m_a,u_ca2_m_a,u_nad_m_a,u_succoa_m_a,u_atp_m_a,u_adp_m_a)+GLUDxm_a(u_nad_m_a,u_nadh_m_a,u_glu_L_c_a,u_akg_m_a))
du[82] = 0
du[83] = 0.5*T2Jcorrection*(ACONTm_a(u_cit_m_a,u_icit_m_a)-ICDHxm_a(u_nad_m_a,u_nadh_m_a,u_icit_m_a))
du[84] = 0.5*T2Jcorrection*(CSm_a(u_oaa_m_a,u_cit_m_a,u_accoa_m_a,u_coa_m_a)-ACONTm_a(u_cit_m_a,u_icit_m_a))
du[85] = 0.5*T2Jcorrection*(PDHm_a(u_nad_m_a,u_pyr_m_a,u_coa_m_a)-CSm_a(u_oaa_m_a,u_cit_m_a,u_accoa_m_a,u_coa_m_a))
du[86] = 0
du[87] = 0
du[88] = 6.96*PYRt2m_a(u_h_m_a,u_pyr_c_a,u_h_c_a,u_pyr_m_a)-T2Jcorrection*(PDHm_a(u_nad_m_a,u_pyr_m_a,u_coa_m_a)+PCm_a(u_co2_m_a,u_oaa_m_a,u_pyr_m_a,u_atp_m_a,u_adp_m_a))
du[89] = T2Jcorrection*(GLNt4_n(u_gln_L_c_n,u_gln_L_e_e)-GLUNm_n(u_gln_L_c_n,u_glu_L_m_n))
du[90] = T2Jcorrection*(-GLNt4_n(u_gln_L_c_n,u_gln_L_e_e)+GLNt4_a(u_gln_L_c_a,u_gln_L_e_e))
du[91] = T2Jcorrection*(-GLNt4_a(u_gln_L_c_a,u_gln_L_e_e)+GLNS_a(u_adp_c_a,u_atp_c_a,u_glu_L_c_a))
du[92] = T2Jcorrection*(0.0266*GLUt6_a(u_na1_c_a,u_notBigg_Va_c_a,u_k_e_e,u_glu_L_syn_syn,u_k_c_a,u_glu_L_c_a)-GLNS_a(u_adp_c_a,u_atp_c_a,u_glu_L_c_a)-GLUDxm_a(u_nad_m_a,u_nadh_m_a,u_glu_L_c_a,u_akg_m_a))
du[93] = 0 # (1/4e-4)*(-dIPump_a-IBK-IKirAS-IKirAV-IleakA-ITRP_a)
du[94] = 0 # vLeakNag-3*vPumpg+vgstim
du[95] = 0 # JNaK_a+(-IleakA-IBK-IKirAS-IKirAV)/(4e-4*843.0*1000.)-RateDecayK_a*(u_k_c_a-u0_ss[95])
du[96] = 0 # SmVn/F*ik_density*(eto_n/eto_ecs)-2*vPumpn*(eto_n/eto_ecs)-2*(eto_a/eto_ecs)*vPumpg-JdiffK-((-IleakA-IBK-IKirAS-IKirAV)/(4e-4*843.0*1000.0))
du[97] = 0
du[98] = 0 # 1/Cm*(-IL-ina_density-ik_density-ICa-ImAHP-dIPump) #+Isyne+Isyni-IM+Iinj)
du[99] = 0 # vLeakNan-3*vPumpn+vnstim
du[100] = 0 # phi*(hinf-u_notBigg_hgate_c_n)/tauh
du[101] = 0 # phi*(ninf-u_notBigg_ngate_c_n)/taun
du[102] = 0 # -SmVn/F*ICa-(u_ca2_c_n-u0_ss[102])/tauCa
du[103] = 0 # phi*(p_inf-u_notBigg_pgate_c_n)/tau_p
du[104] = 0 # psiBK*cosh((u_notBigg_Va_c_a-(-0.5*v5BK*tanh((u_ca2_c_a-Ca3BK)/Ca4BK)+v6BK))/(2*v4BK))*(nBKinf-u_notBigg_nBK_c_a)
du[105] = 0
du[106] = 0 # rhIP3a*((u_notBigg_mGluRboundRatio_c_a+deltaGlutSyn)/(KGlutSyn+u_notBigg_mGluRboundRatio_c_a+deltaGlutSyn))-kdegIP3a*u_notBigg_IP3_c_a
du[107] = 0 # konhIP3Ca_a*(khIP3Ca_aINH-(u_ca2_c_a+khIP3Ca_aINH)*u_notBigg_hIP3Ca_c_a)
du[108] = 0 # beta_Ca_a*(IIP3_a-ICa_pump_a+Ileak_CaER_a)-0.5*ITRP_a/(4e-4*843.0*1000.)
du[109] = 0
du[110] = 0 # (Ca_perivasc/tauTRPCa_perivasc)*(sinfTRPV-u_notBigg_sTRP_c_a)
du[111] = 0 # notBigg_FinDyn_W2017(t)-notBigg_Fout_W2017(t,u_notBigg_vV_b_b)
du[112] = 0 # VprodEET_a*(u_ca2_c_a-CaMinEET_a)-kdeg_EET_a*u_notBigg_EET_c_a
du[113] = notBigg_JdHbin(t,u_o2_b_b)-notBigg_JdHbout(t,u_notBigg_ddHb_b_b)
du[114] = notBigg_JO2art2cap(t,u_o2_b_b)-(eto_n/eto_b)*notBigg_JO2fromCap2n(u_o2_b_b,u_o2_c_n)-(eto_a/eto_b)*notBigg_JO2fromCap2a(u_o2_b_b,u_o2_c_a)
du[115] = notBigg_trGLC_art_cap(t,u_glc_D_b_b)-notBigg_JGlc_be(u_glc_D_ecsEndothelium_ecsEndothelium,u_glc_D_b_b)
du[116] = 0.32*notBigg_JGlc_be(u_glc_D_ecsEndothelium_ecsEndothelium,u_glc_D_b_b)-notBigg_JGlc_e2ecsBA(u_glc_D_ecsBA_ecsBA,u_glc_D_ecsEndothelium_ecsEndothelium)
du[117] = 1.13*notBigg_JGlc_e2ecsBA(u_glc_D_ecsBA_ecsBA,u_glc_D_ecsEndothelium_ecsEndothelium)-notBigg_JGlc_ecsBA2a(u_glc_D_ecsBA_ecsBA,u_glc_D_c_a)-notBigg_JGlc_diffEcs(u_glc_D_ecsBA_ecsBA,u_glc_D_ecsAN_ecsAN)
du[118] = 0.06*notBigg_JGlc_ecsBA2a(u_glc_D_ecsBA_ecsBA,u_glc_D_c_a)-HEX1_a(u_atp_c_a,u_g6p_c_a,u_glc_D_c_a)-notBigg_JGlc_a2ecsAN(u_glc_D_ecsAN_ecsAN,u_glc_D_c_a)
du[119] = 1.35*notBigg_JGlc_a2ecsAN(u_glc_D_ecsAN_ecsAN,u_glc_D_c_a)-notBigg_JGlc_ecsAN2n(u_glc_D_c_n,u_glc_D_ecsAN_ecsAN)+0.08*notBigg_JGlc_diffEcs(u_glc_D_ecsBA_ecsBA,u_glc_D_ecsAN_ecsAN)
du[120] = 0.41*notBigg_JGlc_ecsAN2n(u_glc_D_c_n,u_glc_D_ecsAN_ecsAN)-HEX1_n(u_glc_D_c_n,u_atp_c_n,u_g6p_c_n)
du[121] = HEX1_n(u_glc_D_c_n,u_atp_c_n,u_g6p_c_n)-PGI_n(u_f6p_c_n,u_g6p_c_n)-G6PDH2r_n(u_g6p_c_n,u_nadph_c_n,u_nadp_c_n,u_6pgl_c_n)
du[122] = HEX1_a(u_atp_c_a,u_g6p_c_a,u_glc_D_c_a)-PGI_a(u_g6p_c_a,u_f6p_c_a)-G6PDH2r_a(u_6pgl_c_a,u_nadp_c_a,u_g6p_c_a,u_nadph_c_a)+PGMT_a(u_g1p_c_a,u_g6p_c_a)
du[123] = PGI_n(u_f6p_c_n,u_g6p_c_n)-PFK_n(u_f6p_c_n,u_atp_c_n)-TKT2_n(u_f6p_c_n,u_e4p_c_n,u_xu5p_D_c_n,u_g3p_c_n)+TALA_n(u_f6p_c_n,u_e4p_c_n,u_g3p_c_n,u_s7p_c_n)
du[124] = PGI_a(u_g6p_c_a,u_f6p_c_a)-PFK_a(u_atp_c_a,u_f26bp_c_a,u_f6p_c_a)-PFK26_a(u_adp_c_a,u_atp_c_a,u_f26bp_c_a,u_f6p_c_a)-TKT2_a(u_g3p_c_a,u_e4p_c_a,u_xu5p_D_c_a,u_f6p_c_a)+TALA_a(u_g3p_c_a,u_e4p_c_a,u_s7p_c_a,u_f6p_c_a)
du[125] = PFK_n(u_f6p_c_n,u_atp_c_n)-FBA_n(u_fdp_c_n,u_dhap_c_n,u_g3p_c_n)
du[126] = PFK_a(u_atp_c_a,u_f26bp_c_a,u_f6p_c_a)-FBA_a(u_g3p_c_a,u_fdp_c_a,u_dhap_c_a)
du[127] = PFK26_a(u_adp_c_a,u_atp_c_a,u_f26bp_c_a,u_f6p_c_a)
du[128] = GLCS2_a(u_notBigg_GS_c_a,u_udpg_c_a)-GLCP_a(u_glycogen_c_a,u_notBigg_GPa_c_a,u_camp_c_a,u_notBigg_GPb_c_a)
du[129] = 0
du[130] = 0
du[131] = 10*GLCP_a(u_glycogen_c_a,u_notBigg_GPa_c_a,u_camp_c_a,u_notBigg_GPb_c_a)-PGMT_a(u_g1p_c_a,u_g6p_c_a)-GALUi_a(u_ppi_c_a,u_g1p_c_a,u_udpg_c_a,u_utp_c_a)
du[132] = FBA_n(u_fdp_c_n,u_dhap_c_n,u_g3p_c_n)-GAPD_n(u_nadh_c_n,u_nad_c_n,u_g3p_c_n,u_pi_c_n,u_13dpg_c_n)+TPI_n(u_dhap_c_n,u_g3p_c_n)-TKT2_n(u_f6p_c_n,u_e4p_c_n,u_xu5p_D_c_n,u_g3p_c_n)-TALA_n(u_f6p_c_n,u_e4p_c_n,u_g3p_c_n,u_s7p_c_n)+TKT1_n(u_s7p_c_n,u_r5p_c_n,u_xu5p_D_c_n,u_g3p_c_n)
du[133] = FBA_a(u_g3p_c_a,u_fdp_c_a,u_dhap_c_a)-GAPD_a(u_nadh_c_a,u_nad_c_a,u_pi_c_a,u_13dpg_c_a,u_g3p_c_a)+TPI_a(u_g3p_c_a,u_dhap_c_a)-TKT2_a(u_g3p_c_a,u_e4p_c_a,u_xu5p_D_c_a,u_f6p_c_a)-TALA_a(u_g3p_c_a,u_e4p_c_a,u_s7p_c_a,u_f6p_c_a)+TKT1_a(u_r5p_c_a,u_g3p_c_a,u_s7p_c_a,u_xu5p_D_c_a)
du[134] = FBA_n(u_fdp_c_n,u_dhap_c_n,u_g3p_c_n)-TPI_n(u_dhap_c_n,u_g3p_c_n)
du[135] = FBA_a(u_g3p_c_a,u_fdp_c_a,u_dhap_c_a)-TPI_a(u_g3p_c_a,u_dhap_c_a)
du[136] = GAPD_n(u_nadh_c_n,u_nad_c_n,u_g3p_c_n,u_pi_c_n,u_13dpg_c_n)-PGK_n(u_13dpg_c_n,u_3pg_c_n,u_atp_c_n,u_adp_c_n)
du[137] = GAPD_a(u_nadh_c_a,u_nad_c_a,u_pi_c_a,u_13dpg_c_a,u_g3p_c_a)-PGK_a(u_adp_c_a,u_atp_c_a,u_3pg_c_a,u_13dpg_c_a)
du[138] = GAPD_a(u_nadh_c_a,u_nad_c_a,u_pi_c_a,u_13dpg_c_a,u_g3p_c_a)-LDH_L_a(u_nad_c_a,u_nadh_c_a,u_lac_L_c_a,u_pyr_c_a)-notBigg_vShuttleg(u_nadh_c_a,u_nadh_m_a)
du[139] = 0
du[140] = 0
du[141] = PGK_n(u_13dpg_c_n,u_3pg_c_n,u_atp_c_n,u_adp_c_n)-PGM_n(u_2pg_c_n,u_3pg_c_n)
du[142] = PGK_a(u_adp_c_a,u_atp_c_a,u_3pg_c_a,u_13dpg_c_a)-PGM_a(u_2pg_c_a,u_3pg_c_a)
du[143] = PGM_n(u_2pg_c_n,u_3pg_c_n)-ENO_n(u_pep_c_n,u_2pg_c_n)
du[144] = PGM_a(u_2pg_c_a,u_3pg_c_a)-ENO_a(u_2pg_c_a,u_pep_c_a)
du[145] = ENO_n(u_pep_c_n,u_2pg_c_n)-PYK_n(u_pep_c_n,u_atp_c_n,u_adp_c_n)
du[146] = ENO_a(u_2pg_c_a,u_pep_c_a)-PYK_a(u_adp_c_a,u_atp_c_a,u_pep_c_a)
du[147] = PYK_n(u_pep_c_n,u_atp_c_n,u_adp_c_n)-PYRt2m_n(u_h_m_n,u_pyr_m_n,u_h_c_n,u_pyr_c_n)-LDH_L_n(u_nad_c_n,u_lac_L_c_n,u_nadh_c_n,u_pyr_c_n)
du[148] = PYK_a(u_adp_c_a,u_atp_c_a,u_pep_c_a)-PYRt2m_a(u_h_m_a,u_pyr_c_a,u_h_c_a,u_pyr_m_a)-LDH_L_a(u_nad_c_a,u_nadh_c_a,u_lac_L_c_a,u_pyr_c_a)
du[149] = notBigg_JLacTr_b(u_lac_L_b_b,t)-notBigg_MCT1_LAC_b(u_lac_L_b_b,u_lac_L_e_e)-notBigg_vLACgc(u_lac_L_b_b,u_lac_L_c_a)
du[150] = 0.0275*notBigg_MCT1_LAC_b(u_lac_L_b_b,u_lac_L_e_e)-L_LACt2r_a(u_lac_L_e_e,u_lac_L_c_a)-L_LACt2r_n(u_lac_L_e_e,u_lac_L_c_n)+notBigg_jLacDiff_e(u_lac_L_e_e)
du[151] = 0.8*L_LACt2r_a(u_lac_L_e_e,u_lac_L_c_a)+0.022*notBigg_vLACgc(u_lac_L_b_b,u_lac_L_c_a)+LDH_L_a(u_nad_c_a,u_nadh_c_a,u_lac_L_c_a,u_pyr_c_a)
du[152] = 0.44*L_LACt2r_n(u_lac_L_e_e,u_lac_L_c_n)+LDH_L_n(u_nad_c_n,u_lac_L_c_n,u_nadh_c_n,u_pyr_c_n)
du[153] = G6PDH2r_n(u_g6p_c_n,u_nadph_c_n,u_nadp_c_n,u_6pgl_c_n)+GND_n(u_6pgc_c_n,u_nadph_c_n,u_nadp_c_n,u_ru5p_D_c_n)-notBigg_psiNADPHox_n(u_nadph_c_n)-GTHO_n(u_nadph_c_n,u_gthox_c_n)
du[154] = G6PDH2r_a(u_6pgl_c_a,u_nadp_c_a,u_g6p_c_a,u_nadph_c_a)+GND_a(u_6pgc_c_a,u_nadp_c_a,u_ru5p_D_c_a,u_nadph_c_a)-notBigg_psiNADPHox_a(u_nadph_c_a)-GTHO_a(u_gthox_c_a,u_nadph_c_a)
du[155] = G6PDH2r_n(u_g6p_c_n,u_nadph_c_n,u_nadp_c_n,u_6pgl_c_n)-PGL_n(u_6pgc_c_n,u_6pgl_c_n)
du[156] = G6PDH2r_a(u_6pgl_c_a,u_nadp_c_a,u_g6p_c_a,u_nadph_c_a)-PGL_a(u_6pgc_c_a,u_6pgl_c_a)
du[157] = PGL_n(u_6pgc_c_n,u_6pgl_c_n)-GND_n(u_6pgc_c_n,u_nadph_c_n,u_nadp_c_n,u_ru5p_D_c_n)
du[158] = PGL_a(u_6pgc_c_a,u_6pgl_c_a)-GND_a(u_6pgc_c_a,u_nadp_c_a,u_ru5p_D_c_a,u_nadph_c_a)
du[159] = GND_n(u_6pgc_c_n,u_nadph_c_n,u_nadp_c_n,u_ru5p_D_c_n)-RPE_n(u_xu5p_D_c_n,u_ru5p_D_c_n)-RPI_n(u_r5p_c_n,u_ru5p_D_c_n)
du[160] = GND_a(u_6pgc_c_a,u_nadp_c_a,u_ru5p_D_c_a,u_nadph_c_a)-RPE_a(u_ru5p_D_c_a,u_xu5p_D_c_a)-RPI_a(u_r5p_c_a,u_ru5p_D_c_a)
du[161] = RPI_n(u_r5p_c_n,u_ru5p_D_c_n)-TKT1_n(u_s7p_c_n,u_r5p_c_n,u_xu5p_D_c_n,u_g3p_c_n)
du[162] = RPI_a(u_r5p_c_a,u_ru5p_D_c_a)-TKT1_a(u_r5p_c_a,u_g3p_c_a,u_s7p_c_a,u_xu5p_D_c_a)
du[163] = RPE_n(u_xu5p_D_c_n,u_ru5p_D_c_n)-TKT1_n(u_s7p_c_n,u_r5p_c_n,u_xu5p_D_c_n,u_g3p_c_n)+TKT2_n(u_f6p_c_n,u_e4p_c_n,u_xu5p_D_c_n,u_g3p_c_n)
du[164] = RPE_a(u_ru5p_D_c_a,u_xu5p_D_c_a)-TKT1_a(u_r5p_c_a,u_g3p_c_a,u_s7p_c_a,u_xu5p_D_c_a)+TKT2_a(u_g3p_c_a,u_e4p_c_a,u_xu5p_D_c_a,u_f6p_c_a)
du[165] = TKT1_n(u_s7p_c_n,u_r5p_c_n,u_xu5p_D_c_n,u_g3p_c_n)-TALA_n(u_f6p_c_n,u_e4p_c_n,u_g3p_c_n,u_s7p_c_n)
du[166] = TKT1_a(u_r5p_c_a,u_g3p_c_a,u_s7p_c_a,u_xu5p_D_c_a)-TALA_a(u_g3p_c_a,u_e4p_c_a,u_s7p_c_a,u_f6p_c_a)
du[167] = TKT2_n(u_f6p_c_n,u_e4p_c_n,u_xu5p_D_c_n,u_g3p_c_n)+TALA_n(u_f6p_c_n,u_e4p_c_n,u_g3p_c_n,u_s7p_c_n)
du[168] = TKT2_a(u_g3p_c_a,u_e4p_c_a,u_xu5p_D_c_a,u_f6p_c_a)+TALA_a(u_g3p_c_a,u_e4p_c_a,u_s7p_c_a,u_f6p_c_a)
du[169] = 2*(GTHO_n(u_nadph_c_n,u_gthox_c_n)-GTHP_n(u_gthrd_c_n))+GTHS_n(u_gthrd_c_n)
du[170] = 2*(GTHO_a(u_gthox_c_a,u_nadph_c_a)-GTHP_a(u_gthrd_c_a))+GTHS_a(u_gthrd_c_a)
du[171] = -GTHO_n(u_nadph_c_n,u_gthox_c_n)+GTHP_n(u_gthrd_c_n)
du[172] = -GTHO_a(u_gthox_c_a,u_nadph_c_a)+GTHP_a(u_gthrd_c_a)
du[173] = 0
du[174] = -CKc_n(u_pcreat_c_n,u_atp_c_n,u_adp_c_n)
du[175] = 0
du[176] = -CKc_a(u_adp_c_a,u_atp_c_a,u_pcreat_c_a)
du[177] = (u_ca2_c_a/cai0_ca_ion)*(1+xNEmod*(u[178]/(KdNEmod+u[178])))*ADNCYC_a(u_camp_c_a,u_atp_c_a)-PDE1_a(u_camp_c_a)
du[178] = 0
du[179] = 0
du[180] = 0
du[181] = 0
du[182] = notBigg_psiPHK_a(u_ca2_c_a,u_glycogen_c_a,u_g1p_c_a,u_notBigg_PHKa_c_a,u_notBigg_GPa_c_a,u_udpg_c_a)
du[183] = -notBigg_psiPHK_a(u_ca2_c_a,u_glycogen_c_a,u_g1p_c_a,u_notBigg_PHKa_c_a,u_notBigg_GPa_c_a,u_udpg_c_a)
end
