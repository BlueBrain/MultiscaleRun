# psiGSSGR_n(GSSG_n,NADPH_n), GSSG_n + NADPH_n ⇒ 2GSH_n + NADP_n

Vmf_GSSGR_n = 0.006 # 0.296193997692182 # 8.625213760987834e-6 # 0.0001 #0.0025 #kcat_f_GSSGR_n = 210.0 # Vali PMID: 17936517 10.7 # 0.0025 # 8.925 mM/h Reed2008 #Vmax_GSSGR_n #kcat_r_GSSGR_n = 210.0 # Vali PMID: 17936517 #0.12 #Vmr_GSSGR_n 0.000029 #0.00029 # par[382] # 0.0000005 

# KmGSSGRGSSG_n =  0.11112641937846 # 0.0642364727884081 # 0.0652 #PMID: 4018089 0.107 # mM 0.072 mM 0.107 mM # Reed 2008
# KmGSSGRNADPH_n =  0.0100081583265437 # 0.00883424633339055 # 0.00401498478437164 #0.00852 #PMID: 4018089 0.0104 # mM Reed 2008
# # simplified version, Reed2008; Vali PMID: 17936517 
KmGSSGRGSSG_n =  0.0652 #PMID: 4018089 0.107 # mM 0.072 mM 0.107 mM # Reed 2008
KmGSSGRNADPH_n = 0.00852 #PMID: 4018089 0.0104 # mM Reed 2008


############################

# psiGSSGR_a(GSSG_a,NADPH_a), GSSG_a + NADPH_a ⇒  2GSH_a + NADP_a

Vmf_GSSGR_a = 0.003 #0.323856399322304 #0.0025 #kcat_f_GSSGR_a = 210.0 # Vali PMID: 17936517 4.35 #Vmf_GSSGR_a  0.0025 # 8.925 mM/h Reed2008 #Vmax_GSSGR_n #Vmf_GSSGR_a   1.5 #15.27 # par[381]  

#kcat_r_GSSGR_a = 210.0 # Vali PMID: 17936517 0.05 #Vmr_GSSGR_a 0.000029 #0.00029 # par[382] # 0.0000005 
# KmGSSGRGSSG_a = 0.0825167868170385 #0.0652 #PMID: 4018089 #0.107 # mM 0.072 mM 0.107 mM # Reed 2008
# KmGSSGRNADPH_a = 0.019192925530061 #0.00852 #PMID: 4018089 #0.0104 # mM Reed 2008

KmGSSGRGSSG_a =  0.0652 #PMID: 4018089 #0.107 # mM 0.072 mM 0.107 mM # Reed 2008
KmGSSGRNADPH_a = 0.00852 #PMID: 4018089 #0.0104 # mM Reed 2008


# denom_offset_a = 1.73

# # simplified version, Reed2008; Vali PMID: 17936517 
# psiGSSGR_a1 = (Vmf_GSSGR_a*GSSG_a0*NADPH_a0 ) / ( ( KmGSSGRGSSG_a + GSSG_a0 )*( KmGSSGRNADPH_a + NADPH_a0 )  )   # GSSG_a + NADPH_a ⇒  2GSH_a + NADP_a


############################

# Mulukutla2015, Reed2008
#psiGPX_n(GSH_n), 2GSH_n  ⇒ GSSG_n

V_GPX_n = 0.001072378746836681 # 0.00173937423839227 # 0.00001  #0.00125 #kcat_GPX_n = 2.0 # Vali PMID: 17936517 2.7  0.00125 #4.33 # par[380]  # 4.5 mM/h
KmGPXGSH_n =  0.571655480073944 # 0.571655480073944 # 0.5152645026510133 # 1.33 #0.133 # Vali PMID: 17936517 1.33 #mM # Reed2008 
# KmH2O2_n = 0.00009 # mM # can add H2O2 inhib see Reed 2008 for parameters and eq

############################

# Mulukutla2015, Reed2008
#psiGPX_a(GSH_a), 2GSH_a  ⇒  GSSG_a

V_GPX_a = 0.0011730710542436833 #0.0011730407957506474 #0.0011730286476087722 #0.00149823791984372 #0.00125 #kcat_GPX_a = 2.0 # Vali PMID: 17936517 #1.96 #V_GPX_a 0.00125 #4.33 # par[380]  # 4.5 mM/h
KmGPXGSH_a = 1.13224874593032 #1.33 #0.133 # Vali PMID: 17936517 #1.33 #mM # Reed2008 
# KmH2O2_a = 0.00009 # mM # can add H2O2 inhib see Reed 2008 for parameters and eq

#psiGPX_a1 = V_GPX_a * GSH_a0 / (GSH_a0 + KmGPXGSH_a)


VmaxGSHsyn_n = 1.5e-5
KmGSHsyn_n = 0.03 # approx based on Reed2008 

VmaxGSHsyn_a = 1.5e-5
KmGSHsyn_a = 0.03 # approx based on Reed2008 

glycine_n = 10.0 # PMID: 10719892 #0.924   
glycine_a = 2.0 # PMID: 10719892 #0.924  
glutamylCys_n = 0.021375 # 0.021 #0.022 #0.0098
glutamylCys_a = 0.4 #0.022 #0.0098
KeGSHSyn_n = 5.6
KeGSHSyn_a = 5.6
Km_glutamylCys_GSHsyn_n = 0.022
Km_glycine_GSHsyn_n = 0.3
Km_glutamylCys_GSHsyn_a = 0.022
Km_glycine_GSHsyn_a = 0.3

# glutathioneSyntase1(GSH_n) = VmaxGSHsyn_n*GSH_n/(GSH_n + KmGSHsyn_n)
# glutathioneSyntase1(GSH_n0)
