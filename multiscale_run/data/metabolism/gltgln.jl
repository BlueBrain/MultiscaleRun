# SNAT n: psiSNAT_GLN_n(GLN_out,GLN_n), GLN_out ⇒ GLN_n

TmaxSNAT_GLN_n = 0.07614698345131839 #0.02331661188845118 # < after opt # 0.039  
KmSNAT_GLN_n = 1.1  # Chaudhry 1999
coeff_gln_ratio_n_ecs = 2.5 #1.0506973592641118 # < after opt # 2.5 


# #psiSNAT_GLN_n(GLN_out,GLN_n) = TmaxSNAT_GLN_n*(GLN_out-GLN_n/coeff_gln_ratio_n_ecs)/(KmSNAT_GLN_n+GLN_n)  
# psiSNAT_GLN_n1 = TmaxSNAT_GLN_n*(GLN_out0 - GLN_n0/coeff_gln_ratio_n_ecs)/(KmSNAT_GLN_n+GLN_n0)  

#######################################

# GLS n: psiGLS_n(GLN_n,GLUmito_n), GLN_n ⇒ GLUmito_n

#bigg "psiGLS_n":["GLNtm_n","GLUNm_n","GLUt2m_n"]

VmGLS_n = 330.14406166650446 #0.02921417471855462 # < after opt # 0.01 
KeqGLS_n = 25. #12.0  
KmGLNGLS_n = 12.0  
KiGLUGLS_n = 45.0 


# #psiGLS_n(GLN_n,GLUmito_n) = VmGLS_n*( GLN_n - GLUmito_n/KeqGLS_n )/ (KmGLNGLS_n*(1.0 + GLUmito_n/KiGLUGLS_n) + GLN_n  ) 
# psiGLS_n1 = VmGLS_n*( GLN_n0 - GLUmito_n0/KeqGLS_n )/ (KmGLNGLS_n*(1.0 + GLUmito_n0/KiGLUGLS_n) + GLN_n0  ) 
    

#######################################

# synGlutRelease

glut_vesicle_deltaConc = 0.1 # mM Flanagan2018
coeff_synGlutRelease = 1.5 #0.33

# ! ATTENTION ! v_thr depends on saveat (and tstops)

# # when solving with saveat 1e-4:
# v_thr =  -60. #-2.732

# when solving with saveat 1e-1:
v_thr = 60. #-2.732

#synGlutRelease(V) =  glut_vesicle_deltaConc*exp(-((V+v_thr)/coeff_synGlutRelease)^2) / (coeff_synGlutRelease * 1.772 )


##########################################


# psiEAAT12(Va,Na_a,GLUT_syn,GLUT_a,K_a,K_out),  GLUT_syn ⇒  Va + GLUT_a
# bigg GLUt6(), k_c + glu__L_e + h_e + 3.0 na1_e ⇌ glu__L_c + h_c + 3.0 na1_c + k_e 
# bigg closest rn: GLUt6: k_c + glu__L_e + h_e + 3.0 na1_e ⇌ glu__L_c + h_c + 3.0 na1_c + k_e

# Flanagan 2018

# GLTGLN_r01_EAAT12_a

Na_syn_EAAT = 150.0  # 150 to be same as out #140.0  
H_syn_EAAT = 4e-05  
H_ast_EAAT = 6e-05  
SA_ast_EAAT = 2.8274e-13   # m2
alpha_EAAT = 0.41929231117352916 #0.0011781014781337903 # < after opt # 1e-5 #1e-06  
beta_EAAT = 0.035 #0.0292  
K_ast_EAAT = 100.0  
Vol_syn = 1e-18  
Vol_astEAAT = 3.76e-17

# EAAT1  / GLT-1
#VrevEAAT = (R*T/(2*F))* log( ((Na_syn_EAAT/Na_a)^3) *  (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/GLUT_a)   *   (K_a/K_out )  )

#  VrevEAAT = (R*T/(2*F))* log( clamp(((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))  ,1e-12,  ((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))   ) ) # simplified K
##worked with this before 26may2020 VrevEAAT = (R*T/(2*F))* log( clamp(((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_a/  clamp(K_out,1e-12,K_out))  ,1e-12,  ((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_a/  clamp(K_out,1e-12,K_out))   ) ) # simplified K

#VrevEAAT() = (R*T/(2*F))* log( clamp(((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_a/  clamp(K_out,1e-12,K_out))  ,1e-12,  ((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_a/  clamp(K_out,1e-12,K_out))   ) ) # simplified K
#VEAAT() =  (1/(2*F))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(Va - VrevEAAT)) )  # # Va = Va0
#psiEAAT12() = - ((1/(2*F))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(Va - (R*T/(2*F))* log( clamp(((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))  ,1e-12,  ((Na_syn_EAAT/ clamp(Na_a,1e-12,Na_a)  )^3) * (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/  clamp(GLUT_a,1e-12,GLUT_a)   ) * (K_ast_EAAT/  clamp(K_out,1e-12,K_out))   ) )          )) ) ) / Vol_syn    

#VEAAT(Va,Na_a,GLUT_syn,GLUT_a,K_a,K_out) =  (1/(2*F))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(Va - ((R*T/(2*F))* log( ((Na_syn_EAAT/Na_a)^3) *  (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/GLUT_a)   *   (K_a/K_out )  )))) )  # # Va = Va0

#worked in full sys:
#psiEAAT12(Va,Na_a,GLUT_syn,GLUT_a,K_a,K_out) = - 0.1* ((1/(2*F))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(Va - ((R*T/(2*F))* log( ((Na_syn_EAAT/Na_a)^3) *  (H_syn_EAAT/H_ast_EAAT)  *  (GLUT_syn/GLUT_a)   *   (K_a/K_out )  )))) )) / Vol_syn  #worked with this before 26may2020

# #psiEAAT121 = - ((1/(2*F*1e-3))* SA_ast_EAAT * (  -alpha_EAAT*exp(-beta_EAAT*(Va0 - ((R*T/(2*F*1e-3))*log( ((Na_syn_EAAT/Na_a0)^3) *  (H_syn_EAAT/H_ast_EAAT)  *   (GLUT_syn0/GLUT_a0)   *   (K_a0/K_out0 )  )))) )) 
# psiEAAT121 = - ((1/(2*F*1e-3)) * (  -alpha_EAAT*exp(-beta_EAAT*(Va0 - ((R*T/(2*F*1e-3))*log( ((Na_syn_EAAT/Na_a0)^3) *  (H_syn_EAAT/H_ast_EAAT)  *   (GLUT_syn0/GLUT_a0)   *   (K_a0/K_out0 )  )))) )) 

# #/ Vol_syn  #worked with this before 26may2020

# # check it, should have Vol_syn but Domain error in log if Vol_syn without clamp, but clamp is not easily compatible with Catalyst and not biological


##########################################

# psiGDH_simplif_a(NADmito_a,GLUmito_a,NADHmito_a,AKGmito_a), GLUmito_a + NADmito_a ⇒ AKGmito_a + NADHmito_a
# bigg GDHm glu__L_m + h2o_m + nad_m ⇌ akg_m + h_m + nadh_m + nh4_m
    
# GLTGLN_r02_GDH_a

VmGDH_a = 0.0192685649342926 #0.02  #0.1 #0.02  
KeqGDH_a = 1.5 #1.0 #0.646 #0.34 #0.3 
KiNAD_GDH_a = 1.0  
KmGLU_GDH_a = 3.5      # 0.33 Mezhenska2019
KiAKG_GDH_a = 0.25     # 4.2 Mezhenska2019
KiNADH_GDH_a = 0.004  
KmNADH_GDH_a = 0.04  
KmAKG_GDH_a = 1.1      # 0.36 Mezhenska2019
KiGLU_GDH_a = 3.5      # 9.0  Mezhenska2019

##########################################

# psiGLNsynth_a(GLUT_a,ATP_a,ADP_a),  GLUT_a + ATP_a ⇒ GLN_a + ADP_a

# GLTGLN_r03_GLNsynth_a

VmaxGLNsynth_a = 0.020022679766390446 #0.0013164531825672209 # < after opt #  0.01 #0.039 
KmGLNsynth_a = 2.0 #3.5 #2.0  # 3.5 LISTROM1997
muGLNsynth_a = 0.01  

#Calvetti2011, Pamiljans1961

# bigg GLNS atp_c + glu__L_c + nh4_c ⇌ adp_c + gln__L_c + h_c + pi_c


##########################################

# psiSNAT_GLN_a(GLN_a,GLN_out),  GLN_a ⇒ GLN_out

# GLTGLN_r04_SNAT_a

TmaxSNAT_GLN_a = 0.054730164766604375 #0.039  #0.008 
KmSNAT_GLN_a = 1.1  
coeff_gln_ratio_a_ecs = 1. 

# #SNAT GLN transporter
# psiSNAT_GLN_a1 = TmaxSNAT_GLN_a*(GLN_a0-GLN_out0)/(KmSNAT_GLN_a+GLN_a0) 


# #bigg GLNt4: gln__L_e + na1_e → gln__L_c + na1_c


