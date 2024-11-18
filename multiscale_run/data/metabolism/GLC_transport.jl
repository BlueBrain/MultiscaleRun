C_Glc_a = 4.6 #C_Glc_a = 4.65 #4.75 #5.2 #5.25 #5.5 # DOI 10.1002/glia # 4.75 in Jolivet # 4.8 Winter2017Copasi  #5.5 #Glc_b0 + 0.5    #Glc_b0 + 0.8 # aprox from Calvetti and Nehlig1992  # check it 

# concentration_enzyme_transporter_GLUT1_cap = 0.000208/5 # /5 because five-fold more carriers in parenchymal cells than in endothelium  DOI 10.1002/glia.20375
# kcat_TbGlc = 1149.0*5 # calc from TbGlc 0.239 Jolivet and concGlut1, *5 because /5 in conc because five-fold more carriers in parenchymal cells than in endothelium  DOI 10.1002/glia.20375

# # TmaxBBB = 0.023 mM/s # DOI 10.1002/glia 
# # KztBBB = 5 mM # DOI 10.1002/glia 

# KbGlc_b = 8.0 # GLUT1 on BBB Ronowska2018  Simpson 2007  #4.6 #0.6 #5.0 #4.60 # mM # Km GLC  1-2 mM for GLC influx; Km GLC is 20-30 mM for GLC efflux # Wayne Alberts Basic Neurochemistry 8th edition 2012


# DiNuzzo2010 DiNuzzo2010_1.pdf

# blood -> endoth
TmaxGLCce = 2.21 #0.239 #<JLV 2.2 #2.55 #3.0 #2.5 #5. #1.0 #2.0 #1.2 # 0.4 #0.3 #0.15 #0.1 # ok #0.05 #0.023 #5.67 in DiNuzzo2010
KeG = 10.3 #luT1Barros2007 #8. #10. in DiNuzzo2010 # 8 in Barros2007
ReGoi = 1.
ReGio = 1.
ReGee = 1.

# endoth -> ecsBA
TmaxGLCeb = 20.0 #0.239 #10.0 #4.0 #3.5 #1.5 #0.8 # 1.2 #1.5 #0.6 #0.3 #ok with TmaxGLCce = 0.1 #0.2 # ok w/o diff #0.25 #0.023 #0.2 #0.085 #0.1 #0.026 # estim #6.41 in DiNuzzo2010
KeG2 = 12.5 #ablBarros2007 #8. #10. in DiNuzzo2010 # 8 in Barros2007
ReGoi2 = 1.
ReGio2 = 1.
ReGee2 = 1.


# ecsBA -> a
TmaxGLCba = 8. #2.45 #0.147 #<JLV 20.0 #12.0 #0.1 #0.08 #5.5 #2.5 #1.0 #0.5 #0.4 #0.2 # ok #0.6 #0.34 #0.023 #0.35 #0.35 #0.2 #0.147 #Jlv2015 #0.08 in DiNuzzo2010
KeG3 = 8. #10. in DiNuzzo2010 # 8 in Barros2007
ReGoi3 = 1.
ReGio3 = 0.73
ReGee3 = 0.73


# a -> ecsAN
TmaxGLCai = 0.032 #1.0 #0.3 #0.2 #0.4 #0.6 #0.023 #0.1 #0.147 #0.14 in DiNuzzo2010
KeG4 = 8. #10. in DiNuzzo2010 # 8 in Barros2007
ReGoi4 = 1.
ReGio4 = 1.36
ReGee4 = 1.36


# ecsAN -> n
TmaxGLCin = 0.4 # 0.1 #0.041 #<JLV #0.5 #1.0 #1.1 #1.2 #0.6 #0.023 #0.35 #0.041 #0.58 in DiNuzzo2010
KeG5 = 2.8 #4. #DiNuzzo2010 # 2.8 in Barros2007
ReGoi5 = 1.
ReGio5 = 0.72
ReGee5 = 0.72

# diffusion
kGLCdiff = 0.29 #0.2875 #0.023 #0.03 #0.01 #0.023

# # derived from Barros2007
# KztINb2endo = 5.
# KztINendo2eBA = 5.
# KztINeBA2a = 5.
# KztINa2ecsAN = 5.
# KztINecsAN2n = 5.
# Kitb2endo = 21.
# Kitendo2eBA = 21.
# KitecsBA2a = 21.
# Kita2ecsAN = 21.
# KitecsAN2n = 21.
