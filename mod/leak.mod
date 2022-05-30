COMMENT
Leak channel currents, Na, K, Cl, suppose gCa_leak = 0
This will shift the reversal potential of Cl so the
sum of ionic leak currents matches corresponding nonspecific pas current
Author: Braeden Benedict @ BBP
Date: 31-July-2017
ENDCOMMENT

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
	(molar) = (1/liter)
	(mM) = (millimolar)
	FARADAY = (faraday) (coulombs)
	R = (k-mole) (joule/degC)
}

NEURON {
	SUFFIX leak
  USEION na READ ena, nai, nao WRITE ina
  USEION k READ ek, ki, ko WRITE ik
	USEION cl READ ecl, cli, clo WRITE icl VALENCE -1
	RANGE ina, ik, i, g, erev, icl
}

PARAMETER {
  :PK:PNa:PC1=1:0.04:0.45 from hodgkin and katz, 1949b
  relP_na = 0.04
  relP_k = 1
  relP_cl = 0.45

  g = 3e-5  (S/cm2)
}

ASSIGNED {
  v   (mV)
  ina (mA/cm2)
  ik  (mA/cm2)
  icl (mA/cm2)
  ena (mV)
  ek  (mV)
  ecl (mV)
	i (mA/cm2)
	nafrac
	kfrac
	clfrac
	ina_init (mA/cm2)
	ik_init (mA/cm2)
	icl_init (mA/cm2)
	i_na (mA/cm2)
	i_k (mA/cm2)
	erev (mV)
	nai (mM)
	nao (mM)
	ki (mM)
	ko (mM)
	cli (mM)
	clo (mM)
}

INITIAL {
	nafrac=relP_na/(relP_na+relP_k+relP_cl)
	kfrac = relP_k/(relP_na+relP_k+relP_cl)
	clfrac = relP_cl/(relP_na+relP_k+relP_cl)
	findReversalPotential()
	i = g*(v-erev)
	ik_init = kfrac * g * (v-ek)
	ina_init = nafrac * g * (v - ena)
	icl_init = i-ina_init - ik_init
}

BREAKPOINT {
	findReversalPotential()
	i = g*(v-erev)
	:We will apply the offset to Cl because it is the least important
	i_k = kfrac * g * (v-ek)
  	i_na = nafrac * g * (v - ena)
	icl = i - i_na - i_k - icl_init
	ina = i_na - ina_init
	ik = i_k - ik_init
}

PROCEDURE findReversalPotential() {
	erev = (1e3)*R*(celsius+273.15)/FARADAY*log((relP_na*nao+relP_k*ko+relP_cl*cli)/(relP_na*nai+relP_k*ki+relP_cl*clo))
}
