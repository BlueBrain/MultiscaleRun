COMMENT
TITLE K-Cl cotransporter KCC2
Modified from  Gentiletti D et al (2016) Int J Neural Syst
Author: Braeden Benedict @BBP
10-July-2017
ENDCOMMENT

NEURON {
	SUFFIX kcc2
	USEION k READ ko, ki WRITE ik
	USEION cl READ clo, cli WRITE icl VALENCE -1
	RANGE ik, icl, rate, ratio
	RANGE gKCC
}

UNITS {
	(mV)	= (millivolt)
	(molar) = (1/liter)
	(mM)	= (millimolar)
	(mA)	= (milliamp)
	(mol)	= (1)
	(S) = (siemens)
  FARADAY = (faraday) (coulombs)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	gKCC = 7e-5 (S/cm2) :KCC conductivity (from Dronne et al. 2006)

}

ASSIGNED {
	ik		(mA/cm2)
	icl		(mA/cm2)
	ko		(mM)
	ki		(mM)
  clo   (mM)
  cli   (mM)

	ik_init (mA/cm2)
	icl_init (mA/cm2)
	rate (mA/cm2)
	ratio
	celsius (degC)
}

INITIAL {
	ratio = (ki*cli)/(ko*clo)
	rate = (1000)*gKCC*R*(273.15+celsius)*log(ratio)/FARADAY :U*log(ratio)*(FARADAY*diam/4)*(1e-4)
	ik_init =  rate
	icl_init = -rate
}

BREAKPOINT {
	ratio = (ki*cli)/(ko*clo)
	rate =  (1000)*gKCC*R*(273.15+celsius)*log(ratio)/FARADAY :U*log(ratio)*(FARADAY*diam/4)*(1e-4) <- changed this model to include conductance per unit area
	ik =  rate - ik_init
	icl = -rate - icl_init
}
