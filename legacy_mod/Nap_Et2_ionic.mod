:Comment : mtau deduced from text (said to be 6 times faster than for NaTa)
:TTX blocking mechanism has been removed
:Comment : so I used the equations from NaT and multiplied by 6
:Reference : Modeled according to kinetics derived from Magistretti & Alonso 1999
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21
:Modified by Braeden Benedict to include K currents 26-July-2017
:GHK equation doesn't fit well here so a linear approximation is used for ionic currents
:Permeabilities from Hille, 1972


NEURON	{
	SUFFIX Nap_Et2
	USEION na READ nai, nao, ena WRITE ina
	USEION k READ ki, ko, ek WRITE ik
	RANGE gNap_Et2bar, gNap_Et2, ina, ik, i, erev
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(molar) = (1/liter)
	(mM) = (millimolar)
	FARADAY = (faraday) (coulombs)
	R = (k-mole) (joule/degC)
}

PARAMETER	{
	gNap_Et2bar = 0.00001 (S/cm2)
	relP_na = 1
	relP_k = 0.08514 :Includes activity coefficient of 0.99. 0.08514 = 0.086 * 0.99
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	ik	(mA/cm2)
	i		(mA/cm2) :original current value
	celsius   (degC) :Needs to be changed by user if user does not want the default value
	nai (mM)
	nao (mM)
	ki  (mM)
	ko  (mM)
	erev (mV)
	ek (mV)
	gNap_Et2	(S/cm2)
	mInf
	mTau	(ms)
	mAlpha
	mBeta
	hInf
	hTau	(ms)
	hAlpha
	hBeta
	nafrac
	ik_init (mA/cm2)
	ina_init (mA/cm2)
	gNap_Et20 (S/cm2)
	i_na (mA/cm2)
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNap_Et2 = gNap_Et2bar*m*m*m*h
	findReversalPotential()
	i = gNap_Et2*(v-erev)
	i_na = nafrac*gNap_Et2*(v-ena)
	ik = 	i - i_na - ik_init
	ina = i_na - ina_init
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
	findReversalPotential()
	nafrac = relP_na/(relP_na+relP_k)

	gNap_Et20 = gNap_Et2bar*m*m*m*h
	i = gNap_Et20*(v-erev)
	ina_init = nafrac*gNap_Et20*(v-ena)
	ik_init = i - ina_init
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((celsius-21)/10)

	UNITSOFF
		mInf = 1.0/(1+exp((v- -52.6)/-4.6))
    if(v == -38){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- -38))/(1-(exp(-(v- -38)/6)))
		mBeta  = (0.124 * (-v -38))/(1-(exp(-(-v -38)/6)))
		mTau = 6*(1/(mAlpha + mBeta))/qt

  	if(v == -17){
   		v = v + 0.0001
  	}
    if(v == -64.4){
      v = v+0.0001
    }

		hInf = 1.0/(1+exp((v- -48.8)/10))
    hAlpha = -2.88e-6 * (v + 17) / (1 - exp((v + 17)/4.63))
    hBeta = 6.94e-6 * (v + 64.4) / (1 - exp(-(v + 64.4)/2.63))
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
}

PROCEDURE findReversalPotential() {
	erev = (1e3)*R*(celsius+273.15)/FARADAY*log((relP_na*nao+relP_k*ko)/(relP_na*nai+relP_k*ki))
}
