:Reference :Colbert and Pan 2002
:TTX blocking mechanism has been removed
:comment: took the NaTa and shifted both activation/inactivation by 6 mv
:Modified by Braeden Benedict to include K currents 26-July-2017
:GHK equation doesn't fit well here so a linear approximation is used for ionic currents
:Permeabilities from Hille, 1972

NEURON	{
	SUFFIX NaTs2_t
	USEION na READ nai, nao, ena WRITE ina
	USEION k READ ki, ko, ek WRITE ik
	RANGE gNaTs2_tbar, gNaTs2_t, ina, ik, i, erev
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
	gNaTs2_tbar = 0.00001 (S/cm2)
	relP_na = 1
	relP_k = 0.08514 :Includes activity coefficient of 0.99. 0.08514 = 0.086 * 0.99
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	ik	(mA/cm2)
	i		(mA/cm2) :original current value
	i_sum	(mA/cm2)
	celsius   (degC) :Needs to be changed by user if user does not want the default value
	nai (mM)
	nao (mM)
	ki  (mM)
	ko  (mM)
	erev (mV)
	ek (mV)
	gNaTs2_t	(S/cm2)
	mInf
	mTau	(ms)
	mAlpha
	mBeta
	hInf
	hTau	(ms)
	hAlpha
	hBeta
	kfrac
	nafrac

	ik_init (mA/cm2)
	ina_init (mA/cm2)
	gNaTs2_t0 (S/cm2)
	i_na (mA/cm2)
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gNaTs2_t = gNaTs2_tbar*m*m*m*h
	findReversalPotential()
	i = gNaTs2_t*(v-erev)
	i_na = nafrac*gNaTs2_t*(v-ena)
	ina = i_na - ina_init
	ik = 	i - i_na - ik_init
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

	gNaTs2_t0 = gNaTs2_tbar*m*m*m*h
	i = gNaTs2_t0*(v-erev)
	ina_init = nafrac*gNaTs2_t0*(v-ena)
	ik_init = i - ina_init
}

PROCEDURE rates(){
  LOCAL qt
  qt = 2.3^((celsius-21)/10)

	UNITSOFF
    if(v == -32){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- -32))/(1-(exp(-(v- -32)/6)))
		mBeta  = (0.124 * (-v -32))/(1-(exp(-(-v -32)/6)))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt

    if(v == -60){
      v = v + 0.0001
    }
		hAlpha = (-0.015 * (v- -60))/(1-(exp((v- -60)/6)))
		hBeta  = (-0.015 * (-v -60))/(1-(exp((-v -60)/6)))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
}

PROCEDURE findReversalPotential() {
	erev = (1e3)*R*(celsius+273.15)/FARADAY*log((relP_na*nao+relP_k*ko)/(relP_na*nai+relP_k*ki))
}
