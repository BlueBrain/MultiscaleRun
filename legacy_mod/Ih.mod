COMMENT
Reference: Kole,Hallermann,and Stuart, J. Neurosci. 2006
Modified by Braeden Benedict July-28-17 to specify ionic currents, implemented using a linear approximation.
Reference: Solomon JS, Nerbonne JM. The Journal of Physiology. 1993.
Model does not include the reported dependency on extracellular K concentration
It has been reported that external K concentration affects permeability ratio and zero-current conductance. (Hestrin, 1986), these are not implemented here
ENDCOMMENT

NEURON	{
	SUFFIX Ih
	RANGE gIhbar, gIh, i, erev
	RANGE ik, ina, i_sum
	RANGE ena, ek
	USEION na READ ena WRITE ina
	USEION k READ ek WRITE ik
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
	gIhbar = 0.00001 (S/cm2)
	erev = -45.0 (mV) 	:Overall reversal potential, value found in Kole, could be relooked at.
	relP_na= 0.4 				:From Solomon and Nerbonne. May be dependent on external K concentration.
	relP_k = 1
}

ASSIGNED	{
	v	(mV)
	i	(mA/cm2)
	gIh	(S/cm2)
	mInf
	mTau	(ms)
	mAlpha
	mBeta
	ina      (mA/cm2)
	ik       (mA/cm2)
	i_sum     (mA/cm2)
	nai (mM)
	nao (mM)
	ki  (mM)
	ko  (mM)
  ena (mV)
  ek  (mV)
	nafrac
	kfrac
	i_k (mA/cm2)
	celsius (degC)

	ik_init (mA/cm2)
	ina_init (mA/cm2)
	gIh0 (S/cm2)
}

STATE	{
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gIh = gIhbar*m
	i = gIh*(v-erev) :Original value, should equal i_sum
	i_k = kfrac*gIh*(v-ek)
  	ina = i - i_k - ina_init
	ik = i_k - ik_init
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
	kfrac = relP_k/(relP_na+relP_k)
	gIh0 = gIhbar*m
	i = gIh0*(v-erev)
	ik_init = kfrac*gIh0*(v-ek)
	ina_init = i-ik_init
}

PROCEDURE rates(){
	UNITSOFF
        if(v == -154.9){
            v = v + 0.0001
        }
		mAlpha =  0.001*6.43*(v+154.9)/(exp((v+154.9)/11.9)-1)
		mBeta  =  0.001*193*exp(v/33.1)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
