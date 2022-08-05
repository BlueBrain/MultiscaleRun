COMMENT
an synaptic current with alpha function conductance defined by
        i = g * (v - e)      i(nanoamps), g(microsiemens);
        where
         g = 0 for t < onset and
         g = gmax * (t - onset)/tau * exp(-(t - onset - tau)/tau)
          for t > onset
this has the property that the maximum value is gmax and occurs at
 t = onset + tau.
 ionic version
ENDCOMMENT
					       
NEURON {
	POINT_PROCESS IonSynapse
	RANGE onset, tau, gmax, e, i
	USEION na READ ena, nai, nao WRITE ina
    USEION k READ ek, ki, ko WRITE ik
    USEION ca READ eca, cai, cao WRITE ica
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
    (molar) = (1/liter)
    (mM) = (millimolar)
    R = (k-mole) (joule/degC)
    FARADAY = (faraday) (coulombs)
}

PARAMETER {
	onset=0 (ms)
	tau=.1 (ms)	<1e-3,1e6>
	gmax=0 	(uS)	<0,1e9>
	e=0	(mV)
    relP_na = 1
    relP_k = 1
    relP_ca = 0.075
}

ASSIGNED { 
    v (mV) 
    erev (mV)
    i (nA)  
    g (uS)
    ina (nA)
    ica (nA)
    ik (nA)
    nai (mM)
    nao (mM)
    ki (mM)
    ko (mM)
    cai (mM)
    cao (mM)
    eca (mV)
    ena (mV)
    ek (mV)
    nafrac
    kfrac
    cafrac
}

INITIAL {
    nafrac = relP_na/(relP_na+relP_k+relP_ca)
    kfrac = relP_k/(relP_na+relP_k+relP_ca)
    cafrac = relP_ca/(relP_na+relP_k+relP_ca)
}

BREAKPOINT {
	if (gmax) { at_time(onset) }
	g = gmax * alpha( (t - onset)/tau )
    findReversalPotential()
    i = g*(v-erev)
	ina = nafrac*g*(v - ena)
	ik = kfrac*g*(v - ek)
    ica = i -ina -ik
}

FUNCTION alpha(x) {
	if (x < 0 || x > 10) {
		alpha = 0
	}else{
		alpha = x * exp(1 - x)
	}
}

PROCEDURE findReversalPotential() {
    erev = (1e3)*R*(celsius+273.15)/FARADAY*log((relP_na*nao+relP_k*ko+relP_ca*cao)/(relP_na*nai+relP_k*ki+relP_ca*cai))
}
