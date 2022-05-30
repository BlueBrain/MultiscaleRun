: Chapman JB, Johnson EA, Kootsey JM. (1983)
: Electrical and Biochemical Properties of an Enzyme Model of the Sodium Pump
: J. Membrane Biol. 74, 139-153

: note default step 5 voltage dependence

: extended by Michael Hines as a component of larger models.
: I.e. modifies nai, ki, contributes to ina, ik, and consumes atp
: for investigation of isolated pump, allow clamping of
: nai, ki, atp (note p and adp are constant here)
: initialize to steady state pump with nai, ki, atp clamped.

:Modified by Michiel Camps to integrate in BBP cell types

NEURON {
	SUFFIX nakpump
	USEION na READ nai, nao WRITE ina
	USEION k READ ki, ko WRITE ik
	USEION atp READ atpi WRITE atpi VALENCE -3
	USEION adp READ adpi WRITE adpi VALENCE -2
	USEION p READ pi WRITE pi VALENCE -3
	RANGE inapump, ikpump
	RANGE atpact
	RANGE atpi, adpi, pi
	RANGE ina, ik, totalpump
        RANGE srcrate
        RANGE atpi_clamp, adpi_clamp,pi_clamp :Can be set experimentally, assigned the initial concentrations else
}

UNITS {
	(l) = (liter)
	(mol) = (1)
	(mmol) = (millimol)
	(mM) = (mmol/l)
	(uA) = (microamp)
	(mA) = (milliamp)
	(mV) = (millivolt)
	(um) = (micron)
	F = (faraday)  (kilocoulombs)
	R = (k-mole) (joule/degC)
	PI = (pi) (1)
}

PARAMETER {

	totalpump = 4e-13 (mol/cm2) :1.25e-13 (mol/cm2)

	f1 = 2.5e11 (l3/mol3-s)
	b1 = 1e5 (/s)
	f2 = 1e4 (/s)
	b2 = 1e5 (l/mol-s)
	f3 = 172 (/s)
	b3 = 1.72e4 (l3/mol3-s)
	f4 = 1.5e7 (l2/mol2-s)
	b4 = 2e5 (l/mol-s)
	f5 = 2e6 (l/mol-s)
	b5 = 30 (/s)
	f6 = 1.15e4 (/s)
	b6 = 6e8 (l2/mol2-s)
}

ASSIGNED {
	diam (um)
	v (mV)
	ina (mA/cm2)
	ik (mA/cm2)
	nai (mM)
	nao (mM)
	ki (mM)
	ko (mM)
	inapump (mA/cm2)
	ikpump (mA/cm2)
	atpact (uA/cm2)

	srcrate (/s)

	:compensation currents
	ik_init (mA/cm2)
	ina_init (mA/cm2)
        adpi_clamp (mM)
        atpi_clamp (mM)
        pi_clamp (mM)
}

STATE {
	eatp (mol/cm2)
	na3eatp (mol/cm2)
	na3ep (mol/cm2)
	ep (mol/cm2)
	k2e (mol/cm2)
	k2eatp (mol/cm2)
	atpi (mM)
	adpi (mM)
	pi (mM)
}

LOCAL volin, volout, surf

INITIAL {
	atpi_clamp = atpi
	adpi_clamp = adpi
	pi_clamp = pi
	volin = PI*diam*diam/4 : cross section area
	volout = 1 (um2)
	surf = PI*diam*(1e7) : circumference

	:clamp atp, adp, p to initial values
	srcrate = 10e15
	FROM i = 1 TO 1 {
		SOLVE scheme STEADYSTATE sparse
	}

	: set to 0 to unclamp atp, adp, p
	srcrate = 0

	SOLVE scheme METHOD sparse
	ina_init = 3*atpact*(1e-3)
	ik_init = -2*atpact*(1e-3)
}

BREAKPOINT {
	SOLVE scheme METHOD sparse

	inapump = 3*atpact*(1e-3)
	ikpump = -2*atpact*(1e-3)
	:Flux in both directions is allowed, but should be very unlikely
	ina = out_rect(inapump - ina_init)
	ik = in_rect(ikpump - ik_init)
}

KINETIC scheme {
	LOCAL x
	x = F/surf*(1e9)

	COMPARTMENT volin { nai ki atpi adpi pi }
	COMPARTMENT volout { nao ko }
	COMPARTMENT surf*(1e3) { eatp na3eatp na3ep ep k2e k2eatp }
	~ eatp + 3 nai <-> na3eatp	(f1*surf*(1e-9), b1*surf*(1e0))
	~ na3eatp <-> na3ep + adpi	(f2*surf*(1e0), b2*surf*(1e-3))
	~ na3ep <-> ep + 3 nao	(f3*surf*(1e0), b3*surf*(1e-9))
	~ ep + 2 ko <-> k2e + pi	(f4*surf*(1e-6), b4*surf*(1e-3))
	~ k2e + atpi <-> k2eatp		(f5*surf*(1e-3), b5*surf*(1e0))
	~ k2eatp <-> eatp + 2 ki	(f6*surf*(1e0), b6*surf*(1e-6))
	atpact = (f_flux - b_flux)*x
	CONSERVE eatp+na3eatp+na3ep+ep+k2e+k2eatp = totalpump*surf*(1e3)

:Clamping of nucleotide concentrations
	COMPARTMENT volin {atpi_clamp adpi_clamp pi_clamp}
	~ atpi_clamp <-> atpi (srcrate,srcrate)
	~ adpi_clamp <-> adpi (srcrate, srcrate)
	~ pi_clamp <-> pi (srcrate, srcrate)
}

FUNCTION in_rect(x) {
    if ( x > 0 ) {
        in_rect = 0
    }   
    else {
        in_rect = x
    }
}
FUNCTION out_rect(x) {
    if ( x < 0 ) {
        out_rect = 0
    }   
    else {
        out_rect = x
    }
}
