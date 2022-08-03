: Intracellular ion accumulation
: Author: Braeden Benedict
: Date: 27-06-2017
: Does not include changing external concentrations or any diffusion

NEURON {
 SUFFIX internal_ions
 RANGE i_Tot
 USEION na READ ina WRITE nai
 USEION k READ ik WRITE ki
 USEION ca READ ica WRITE cai
 USEION cl READ icl WRITE cli VALENCE -1
 RANGE gamma, decay, depth
}

DEFINE Nannuli 2

UNITS {
 (mA) = (milliamp)
 FARADAY = (faraday) (coulombs)
 (molar) = (1/liter)
 (mM) = (millimolar)
 PI = (pi) (1)
 (um) = (micron)
}

ASSIGNED {
  i_cap (mA/cm2)
  ina (mA/cm2)
  ik (mA/cm2)
  ica (mA/cm2)
  icl (mA/cm2)
  diam (micron)
  depth (micron)
  i_Tot (mA/cm2)
  conversionFactor (1/cm/coulombs)
  conversionFactorCa (1/cm/coulombs)
}

PARAMETER {
  DCa = 0.53 (um2/ms)
  DNa = 1.33 (um2/ms)
  DK = 1.96 (um2/ms)
  DCl = 2.03 (um2/ms)
  gamma = 0.05 :percent of free calcium (not buffered)
  : Include 'diffusion' of Ca to inner compartment as exponential decay
  decay = 80 (ms) : rate of removal of calcium, taken from CaDynamics_E24
  cai_base = 0.00005 (mM)
}

STATE {
  nai (mM)
  ki (mM)
  cai (mM)
  cli (mM)
}

LOCAL volin, volshell, surf

INITIAL {
  depth = 0.1 :diam/4/(Nannuli-1)
  volin = PI*diam*diam/4 : Surface area (volume per unit length)
  volshell = PI*((diam*diam) - (diam-depth)*(diam-depth))/4 : Surface area ca-shell
  surf = PI*diam : circumference (segment surface per unit length)
  conversionFactor = (1e4)*4/diam/FARADAY
  conversionFactorCa = gamma*(1e4)/(2*FARADAY*depth)
  :Ca dynamics have been implemented within a shell rather than the entire volume
  :The values assigned to gamma and depth in CaDynamics_BB must also be assigned here
}

BREAKPOINT {
  SOLVE state METHOD sparse
  if ( nai <= 0 ) {
    nai = 0.01
  }
  i_Tot = ina + ik + ica + icl :Sum of all ionic currents, for user's convenience
}

KINETIC state {
 COMPARTMENT volin {nai ki cli}
 LONGITUDINAL_DIFFUSION DNa*volin {nai}
 LONGITUDINAL_DIFFUSION DK*volin {ki}
 LONGITUDINAL_DIFFUSION DCl*volin {cli}
 ~ nai << (-ina * conversionFactor*volin)
 ~ ki << (-ik  * conversionFactor*volin)
 ~ cli << (icl * conversionFactor*volin)
 COMPARTMENT volshell {cai}
 LONGITUDINAL_DIFFUSION DCa*volshell {cai}
 ~ cai << ((-ica * conversionFactorCa - (cai - cai_base)/decay)*volshell)
}
