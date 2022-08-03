:Na-K-Cl cotransporter

NEURON {
  SUFFIX nakcc
  USEION na READ nai, nao WRITE ina
  USEION k READ ki, ko WRITE ik
  USEION cl READ cli, clo WRITE icl VALENCE -1
  RANGE gNKCC, rate, ratio
}

UNITS {
  (mV) = (millivolts)
  (mA) = (milliamp)
  (S) = (siemens)
  (molar) = (1/liter)
  (mM) = (millimolar)
  R = (k-mole) (joule/degC)
  FARADAY = (faraday) (coulombs)
}

PARAMETER {
  gNKCC = 2e-6 (S/cm2)
}

ASSIGNED {
  ina   (mA/cm2)
  ik		(mA/cm2)
  icl		(mA/cm2)
  nao   (mM)
  nai   (mM)
  ko		(mM)
  ki		(mM)
  clo   (mM)
  cli   (mM)

  rate (mA/cm2)
  ratio
  celsius (degC)
  ina_init (mA/cm2)
  ik_init (mA/cm2)
  icl_init (mA/cm2)
}

INITIAL {
  ratio = (ki*nai*cli*cli)/(ko*nao*clo*clo)
  rate = gNKCC*R*(273.15+celsius)*log(ratio)/FARADAY
  ik_init = rate
  ina_init = rate
  icl_init = -2*rate
}

BREAKPOINT {
  ratio = (ki*nai*cli*cli)/(ko*nao*clo*clo)
  rate = gNKCC*R*(273.15+celsius)*log(ratio)/FARADAY
  ik = rate - ik_init
  ina = rate - ina_init
  icl = -2*rate - icl_init
}
