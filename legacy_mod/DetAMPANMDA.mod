COMMENT
/**
 * @file DetAMPANMDA.mod
 * @brief Adapted from ProbAMPANMDA_EMS.mod by Eilif, Michael and Srikanth
 * @author chindemi
 * @date 2014-05-25
 * @remark Copyright © BBP/EPFL 2005-2014; All rights reserved. Do not distribute without further notice.
 */
ENDCOMMENT


TITLE AMPA and NMDA receptor with presynaptic short-term plasticity


COMMENT
AMPA and NMDA receptor conductance using a dual-exponential profile
presynaptic short-term plasticity based on Fuhrmann et al. 2002, deterministic
version.
ENDCOMMENT


NEURON {
    THREADSAFE

    POINT_PROCESS DetAMPANMDA_ionic
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA
    RANGE Use, u, Dep, Fac, u0, mg, NMDA_ratio
    RANGE i, i_AMPA, i_NMDA, g_AMPA, g_NMDA
    RANGE ina, ik, ica, ina_NMDA, ik_NMDA, ica_NMDA, ina_AMPA, ik_AMPA, ica_AMPA, i_sum
    RANGE absPerm_na_NMDA, absPerm_na_AMPA, e_NMDA, e_AMPA
    RANGE nai,nao,ki,ko,cai,cao
    USEION na READ ena, nai, nao WRITE ina
    USEION k READ ek, ki, ko WRITE ik
    USEION ca READ eca, cai, cao WRITE ica
    RANGE synapseID, verboseLevel
}


PARAMETER {
    tau_r_AMPA = 0.2   (ms)  : Dual-exponential conductance profile
    tau_d_AMPA = 1.7   (ms)  : IMPORTANT: tau_r < tau_d
    tau_r_NMDA = 0.29  (ms)  : Dual-exponential conductance profile
    tau_d_NMDA = 43    (ms)  : IMPORTANT: tau_r < tau_d
    Use = 1.0          (1)   : Utilization of synaptic efficacy
    Dep = 100          (ms)  : Relaxation time constant from depression
    Fac = 10           (ms)  : Relaxation time constant from facilitation
    e = 0              (mV)  : AMPA and NMDA reversal potential
    mg = 1             (mM)  : Initial concentration of mg2+
    gmax = .001        (uS)  : Weight conversion factor (from nS to uS)
    u0 = 0                   : Initial value of u, which is the running value of Use
    NMDA_ratio = 0.71  (1)   : The ratio of NMDA to AMPA
    synapseID = 0
    verboseLevel = 0
    relP_na_NMDA = 1
    relP_k_NMDA = 1
    relP_ca_NMDA = 0.075
    relP_na_AMPA = 1
    relP_k_AMPA = 1
    relP_ca_AMPA = 0.00118
    e_NMDA = 0 (mV)  :NMDA reversal potential
    e_AMPA = 0 (mV)  :AMPA reversal potential
}


ASSIGNED {
    v (mV)
    i (nA)
    i_AMPA (nA)
    i_NMDA (nA)
    g_AMPA (uS)
    g_NMDA (uS)
    g (uS)
    factor_AMPA
    factor_NMDA
    mggate

    ina      (nA)
    ik       (nA)
    ica       (nA)
    ina_NMDA  (nA)
    ik_NMDA   (nA)
    ica_NMDA  (nA)
    ina_AMPA  (nA)
    ik_AMPA   (nA)
    ica_AMPA  (nA)
    i_sum     (nA)
    celsius   (degC) :Needs to be changed by user if user does not want the default value
    ena (mV)
    ek  (mV)
    eca (mV)
    nai (mM)
    nao (mM)
    ki  (mM)
    ko  (mM)
    cai (mM)
    cao (mM)
    nafrac_AMPA
    kfrac_AMPA
    cafrac_AMPA
    nafrac_NMDA
    kfrac_NMDA
    cafrac_NMDA
}

STATE {
    A_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_r_AMPA
    B_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_d_AMPA
    A_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_r_NMDA
    B_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_d_NMDA
}


INITIAL{
    LOCAL tp_AMPA, tp_NMDA

    A_AMPA = 0
    B_AMPA = 0

    A_NMDA = 0
    B_NMDA = 0

    tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA) :time to peak of the conductance
    tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA) :time to peak of the conductance

    factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA) :AMPA Normalization factor - so that when t = tp_AMPA, gsyn = gpeak
    factor_AMPA = 1/factor_AMPA

    factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA) :NMDA Normalization factor - so that when t = tp_NMDA, gsyn = gpeak
    factor_NMDA = 1/factor_NMDA

    nafrac_AMPA=relP_na_AMPA/(relP_na_AMPA+relP_k_AMPA+relP_ca_AMPA)
    kfrac_AMPA=relP_k_AMPA/(relP_na_AMPA+relP_k_AMPA+relP_ca_AMPA)
    cafrac_AMPA=relP_ca_AMPA/(relP_na_AMPA+relP_k_AMPA+relP_ca_AMPA)
    nafrac_NMDA=relP_na_NMDA/(relP_na_NMDA+relP_k_NMDA+relP_ca_NMDA)
    kfrac_NMDA=relP_k_NMDA/(relP_na_NMDA+relP_k_NMDA+relP_ca_NMDA)
    cafrac_NMDA=relP_ca_NMDA/(relP_na_NMDA+relP_k_NMDA+relP_ca_NMDA)
}


BREAKPOINT {
    SOLVE state METHOD cnexp
    mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 3.57 (mM))) :mggate kinetics - Jahr & Stevens 1990
    g_AMPA = gmax*(B_AMPA-A_AMPA) :compute time varying conductance as the difference of state variables B_AMPA and A_AMPA
    g_NMDA = gmax*(B_NMDA-A_NMDA) * mggate :compute time varying conductance as the difference of state variables B_NMDA and A_NMDA and mggate kinetics
    g = g_AMPA + g_NMDA
    i_AMPA = g_AMPA*(v-e_AMPA) :compute the AMPA driving force based on the time varying conductance, membrane potential, and AMPA reversal
    i_NMDA = g_NMDA*(v-e_NMDA) :compute the NMDA driving force based on the time varying conductance, membrane potential, and NMDA reversal
    i = i_AMPA + i_NMDA

    ina_NMDA = nafrac_NMDA*g_NMDA*(v-ena+findOffsetNa(relP_na_NMDA,relP_k_NMDA,relP_ca_NMDA,e_NMDA))
    ik_NMDA = kfrac_NMDA*g_NMDA*(v-ek)
    ica_NMDA = cafrac_NMDA*g_NMDA*(v-eca)

    ina_AMPA = nafrac_AMPA*g_AMPA*(v-ena+findOffsetNa(relP_na_AMPA,relP_k_AMPA,relP_ca_AMPA,e_AMPA))
    ik_AMPA = kfrac_AMPA*g_AMPA*(v-ek)
    ica_AMPA = cafrac_AMPA*g_AMPA*(v-eca)

    ina = ina_NMDA + ina_AMPA
    ik = ik_NMDA + ik_AMPA
    ica = ica_NMDA + ica_AMPA
}


DERIVATIVE state{
    A_AMPA' = -A_AMPA/tau_r_AMPA
    B_AMPA' = -B_AMPA/tau_d_AMPA
    A_NMDA' = -A_NMDA/tau_r_NMDA
    B_NMDA' = -B_NMDA/tau_d_NMDA
}


NET_RECEIVE (weight,weight_AMPA, weight_NMDA, R, Pr, u, tsyn (ms)){
    LOCAL result
    weight_AMPA = weight
    weight_NMDA = weight * NMDA_ratio

    INITIAL{
            R=1
            u=u0
            tsyn=t
    }

    : Disable in case of t < 0 (in case of ForwardSkip) which causes numerical
    : instability if synapses are activated.
    if(t < 0 ) {
    VERBATIM
        return;
    ENDVERBATIM
    }

    : calc u at event-
    if (Fac > 0) {
        u = u*exp(-(t - tsyn)/Fac) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
    } else {
        u = Use
    }
    if(Fac > 0){
        u = u + Use*(1-u) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
    }

    R  = 1 - (1-R) * exp(-(t-tsyn)/Dep) :Probability R for a vesicle to be available for release, analogous to the pool of synaptic
                                        :resources available for release in the deterministic model. Eq. 3 in Fuhrmann et al.
    Pr  = u * R                         :Pr is calculated as R * u (running value of Use)
    R  = R - u * R                      :update R as per Eq. 3 in Fuhrmann et al.

    if( verboseLevel > 0 ) {
        printf("Synapse %f at time %g: R = %g Pr = %g erand = %g\n", synapseID, t, R, Pr, result )
    }

    tsyn = t

    A_AMPA = A_AMPA + Pr*weight_AMPA*factor_AMPA
    B_AMPA = B_AMPA + Pr*weight_AMPA*factor_AMPA
    A_NMDA = A_NMDA + Pr*weight_NMDA*factor_NMDA
    B_NMDA = B_NMDA + Pr*weight_NMDA*factor_NMDA

    if( verboseLevel > 0 ) {
        printf( " vals %g %g %g %g\n", A_AMPA, weight_AMPA, factor_AMPA, weight )
    }
}


FUNCTION toggleVerbose() {
    verboseLevel = 1-verboseLevel
}


FUNCTION findOffsetNa(relP_na, relP_k, relP_ca, e (mV)) (mV) {
	findOffsetNa = (-relP_ca*(e-eca)-relP_k*(e-ek))/(relP_na) + ena - e
}
