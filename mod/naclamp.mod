COMMENT
Since this is an electrode current, positive amplitudes depolarize the cell.
This mechanism is a modified version of IClamp which will inject
K or Cl current instead of nonspecfied current.
This can be used with any model in which intracellular ion concentrations are being tracked, or as a general substitution for IClamp.
ENDCOMMENT

NEURON {
 	POINT_PROCESS IonIClamp
 	RANGE del, dur, amp, icl, ik,i
    USEION k WRITE ik
    USEION cl WRITE icl
}

UNITS {
  (nA) = (nanoamp)
}

PARAMETER {
	del (ms)
	dur (ms)	<0,1e9>
	amp (nA)
}

ASSIGNED {
  icl (nA)
  ik (nA)
  qq (ms)
  i (nA)
}

BREAKPOINT {
	at_time(del)
	at_time(del+dur)

	if (t < del + dur && t >= del) {
  	 if (amp>0) {
      ik = -amp 
      icl = 0
    } else if (amp<0) {
      ik = 0
      icl = -amp
    } else{
      ik=0
      icl=0
    }
  } else{
	 icl = 0
     ik = 0
  }
  : Total current can be recorded (positive = depolarizing)
  i = -ik - icl
}
