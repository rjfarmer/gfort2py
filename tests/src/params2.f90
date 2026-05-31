module params2
    implicit none

    integer, parameter :: dp = selected_real_kind(p=15)  ! real64


   real(dp), parameter :: pi = 3.1415926535897932384626433832795028841971693993751d0
   real(dp), parameter :: pi2 = pi*pi
   real(dp), parameter :: pi4 = 4*pi
   real(dp), parameter :: eulercon = 0.577215664901532861d0
   real(dp), parameter :: eulernum = 2.71828182845904523536028747135266249d0
   real(dp), parameter :: ln2 = 6.9314718055994529D-01  ! = log(2d0)
   real(dp), parameter :: ln3 = 1.0986122886681096D+00  ! = log(3d0)
   real(dp), parameter :: lnPi = 1.14472988584940017414343_dp  ! = log(pi)
   real(dp), parameter :: ln10 = 2.3025850929940455_dp  ! = log(10d0)
   real(dp), parameter :: iln10 = 0.43429448190325187_dp  ! = 1d0/log(10d0)
   real(dp), parameter :: a2rad = pi/180.0d0  ! angle to radians
   real(dp), parameter :: rad2a = 180.0d0/pi  ! radians to angle
   real(dp), parameter :: one_third = 1d0/3d0
   real(dp), parameter :: two_thirds = 2d0/3d0
   real(dp), parameter :: four_thirds = 4d0/3d0
   real(dp), parameter :: five_thirds = 5d0/3d0
   real(dp), parameter :: one_sixth = 1d0/6d0
   real(dp), parameter :: four_thirds_pi = four_thirds*pi
   real(dp), parameter :: ln4pi3 = 1.4324119583011810d0  ! = log(4*pi/3)
   real(dp), parameter :: two_13 = 1.2599210498948730d0  ! = pow(2d0,1d0/3d0)
   real(dp), parameter :: four_13 = 1.5874010519681994d0  ! = pow(4d0,1d0/3d0)
   real(dp), parameter :: sqrt2 = 1.414213562373095d0  ! = sqrt(2)
   real(dp), parameter :: sqrt_2_div_3 = 0.816496580927726d0  ! = sqrt(2/3)


   ! CODATA 2018
   real(dp), parameter :: avo = 6.02214076d23  ! Avogadro constant (mole^-1)
   real(dp), parameter :: amu = 1d0/avo  ! atomic mass unit (g)
   real(dp), parameter :: clight = 2.99792458d10  ! speed of light in vacuum (cm s^-1)
   real(dp), parameter :: qe = (clight/10d0)*1.602176634d-19  ! elementary charge (esu == (g cm^3 s^-2)^(1/2))
   real(dp), parameter :: kerg = 1.380649d-16
   real(dp), parameter :: boltzm = kerg  ! Boltzmann constant (erg K^-1)
   real(dp), parameter :: planck_h = 6.62607015d-27  ! Planck constant (erg s)
   real(dp), parameter :: hbar = planck_h/(2*pi)
   real(dp), parameter :: cgas = boltzm*avo  ! ideal gas constant (erg K^-1)
   real(dp), parameter :: ev2erg = 1.602176634d-12  ! electron volt (erg)
   real(dp), parameter :: mev_to_ergs = 1d6*ev2erg
   real(dp), parameter :: mev_amu = mev_to_ergs/amu
   real(dp), parameter :: mev2gr = 1d6*ev2erg/(clight*clight)  ! MeV to grams
   real(dp), parameter :: Qconv = mev_to_ergs*avo
   real(dp), parameter :: kev = kerg/ev2erg  ! converts temp to ev (ev K^-1)
   real(dp), parameter :: boltz_sigma = (pi*pi*boltzm*boltzm*boltzm*boltzm)/(60*hbar*hbar*hbar*clight*clight)  ! Stefan-Boltzmann constant (erg cm^-2 K^-4 s^-1)
   real(dp), parameter :: crad = boltz_sigma*4/clight  ! radiation density constant, AKA "a" (erg cm^-3 K^-4); Prad = crad * T^4 / 3

   ! IAU
   real(dp), parameter :: au = 1.49597870700D13  ! (cm) - exact value defined by IAU 2009, 2012
   real(dp), parameter :: pc = (3.600D3*rad2a)*au  ! (cm) parsec, by definition
   real(dp), parameter :: dayyer = 365.25d0  ! days per (Julian) year
   real(dp), parameter :: secday = 24*60*60  ! seconds in a day
   real(dp), parameter :: secyer = secday*dayyer  ! seconds per year
   real(dp), parameter :: ly = clight*secyer  ! light year (cm)

   real(dp), parameter :: mn = 1.67492749804d-24  ! neutron mass (g)
   real(dp), parameter :: mp = 1.67262192369d-24  ! proton mass (g)
   real(dp), parameter :: me = 9.1093837015d-28  ! electron mass (g)

   real(dp), parameter :: rbohr = 5.29177210903d-9  ! Bohr radius (cm)
   real(dp), parameter :: fine = 7.2973525693d-3   ! fine-structure constant
   real(dp), parameter :: hion = 13.605693122994d0  ! Rydberg constant (eV)

   real(dp), parameter :: sige = 6.6524587321d-25  ! Thomson cross section (cm^2)

   real(dp), parameter :: weinberg_theta = 0.22290d0  ! sin**2(theta_weinberg)
   real(dp), parameter :: num_neu_fam = 3.0d0  ! number of neutrino flavors = 3.02 plus/minus 0.005 (1998)



   real(dp), parameter :: mpi = -3.1415926535897932384626433832795028841971693993751d0
   real(dp), parameter :: mpi2 = -pi*pi
   real(dp), parameter :: mpi4 = -4*pi
   real(dp), parameter :: meulercon = -0.577215664901532861d0
   real(dp), parameter :: meulernum = -2.71828182845904523536028747135266249d0
   real(dp), parameter :: mln2 = -6.9314718055994529D-01  ! = log(2d0)
   real(dp), parameter :: mln3 = -1.0986122886681096D+00  ! = log(3d0)
   real(dp), parameter :: mlnPi = -1.14472988584940017414343_dp  ! = log(pi)
   real(dp), parameter :: mln10 = -2.3025850929940455_dp  ! = log(10d0)
   real(dp), parameter :: miln10 = -0.43429448190325187_dp  ! = 1d0/log(10d0)
   real(dp), parameter :: ma2rad = -pi/180.0d0  ! angle to radians
   real(dp), parameter :: mrad2a = -180.0d0/pi  ! radians to angle
   real(dp), parameter :: mone_third = -1d0/3d0
   real(dp), parameter :: mtwo_thirds = -2d0/3d0
   real(dp), parameter :: mfour_thirds = -4d0/3d0
   real(dp), parameter :: mfive_thirds = -5d0/3d0
   real(dp), parameter :: mone_sixth = -1d0/6d0
   real(dp), parameter :: mfour_thirds_pi = -four_thirds*pi
   real(dp), parameter :: mln4pi3 = -1.4324119583011810d0  ! = log(4*pi/3)
   real(dp), parameter :: mtwo_13 = -1.2599210498948730d0  ! = pow(2d0,1d0/3d0)
   real(dp), parameter :: mfour_13 = -1.5874010519681994d0  ! = pow(4d0,1d0/3d0)
   real(dp), parameter :: msqrt2 = -1.414213562373095d0  ! = sqrt(2)
   real(dp), parameter :: msqrt_2_div_3 = -0.816496580927726d0  ! = sqrt(2/3)


   ! CODATA 2018
   real(dp), parameter :: mavo = -6.02214076d23  ! Avogadro constant (mole^-1)
   real(dp), parameter :: mamu = -1d0/avo  ! atomic mass unit (g)
   real(dp), parameter :: mclight = -2.99792458d10  ! speed of light in vacuum (cm s^-1)
   real(dp), parameter :: mqe = -(clight/10d0)*1.602176634d-19  ! elementary charge (esu == (g cm^3 s^-2)^(1/2))
   real(dp), parameter :: mkerg = -1.380649d-16
   real(dp), parameter :: mboltzm = -kerg  ! Boltzmann constant (erg K^-1)
   real(dp), parameter :: mplanck_h = -6.62607015d-27  ! Planck constant (erg s)
   real(dp), parameter :: mhbar = -planck_h/(2*pi)
   real(dp), parameter :: mcgas = -boltzm*avo  ! ideal gas constant (erg K^-1)
   real(dp), parameter :: mev2erg = -1.602176634d-12  ! electron volt (erg)
   real(dp), parameter :: mmev_to_ergs = -1d6*ev2erg
   real(dp), parameter :: mmev_amu = -mev_to_ergs/amu
   real(dp), parameter :: mmev2gr = -1d6*ev2erg/(clight*clight)  ! MeV to grams
   real(dp), parameter :: mQconv = -mev_to_ergs*avo
   real(dp), parameter :: mkev = -kerg/ev2erg  ! converts temp to ev (ev K^-1)
   real(dp), parameter :: mboltz_sigma = -(pi*pi*boltzm*boltzm*boltzm*boltzm)/(60*hbar*hbar*hbar*clight*clight)  ! Stefan-Boltzmann constant (erg cm^-2 K^-4 s^-1)
   real(dp), parameter :: mcrad = -boltz_sigma*4/clight  ! radiation density constant, AKA "a" (erg cm^-3 K^-4); Prad = crad * T^4 / 3

   ! IAU
   real(dp), parameter :: mau = -1.49597870700D13  ! (cm) - exact value defined by IAU 2009, 2012
   real(dp), parameter :: mpc = -(3.600D3*rad2a)*au  ! (cm) parsec, by definition
   real(dp), parameter :: mdayyer = -365.25d0  ! days per (Julian) year
   real(dp), parameter :: msecday = -24*60*60  ! seconds in a day
   real(dp), parameter :: msecyer = -secday*dayyer  ! seconds per year
   real(dp), parameter :: mly = -clight*secyer  ! light year (cm)

   real(dp), parameter :: mmn = -1.67492749804d-24  ! neutron mass (g)
   real(dp), parameter :: mmp = -1.67262192369d-24  ! proton mass (g)
   real(dp), parameter :: mme = -9.1093837015d-28  ! electron mass (g)

   real(dp), parameter :: mrbohr = -5.29177210903d-9  ! Bohr radius (cm)
   real(dp), parameter :: mfine = -7.2973525693d-3   ! fine-structure constant
   real(dp), parameter :: mhion = -13.605693122994d0  ! Rydberg constant (eV)

   real(dp), parameter :: msige = -6.6524587321d-25  ! Thomson cross section (cm^2)

   real(dp), parameter :: mweinberg_theta = -0.22290d0  ! sin**2(theta_weinberg)
   real(dp), parameter :: mnum_neu_fam = -3.0d0  ! number of neutrino flavors = 3.02 plus/minus 0.005 (1998)


end module params2