import numpy as np

def extinction_amores(glong,glat,distance):
#Interstellar Extinction in the Galaxy (Amores & L�pine - 2004)
#This program corresponds to the Axysimetric Model (Model A)
#If you have any difficulty, sugestion | comments, please contact:
#jacques@astro.iag.usp.br     |     amores@astro.iag.usp.br
#You enter longitude, latitude & distance of a point in the Galaxy & get extinction
#Converted to python by A. G.Sreejith    

    r0=7.5 #adopted distance of the Galactic center
    conv=np.pi/180.

    step  = 0.05     #steps of the gas density integration to obtain column density, in pc
    #glong=100.0        #galactic longitude# an arbitrary value given here
    #glat=0.0            #galactic latitude
    #dist = 20.0       #distance of the point to which we will calculate the extinction in kpc

    #print,'Interstellar Extinction in the Galaxy (Amores & L�pine - 2005, AJ, 130, 679)'
    #
    #read,glong,glat,PROMPT='Give the galactic longitude & latitude (Degrees,Degrees)....:  '
    #read,dist,PROMPT='Distance [kpc](positive value)...�

    dist=distance
    nstep=int(dist/step)

    if nstep == 0:
        nstep = 1

    #computes  trigonometric functions only once
    yproj=np.cos(glong*conv)
    xproj=np.sin(glong*conv)
    bproj=np.sin(glat*conv)
    dproj=np.cos(glat*conv)
    av=0.0                  #for the integration of the colunar density

    #declaring & puting values in the variables. The arrays will contain the
    #value of quantities like galactic radius | gas density for each step along the line-of sight
    #if you work with other language you should probably define these quantities in a loop
    dis= np.zeros(nstep)
    x  = np.zeros(nstep)
    y  = np.zeros(nstep)
    yy = np.zeros(nstep)
    r  = np.zeros(nstep)
    z  = np.zeros(nstep)
    zCO= np.zeros(nstep)
    zH = np.zeros(nstep)
    ah1= np.zeros(nstep)
    aco= np.zeros(nstep)
    zmet=np.zeros(nstep)
    agas=np.zeros(nstep)
    ipas=np.arange(0,nstep)/1 +1  
    # generates an array with a sequence of numbers, used as index for
    # distance along line-of-sight
    nel=len(ipas)
    
    dis=ipas*step - step
    x=(dis*xproj)*dproj
    y=dis*yproj*dproj
    yy=r0-y
    r=np.sqrt(x*x+yy*yy)
    z=dis*bproj
    
    zCO=0.036*np.exp(0.08*r)        #H2 scale-height
    zH = zCO*1.8                    #H1 scale-height (Guilbert 1978)
    zc = 0.02                       #shift takes in to account that the sun is not precisely in the galactic plane
    
    ah1=0.7*np.exp(-r/7.0-((1.9/r)**2))  #function that calculates the HI density
    aco = 58.*np.exp(-r/1.20-((3.50/r)**2)) + 240.*np.exp(-(r**2/0.095)) # H2 density# last term is for galactic center
    
    ah1[0] = 0.0
    aco[0] = 0.0
    
    for i in range(0, nel):
        if r[i] <= 1.2:  zmet[i] = 9.6
        if  1.2  < r[i] <= 9.0 : zmet[i] = (r0/r[i])**0.5
        if r[i] > 9.0:  zmet[i] = (r0/r[i])**0.1

    # this defines the metallicity correction, see section 3 of the paper
    
        
    gam1=1.0
    gam2=2.0
    
    #See the final tuning (section 4.1) correction factor for interval l=120-200
    
    tune=1.
    if 120 <= glong <= 200  : tune=2.
    agas=gam1*(ah1*zmet*np.exp(-0.5*((z-zc)/zH)**2))+gam2*aco*np.exp(-0.5*((z-zc)/zCO)**2)

    av=np.sum(agas)*step*3.086*.57*tune

    # "total" instruction gives the sum of the array elements
    # it is equivaletn to integrate along the line-of-sight. The step is in units of kpc=
    #3.08 *10**21 cm & the conversion factor gamma= .53 10**-21 mag cm2

    rs = 3.05  #ratio between total to selective extinction
    ebv = av/rs
    #print('Ebv')
    #print(ebv)
   
    #status = Check_Math()         # Get status & reset accumulated math error register.
    #IF(status AND NOT floating_point_underflow) NE 0 THEN $
    #  Message, 'IDL Check_Math() error: ' + StrTrim(status, 2)

    return ebv,av

