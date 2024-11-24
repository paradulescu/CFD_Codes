import jax.numpy as jnp

#-----------------------------------------------------------
# This function calculates the 2D Roe flux
# INPUTS: UL, UR = left, right states, as 4x1 vectors
#         gamma = ratio of specific heats (e.g. 1.4)
#         n = left-to-right unit normal 2x1 vector
# OUTPUTS: F = numerical normal flux (4x1 vector)
#          smag = max wave speed estimate

def FluxFunction(UL, UR, gamma, n):

    gmi = gamma-1.0

    # process left state
    rL = UL[0]
    uL = UL[1]/rL
    vL = UL[2]/rL
    unL = uL*n[0] + vL*n[1]
    qL = jnp.sqrt(UL[1]**2 + UL[2]**2)/rL
    pL = (gamma-1)*(UL[3] - 0.5*rL*qL**2)
    rHL = UL[3] + pL
    HL = rHL/rL
    cL = jnp.sqrt(gamma*pL/rL)

    # left flux
    FL = jnp.array([rL*unL, UL[1]*unL + pL*n[0], UL[2]*unL + pL*n[1], rHL*unL])

    # process right state
    rR = UR[0]
    uR = UR[1]/rR
    vR = UR[2]/rR
    unR = uR*n[0] + vR*n[1]
    qR = jnp.sqrt(UR[1]**2 + UR[2]**2)/rR
    pR = (gamma-1)*(UR[3] - 0.5*rR*qR**2)
    rHR = UR[3] + pR
    HR = rHR/rR
    cR = jnp.sqrt(gamma*pR/rR)

    # right flux
    FR = jnp.array([rR*unR, UR[1]*unR + pR*n[0], UR[2]*unR + pR*n[1], rHR*unR])

    # difference in states
    du = UR - UL

    # Roe average
    di     = jnp.sqrt(rR/rL)
    d1     = 1.0/(1.0+di)
    
    ui     = (di*uR + uL)*d1
    vi     = (di*vR + vL)*d1
    Hi     = (di*HR + HL)*d1

    af     = 0.5*(ui*ui+vi*vi )
    ucp    = ui*n[0] + vi*n[1]
    c2     = gmi*(Hi - af)
    ci     = jnp.sqrt(c2)
    ci1    = 1.0/ci
    
    # eigenvalues
    l = jnp.array([ucp+ci, ucp-ci, ucp])

    # entropy fix
    epsilon = ci*.1
    
    l = jnp.where( (l < epsilon) & (l > -epsilon), 0.5*(epsilon + l * l /epsilon), l)

    l = jnp.abs(l); l3 = l[2]

    # average and half-difference of 1st and 2nd eigs
    s1    = 0.5*(l[0] + l[1])
    s2    = 0.5*(l[0] - l[1])

    # left eigenvector product generators
    G1    = gmi*(af*du[0] - ui*du[1] - vi*du[2] + du[3])
    G2    = -ucp*du[0]+du[1]*n[0]+du[2]*n[1]

    # required functions of G1 and G2 (again, see Theory guide)
    C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
    C2    = G1*s2*ci1          + G2*(s1-l3)

    # flux assembly
    F = jnp.array([0.5*(FL[0]+FR[0])-0.5*(l3*du[0] + C1   ),
                   0.5*(FL[1]+FR[1])-0.5*(l3*du[1] + C1*ui + C2*n[0]),
                   0.5*(FL[2]+FR[2])-0.5*(l3*du[2] + C1*vi + C2*n[1]),
                   0.5*(FL[3]+FR[3])-0.5*(l3*du[3] + C1*Hi + C2*ucp  )])
    
    # max wave speed
    smag = jnp.max(l)

    return F, smag