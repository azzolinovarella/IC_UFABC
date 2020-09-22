import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, acos, atan


def ex_res_2_6(plot=False) -> pd.DataFrame:
    """
    Exercise:

    EXAMPLE 2.6
    Consider the crank-rocker four-bar linkage shown in Fig. 2.25 (pag. 88) with RBA = 100 mm, RCB =
    250 mm, RCD = 300 mm, and RDA = 200 mm. The paths of coupler pins B and C are
    shown by the circle and the circular arc, respectively. The location of coupler point P is
    given by RPB = 150 mm and α = ∠CBP = −45◦. Calculate the coordinates of coupler
    88 POSITION, POSTURE, AND DISPLACEMENT point P and plot the path of this point (the coupler curve)
    for a complete rotation of the crank.

    Let's use the equations:
        -> S = sqrt(r1**2 + r2**2 − 2*r1*r2*cos(theta2))
        -> beta = acos((r1**2 + S**2 - r2**2)/(2*r1*S))
        -> psi = acos((r3**2 + S**2 - r4**2)/(2*r3*S))
        -> lambda = acos((r4**2 + S**2 - r3**2)/(2*r4*S))
        -> gama = +/- acos((r3**2 + r4**2 - S**2)/(2*r3*r4))
        -> theta3 =
            * if 0 <= theta2 <= pi: - beta +/- psi
            * else: beta +/- psi
        -> Rp = sqrt(Ra ** 2 + Rpa ** 2 + 2*Ra*Rpa*cos(theta3 + alfa - theta2))
        -> theta6 = atan((Ra*sin(theta2) + Rpa*sin(theta3 + alfa))/(Ra*cos(theta2) + Rpa*cos(theta3 + alfa)))
    """
    # For our case we have that:
    r1 = 200  # ||AD|| in our example
    r2 = 100  # ||AB|| in our example
    r3 = 250  # ||BC|| in our example
    r4 = 300  # ||DC|| in our example
    Ra = r2  # ||AB|| in our example
    Rpa = 150  # ||BP|| in our example
    alfa = - pi / 4  # CBP in our example

    theta2 = pd.Series(np.linspace(0, 2 * pi, 361))  # Creating a Series with 361 elements, each one From 0 to 360 degree
    S = theta2.apply(lambda t2: np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * cos(t2)))
    beta = S.apply(lambda s: acos((r1 ** 2 + s ** 2 - r2 ** 2) / (2 * r1 * s)))
    psi = S.apply(lambda s: acos((r3 ** 2 + s ** 2 - r4 ** 2) / (2 * r3 * s)))
    theta3 = pd.Series((- beta[i] + psi[i] if 0 <= theta2[i] <= pi else beta[i] + psi[i] for i in range(0, len(beta))))
    Rp = pd.Series((np.sqrt(Ra ** 2 + Rpa ** 2 + 2 * Ra * Rpa * cos(t3 + alfa - t2)) for t3, t2 in zip(theta3, theta2)))
    theta6 = pd.Series((atan((Ra * sin(t2) + Rpa * sin(t3 + alfa)) / (Ra * cos(t2) + Rpa * cos(t3 + alfa))) for t3, t2 in zip(theta3, theta2)))
    Rpx = pd.Series((R * cos(t6) for R, t6 in zip(Rp, theta6)))
    Rpy = pd.Series((R * sin(t6) for R, t6 in zip(Rp, theta6)))

    df = pd.DataFrame.from_dict({"theta2 (deg)": theta2.apply(lambda angle: angle * 180 / pi),
                                 "theta3 (deg)": theta3.apply(lambda angle: angle * 180 / pi).round(1),
                                 "Rp (mm)": Rp.round(1),
                                 "theta6 (deg)": theta6.apply(lambda angle: angle * 180 / pi).round(1),
                                 "Rpx (mm)": Rpx.round(1),
                                 "Rpy (mm)": Rpy.round(1)}).set_index(['theta2 (deg)'])

    if plot is True:
        plt.figure(1)
        plt.plot(Rpx, Rpy)
        plt.title('Curve made by the point P')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.xlim((0, 250))
        plt.ylim((0, 150))
        plt.grid()

    return df


def ex_prop_2_23() -> pd.DataFrame:
    pass


if __name__ == '__main__':
    # Exercicio 2.6 (pag. 87)
    print(ex_res_2_6(plot=True)[0:90:10])
    plt.show()

