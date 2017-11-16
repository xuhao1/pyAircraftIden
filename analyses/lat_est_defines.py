#
# def lat_aoa():
#     # X = [V,aoa,the,q]
#
#     V0 = 52.7
#     Zaldot, Maldot, V0 = sp.symbols('Zaldot Maldot V0')
#
#     M = sp.Matrix([[1, 0, 0, 0],
#                    [0, V0 - Zaldot, 0, 0],
#                    [0, 0, 1, 0],
#                    [0, -Maldot, 0, 1]])
#
#     g = 9.78
#     Xv, Xtv, Xa, al0 = sp.symbols('Xv Xtv Xa al0')
#     Zv, Zv, Zq, Za = sp.symbols("Zv Zv Zq Za")
#     Mv, Ma, Mq = sp.symbols("Mv Ma Mq")
#
#     al0 = 0.015464836009954524
#     F = sp.Matrix([[Xv + Xtv * sp.cos(al0), Xa, -g, 0],
#                    [Zv - Xtv * sp.sin(al0), Za, 0, V0 + Zq],
#                    [0, 0, 0, 1],
#                    [Mv, Ma, 0, Mq]])
#
#     Xele, Mele = sp.symbols('Xele,Mele')
#     G = sp.Matrix([[Xele * sp.cos(al0)],
#                    [0],
#                    [0],
#                    [Mele]])
#
#     # direct using u w q th for y
#     H0 = sp.Matrix([[1, 0, 0, 0],
#                     [0, 1, 0, 0],
#                     [0, 0, 1, 0],
#                     [0, 0, 0, 1]])
#
#     H1 = sp.Matrix.zeros(4, 4)
#
#     syms = [Zaldot, Maldot,
#             Xv, Xtv, Xa,
#             Zv, Zq, Za,
#             Mv, Ma, Mq,
#             Xele, Mele]


def lat_dyn_uw():
    # X = [u,w,q,th]
    Xwdot, Zwdot, Mwdot = sp.symbols('Xwdot Zwdot Mwdot')

    M = sp.Matrix([[1, -Xwdot, 0, 0],
                   [0, 1 - Zwdot, 0, 0],
                   [0, -Mwdot, 1, 0],
                   [0, 0, 0, 1]])

    g = 9.78
    Xu, Xw, Xq, W0, th0 = sp.symbols('Xu Xw Xq W0 th0')
    Zu, Zw, Zq, U0 = sp.symbols('Zu Zw Zq U0')
    Mu, Mw, Mq = sp.symbols('Mu Mw Mq')

    F = sp.Matrix([[Xu, Xw, Xq - W0, -g * sp.cos(th0)],
                   [Zu, Zw, Zq + U0, -g * sp.sin(th0)],
                   [Mu, Mw, Mq, 0],
                   [0, 0, 1, 0]])

    Xele, Zele, Mele = sp.symbols('Xele,Zele,Mele')
    G = sp.Matrix([[Xele],
                   [Zele],
                   [Mele],
                   [0]])

    # direct using u w q th for y
    H0 = sp.Matrix([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    H1 = sp.Matrix.zeros(4, 4)

    syms = [Xwdot, Zwdot, Mwdot,
            Xu, Xw, Xq, W0, th0,
            Zu, Zw, Zq, U0,
            Mu, Mw, Mq,
            Xele, Zele, Mele]



    # V0, al0]
