import schurcorr as sc

alpha = [0.2, -0.3, 0.4]
r = sc.from_pacf(alpha)

lo, hi = sc.admissible_bounds(r)

print("r =", r)
print("lower =", lo)
print("upper =", hi)
print("x =", sc.sh_coordinates(r))
print("admissible =", sc.check_admissibility(r))


# import schurcorr as sc
#
# alpha = [0.2, -0.3, 0.4]
# r = sc.from_pacf(alpha)
# alpha_back = sc.pacf(r)
#
# print("r =", r)
# print("alpha_back =", alpha_back)
# print("logJ =", sc.log_jacobian(alpha))
