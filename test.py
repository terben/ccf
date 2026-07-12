import schurcorr as sc

alpha = [0.2, -0.3, 0.4]
r = sc.from_pacf(alpha)

lower, upper = sc.admissible_bounds(r)

print("r =", r)
print("lower =", lower)
print("upper =", upper)
print("x =", sc.sh_coordinates(r))
print("admissible =", sc.check_admissibility(r))
print("invalid =", sc.check_admissibility([0.9, -0.9]))

try:
    sc.check_admissibility(r, atol=-1.0)
except ValueError as error:
    print(error)
