import numpy as np

LX = 101
LY = 100

q = 9
ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]) * 1.0

is_solid_node = np.zeros((LX, LY), dtype=bool)
is_solid_node[20:30, 45:55] = True

ftemp = np.zeros((q, LX, LY))
ux = np.zeros(LX, LY)
uy = np.zeros(LX, LY)
rho = np.zeros(LX, LY)


def calculate_macros(f):
    for j in range(LY):
        for i in range(LX):
            if not is_solid_node[j, i]:
                for a in range(q):
                    rho[j][i] += f[j][i][a]
                    uy[j][i] += ex[a]*f[j][i][a]
                    uy[j][i] += ey[a]*f[j][i][a]
                ux[j][i] /= rho[j][i]
                uy[j][i] /= rho[j][i]
    return rho, ux, uy


def stream(f):
    for j in range(LY):
        jn = j-1
        if j <= 0:
            jn = LY-1
        jp = j+1
        if j >= LY-1:
            jp = 0
        for i in range(LX):
            if not is_solid_node[j, i]:
                i_n = i-1
                if i <= 0:
                    i_n = LX-1
                ip = i+1
                if i >= LX-1:
                    ip = 0
                ftemp[j][i ][0] = f[j][i][0]
                ftemp[j][ip][1] = f[j][i][1]
                ftemp[jp][i][2] = f[j][i][2]
                ftemp[j ][i_n][3] = f[j][i][3]
                ftemp[jn][i ][4] = f[j][i][4]
                ftemp[jp][ip][5] = f[j][i][5]
                ftemp[jp][i_n][6] = f[j][i][6]
                ftemp[jn][i_n][7] = f[j][i][7]
                ftemp[jn][ip][8] = f[j][i][8]

