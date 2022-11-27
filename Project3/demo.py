import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plt2
import time

###############ergasia3
#erwthma A1: thelw entash apo diaxyto fws apo periballon sthn eksodo
def ambient_light(k_a,I_a):
    I = k_a * I_a
    return I

#erwthma A2: thelw entash apo diaxyth anaklash sthn eksodo opou P syntetagmenes shmeiou endiaferontos, N monadiaio dianysma ka8eto sthn epifaneia tou P, color 3x1 to RGB tou shmeiou P,  
#k_d syntelesths diaxyths anaklashs. telos dinontai oi syntetagmenes ths phghs fwtismou kai tis entaseis auths
def diffuse_light(P,N,color,k_d,light_positions,light_intensities):
    SP = light_positions - P
    L = SP / np.linalg.norm(SP) #monadiaio apo P pros Source
    L = np.reshape(L, (3, ))
    cosa = np.dot(N, L)
    #print(np.shape(color))
    I = light_intensities * k_d * cosa
    I = color * I #trixrwmatikh entash
    return I.reshape(3,)

#erwthma A3: thelw entash apo katoptrikh anaklash basei autwn pou eixa omoiws me prin alla exontas kai tis syntetagmenes tis cameras  sto 3x1 cam_pos, to k_s ws syntelesth katoptrikhs anaklashs, 
#n ton ek8eth tou Phong kai color 3x1 to RGB tou shmeiou P 
def specular_light(P,N, color, cam_pos, k_s, n, light_positions, light_intensities):
    CP = cam_pos - P
    V = CP / np.linalg.norm(CP) #monadiaio apo P pros Camera
    V = np.reshape(V, (3, ))
    SP = light_positions - P
    L = SP / np.linalg.norm(SP) #monadiaio apo P pros Source
    L = np.reshape(L, (3, ))
    cosa = np.dot(N, L)
    cosb_a = np.dot(2* N *cosa - L, V)
    I = light_intensities * k_s * cosb_a ** n
    I = color * I #trixrwmatikh entash
    return I.reshape(3,)

#erwthma B1: thelw na ypologisw ta ka8eta dianysmata se mia epifaneia(apoteleitai apo N trigwna) enos 3D antikeimenou
#pairnei sto vertices(3xN) tis syntetagmenes twn koryfwn tou antikeimenou kai sto face_indices periexei tous auxontes ari8mous twn koryfwn tou k-ostou trigwnou
def calculate_normals(vertices, face_indices):
    normals = np.zeros(vertices.shape)
    for triangle in face_indices:
        index = vertices[triangle]
        A = index[0]
        B = index[1]
        C = index[2]
        AB = (A - B)
        AC = (A - C)
        nAB = AB / np.linalg.norm(AB) 
        nAC = AC / np.linalg.norm(AC) 
        ABxAC = np.cross(nAB, nAC) 
        normals[triangle] += ABxAC

    for i in range(len(normals)):
        normals[i] /= np.linalg.norm(normals[i])
    return normals

#erwthma B2: 
def render_object(shader, focal, eye, lookat, up, bg_color, M,N,H,W, verts, vert_colors, face_indices, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, lighter):
    normals = calculate_normals(verts, face_indices)
    verts2d, depth = project_cam_lookat(focal, eye, lookat, up, verts)
    verts2d = rasterize(verts2d, M, N, H, W)
    #texnhth peristrofh gia na fainetai isxio an theloume
    #verts2d[:, 0], verts2d[:, 1] = verts2d[:, 1], verts2d[:, 0].copy() 
    verts2d = np.array(verts2d).astype(int)
    #ypologizw to meso ba8os kai taksinomw ta ba8h xrwmatos apo mikrotera se megalytera se mikrotera
    new_depth = np.array(np.mean(depth[face_indices], axis=1))
    faces = list(np.flip(np.argsort(new_depth)))
    #8elw canva me bg_color fonto apo pisw me analysh MxN
    X = [[[bg_color[i] for i in range(3)] for j in range(M)] for k in range(N)]

    if shader == 'gouraud':
        for triangle in faces:
            verts_idx = face_indices[triangle]
            triangle_vertsc = np.array(vert_colors[verts_idx])
            triangle_vertsp = np.array(verts2d[verts_idx])
            bcoords = np.mean(verts[verts_idx], axis=0)
            img = shade_gouraud(triangle_vertsp, normals[verts_idx], triangle_vertsc, bcoords, eye, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, X, lighter)
                            
    if shader == 'phong':
        for triangle in faces:
            verts_idx = face_indices[triangle]
            triangle_vertsc = np.array(vert_colors[verts_idx])
            triangle_vertsp = np.array(verts2d[verts_idx])
            bcoords = np.mean(verts[verts_idx], axis=0)
            img = shade_phong(triangle_vertsp, normals[verts_idx], triangle_vertsc, bcoords, eye, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, X, lighter)
    return img

#boh8htikh synarthsh gia ton ypologismo tou fwtismou basei tou lighter mode pou 8eloume
def color_lighter(P, N, color, cam_pos, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, lighter):
    if lighter == 'combined':
        b = ambient_light(k_a, I_a) + diffuse_light(P, N, color, k_d, light_positions, light_intensities) + specular_light(P, N, color, cam_pos, k_s, n, light_positions, light_intensities)
    elif lighter == 'ambient':
        b = ambient_light(k_a, I_a) 
    elif lighter == 'diffusion':
        b = diffuse_light(P, N, color, k_d, light_positions, light_intensities)  
    elif lighter == 'specular':
        b = specular_light(P, N, color, cam_pos, k_s, n, light_positions, light_intensities)
    else:
        print("Incorrect lighter mode given. You can only use : 'combined' or 'ambient' or 'diffusion' or 'specular'! Now exiting...\n")
        exit(0)
    #gia ta koresmena shmeia
    b = np.where(b<=1, b, 1)
    b = np.where(b>=0, b, 0)
    return b

#erwthma B3:
def shade_gouraud(vertsp, vertsn, vertsc, bcoords, cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, X: list, lighter):
    #fwtismos
    vertsc[0] = color_lighter(bcoords, vertsn[0], vertsc[0], cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter)
    vertsc[1] = color_lighter(bcoords, vertsn[1], vertsc[1], cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter)
    vertsc[2] = color_lighter(bcoords, vertsn[2], vertsc[2], cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter)
    vcolors = vertsc.tolist()

    verts2d = vertsp.tolist()
    #bresenham_line gia na sxhmatisw thn ka8e eu8ygrammh akmh tou trigwnou kai pairnw tous pinakes twn shmeiwn twn akmwn
    e1 = bresenham_line(verts2d[0], verts2d[1])
    e2 = bresenham_line(verts2d[1], verts2d[2])
    e3 = bresenham_line(verts2d[2], verts2d[0])
    #apo autous tous pinakes taksinomw tis syntetagmenes twn shmeiwn ths akmhs ana grammes tou pinaka wste na brw thn arxh kai to telos ths grammhs sarwshs
    e1 = sorted(e1, key=lambda x: x[0])
    e2 = sorted(e2, key=lambda x: x[0])
    e3 = sorted(e3, key=lambda x: x[0])

    #gia gouraud ana sarwsh grammhs y thelw parembolh gia ta A(a,y) k B(b,y) kai meta basei autwn sta shmeia ths idias grammhs P(blabla,y) https://prnt.sc/bIZtynqgdfoi
    #gia ka8e akmh tou trigwnou :a)ypologizw xrwma gia ta shmeia ths akmhs ,b)an den einai koryfh tou trigwnou tote kanw parembolh , c)alliws to pairnw apo ta gnwsta
    #a)
    for point in e1:
        #b)
        if 0 <= point[1] <= len(X[1]) - 1 and 0 <= point[0] <= len(X[0]) - 1:
            if point not in verts2d:
                #an h klish > 1 tote xreiazetai antallagh rolwn x kai y 
                if abs(verts2d[0][0] - verts2d[1][0]) > abs(verts2d[0][1] - verts2d[1][1]):
                    X[point[1]][point[0]] = interpolate_color(verts2d[0][0], verts2d[1][0], point[0], vcolors[0], vcolors[1])
                else:
                    X[point[1]][point[0]] = interpolate_color(verts2d[0][1], verts2d[1][1], point[1], vcolors[0], vcolors[1])
            #c)
            else:
                index = verts2d.index(point)
                X[point[1]][point[0]] = vcolors[index]

    #a)
    for point in e2:
        #b)
        if 0 <= point[1] <= len(X[1]) - 1 and 0 <= point[0] <= len(X[0]) - 1:
            if point not in verts2d:
                #an h klish > 1 tote xreiazetai antallagh rolwn x kai y 
                if abs(verts2d[1][0] - verts2d[2][0]) > abs(verts2d[1][1] - verts2d[2][1]):
                    X[point[1]][point[0]] = interpolate_color(verts2d[1][0], verts2d[2][0], point[0], vcolors[1], vcolors[2])
                else:
                    X[point[1]][point[0]] = interpolate_color(verts2d[1][1], verts2d[2][1], point[1], vcolors[1], vcolors[2])
            #c)                                                    
            else:
                index = verts2d.index(point)
                X[point[1]][point[0]] = vcolors[index]

    #a)
    for point in e3:
        #b)
        if 0 <= point[1] <= len(X[1]) - 1 and 0 <= point[0] <= len(X[0]) - 1:
            if point not in verts2d:
                if abs(verts2d[2][0] - verts2d[0][0]) > abs(verts2d[2][1] - verts2d[0][1]):
                    #an h klish > 1 tote xreiazetai antallagh rolwn x kai y 
                    X[point[1]][point[0]] = interpolate_color(verts2d[2][0], verts2d[0][0], point[0], vcolors[2], vcolors[0])
                else:
                    X[point[1]][point[0]] = interpolate_color(verts2d[2][1], verts2d[0][1], point[1], vcolors[2], vcolors[0])
            #c)
            else:
                index = verts2d.index(point)
                X[point[1]][point[0]] = vcolors[index]

    bottom_scanline = min(e1[0][0], e2[0][0], e3[0][0]) #h arxikh grammh sarwshs basei ths mikroterhs timhs twn prwtwn shmeiwn twn akmwn
    top_scanline = max(e1[-1][0], e2[-1][0], e3[-1][0]) #h teleutaia grammh sarwshs basei ths megalyterhs timhs twn teleutaiwn shmeiwn twn akmwn

    #lista energwn shmeiwn kai ananewsh auths se ka8e nea grammh sarwshs
    for scanline in range(bottom_scanline, top_scanline + 1):
        if 0 <= scanline <= len(X[0]) - 1:
            active_points = [] 
            for point in e1:
                if point[0] == scanline:
                    active_points.append(point)
            for point in e2:
                if point[0] == scanline:
                    active_points.append(point)
            for point in e3:
                if point[0] == scanline:
                    active_points.append(point)

            #an brw koryfh tote paw sthn epomenh epanalhpsh
            if len(active_points) == 1:
                continue
            else:
                #apo thn lista energwn shmeiwn pairnw thn elaxisth timh kata sthlh
                min_col_active = np.array(active_points).min(axis=0)[1]

                #apo thn lista energwn shmeiwn pairnw thn megisth timh kata sthlh 
                max_col_active = np.array(active_points).max(axis=0)[1]

                #zwgrafise ta energa shmeia metaksy twn oriakwn sthlwn
                for i in range(min_col_active + 1, max_col_active):
                    
                    if 0 <= i <= len(X[1]) - 1:
                        if [i, scanline] in active_points:
                            continue
                        else:
                            if max_col_active > len(X[1]) - 1:
                                max_col_active = len(X[1]) - 1
                            if 0 > max_col_active:
                                max_col_active = 0
                            if 0 <=min_col_active <= len(X[1]) - 1:
                                X[i][scanline] = interpolate_color(min_col_active, max_col_active, i, X[min_col_active][scanline], X[max_col_active][scanline])

    Y = X
    return Y

#boh8htikh synarthsh gia to kanoniko dianysma tou x shmeiou
#x shmeio parembolhs, x1 kai x2 : basei autwn tha ginei h parembolh. oi syntetagmenes autes anaferontai eite gia orizonties eite gia ka8etes.
#N1 kai N2 einai ta kanonika dianysmata twn x1 kai x2 antistoixa. 
def interpolate_normal(x1, x2, x, N1, N2):
    #gia kalyterh ypologistikh isxy kai gia antimetwpish tou "RuntimeWarning: invalid value encountered in true_divide"
    if abs(x1 - x2) < 1e-3:
        return N1
    #akolou8w thn logikh tou kyrtou grammikoy syndyasmou n12= lambda*n1 + (1-lambda)*n2,opou...
    lambd = (x2 - x) / (x2 - x1)
    N = np.array(lambd * N1 + (1 - lambd) * N2)
    value = N / np.linalg.norm(N)
    return value

#erwthma B4:
def shade_phong(vertsp, vertsn, vertsc, bcoords, cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, X: list, lighter):
    verts2d = vertsp.tolist()
    #bresenham_line gia na sxhmatisw thn ka8e eu8ygrammh akmh tou trigwnou kai pairnw tous pinakes twn shmeiwn twn akmwn
    e1 = bresenham_line(verts2d[0], verts2d[1])
    e2 = bresenham_line(verts2d[1], verts2d[2])
    e3 = bresenham_line(verts2d[2], verts2d[0])

    #apo autous tous pinakes taksinomw tis syntetagmenes twn shmeiwn ths akmhs ana grammes tou pinaka wste na brw thn arxh kai to telos ths grammhs sarwshs
    e1 = sorted(e1, key=lambda x: x[0])
    e2 = sorted(e2, key=lambda x: x[0])
    e3 = sorted(e3, key=lambda x: x[0])

    #gia  phong ana sarwsh grammhs y thelw parembolh gia ta A(a,y) k B(b,y) kai meta basei autwn sta shmeia ths idias grammhs P(blabla,y)
    #gia ka8e akmh tou trigwnou :a)ypologizw fwteinothta gia ta shmeia ths akmhs ,b)an den einai koryfh tou trigwnou tote kanw parembolh se kanoniko dianysma kai xrwma kai meta pairnw entash fwteinothtas, c)alliws to pairnw apo ta gnwsta thn fwteinothta
    #a)
    for point in e1:
        #b)
        if 0 <= point[1] <= len(X[1]) - 1 and 0 <= point[0] <= len(X[0]) - 1:
            if point not in verts2d:
                #an h klish > 1=> dy > dx tote xreiazetai antallagh rolwn x kai y 
                if abs(verts2d[0][0] - verts2d[1][0]) > abs(verts2d[0][1] - verts2d[1][1]):
                    vertN = interpolate_normal(verts2d[0][0], verts2d[1][0], point[0], vertsn[0], vertsn[1])
                    vertC = interpolate_color(verts2d[0][0], verts2d[1][0], point[0], vertsc[0].tolist(), vertsc[1].tolist()) #vertsc[0].tolist(), vertsc[1].tolist()
                    
                    X[point[1]][point[0]] = color_lighter(bcoords, vertN, vertC,cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter) #.tolist()
                else:
                    vertN = interpolate_normal(verts2d[0][1], verts2d[1][1], point[1], vertsn[0], vertsn[1])
                    vertC = interpolate_color(verts2d[0][1], verts2d[1][1], point[1], vertsc[0].tolist(), vertsc[1].tolist())#vertsc[0], vertsc[1]) #vertsc[0].tolist(), vertsc[1].tolist()
                    X[point[1]][point[0]] = color_lighter(bcoords, vertN, vertC,cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()
            #c)
            else:
                index = verts2d.index(point)
                X[point[1]][point[0]] = color_lighter(bcoords, vertsn[index, :], vertsc[index, :],cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()

    #a)
    for point in e2:
        #b)
        if 0 <= point[1] <= len(X[1]) - 1 and 0 <= point[0] <= len(X[0]) - 1:
            if point not in verts2d:
                #an h klish > 1 tote xreiazetai antallagh rolwn x kai y 
                if abs(verts2d[1][0] - verts2d[2][0]) > abs(verts2d[1][1] - verts2d[2][1]):
                    vertN = interpolate_normal(verts2d[1][0], verts2d[2][0], point[0], vertsn[1], vertsn[2])
                    vertC = interpolate_color(verts2d[1][0], verts2d[2][0], point[0], vertsc[1].tolist(), vertsc[2].tolist())
                    X[point[1]][point[0]] = color_lighter(bcoords, vertN, vertC,cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()
                else:
                    vertN = interpolate_normal(verts2d[1][1], verts2d[2][1], point[1], vertsn[1], vertsn[2])
                    vertC = interpolate_color(verts2d[1][1], verts2d[2][1], point[1], vertsc[1].tolist(), vertsc[2].tolist())
                    X[point[1]][point[0]] = color_lighter(bcoords, vertN, vertC,cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()
            #c)                                                    
            else:
                index = verts2d.index(point)
                X[point[1]][point[0]] = color_lighter(bcoords, vertsn[index, :], vertsc[index, :],cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()

    #a)
    for point in e3:
        #b)
        if 0 <= point[1] <= len(X[1]) - 1 and 0 <= point[0] <= len(X[0]) - 1:
            if point not in verts2d:
                #an h klish > 1 tote xreiazetai antallagh rolwn x kai y
                if abs(verts2d[2][0] - verts2d[0][0]) > abs(verts2d[2][1] - verts2d[0][1]):
                    vertN = interpolate_normal(verts2d[2][0], verts2d[0][0], point[0], vertsn[2], vertsn[0])
                    vertC = interpolate_color(verts2d[2][0], verts2d[0][0], point[0], vertsc[2].tolist(), vertsc[0].tolist())
                    X[point[1]][point[0]] = color_lighter(bcoords, vertN, vertC,cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()
                else:
                    vertN = interpolate_normal(verts2d[2][1], verts2d[0][1], point[1], vertsn[2], vertsn[0])
                    vertC = interpolate_color(verts2d[2][1], verts2d[0][1], point[1], vertsc[2].tolist(), vertsc[0].tolist())
                    X[point[1]][point[0]] = color_lighter(bcoords, vertN, vertC,cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()
            #c)
            else:
                index = verts2d.index(point)
                X[point[1]][point[0]] = color_lighter(bcoords, vertsn[index, :], vertsc[index, :],cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, lighter).tolist()

    bottom_scanline = min(e1[0][0], e2[0][0], e3[0][0]) #h arxikh grammh sarwshs basei ths mikroterhs timhs twn prwtwn shmeiwn twn akmwn
    top_scanline = max(e1[-1][0], e2[-1][0], e3[-1][0]) #h teleutaia grammh sarwshs basei ths megalyterhs timhs twn teleutaiwn shmeiwn twn akmwn

    #lista energwn shmeiwn kai ananewsh auths se ka8e nea grammh sarwshs
    for scanline in range(bottom_scanline, top_scanline + 1):
        if 0 <= scanline <= len(X[0]) - 1:
            active_points = [] 
            for point in e1:
                if point[0] == scanline:
                    active_points.append(point)
            for point in e2:
                if point[0] == scanline:
                    active_points.append(point)
            for point in e3:
                if point[0] == scanline:
                    active_points.append(point)
            #an brw koryfh tote paw sthn epomenh epanalhpsh
            if len(active_points) == 1:
                continue
            else:
                #apo thn lista energwn shmeiwn pairnw thn elaxisth timh kata sthlh
                a = np.array(active_points)
                min_col_active = a.min(axis=0)[1]        
                #apo thn lista energwn shmeiwn pairnw thn megisth timh kata sthlh 
                max_col_active = a.max(axis=0)[1]
                for i in range(min_col_active + 1, max_col_active):
                    
                    if 0 <= i <= len(X[1]) - 1:
                        if [i, scanline] in active_points:
                            continue
                        else:
                            if max_col_active > len(X[1]) - 1:
                                max_col_active = len(X[1]) - 1
                            if 0 > max_col_active:
                                max_col_active = 0
                            if 0 <=min_col_active <= len(X[1]) - 1:
                                X[i][scanline] = interpolate_color(min_col_active, max_col_active, i, X[min_col_active][scanline], X[max_col_active][scanline])
                                
    Y = X
    return Y

###############ergasia2

#erwthma A: thelw eksodo to neo synolo shmeiwn c_q(Nx3) basei enos affine metasxhmatismou sta c_p
#metasxhmatizei ena synolo shmeiwn c_p(3xN) peristrefontas to gyrw apo aksona peristrofhs u(3x1) kata gwnia theta kai metatopizontas ystera basei tou dianysmatos metatopishs t(3x1)
def affine_transform(c_p, u, theta, t=3 * [0]):
    u = np.array(u) / np.linalg.norm(u)
    if c_p.ndim > 1:
        c_p = np.hstack((c_p, np.ones((c_p.shape[0], 1))))
    else:
        c_p = np.append(c_p, 1)
    T_h = np.eye(4)
    T_h[0:3, 3] = np.array(t)
    b1 = np.array([(1 - np.cos(theta)) * u[0] ** 2 + np.cos(theta),
                      (1 - np.cos(theta)) * u[0] * u[1] - np.sin(theta) * u[2],
                      (1 - np.cos(theta)) * u[0] * u[2] + np.sin(theta) * u[1]])
    b2 = np.array([b1[1],
                      (1 - np.cos(theta)) * u[1] ** 2 + np.cos(theta),
                      (1 - np.cos(theta)) * u[1] * u[2] - np.sin(theta) * u[0]])
    b3 = np.array([b1[2], b2[2], (1 - np.cos(theta)) * u[2] ** 2 + np.cos(theta)])
    T_h[0:3, 0:3] = np.vstack((b1, b2, b3))

    c_q = np.dot(T_h, c_p.T).T
    if c_q.ndim > 1:
        c_q = np.delete(c_q, 3, 1)
    else:
        c_q = np.array(c_q[0:3])
    return c_q

#boh8htikh synarthsh gia diatetagmena kai ametablhta antikeimena
def create_tuples(*items):
    return tuple(item for item in items)

#erwthma B: thelw sthn eksodo to neo synolo shmeiwn d_p(Nx3) ekfrasmeno ws pros to neo diaforetiko systhma syntetagmenwn
#metasxhmatizei ena synolo shmeiwn c_p(3xN) basei tou pinaka peristrofhs R(3x3) kai tis syntetagmenes c_0(3x1) tou dianysmatos metatopishs v_0 ws to palio systhma syntetagmenwn
def system_transform(c_p, R, c_0):
    c_p, R, c_0 = create_tuples(c_p, R, c_0)
    if c_p.ndim == 1:
        c_p = (c_p - c_0)
    else:
        c_p = np.array([c_p[i, :] - c_0 for i in range(c_p.shape[0])])
    return np.dot(R, c_p.T).T

#erwthma C: thelw sthn eksodo tis prooptikes proboles twn koryfwn apo 3D shmeia se verts2d kai to ba8os autwn ws depth
#pairnei f apostash petasmatos apo kentro cameras,c_x/y/z syntetagmenes dianysmatwn x/y/z ths cameras ws pros WCS kai p ta shmeia pros probolh 
def project_cam(f, c_v, c_x, c_y, c_z, p):
    c_v, c_x, c_y, c_z, p = create_tuples(c_v, c_x, c_y, c_z, p)
    R = np.vstack((c_x, c_y, c_z)).T
    p = system_transform(p, R, c_v)
    depth = p[:, 2]
    x_perspective = (f / depth) * p[:, 1]
    y_perspective = (f / depth) * p[:, 0]
    verts2d = np.vstack((x_perspective, y_perspective))
    return verts2d.T, depth

#erwthma D: thelw na stoxeusw shmeio me thn camera kai kanw prooptikh probolh tou 3D shmeiou sthn camera
#c_org h 8esh ths cameras ws pros thn skhnh, c_up to upvector ths cameras, c_lookat to stoxeumeno shmeio estiashs
def project_cam_lookat(f, c_org, c_lookat, c_up, verts3d):
    c_lookat = np.array(c_lookat) + np.array(c_org)
    c_z = c_lookat / np.linalg.norm(c_lookat)
    t = np.array(c_up - np.dot(c_up, c_z) * c_z)
    c_y = t / np.linalg.norm(t)
    c_x = np.cross(c_y, c_z)
    c_x, c_y, c_z = create_tuples(c_x, c_y, c_z)
    return project_cam(f, c_org, c_x, c_y, c_z, verts3d)

#erwthma E: thelw na antistoixisw diastaseis petasmatos cameras se diastaseis canva/pixels eikonas
#dinw diastaseis cameras kai eikonas HxW(YpsosxPlatos) kai ta pros probolh(2D syntetagmenes) shmeia 
def rasterize(verts2d, img_h, img_w, cam_h, cam_w):
    verts_rast = np.zeros((len(verts2d), 2))
    w_anal = img_w / cam_w
    h_anal = img_h / cam_h
    for i in range(len(verts2d)):
        verts_rast[i, 0] = np.around((verts2d[i, 0] + cam_h / 2) * h_anal)
        verts_rast[i, 1] = np.around((-verts2d[i, 1] + cam_w / 2) * w_anal)

    return verts_rast

#erwthma Z: thelw na apeikonisw se fwtografia eikonas to 3D antikeimeno mou afou kanw prooptikh probolh
#dinw ta 3D shmeia(verts3d-Nx3),tis koryfes twn sxhmatismenwn trigwnwn(faces-Nx3),to RGB xrwma ka8e koryfhs(vcolors-Nx3) kai ta alla gnwsta opws panw
# def render_object(verts3d, faces, vcolors, img_h, img_w, cam_h, cam_w, f, c_org, c_lookat, c_up):
#     verts2d, depth = project_cam_lookat(f, c_org, c_lookat, c_up, verts3d)
#     verts2d = rasterize(verts2d, img_h, img_w, cam_h, cam_w)
#     #texnhth peristrofh gia na fainetai isxio an theloume
#     #verts2d[:, 0], verts2d[:, 1] = verts2d[:, 1], verts2d[:, 0].copy() 
#     verts2d = np.array(verts2d).astype(int)
#     verts2d = verts2d.tolist()
#     depth = depth.tolist()
#     faces = faces.tolist()
#     vcolors = vcolors.tolist()
#     return render(verts2d, faces, vcolors, depth, "gouraud")


###############ergasia1
#synarthseis ypologismou

#erwthma A: thelw eksodo thn value pou einai triada xrwmatos(1x3) gia to shmeio grammikhs parembolhs x.

#x shmeio parembolhs, x1 kai x2 : basei autwn tha ginei h parembolh. oi syntetagmenes autes anaferontai eite gia orizonties eite gia ka8etes.
#c1 kai c2 einai 3D(1x3) times xrwmatos twn x1 kai x2 antistoixa. 
def interpolate_color(x1, x2, x, c1: list, c2: list):
    #akolou8w thn logikh tou kyrtou grammikoy syndyasmou c12= lambda*c1 + (1-lambda)*c2,opou...
    lambd = (x2 - x) / (x2 - x1)
    value = lambd * np.array(c1) + (1 - lambd) * np.array(c2)
 
    return value

#erwthma B: na kanw plhrwsh trigwnwn kai na tropopoiei ton img MxNx3(YpsosxPlatosxXrwmata) pou tha exei tis ypologismenes times (Ri,Gi,Bi) alla kai ta proyparxonta trigwna ths eisodou img(akolou8w thn
#logikh ths epikalypshs gia tyxon koinwn xrwmatismenwn shmeiwn pou yphrxan hdh apo alles plhrwseis)

#algorithmos tou Bresenham gia grammh kai epistrefei tis syntetagmenes twn shmeiwn tou eu8ygrammou tmhmatos https://codingee.com/computer-graphics-program-to-implement-bresenhams-line-generation-algorithm/
def bresenham_line(verts2d1: list, verts2d2: list):
    x1 = verts2d1[0]
    x2 = verts2d2[0]
    dx = abs(x2 - x1)

    y1 = verts2d1[1]
    y2 = verts2d2[1]
    dy = abs(y2 - y1)

    x, y = x1, y1
    #an h klish > 1 tote xreiazetai antallagh rolwn x kai y
    swapper = False
    if dy > dx: #gia na apofygw thn logikh ths diaireshs pou prokyptei sfalma gia dx = 0
        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        swapper = True

    p = 2 * dy - dx

    c2d = [] #zeugari syntetagmenwn pou tha epistrepsw
    if swapper:
        c2d.append([y, x])
    else:
        c2d.append([x, y])

    for k in range(dx):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2*(dy - dx)
        else:
            p = p + 2*dy
        x = x + 1 if x < x2 else x - 1
        

        if swapper:
            c2d.append([y, x])
        else:
            c2d.append([x, y])

    return c2d

#8ewrw canva MxN kai h img exei sthn ousia MxNx3 me tyxon proyparxonta trigwna. vert2d int 3x2 opou se ka8e grammh tou exei tis 2D syntetagmenes mias koryfhs trigwnou
#vcolors einai 3x3 opou exei tis 3 koryfes kai se ka8e grammh tou exei to xrwma mias koryfhs me thn logikh RGB(times [0,1]). shade_t einai String gia mode "flat" h "gouraud" 
def shade_triangle(img: list, verts2d: list, vcolors: list, shade_t: str):

    #bresenham_line gia na sxhmatisw thn ka8e eu8ygrammh akmh tou trigwnou kai pairnw tous pinakes twn shmeiwn twn akmwn
    e1 = bresenham_line(verts2d[0], verts2d[1])
    e2 = bresenham_line(verts2d[1], verts2d[2])
    e3 = bresenham_line(verts2d[2], verts2d[0])
    #apo autous tous pinakes taksinomw tis syntetagmenes twn shmeiwn ths akmhs ana grammes tou pinaka wste na brw thn arxh kai to telos ths grammhs sarwshs
    e1 = sorted(e1, key=lambda x: x[0])
    e2 = sorted(e2, key=lambda x: x[0])
    e3 = sorted(e3, key=lambda x: x[0])
    #gia flat thelw sthn ousia a)meso oro twn 3 koryfwn gia ka8e xrwma(R,G,B) kai b)na xrwmatisw ola ta shmeia me to idio xrwma
    color_mean = [0, 0, 0]
    if shade_t == "flat":
        #a)
        color_mean[0] = (vcolors[0][0] + vcolors[1][0] + vcolors[2][0])/ 3
        color_mean[1] = (vcolors[0][1] + vcolors[1][1] + vcolors[2][1])/ 3
        color_mean[2] = (vcolors[0][2] + vcolors[1][2] + vcolors[2][2])/ 3
        #b)
        for point in e1:
            img[point[1]][point[0]] = list(color_mean)
        for point in e2:
            img[point[1]][point[0]] = list(color_mean)
        for point in e3:
            img[point[1]][point[0]] = list(color_mean)

    #gia gouraud ana sarwsh grammhs y thelw parembolh gia ta A(a,y) k B(b,y) kai meta basei autwn sta shmeia ths idias grammhs P(blabla,y) https://prnt.sc/bIZtynqgdfoi
    #gia ka8e akmh tou trigwnou :a)ypologizw xrwma gia ta shmeia ths akmhs ,b)an den einai koryfh tou trigwnou tote kanw parembolh , c)alliws to pairnw apo ta gnwsta
    elif shade_t == "gouraud":
        #a)
        for point in e1:
            #b)
            if 0 <= point[1] <= len(img[1]) - 1 and 0 <= point[0] <= len(img[0]) - 1:
                if point not in verts2d:
                    #an h klish > 1 tote xreiazetai antallagh rolwn x kai y 
                    if abs(verts2d[0][0] - verts2d[1][0]) > abs(verts2d[0][1] - verts2d[1][1]):
                        img[point[1]][point[0]] = interpolate_color(verts2d[0][0], verts2d[1][0], point[0], vcolors[0], vcolors[1])
                    else:
                        img[point[1]][point[0]] = interpolate_color(verts2d[0][1], verts2d[1][1], point[1], vcolors[0], vcolors[1])
                #c)
                else:
                    index = verts2d.index(point)
                    img[point[1]][point[0]] = vcolors[index]

        #a)
        for point in e2:
            #b)
            if 0 <= point[1] <= len(img[1]) - 1 and 0 <= point[0] <= len(img[0]) - 1:
                if point not in verts2d:
                    #an h klish > 1 tote xreiazetai antallagh rolwn x kai y 
                    if abs(verts2d[1][0] - verts2d[2][0]) > abs(verts2d[1][1] - verts2d[2][1]):
                        img[point[1]][point[0]] = interpolate_color(verts2d[1][0], verts2d[2][0], point[0], vcolors[1], vcolors[2])
                    else:
                        img[point[1]][point[0]] = interpolate_color(verts2d[1][1], verts2d[2][1], point[1], vcolors[1], vcolors[2])
                #c)                                                    
                else:
                    index = verts2d.index(point)
                    img[point[1]][point[0]] = vcolors[index]

        #a)
        for point in e3:
            #b)
            if 0 <= point[1] <= len(img[1]) - 1 and 0 <= point[0] <= len(img[0]) - 1:
                if point not in verts2d:
                    if abs(verts2d[2][0] - verts2d[0][0]) > abs(verts2d[2][1] - verts2d[0][1]):
                        # If gradient > 1
                        img[point[1]][point[0]] = interpolate_color(verts2d[2][0], verts2d[0][0], point[0], vcolors[2], vcolors[0])
                    else:
                        img[point[1]][point[0]] = interpolate_color(verts2d[2][1], verts2d[0][1], point[1], vcolors[2], vcolors[0])
                #c)
                else:
                    index = verts2d.index(point)
                    img[point[1]][point[0]] = vcolors[index]
    else:
        print("Incorrect shading mode given. You can only use 'flat' or 'gouraud'! Now exiting...\n")
        exit(0)

    bottom_scanline = min(e1[0][0], e2[0][0], e3[0][0]) #h arxikh grammh sarwshs basei ths mikroterhs timhs twn prwtwn shmeiwn twn akmwn
    top_scanline = max(e1[-1][0], e2[-1][0], e3[-1][0]) #h teleutaia grammh sarwshs basei ths megalyterhs timhs twn teleutaiwn shmeiwn twn akmwn

    #lista energwn shmeiwn kai ananewsh auths se ka8e nea grammh sarwshs
    for scanline in range(bottom_scanline, top_scanline + 1):
        if 0 <= scanline <= len(img[0]) - 1:
            active_points = [] 
            for point in e1:
                if point[0] == scanline:
                    active_points.append(point)
            for point in e2:
                if point[0] == scanline:
                    active_points.append(point)
            for point in e3:
                if point[0] == scanline:
                    active_points.append(point)

            #an brw koryfh tote paw sthn epomenh epanalhpsh
            if len(active_points) == 1:
                continue
            else:
                #apo thn lista energwn shmeiwn pairnw thn elaxisth timh kata sthlh
                min_col_active = np.array(active_points).min(axis=0)[1]

                #apo thn lista energwn shmeiwn pairnw thn megisth timh kata sthlh 
                max_col_active = np.array(active_points).max(axis=0)[1]

                #zwgrafise ta energa shmeia metaksy twn oriakwn sthlwn
                for i in range(min_col_active + 1, max_col_active):
                    
                    if 0 <= i <= len(img[1]) - 1:
                        if [i, scanline] in active_points:
                            continue
                        else:
                            if shade_t == "flat":
                                img[i][scanline] = list(color_mean)
                            elif shade_t == "gouraud":
                                if max_col_active > len(img[1]) - 1:
                                    max_col_active = len(img[1]) - 1
                                if 0 > max_col_active:
                                    max_col_active = 0
                                if 0 <=min_col_active <= len(img[1]) - 1:
                                    img[i][scanline] = interpolate_color(min_col_active, max_col_active, i, img[min_col_active][scanline], img[max_col_active][scanline])

#erwthma C: na kanw teliko xrwmatismo stoixeiwn kai na epistrefei thn img MxNx3, tetoia wste na exei K xrwmatismena trigwna ta opoia proballoun 3D antikeimeno se 2D

#verts2d me Lx2 periexei tis 2D syntetagmenes twn L koryfwn ths eikonas. faces me Kx3 periexei tis 3 koryfes twn K sxhmatismenwn trigwnwn
#depth me Lx1 dhlwnei to ba8os ka8e koryfhs prin thn 2D probolh tou 3D antikeimenou(makrinotero~megalytero ba8os->kontinotero~mikrotero ba8os). shade_t einai String gia mode "flat" h "gouraud" 
def render(verts2d: list, faces: list, vcolors: list, depth: list, shade_t: str):
    #8elw canva me aspro fonto apo pisw me analysh 512x512
    m = 512
    n = 512
    img = [[[1.0 for i in range(3)] for j in range(m)] for k in range(n)]

    triangle_colors = []
    triangle_depths = []
    for triangle in faces:
        #se poia 8esh einai to trigwno sthn lista faces
        index = faces.index(triangle)
        #sthn antistoixh 8esh bazw tis syntetagmenes twn triwn koryfwn
        faces[index] = [verts2d[triangle[0]], verts2d[triangle[1]], verts2d[triangle[2]]]

        #ypologizw to ba8os xrwmatos tou trigwnou pou sxhmatizetai basei twn triwn autwn koryfwn
        new_depth = (depth[triangle[0]] + depth[triangle[1]] + depth[triangle[2]]) / 3
        triangle_colors.append([vcolors[triangle[0]], vcolors[triangle[1]], vcolors[triangle[2]]])
        triangle_depths.append(new_depth)

    zipped = zip(triangle_depths, faces, triangle_colors)
    triangle_depths, faces, triangle_colors = zip(*sorted(zipped, key=lambda x: -x[0])) #taksinomw ta ba8h xrwmatos apo mikrotera se megalytera se mikrotera

    for triangle in faces:
        shade_triangle(img, triangle, triangle_colors[faces.index(triangle)], shade_t)

    return img


##############################################

#antlhsh dedomenwn
start = time.time()
data = np.load("h3.npy", allow_pickle=True).tolist()
#print(data)
verts = np.array(data['verts']) #3ades
vertex_colors = np.array(data['vertex_colors']) #3ades
face_indices = np.array(data['face_indices']) #3ades
depth = np.array(data['depth']) #ba8h twn antikeimenwn
cam_eye = np.array(data['cam_eye']) #0 0 15000
cam_up = np.array(data['cam_up']) #0 1 0
cam_lookat = np.array(data['cam_lookat']) #0 0 0
#ta k 0.1 , 0.8 , 1.0
ka = np.array(data['ka'])
kd = np.array(data['kd'])
ks = np.array(data['ks'])
n = np.array(data['n']) #1
light_positions = np.array(data['light_positions']) #0 1000 0
light_intensities = np.array(data['light_intensities']) #1 1 1
Ia = np.array(data['Ia']) #0.2 0.2 0.2
# img512x512 se cam15x15
M = np.array(data['M'])
N = np.array(data['N'])
W = np.array(data['W'])
H = np.array(data['H'])
bg_color = np.array(data['bg_color']) # 1 1 1
f = 70 #apo prohgoumenh ekfwnhsh focal length
end = time.time()
print('Data management execution time : ', end - start)

#peiramata gouraud
start = time.time()
img = render_object('gouraud', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'combined')
end = time.time()
plt.figure(1)
plt.imshow(img)
plt.title('Gouraud Shading Combined Lighting')
plt.show()
plt2.imsave('g1.png',np.array(img))
print('Figure1 rendering execution time : ', end - start)
##
start = time.time()
img = render_object('gouraud', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'ambient')
end = time.time()
plt.figure(2)
plt.imshow(img)
plt.title('Gouraud Shading Ambient Lighting')
plt.show()
plt2.imsave('g2.png',np.array(img))
print('Figure2 rendering execution time : ', end - start)
##
start = time.time()
img = render_object('gouraud', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'diffusion')
end = time.time()
plt.figure(3)
plt.imshow(img)
plt.title('Gouraud Shading Diffusion Lighting')
plt.show()
plt2.imsave('g3.png',np.array(img))
print('Figure3 rendering execution time : ', end - start)
##
start = time.time()
img = render_object('gouraud', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'specular')
end = time.time()
plt.figure(4)
plt.imshow(img)
plt.title('Gouraud Shading Specular Lighting')
plt.show()
plt2.imsave('g4.png',np.array(img))
print('Figure4 rendering execution time : ', end - start)

#peiramata phong
start = time.time()
img = render_object('phong', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'combined')
end = time.time()
plt.figure(1)
plt.imshow(img)
plt.title('Phong Shading Combined Lighting')
plt.show()
plt2.imsave('ph1.png',np.array(img))
print('Figure1 rendering execution time : ', end - start)
#
start = time.time()
img = render_object('phong', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'ambient')
end = time.time()
plt.figure(2)
plt.imshow(img)
plt.title('Phong Shading Ambient Lighting')
plt.show()
plt2.imsave('ph2.png',np.array(img))
print('Figure2 rendering execution time : ', end - start)
#
start = time.time()
img = render_object('phong', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'diffusion')
end = time.time()
plt.figure(3)
plt.imshow(img)
plt.title('Phong Shading Diffusion Lighting')
plt.show()
plt2.imsave('ph3.png',np.array(img))
print('Figure3 rendering execution time : ', end - start)
#
start = time.time()
img = render_object('phong', f, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, 'specular')
end = time.time()
plt.figure(4)
plt.imshow(img)
plt.title('Phong Shading Specular Lighting')
plt.show()
plt2.imsave('ph4.png',np.array(img))
print('Figure4 rendering execution time : ', end - start)

print("Program has been exited successfully!")

