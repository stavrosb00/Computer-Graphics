import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plt2
import time



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
                if [i, scanline] in active_points:
                    continue
                else:
                    if shade_t == "flat":
                        img[i][scanline] = list(color_mean)
                    elif shade_t == "gouraud":
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

#antlhsh dedomenwn kai metatroph se listes
start = time.time()
data = np.load('hw1.npy', allow_pickle=True)
#print(data)
vcolors = data[()]['vcolors'].tolist()
faces = data[()]['faces'].tolist()
depth = data[()]['depth'].tolist()
verts2d = data[()]['verts2d'].astype(int).tolist()

#print(verts2d)
#print(vcolors)
#print(faces)
#print(depth)
end = time.time()
print('Data management execution time : ', end - start)
start = time.time()
img = render(verts2d, faces, vcolors, depth, "gouraud")
end = time.time()
print('Rendering execution time : ', end - start)
plt.imshow(img, interpolation='nearest')
plt.title('Gouraud Shading')
plt.show()
plt2.imsave('shade_gouraud.png',np.array(img))
print("Program has been exited successfully!")