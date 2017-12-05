# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv, xyz2rgb
import os
import time


def mask(A, w):
	v = A.shape[0]
	h = A.shape[1]
	hor = int(h/w)
	ver = int(v/w)
	return np.array([ver, hor])


def mask_esp(A):
	v = A.shape[0]
	h = A.shape[1]
	mask = np.ones((v, h))
	maskc = np.zeros((v, h))
	for i in range(v):
    		for j in range(h):
        		r = (i-int(v/2))**2/(3*v/8)**2+(j-int(h/2))**2/(3*h/8)**2
            		if r<=1:
               			maskc[i, j] = 1
	mask = mask-maskc
	mask[int(v/2):-1, :] = 0
	mask[:, int(h/2):-1] = 0
	return mask, np.fliplr(mask), np.flipud(mask), np.flipud(np.fliplr(mask)), maskc


def histo(A, mas='4'):
	
	arrh = np.array([])
	arrs = np.array([])
	arrv = np.array([])
	if mas=='4' or mas=='5' or mas=='6':
		if mas=='4':
			n = 4
		elif mas=='5':
			n = 5
		elif mas=='6':
			n = 6
		m = mask(A, n)
		for i in range(n): # recorre imagen
			for j in range(n):
				f = A[m[0]*i:m[0]*(i+1), m[1]*j:m[1]*(j+1), :]
				g1 = np.ravel(f[:,:,0])
				g2 = np.ravel(f[:,:,1])
				g3 = np.ravel(f[:,:,2])
				valh, edg = np.histogram(g1, bins=8, density=True)
				vals, edg = np.histogram(g2, bins=12, density=True)
				valv, edg = np.histogram(g3, bins=3, density=True)			
				arrh = np.concatenate((arrh, valh))
				arrs = np.concatenate((arrs, vals))
				arrv = np.concatenate((arrv, valv))

	elif mas=='e':
		a, b, c, d, e = mask_esp(A)
		M = [a, b, c, d, e]
		for i in M:	
			f1 = A[:,:,0]*i
			f2 = A[:,:,1]*i
			f3 = A[:,:,2]*i
			f1 = np.ravel(f1)
			f2 = np.ravel(f2)
			f3 = np.ravel(f3)
			arg = np.nonzero(np.ravel(i)==0)[0]
			f1 = np.delete(f1, arg)
			f2 = np.delete(f2, arg)
			f3 = np.delete(f3, arg)
			valh, edg = np.histogram(f1, bins=8, density=True)
			vals, edg = np.histogram(f2, bins=12, density=True)
			valv, edg = np.histogram(f3, bins=3, density=True)			
			arrh = np.concatenate((arrh, valh))
			arrs = np.concatenate((arrs, vals))
			arrv = np.concatenate((arrv, valv))
	return arrh, arrs, arrv
	

def carac(A, mas='4'):
	A = rgb2hsv(A)	
	B = histo(A, mas=mas)
	return B


def distancia(X, Y):
	dis = np.sqrt(np.sum((X - Y)**2))
	return dis


def ranking(A, W, w): 
	# A: vector de distancias
	# W: vector de clases de database
	# w: clase de query
	arg = np.argsort(A) # argumentos del orden min de distancia
	W1 = W[arg] # vector de clases de database ordenado segun arg
	arg1 = np.nonzero(W1==w)[0] # determina las posiciones de los elementos de la clase
	N = np.sum(W==w) # numero de muestras por clase		
	r = np.sum(arg1)/N	
	return r


def ranking_norma(A, W, w): 
	# A: vector de distancias
	# W: vector de clases de database
	# w: clase de query
	arg = np.argsort(A) # argumentos del orden min
	W1 = W[arg] # vector de clases de database ordenado segun arg
	arg1 = np.nonzero(W1==w)[0] # determina las posiciones de los elementos de la clase
	N = np.sum(W==w) # numero de muestras por clase	
	n = len(W) # numero total de imagenes en database	
	r = (np.sum(arg1) - N*(N + 1)/2)/N/n	
	return r


def IRP(A1, A2, Q1, Q2, v):
	# IRP: inverse rank position
	# A1: vector de clases de database
	# A2: vector de id de database
	# Q1, Q2: vector de distancias de cada metodo
	# v: imagen de query a analizar
	# return array con los nombres de las primeras 10 imagenes segun irp
	n = len(Q1[v]) # numero de imaganes en database
	arg1 = np.argsort(Q1[v]) # ordena las distancias 
	arg2 = np.argsort(Q2[v])
	irp = []
	for i in range(n): # recorre imagenes en database
		a1 = np.nonzero(arg1==i)[0][0] + 1 # determina la posicion de la imagen de database i
		a2 = np.nonzero(arg2==i)[0][0] + 1
		p1 = a1 
		p2 = a2
		irp.append(1/((1/p1)+(1/p2))) # irp
	irp = np.array(irp)
	argi = np.argsort(irp) # ordena imagenes segun irp
	Name = []
	for i in range(10):
		name = str(int(A1[argi][i]))+A2[argi][i] # escribe nombre de la imagen
		Name.append(name)
	Name = np.array(Name)
	return Name


def dist_fft(A1, A2):
	return np.sum((A1 - A2)**2)
	

direct1 = '/home/mauricio/Documents/Imagenes/img_database' 
direct2 = '/home/mauricio/Documents/Imagenes/img_query'	
direct11 = '/home/mauricio/Documents/Imagenes/fft_database' 
direct22 = '/home/mauricio/Documents/Imagenes/fft_query'

mascaras = np.array(['4', '5', '6', 'e'])
# extracion de caracteristicas segun mascara

for masc in mascaras:

	# imagenes de database
	H = []
	S = []
	V = []
	clases = []
	idb = []
	Ti = time.time()
	j = 0
	print 'Creando histogramas database mascara '+masc
	for i in os.listdir(direct1):
		ti = time.time()
		Im = imread(direct1+'/'+i)
		hh, hs, hv = carac(Im, mas=masc)
		H.append(hh)
		S.append(hs)
		V.append(hv)
		clase = int(i[:3])
		clases.append(clase)
		idb.append(i[3:6])
		tf = time.time()
		dt = tf-ti
		print 'tiempo', dt, 's'
		if j==20:
			break
		j += 1
	Tf = time.time()
	dT = Tf - Ti
	print 'tiempo total', dT/60, 'min'
	H = np.array(H)
	S = np.array(S)
	V = np.array(V)
	clases = np.array(clases)
	idb = np.array(idb)

	np.savetxt('H_database_mask_'+masc, H)
	np.savetxt('S_database_mask_'+masc, S)
	np.savetxt('V_database_mask_'+masc, V)
	np.savetxt('clase_database_mask_'+masc, clases)
	np.savetxt('id_database_mask_'+masc, idb, delimiter=" ", fmt="%s")

	#print 'estimado', dt*407/60, 'min'

	# imagenes de query

	Hq = []
	Sq = []
	Vq = []
	clasesq = []
	idq = []
	Tiq = time.time()
	j = 0
	print 'Creando histogramas query mascara '+masc
	for i in os.listdir(direct2):
		tiq = time.time()
		Im = imread(direct2+'/'+i)
		hh, hs, hv = carac(Im, mas=masc)
		Hq.append(hh)
		Sq.append(hs)
		Vq.append(hv)
		claseq = int(i[:3])
		clasesq.append(claseq)
		idq.append(i[3:6])
		tfq = time.time()
		dtq = tf-ti
		print 'tiempo', dtq, 's'
		if j==5:
			break
		j += 1
	Tfq = time.time()
	dTq = Tfq - Tiq
	print 'tiempo total', dTq/60, 'min'
	Hq = np.array(Hq)
	Sq = np.array(Sq)
	Vq = np.array(Vq)
	clasesq = np.array(clasesq)
	idq = np.array(idq)

	np.savetxt('H_query_mask_'+masc, Hq)
	np.savetxt('S_query_mask_'+masc, Sq)
	np.savetxt('V_query_mask_'+masc, Vq)
	np.savetxt('clase_query_mask_'+masc, clasesq)
	np.savetxt('id_query_mask_'+masc, idq, delimiter=" ", fmt="%s")

# recorre las imagenes de query y determina la imagen de database mas cercana
# en funcion de minimizar la distancia entre los 3 histogramas

DD = [] # guarda distancias de cada metod

index = np.random.randint(low=0, high=150) # selecciona imagen de query en random 
# indice de imagen a utilizar (no corresponde a la tercera imagen de
# query, sino que a la tercera imagen en ser procesada por el programa.)

Repo = []
for masc in mascaras:
	print 'mascara ', masc
	
	H = np.loadtxt('H_database_mask_'+masc)
	S = np.loadtxt('S_database_mask_'+masc)
	V = np.loadtxt('V_database_mask_'+masc)
	clase = np.loadtxt('clase_database_mask_'+masc)
	idb = np.genfromtxt('id_database_mask_'+masc, dtype='str')

	Hq = np.loadtxt('H_query_mask_'+masc)
	Sq = np.loadtxt('S_query_mask_'+masc)
	Vq = np.loadtxt('V_query_mask_'+masc)
	claseq = np.loadtxt('clase_query_mask_'+masc)
	idq = np.genfromtxt('id_query_mask_'+masc, dtype='str')

	D = [] # vector de distancias
	for i1, i2, i3 in zip(Hq, Sq, Vq):
		l = []
		for j1, j2, j3 in zip(H, S, V):
			dh = distancia(j1, i1)
			ds = distancia(j2, i2)
			dv = distancia(j3, i3)
			prom = (dh + ds + dv)/3
			l.append(prom)
		l = np.array(l)
		D.append(l)
	D = np.array(D)
	DD.append(D)			

	# rankings

	# primero
	R = []
	Rn = []
	for i in range(D.shape[0]):
		r = ranking(D[i], clase, claseq[i])
		rn = ranking_norma(D[i], clase, claseq[i])
		R.append(r)
		Rn.append(rn)
	R = np.array(R)
	Rn = np.array(Rn)
	np.savetxt('rank_mask_'+masc, R)
	np.savetxt('rank_norm_mask_'+masc, Rn)
	
	# resultados con imagenes

	num = 10 # numero de imagenes a seleccionar
	#index = np.random.randint(low=0, high=D.shape[0]) # selecciona imagen de query en random	

	img_cl_q = int(claseq[index]) # codigo de clase de las imagen query seleccionada
	img_id_q = idq[index] # codigo de identificacion de la imagen query seleccionada

	# guarda imagen query seleccionada
	img = imread(direct2+'/'+str(img_cl_q)+img_id_q+'.jpg')
	plt.clf()
	plt.imshow(img)
	plt.axis('off')
	plt.savefig('img_query')

	# resultados relevancia

	args = np.argsort(D[index]) # args del orden en relevancia
	cl_img = clase[args][:num] # codigo de clase de las imagenes seleccionadas
	id_img = idb[args][:num] # codigo de identificacion
	
	plt.clf()
	for i in range(num):
		img = imread(direct1+'/'+str(int(cl_img[i]))+id_img[i]+'.jpg')
		plt.subplot(2, 5, i+1)
	 	plt.imshow(img) # aspect='auto'
		plt.title(str(int(i+1)))
		plt.axis('off')
	plt.savefig('relevancia_mask_'+masc)	

	# resultados ranking

	print 'posicion promedio de la clase de consulta', R[index]

	# resultados ranking normalizado

	print 'posicion promedio de la clase de consulta (normalizada)', Rn[index]
	
	Repo.append(np.array([R[index], Rn[index]]))
		
Repo = np.array(Repo)
DD = np.array(DD)
np.savetxt('img_repo', Repo)

# seleccion mejores dos metodos

proms = np.mean(np.mean(DD, axis=1), axis=1) # distancia promedio por metodo
# respecto a todas imagenes query y database
print 'metodos mejores', mascaras[np.argsort(proms)][:2]

# realiza rank irp 
print 'rank irp'
m1 = np.argsort(proms)[0]
m2 = np.argsort(proms)[1]
images = IRP(clase, idb, DD[m1], DD[m2], index)

plt.clf()
k = 0
for i in images:
	img = imread(direct1+'/'+i+'.jpg')
	plt.subplot(2, 5, k+1)
 	plt.imshow(img) # aspect='auto'
	plt.title(str(int(k+1)))
	plt.axis('off')
	k += 1
plt.savefig('ranking_irp_masks_'+str(m1)+'_'+str(m2))

# otra caracteristica
# transformada de Fourier 2D

# recorre las imagenes query y las database y calcula y guarda fft2
for i in os.listdir(direct1):
	print i
	imd = imread(direct1+'/'+i, as_grey=True)
	imd_fft = np.abs(np.fft.fft2(imd, s=[400 ,400]))
	imd_fft = imd_fft/np.max(imd_fft)
	np.savetxt('/home/mauricio/Documents/Imagenes/fft_database/'+i[:-4], imd_fft)

for i in os.listdir(direct2):
	print i
	imq = imread(direct2+'/'+i, as_grey=True)
	imq_fft = np.abs(np.fft.fft2(imq, s=[400 ,400]))
	imq_fft = imq_fft/np.max(imq_fft)
	np.savetxt('/home/mauricio/Documents/Imagenes/fft_query/'+i[:-4], imq_fft)
"""
"""
# calcula distancias fft2
L = []
for i in os.listdir(direct22):
	print i
	l = []
	imq_fft = np.loadtxt(direct22+'/'+i)
	for j in os.listdir(direct11):
		imd_fft = np.loadtxt(direct11+'/'+j)
		l.append(dist_fft(imq_fft, imd_fft))
	l = np.array(l)
	L.append(l)
L = np.array(L)
np.savetxt('distancias_fft2', L)

# guarda los nombres de las imagenes fft2 de database y query
names_fft_b = []
for i in os.listdir(direct11):
	names_fft_b.append(i) 
names_fft_b = np.array(names_fft_b)

names_fft_q = []
for i in os.listdir(direct22):
	names_fft_q.append(i) 
names_fft_q = np.array(names_fft_q)

# imagen seleccionada
plt.clf()
img = imread(direct2+'/'+names_fft_q[index]+'.jpg')
plt.imshow(img)
plt.axis('off')
plt.savefig('img_fft')

# convierte strigs de names a int y str de clase e id

fft_cl_b = []
fft_cl_q = []
fft_id_b = []
fft_id_q = []
for i in names_fft_b:
	fft_cl_b.append(int(i[:3]))
	fft_id_b.append(i[3:6])
for i in names_fft_q:
	fft_cl_q.append(int(i[:3]))
	fft_id_q.append(i[3:6])
fft_cl_b = np.array(fft_cl_b)
fft_cl_q = np.array(fft_cl_q)
fft_id_b = np.array(fft_id_b)
fft_id_q = np.array(fft_id_q)

# resultados ranking normalizado

Rf = []
Rfn = []
for i in range(L.shape[0]):
	rf = ranking(L[i], fft_cl_b, fft_cl_q[i])
	rfn = ranking_norma(L[i], fft_cl_b, fft_cl_q[i])
	Rf.append(rf)
	Rfn.append(rfn)
Rf = np.array(Rf)
Rfn = np.array(Rfn)

print 'rank:', Rf[index]
print 'rank normal:', Rfn[index]

# resultados espectral

L = np.loadtxt('distancias_fft2')
args = np.argsort(L[index]) # args del orden en relevancia
names_1 = names_fft_b[args][:num] # codigo de clase de las imagenes seleccionadas
plt.clf()
for i in range(num):
	img = imread(direct1+'/'+names_1[i]+'.jpg')
	plt.subplot(2, 5, i+1)
 	plt.imshow(img) # aspect='auto'
	plt.title(str(int(i+1)))
	plt.axis('off')
plt.savefig('relevancia_fft_mask')

