#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:20:08 2019

@author: clement.metz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import os
import cv2

parent_path = 'photos/03_05/x1' # Beaucoup de cellules
listpath = os.listdir('./'+parent_path)
for p in listpath:
    path = parent_path + '/' + p
    try:
        os.mkdir(path[:-4])
    except FileExistsError:
        pass
    #path='NR_A,B,C.jpg' #Tâche noire en haut
    #path='NR_AetD.jpg' # Amas de cellules divisé en régions
    #path='x0.75_A.jpg' #peu de cellules
    #path='testtas-2.jpg'
    #path='x0.75_A bis.jpg' #une seule cellule
    img = cv2.imread(path)
    gdgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Resize
    dim = img.shape
    factdiv = 5
    newDim = (int(dim[1] / factdiv), int(dim[0] / factdiv))
    img = cv2.resize(img, dsize=newDim, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray, cmap='gray')
    #plt.show()
    normalizedImg = np.zeros(gray.shape)
    normalizedImg = cv2.normalize(gray, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    
    #plt.imshow(normalizedImg, cmap='gray')
    #plt.show()
    ret, thresh = cv2.threshold(normalizedImg,0,1 ,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    #dist_transform[dist_transform>10] /=5
    coeff = min(dist_transform.max(), 10)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*coeff,255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    #plt.imshow(unknown, cmap='gray')
    #plt.show()
    
    
    
    distance = dist_transform
    #distance = ndi.distance_transform_edt(sure_fg)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=sure_fg)
    markers = ndi.label(local_maxi)[0]
    
    
    labels = watershed(-distance, markers, mask=sure_fg)
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(normalizedImg, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap='gray', interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap="jet", interpolation='nearest')
    ax[2].set_title('Separated objects')
    
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout()
    plt.show()
    
    ###############################################################################
    #              ROGNAGE                                                        #
    ###############################################################################
    
    #Liste des valeurs comprises dans les labels
    valeurs_wat = []
    allcells = []
    carres = []   #liste des coordonnées de chaque région
    for i in range(1, labels.max()+1):
        for ligne in labels:
            if i in ligne:
                if i not in valeurs_wat:
                    valeurs_wat.append(i)
    
    print(f"{len(valeurs_wat)} régions détectées")
    print(f'------ Rognage ------\n')
    
    lastprint = 0 #Dernier pourcentage affiché sur la console
    print(f'0 % effectués, {len(valeurs_wat)} régions à traiter, {len(allcells)} cellules détectées')
    
    #labels = np.array([[0,0,0,0,0],[0 ,0, 1, 0, 0],[0, 1, 1, 1, 0],[0, 0, 1, 0, 0],[0,0,0,0,0]])
    labelsT = labels.transpose()
    for i in valeurs_wat:                       #Pour chaque cellule à ségmenter (pour chaque couleur
        #trouver le pixel du haut gauche
        pixHaut = [0, 0] #[y, x]
        found = False #booléen qui annonce si on a trouvé le pixel ou pas
        #Trouver la ligne
        for ligne in range(labels.shape[0]):
            for colonne in range(labels.shape[1]):
                if i == labels[ligne, colonne] and not found:
                    pixHaut[0] = ligne
                    found = True
                    break
            if found:
                break
        found = False
        #Trouver la colonne
        for ligne in range(labelsT.shape[0]):
            for colonne in range(labelsT.shape[1]):
                if i == labelsT[ligne, colonne] and not found:
                    pixHaut[1] = ligne
                    found = True
                    break
            if found:
                break
        found = False
        #Trouver le pixel du bas droit
        pixBas = [0, 0]
        
        for ligne in range(labels.shape[0]-1, -1, -1):
            for colonne in range(labels.shape[1]):
                if i == labels[ligne, colonne] and not found:
                    pixBas[0] = ligne
                    found = True
                    break
            if found:
                break
        found = False
        
        #Trouver la colonne
        for ligne in range(labelsT.shape[0]-1, -1, -1):
            for colonne in range(labelsT.shape[1]):
                if i == labelsT[ligne, colonne] and not found:
                    pixBas[1] = ligne
                    found = True
                    break
            if found:
                break
        
        
        # Création du rectangle de rognage
        #marges
        
        marge = 16
        arg1 = max(min([pixHaut[0], pixBas[0]])-marge, 0)
        arg2 = min(max([pixHaut[0], pixBas[0]])+marge, labels.shape[0])
        arg3 = max(min([pixHaut[1], pixBas[1]])-marge, 0)
        arg4 = min(max([pixHaut[1], pixBas[1]])+marge, labels.shape[1])
        
        #Carrétisation de l'image
        if arg2 - arg1 > arg4-arg3:
            while not(arg2 - arg1 == arg4-arg3):
                if arg4 == labels.shape[1]:
                    arg3-=1
                else:
                    arg4+=1
        else:
            while not(arg2 - arg1 == arg4-arg3):
                if arg2 == labels.shape[0]:
                    arg1 -= 1
                else:
                    arg2+=1
        
        cell = gray[arg1:arg2+1, arg3:arg4+1]
        added = False
        #suppression des images trop petites, trop grandes ou trop sombres
        if(68 > cell.shape[0] > 35 and cell.mean() > 90):
            allcells.append(cell)
            added = True
            carres.append([arg1, arg2+1, arg3, arg4+1])
            
        #Affichage de l'avancement de l'algorithme
        
        pct = int(100 * i / len(valeurs_wat))
        if added:
            print('+', end='')
        else:
            print('#', end='')
        if (pct % 10 == 0) and lastprint != pct:
            print(f'\n{pct} % effectués, {i}/{len(valeurs_wat)} régions traitées, {len(allcells)} cellules détectées')
            lastprint = pct
    
    danger = False
    print('')
    if len(allcells) / len(valeurs_wat) <= 0.6:
        print("/!\\ ATTENTION /!\\")
        print("Il est possible qu'un amas de cellules ait falcifié l'analyse ou que l'image ait été difficilement traitée.\n")
        danger = True
            
    grandscarres = np.array(carres) * 5
    allgrandcell = []
    for mask in grandscarres:
        allgrandcell.append(gdgray[mask[0]:mask[1], mask[2]:mask[3]])
        
    # Suppression des éventuels doublons
    print(f'------ Suppression des éventuels doublons ------\n')
    to_delete = []
    seuil_acceptable = 0.7
    cpti = 0
    for i in carres:
        cptj = cpti+1
        # éviter de traîter les doublons déjà supprimés
        if cpti not in to_delete:
            for j in carres[cpti+1:]:
                if not (j[1] < i[0] or i[1] < j[0] or i[3] < j[2] or j[3] < i[2]):
                    #recouvrement
                    #Calcul de la zone de recouvrement
                    dy = 0
                    dx = 0
                    #en y 
                    if i[0] < j[0] and j[1] < i[1]:
                        #j inclus a dans i
                        dy = j[1] - j[0]
                    elif j[0] < i[0] and i[1] < j[1]:
                        #i inclus dans i
                        dy = i[1] - i[0]
                    elif j[1] - i[0] < min(j[1] - j[0], i[1] - i[0]):
                        #j plus haut que i
                        dy = j[1] - i[0]
                    else:
                        #i est alors forcément plus haut que i
                        dy = i[1] - j[0]
                        
                    #en x 
                    if i[2] < j[2] and j[3] < i[3]:
                        #j inclus a dans i
                        dx = j[3] - j[2]
                    elif j[2] < i[2] and i[3] < j[3]:
                        #i inclus dans i
                        dx = i[3] - i[2]
                    elif j[3] - i[2] < min(j[3] - j[2], i[3] - i[2]):
                        #j plus à gauche que i
                        dx = j[3] - i[2]
                    else:
                        #i est alors forcément plus à gauche que i
                        dx = i[3] - j[2]
                        
                    #Calcul du pourcentage de recouvrement de i
                    recouvrement = dy * dx / ((i[1] - i[0]) * (i[3] - i[2]))
                    if recouvrement > seuil_acceptable and cptj not in to_delete:
                        to_delete.append(cptj)
                cptj += 1
        cpti += 1
    
    #suppression des doublons par la fin pour être sûr de supprimer les bons éléments
    to_delete.sort()
    for d in to_delete[::-1]:
        allcells.pop(d)
        allgrandcell.pop(d)
    
    allcells*=factdiv
    
    print("100% effectués")
    print(f"{len(to_delete)} images présentant plus de {int(seuil_acceptable*100)}% de similarité avec au moins une autre image ont été supprimées")
    print(f"{len(allgrandcell)} cellules détectées sur la photo\n")
    
    # Mise à la bonne taille pour l'IA
    print(f'------ Normalisation de la taille des images des cellules ------\n')
    picallcells = []
    for image in allgrandcell:
        picallcells.append(Image.fromarray(image).resize((80, 80), resample=Image.LANCZOS))
    print("100% effectués\n")
    
    
    num = 0
    for im in picallcells:
        plt.imshow(im, cmap = 'gray')
        plt.show()
        im.save(path[0:-4]+"/cell"+str(num)+".png", "PNG", quality=100, optimize=True, progressive=True)
        num+=1
    
    #######################################################################################################
    #   COMPTE RENDU
    
    file = open(path[:-4] + "/Rapport.txt", 'w')
    file.write("Rapport d'analyse d'image\n")
    file.write("-------------------------\n\n")
    file.write(f"{len(allgrandcell)} cellules détectées sur la photo\n")
    if danger:
        file.write("/!\\ ATTENTION /!\\\n")
        file.write("Il est possible qu'un amas de cellules ait falcifié l'analyse ou\nque l'image ait été difficilement traitée.\n")
    else:
        file.write("L'analyse semble s'être déroulée correctement.\n")
    file.write("\nDétails du traîtement\n")
    file.write('---------------------\n\n')
    file.write("- Fichier : " + os.getcwd() + '/' + path + "\n")
    file.write("- Nombre de régions détectées par Watershed : " + str(len(valeurs_wat))+"\n")
    file.write(f"- Nombre de cellules supprimées après filtrage\n  (trop grandes, trop petites ou trop sombres) : {len(valeurs_wat)-len(allgrandcell)-len(to_delete)}\n")
    file.write(f"- {len(to_delete)} images présentant plus de {int(seuil_acceptable*100)}% de similarité\n  avec au moins une autre image ont été supprimées\n")
    file.close()







