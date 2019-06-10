#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import os
import cv2

def merge_elements(d2):
    '''
    Fonction sale et à optimiser qui étant donné une liste de liste de doublons
    retourne une liste qui fusionne tous les doublons dans une seule liste
    bon ça veut rien dire mais tkt ça marche
    '''
    finished = False
    while not finished:
        d3 = []
        for d in d2:
            liste = [k for k in d]
            for num in d:
                for db in d2:
                    if num in db and db != d:
                        for elem in db:
                            liste.append(elem)
            liste = list(set(liste))
            liste.sort()
            d3.append(liste)
        oldsize = len(d3)
        d3 = [list(item) for item in set(tuple(row) for row in d3)]
        if len(d3) == oldsize:
            finished = True
        d2 = d3
    d3.sort()
    return d3

def segmenter(path):
    global picallcells
    img = cv2.imread(path)
    gdgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Resize
    dim = img.shape
    factdiv = dim[1] / 518
    newDim = (518, 388)
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
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    
    #dist_transform[dist_transform>10] /=5
    coeff = min(dist_transform.max(), 10)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*coeff,255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    #plt.imshow(unknown, cmap='gray')
    #plt.show()
    
    
    
    distance = dist_transform
    #distance = ndi.distance_transform_edt(sure_fg)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=sure_fg)
    markers = ndi.label(local_maxi)[0]
    
    
    labels = watershed(-distance, markers, mask=sure_fg)
    
    ###############################################################################
    #              ROGNAGE                                                        #
    ###############################################################################
    
    #Liste des valeurs comprises dans les labels
    valeurs_wat = list(set(labels.flatten().tolist()))[1:]
    allcells = []
    carres = []   #liste des coordonnées de chaque région
    print(f"{len(valeurs_wat)} régions détectées")
    print(f'------ Rognage ------\n')
    
    labelsT = labels.transpose()
    water_remaining = [] # liste des valeurs de watershed considérées comme des cellules
    for i in valeurs_wat:                       #Pour chaque cellule à ségmenter (pour chaque couleur
        #trouver le pixel du haut gauche
        nb_it = 3#((np.count_nonzero(labels==i)>100) + (np.count_nonzero(labels==i)>50) +1)
        temp_labels =cv2.dilate((labels==i).astype(np.uint8),np.ones((5,5), np.uint8),iterations = nb_it)
        pixHaut = [np.min(np.where(temp_labels==1)[0]), np.min(np.where(temp_labels==1)[1])]
       
        #Trouver le pixel du bas droit
        pixBas = [np.max(np.where(temp_labels==1)[0]), np.max(np.where(temp_labels==1)[1])]
        # Création du rectangle de rognage
        #marges
        
        marge = 0
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
        if(cell.shape[0] > 0): #and cell.mean() > 90):
            allcells.append(cell)
            added = True
            carres.append([arg1, arg2+1, arg3, arg4+1])
            water_remaining.append(i)
        
    danger = False
    print(f'{len(water_remaining)} cellules ajoutées')
    if len(allcells) / len(valeurs_wat) <= 0.6:
        print("/!\\ ATTENTION /!\\")
        print("Il est possible qu'un amas de cellules ait falcifié l'analyse ou que l'image ait été difficilement traitée.\n")
        danger = True
        
    temp = (np.array(carres)).astype(int)        
    carres = np.zeros((temp.shape[0], temp.shape[1] + 1))
    carres[:,:-1] = temp
    carres[:,-1] = np.array(water_remaining)
    carres = carres.astype(int)
    
    
    # Fusion des éventuels doublons
    print(f'------ Fusion des éventuels doublons ------\n')
    to_delete = []
    doublons = [] # liste de valeurs de watershed à fusionner
    seuil_acceptable = 0.5
    cpti = 0
    for i in carres:
        doublons_i = [i[4]] # On ajoute dans un premier temps le num du label de i
        cptj = cpti+1
        # éviter de traîter les doublons déjà supprimés
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
                if recouvrement > seuil_acceptable:
                    to_delete.append(cptj)
                    doublons_i.append(j[4])
            cptj += 1
        doublons.append(doublons_i)
        cpti += 1
    
    doublons = merge_elements(doublons) # Tassage des zones à fusionner
    # pour chaque zone à fusionner
    for d in doublons:
        if len(d) > 1:
            for value in d[1:]:
                # Fusion des zones
                labels[labels == value] = d[0]
    print("100 % effectués")
    
    #Liste des valeurs comprises dans les labels
    valeurs_wat = list(set(labels.flatten().tolist()))[1:]
    allcells = []
    carres = []   #liste des coordonnées de chaque région
    print(f"{len(valeurs_wat)} restantes après fusion")
    print(f'------ Mise à jour des régions ------\n')
    
    water_remaining = [] # liste des valeurs de watershed considérées comme des cellules
    for i in valeurs_wat:                       #Pour chaque cellule à ségmenter (pour chaque couleur
        #trouver le pixel du haut gauche
        temp_labels =cv2.dilate((labels==i).astype(np.uint8),np.ones((5,5), np.uint8),iterations = 7)
        pixHaut = [np.min(np.where(temp_labels==1)[0]), np.min(np.where(temp_labels==1)[1])]
       
        #Trouver le pixel du bas droit
        pixBas = [np.max(np.where(temp_labels==1)[0]), np.max(np.where(temp_labels==1)[1])]
        # Création du rectangle de rognage
        #marges
        
        
        marge = 0
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
        #suppression des images trop petites, trop grandes ou trop sombres
        if(cell.shape[0] > 0): #and cell.mean() > 90):
            allcells.append(cell)
            carres.append([arg1, arg2+1, arg3, arg4+1])
            water_remaining.append(i)
        
    danger = False
    print('')
    if len(allcells) / len(valeurs_wat) <= 0.6:
        print("/!\\ ATTENTION /!\\")
        print("Il est possible qu'un amas de cellules ait falcifié l'analyse ou que l'image ait été difficilement traitée.\n")
        danger = True
        
    temp = (np.array(carres)).astype(int)        
    carres = np.zeros((temp.shape[0], temp.shape[1] + 1))
    carres[:,:-1] = temp
    carres[:,-1] = np.array(water_remaining)
    carres = carres.astype(int)
    
    temp = (np.array(carres[:, :-1]) * factdiv).round().astype(int)
    grandscarres = np.zeros((temp.shape[0], temp.shape[1] + 1))
    grandscarres[:,:-1] = temp
    grandscarres[:,-1] = np.array(water_remaining)
    grandscarres = grandscarres.astype(int)
    # Mise à jour de la taille des labels
    grands_labels = cv2.resize(labels.astype(np.float32), dsize = (dim[1], dim[0]), interpolation=cv2.INTER_NEAREST)
    # grands carrés (taille originale) : [xs, xe, ys, ye, num_label]
    allgrandcell = []
    for mask in grandscarres:
        allgrandcell.append(np.multiply(gdgray, cv2.dilate((grands_labels==mask[4]).astype(np.uint8),np.ones((5,5), np.uint8),iterations = 7))[mask[0]:mask[1], mask[2]:mask[3]])
    
    print("100% effectués")
    print(f"{len(to_delete)} images présentant plus de {int(seuil_acceptable*100)}% de similarité avec au moins une autre image ont été supprimées")
    print(f"{len(allgrandcell)} cellules détectées sur la photo\n")
    
    
    # Mise à la bonne taille pour l'IA
    print(f'------ Normalisation de la taille des images des cellules ------\n')
    picallcells = []
    cpt_gd_cells = 0
    for image in allgrandcell:
        if max(image.shape[0], image.shape[1]) < 100:
            bkgrd = np.zeros((100,100))
            ligim = image.shape[0]
            colim = image.shape[1]
            lower = 50 - (ligim // 2)
            upper = 50 + ligim-(ligim // 2)
            left = 50 - (colim // 2)
            right = 50 + colim-(colim // 2)
            bkgrd[lower:upper, left:right] = image
        else:
            bkgrd = image
            cpt_gd_cells += 1
        picallcells.append(Image.fromarray(bkgrd).resize((80, 80), resample=Image.LANCZOS).convert("L"))
    print("100% effectués\n")
    if cpt_gd_cells >1:
        danger = True
    for im in allgrandcell:
        pass
        #plt.imshow(im, cmap='gray')
        #plt.show()
    
    num = 0
    for im in picallcells:
        #plt.imshow(im, cmap = 'gray')
        #plt.show()
        #im.save(path[0:-4]+"/cell"+str(num)+".png", "PNG", quality=100, optimize=True, progressive=True)
        num+=1
    
    
    #######################################################################################################
    #   COMPTE RENDU
    
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(normalizedImg, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap='gray', interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap="jet", interpolation='nearest')
    ax[2].set_title('Separated objects')
    # Get the color map by name:
    cm = plt.get_cmap('jet')
    
    # Apply the colormap like a function to any array:
    colored_image = cm(labels)
    
    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it:
    photo = ImageTk.PhotoImage(Image.fromarray((colored_image[:, :, :] * 255).astype(np.uint8)))
    canvas_image.create_image(0,0,image=photo, anchor=NW)
    canvas_image.image=photo
    canvas_image.grid(row=0, column=0)
    
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout()
       # plt.show()
    fig.savefig(path[0:-4]+"_Details.png", format="png", dpi=200)
    
    file = open(path[:-4] + "_Rapport.txt", 'w')
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
    file.write(f"- Nombre de cellules supprimées après filtrage\n  (trop grandes, trop petites ou trop sombres) : {len(valeurs_wat)-len(allgrandcell)}\n")
    file.close()
    
    print("finished")

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk
from PIL import Image
from tensorflow.keras import models
import datetime
# Création de la fenêtre
window = Tk()
window.style = ttk.Style()
window.style.theme_use("clam")
window.title("Analyse Vitellogénèse")
window.geometry("518x600")
window.config(background="#4065A4")
# Chargement IA
classes = models.load_model("./models/final.h5")
interesting = models.load_model("./models/interesting.h5")

canvas_image = Canvas(window, width = 518, height = 388,bg="#305594", bd=0, highlightthickness=0)
defau = Image.open("default.png")
name = ""
picallcells = []



def prediction():
    indet=0
    regression=0
    sta=0
    stb=0
    stc=0
    std=0
    vit=0
    for image in picallcells:
        x = np.array(image)
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 3)
        # Si intéressant
        if interesting.predict(x).argmax() == 0:
            y = classes.predict(x).argmax()
            print(classes.predict(x))
            if y == 0:
                regression+=1
            elif y == 1:
                sta+=1
            elif y == 2:
                stb += 1
            elif y == 3:
                stc+=1
            elif y == 4:
                std+=1
            else:
                vit+=1
        else:
            indet+=1
    print_on_label(label_a, "Stade A = "+str(sta))
    print_on_label(label_b, "Stade B = "+str(stb))
    print_on_label(label_c, "Stade C = "+str(stc))
    print_on_label(label_d, "Stade D = "+str(std))
    print_on_label(label_vit, "Vitello = "+str(vit))
    print_on_label(label_reg, "Regression = "+str(regression))
    print_on_label(label_pasint, "Pas int. = "+str(indet))
        
    file = open(name[:-4] + "_Stats.csv", 'w')
    file.write("Reg;Stade A;Stade B;Stade C;Stade D;Vitello;Indéterminé\n")
    file.write(f"{regression};{sta};{stb};{stc};{std};{vit};{indet}\n")
    file.write("\n" + str(datetime.datetime.now()) + f";{name}\n")

ph_def = ImageTk.PhotoImage(defau)
canvas_image.create_image(159,94, image=ph_def, anchor=NW)
canvas_image.image=ph_def
canvas_image.grid(row=0, column = 0)

def open_image():
    global name
    name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                           filetypes =(("Image File", "*.png *.jpg"),("All Files","*.*")),
                           title = "Choose a file."
                           )
    try:
        image = Image.open(name).resize((518,388))
        print("Image ouverte correctement")
        photo = ImageTk.PhotoImage(image)
        canvas_image.create_image(0,0,image=photo, anchor=NW)
        canvas_image.image=photo
        #canvas_image.grid(row=0, column = 0)
    except:
        print("Erreur Fichier")
    

open_image_button = Button(window, text="Ouvrir une image", command = open_image, bg="#4065A4");
segment_image_button = Button(window, text="Ségmenter", command = lambda: segmenter(name))
prediction_button = Button(window, text="Classer", command=prediction, bg="#4065A4")
open_image_button.grid(row =9,column=0)
segment_image_button.grid(row =10,column=0)
prediction_button.grid(row =11,column=0)

def print_on_label(label, toprint):
    label.config(text=toprint)

label_a = Label(window, text="Stade A :", bg= "#4065A4")
label_a.grid(row = 2, column=0)
label_b = Label(window, text="Stade B :", bg= "#4065A4")
label_b.grid(row = 3, column=0)
label_c = Label(window, text="Stade C :", bg= "#4065A4")
label_c.grid(row = 4, column=0)
label_d = Label(window, text="Stade D :", bg= "#4065A4")
label_d.grid(row = 5, column=0)
label_reg = Label(window, text="Regression :", bg= "#4065A4")
label_reg.grid(row = 6, column=0)
label_vit = Label(window, text="Vitello :", bg= "#4065A4")
label_vit.grid(row = 7, column=0)
label_pasint = Label(window, text="Pas int. :", bg= "#4065A4")
label_pasint.grid(row = 8, column=0)

window.mainloop()