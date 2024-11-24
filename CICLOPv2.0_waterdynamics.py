#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:47:18 2024

@author: abelxf
"""

import gc
import sys
import numpy as np
import os
from os import path
import math
import copy
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import argparse
import pandas as pnd
import time
from numpy import random
import MDAnalysis as mda
import scipy
import csv
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d, convex_hull_plot_2d
from shapely.geometry import Point, Polygon
from functools import reduce
import operator
import itertools
from functools import lru_cache
import concurrent.futures
from multiprocessing import Pool
from MDAnalysis.coordinates.XTC import XTCWriter
from tqdm import tqdm
#from collections import Counter
from datetime import datetime
from scipy.stats import linregress


time_0 = time.time()

print("**************************************************************************")
print()
print("                               C I C L O P                                ")
print("           Characterization of Inner Cavity Lining of Proteins            ")
print()
print("  A tool for the automatic detection and quantification of inside lining  ")
print("                    residues in 3D protein structures                     ")
print("                           and cavity water                               ")
print()
print("                 Parth Garg,Arjun Ray,Abel Xavier Francis                 ")
print("                       ciclop.raylab@iiitd.ac.in                          ")
print()
print("**************************************************************************")


if __name__ == "__main__":  # Adding all the flags and arguments to be parsed through the command line for different functions to be executed
    parser = argparse.ArgumentParser()
    # Basic Arguments for file names, water models and alignment
    # New
    parser.add_argument('-w', type=str, choices=['spc', 'spce', 'tip3p', 'tip4p', 'tip5p', 'no_water'], required=True,
                        help='Specify the water model used to solvate the system. Select no_water if system is not solvated')
    parser.add_argument('-f', '--f', type=str,
                        help='Input file to be read in pdb format and water in tip4p configuration.')
    parser.add_argument('-o', '--o', type=str,
                        help='Output file to write the final pdb to')
    parser.add_argument('-layer', '--layer', action='store_true',
                        help='Use this flag to output a layer of water.')
    parser.add_argument('-ls', '--layer_start', type=int,
                        help='Provide starting layer value')
    parser.add_argument('-le', '--layer_end', type=int,
                        help='Provide ending point for layers')
    parser.add_argument('-layer_range', '--layer_range', action='store_true',
                        help='Use this flag to output a range of layers')
    parser.add_argument('-dens_plot', '--density_plot', action="store_true",
                        help='Save plot of water number density vs distance from the cavity wall')
    parser.add_argument('--vx_dim', type=int, default=1,
                        help=r'Provide voxel dimension.Default is 1$\AA$,and best results are obtained with this setting')
    parser.add_argument('-traj', '--traj', type=str,
                        help='Provide trajectory file for analysis in .xtc format')
    parser.add_argument('-s', '--s', type=str, help='Provide structure file')
    parser.add_argument('-t0', '--t0', type=float,
                        help='Provide the time at which analysis starts')
    parser.add_argument('-tf', '--tf', type=float,
                        help='Provide the time at which analysis stops')
    parser.add_argument('-step', '--step', type=float,
                        help='Provide the time steps used to traverse the time period')
    parser.add_argument('-to_csv', '--to_csv', type=str,
                        help='Provide name for the csv file')
    parser.add_argument('-no_csv', action='store_true',
                        help='Use this flag if you do not want to write out a csv file')
    parser.add_argument('-log_file', '--log_file', type=str, default=  'CICLOP_log-' + datetime.now().strftime("%Y-%m-%d") + "-" +  time.strftime("%H-%M-%S") + ".log", help='Provide a prefered logfile name')

    # Alignment Flags
    parser.add_argument('-noalign', action='store_true',
                        help='Use this flag to skip alignment of the protein axis to the Z-Axis')
    parser.add_argument('-axis_x', type=float, help = "Provide X-Component of protein axis vector")
    parser.add_argument('-axis_y', type=float, help = "Provide Y-Component of protein axis vector")
    parser.add_argument('-axis_z', type=float, help = "Provide Z-Component of protein axis vector")

    # Arguments for Selections(Chains, Specific Waters)
    parser.add_argument('-select_chain', '--select_chain', type=str,
                        help='Provide string of chains to be selected for analysis')
    parser.add_argument('-show', '--show', action="store_true",
                        help="Use this flag to display your plot")
    parser.add_argument('-all_water', action="store_true", help = "Use this flag to select all the water within the cavity for analysis")

    # Arguments for Plotting
    #parser.add_argument('-name', '--name', type=str,
     #                   help="Use this flag to give desired name to output")
    parser.add_argument('-grid', '--grid', action="store_true",
                        help="Use this flag to add grid to plots")
    parser.add_argument('-dpi', '--dpi', type=int, default=400,
                        help="Use this flag to set output plot resolution")

    # Arguments for Analyses
    parser.add_argument('-cav_diff', '--cavity_diffusion', action="store_true",
                        help='Provide output png to plot the regression line to find the diffusion coeffecient of cavity water.')
    parser.add_argument('-fit_limit', '--fit_limit', type=float, default=0.1,
                        help="Use this flag to determine the fraction of MSD data to be ignored from the beginning and end.")
    
    parser.add_argument('-water_prop', '--water_prop', action = "store_true",
                        help='Provide output file to study water trajectory')
    
    
    # Radius and Volume Plots
    parser.add_argument('-vol_profile', '--vol_profile', action="store_true",
                        help='Provide output file for the volume profile plot. Set alignment to default')
    parser.add_argument('-rad_profile', '--rad_profile', action="store_true",
                        help='Provide output file for the radius profile plot. Set alignemnt to default')
    #parser.add_argument('-errorbar', '--errorbar', action="store_true",
     #                   help='Provide the png file to which the final errorbar plot is to be saved to')
    #parser.add_argument('-rad_errorbar', '--rad_errorbar', action="store_true",
     #                   help='Provide the png file to which the final radius errorbar plot is to be saved to')
    #parser.add_argument('-vol_errorbar', '--vol_errorbar', action="store_true",
     #                   help='Provide the png file to which the final volume errorbar plot is to be saved to')

    # Charge Plots
    parser.add_argument('-charge_plot', '--charge_plot', action="store_true",
                        help='Use this flag to obtain charge distribution plot along the cavity axis')
    #parser.add_argument('-charge_errorbar', '--charge_errorbar',action="store_true", help='Produces charge errorbar plot')

    # Display
    
    # Autocorrelation Functions
    parser.add_argument('-res_time', '--residence_time', action="store_true",
                        help="Use this flag to obtain residence times of water molecules at different distance from the protein cavity wall.")
    parser.add_argument('-VACF', '--VACF', action="store_true",
                        help='Use this flag to find the diffusion constant using the Velocity Autocorrelation Function')
    parser.add_argument('-rel_plot', '--relaxation_plot', action="store_true",
                        help="Provide and output .png file to write out the relaxation time plot for water within 5 Angstroms from the protein inner surface")
    parser.add_argument('-second_order', action="store_true",
                        help="Use this flag to output second order correlation function for dipole relaxations")

    # multithreading
    parser.add_argument('-multi_process', action="store_true",
                        help="Use this flag to implement multiprocessing")
    parser.add_argument('-frame_storage', type=int, default=10,
                        help="Provide the number of cores you would like to use for multiprocessing")

    parser.add_argument('-select', '--select', type = str, default = 'protein', help = 'Use this flag to select atoms for analysis using the MDAnalysis selection language')
    
    
    parser.add_argument('-axis_dens', '--axis_dens', action = "store_true", help = 'Use this flag to produce water density profile along the protein axis')
    
    parser.add_argument('-dist_check', '--dist_check', action = "store_true")
    
    args = parser.parse_args()


if args.traj != None:
    args.t0 = round(args.t0, 5)
    args.tf = round(args.tf, 5)
    args.step = round(args.step, 5)


# Name for the log file
name = args.log_file

log_file = open(name, 'w')
log_file.write(
    "**************************************************************************\n")
log_file.write("\n")
log_file.write(
    "                               C I C L O P                                \n")
log_file.write(
    "           Characterization of Inner Cavity Lining of Proteins            \n")
log_file.write("\n")
log_file.write(
    "  A tool for the automatic detection and quantification of inside lining  \n")
log_file.write(
    "                    residues in 3D protein structures                     \n")
log_file.write(
    "                           and cavity water                               \n")
log_file.write("\n")
log_file.write(
    "                 Parth Garg,Arjun Ray,Abel Xavier Francis                 \n")
log_file.write(
    "                       ciclop.raylab@iiitd.ac.in                          \n")
log_file.write('\n')
log_file.write(
    "**************************************************************************\n")
log_file.write('\n')
log_file.write(
    '                            RESULTS                                       \n')
log_file.write(
    '                            _______                                       \n')
log_file.write(
    '                            _______                                       \n')
log_file.write('\n')


log_file.write('Command Line:\n')
log_file.write('\n')

Arguments_Parsed = sys.argv
for argument in Arguments_Parsed:
    if type(argument) != str:
        argument = str(argument)

Arguments_Parsed = ' '.join(Arguments_Parsed)
log_file.write(Arguments_Parsed)
log_file.write('\n')

dic_layer = {}  # used only in avg_exchange_analysis


class voxel():
    def __init__(self, i, j, k):
        self.left_bound = None
        self.right_bound = None
        self.up_bound = None
        self.down_bound = None
        self.height_bound = None
        self.depth_bound = None
        self.contained_atoms = []
        self.contained_Residues = []
        self.isEmpty = True
        self.encountered = False
        self.Position = [i, j, k]
        self.Inner_Cavity = False
        self.layer = 0
        self.water = []
        self.OW_atoms = []
        self.grid = None


def slope(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    if (x1 == x2):
        if (y1 < y2):
            return np.inf
        else:
            return -np.inf
    else:
        return 1.0 * (y1-y2)/(x1-x2)


def Triangle_area(p1, p2, p3):
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    x3 = p3[0]
    y3 = p3[1]

    a = math.sqrt(((x1 - x2)**2) + ((y1-y2)**2))

    b = math.sqrt(((x2 - x3)**2) + ((y2-y3)**2))

    c = math.sqrt(((x3 - x1)**2) + ((y3-y1)**2))

    s = (a + b + c)/2
    return math.sqrt(s*(s-a)*(s-b)*(s-c))


def Plot(X, Y, Args, x_label, y_label, Title, Name, Legend):
    plt.figure(num=0, dpi=Args.dpi)
    
    if len(np.shape(X)) == 1:
        plt.plot(X, Y, color='blue')
    else:
        for i in range(0, len(X)):
            # print('Lol ', i)
            plt.plot(X[i], Y[i])

    if Legend != None:
        plt.legend(Legend)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(Title)
    if Args.grid == True:
        plt.grid()
    if Args.show == True:
        plt.show()
    if Args.o == None:
        plt.savefig(Name, dpi=Args.dpi)
        print("Output File Written: " + Name)
        log_file.write("Output File Written: " + Name + "\n")
        log_file.write("\n")
        log_file.flush()
    else:
        plt.savefig(Args.o, dpi=Args.dpi)
        print("Output File Written: " + Args.o)
        log_file.write("Output File Written: " + Args.o + "\n")
        log_file.write("\n")
        log_file.flush()
        
def Plot_Errorbar(X, Y, y_err, Args, x_label, y_label, Title, Name):
    plt.figure(num=0, dpi=120)
    plt.errorbar(X, Y, yerr=y_err, xerr=None, color="blue",
                 ecolor="red", markersize=4, capsize=3, ls="-")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(Title)
    if Args.grid == True:
        plt.grid()
    if Args.show == True:
        plt.show()
    if Args.o == None:
        plt.savefig(Name, dpi=Args.dpi)
        print("Output File Written: " + Name)
        log_file.write("Output File Written: " + Name + "\n")
        log_file.write("\n")
        log_file.flush()
    else:
        plt.savefig(Args.o, dpi=Args.dpi)
        print("Output File Written: " + Args.o)
        log_file.write("Output File Written: " + Args.o + "\n")
        log_file.write("\n")
        log_file.flush()
        
def get_pdb_line(atom_list, water_list, Args):
    Out = open(Args.o, 'w')

    Out_List = []

    IDs = []

    if atom_list != None:
        Atoms = list(atom_list.keys())
        for Atom in Atoms:
            if atom_list[Atom][1] == True:
                Atom.position = atom_list[Atom][0]
                Atom.tempfactor = 9999
                Out_List.append(Atom)
                IDs.append(Atom.index)
            else:
                Atom.position = atom_list[Atom][0]
                Atom.tempfactor = -1
                Out_List.append(Atom)
                IDs.append(Atom.index)

    if water_list != None:
        O_Atoms = water_list.keys()
        for O in O_Atoms:
            O.position = water_list[O][0]
            layer = water_list[O][2]
            if 1 <= layer < 10:
                O.tempfactor = 1000
            elif 10 <= layer < 20:
                O.tempfactor = 4000
            elif layer >= 20:
                O.tempfactor = 7000
            IDs.append(O.index)
            Out_List.append(O)
            for atom in water_list[O][1]:
                atom.tempfactor = 250*water_list[O][2]
                Out_List.append(atom)
                IDs.append(atom.index)

    Out_Group = mda.core.groups.AtomGroup(Out_List)
    Out_Group.write(Args.o)

    Out.close()
    log_file.write("B-Factor values have been modified:\n")
    log_file.write("B-Factor = 9999 for Cavity Lining Atoms\n")
    log_file.write("B-Factor = 1000 for Water between 1 and 10 Angstroms from Cavity Lining\n")
    log_file.write("B-Factor = 4000 for Water between 10 and 20 Angstroms from Cavity Lining\n")
    log_file.write("B-Factor = 7000 for Water at a distance greater than 20 Angstroms from the Cavity Lining\n")
    
    #log_file.write("Output File Written: " + Args.o + "\n")
    #log_file.write("\n")
    #log_file.flush()
    return ()


def Produce_Atom_Lists(ag, t, Water_Model, Args):
    # if(Args.traj != None):
    #     u.trajectory[t]
    # frame.trajectory[0]
    all_atoms = {}
    
    atoms = ag.select_atoms(Args.select)
    log_file.write("Group selected for analysis is: " + Args.select + "\n")
    log_file.write("\n")
    log_file.flush()
    
    for atom in atoms:
        all_atoms[atom] = [
            [atom.position[0], atom.position[1], atom.position[2]], False, False]
    # print('The number of atoms in the selected chains are:',len(list(all_atoms.keys())))
    log_file.write('Total number of protein atoms are:' +
                   str(len(list(all_atoms.keys()))) + '\n')
    log_file.flush()
    ##############################################################################################

    non_O_atoms = ag.select_atoms('resname SOL and not name OW')
    O_ls = ag.select_atoms('name OW and resname SOL')
    # print('Number of solvent molecules:',len(O_ls))

    # print("Bleh bleh bleh")
    # print(O_ls[0].position, O_ls[1].position)

    O_Atoms = {}
    ind = 0
    if Water_Model == 'tip4p':
        for i in range(0, len(non_O_atoms), 3):
            O_Atoms[O_ls[ind]] = ([O_ls[ind].position[0], O_ls[ind].position[1], O_ls[ind].position[2]], [
                                  non_O_atoms[i], non_O_atoms[i+1], non_O_atoms[i+2]], 0, False, False)
            ind += 1

    elif (Water_Model == 'spc' or Water_Model == 'spce' or Water_Model == 'tip3p'):
        for i in range(0, len(non_O_atoms), 2):
            O_Atoms[O_ls[ind]] = ([O_ls[ind].position[0], O_ls[ind].position[1], O_ls[ind].position[2]], [
                                  non_O_atoms[i], non_O_atoms[i+1]], 0, False, False)
            ind += 1
    elif Water_Model == 'tip5p':
        for i in range(0, len(non_O_atoms), 4):
            O_Atoms[O_ls[ind]] = ([O_ls[ind].position[0], O_ls[ind].position[1], O_ls[ind].position[2]], [
                                  non_O_atoms[i], non_O_atoms[i+1], non_O_atoms[i+2], non_O_atoms[i+3]], 0, False, False)
            ind += 1

    log_file.write(
        "Number of solvent molecules in the system is:" + str(len(O_Atoms)) + "\n")
    log_file.flush()


# Finding the axis of the protein by finding the best fit line of all the carbon atoms that are present in the secondary structure of the protein and by default aligning it to the negative Z-Axis
    if Args.axis_x == None and Args.axis_z == None and Args.axis_y == None and Args.noalign == False:
        log_file.write("Aligning structure automatically \n")
        log_file.write("\n")
        log_file.flush()
        flt_atoms = atoms.select_atoms('name CA or name CB')
        p = np.min(flt_atoms.positions, axis=0)

        data = (flt_atoms.positions - p)
        datamean = data.mean(axis=0)

        uu, dd, vv = np.linalg.svd(data - datamean)
        vv[0::2] = -vv[0::2]
        # Generate line points

        linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

        linepts += datamean
        # print(linepts)
        pt1 = linepts[0]
        pt2 = linepts[1]

        # The unit vector parallel to the axis of the protein
        vector_query = [abs(pt1[0] - pt2[0]),
                        abs(pt1[1] - pt2[1]), abs(pt1[2] - pt2[2])]
    elif Args.axis_x != None and Args.axis_y != None and Args.axis_z != None:
        log_file.write("Aligning structure manually \n")
        log_file.write("\n")
        log_file.flush()
        vector_query = [Args.axis_x, Args.axis_y, Args.axis_z]

    # print('Axis of the protein is',vector_query)

    # Finding a vector perpendicular to the axis.
    # Assume the vector to be [a, b, c]. We require it to be normalized. Hence a^{2}+b^{2}+c^{2} = 1.
    # Second Condition is a.vector_query[0] + b.vector_query[1] + c.vector_query[2] = 0. Essentially a plane perpendicular to the axis.
    # We fix one of the components and then solve for the other two. We fix the Z-Component to zero, i.e, c = 0

    a = vector_query[1]/(np.sqrt((vector_query[0]**2)+(vector_query[1]**2)))

    b = -vector_query[0]/(np.sqrt((vector_query[0]**2)+(vector_query[1]**2)))

    perpendicular_vector = [a, b, 0]

    vector_target = [0, 0, -1]  # Target vector

    log_file.write("Axis vector is: [" + str(vector_query[0]) + ',' +
                   str(vector_query[1]) + ',' + str(vector_query[2]) + ']\n')
    
    log_file.flush()
    

    v1, v2 = (vector_query / np.linalg.norm(vector_query)
              ).reshape(3), (vector_target / np.linalg.norm(vector_target)).reshape(3)

    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot_mat = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    # print("rot_matrix: ")
    # print(rot_mat)

    log_file.write('Rotation matrix used for alignment: \n')
    log_file.write('\n')
    log_file.write('\n')
    log_file.write('[[ '+str(rot_mat[0][0])+' ' +
                   str(rot_mat[0][1])+' '+str(rot_mat[0][2])+' ]\n')
    log_file.write(' [ '+str(rot_mat[1][0])+' ' +
                   str(rot_mat[1][1])+' '+str(rot_mat[1][2])+' ]\n')
    log_file.write(' [ '+str(rot_mat[2][0])+' ' +
                   str(rot_mat[2][1])+' '+str(rot_mat[2][2])+' ]]\n')
    log_file.write('\n')
    log_file.write('\n')
    log_file.flush()

    # for atom in all_atoms:
    #     ls = np.dot(rot_mat,atom.position)
    #     atom.position[0] = ls[0]
    #     atom.position[1] = ls[1]
    #     atom.position[2] = ls[2]
    atoms.positions = np.transpose(
        np.dot(rot_mat, np.transpose(atoms.positions)))
    atoms.positions = np.round(atoms.positions, 3)
    # print("Atom positions: ")
    # print(atoms.positions)
    O_ls.positions = np.transpose(
        np.dot(rot_mat, np.transpose(O_ls.positions)))
    O_ls.positions = np.round(O_ls.positions, 3)

    val = O_Atoms.values()
    O_Atoms = dict(zip(O_ls, val))
    atm_val = all_atoms.values()
    all_atoms = dict(zip(atoms, atm_val))

    del atoms
    del non_O_atoms
    del O_ls
    del ind
    # del flt_atoms
    # del data

    
    '''    
    for O in list(O_Atoms.keys()):
        for atom in O_Atoms[O][1]:
            position_new = np.dot(atom.position, rot_mat)
            atom.position[0] 
    '''
    return (all_atoms, O_Atoms, vector_query, perpendicular_vector)


# Function to produce O Atom lists without transformations
def O_Atoms_List(u, t, Args):
    if (Args.traj != None):
        u.trajectory[t]
    non_O_atoms = u.select_atoms('resname SOL and not name OW')
    O_ls = u.select_atoms('name OW and resname SOL')
    # print('Number of solvent molecules:',len(O_ls))

    O_Atoms = {}
    ind = 0
    if Args.w == 'tip4p':
        for i in range(0, len(non_O_atoms), 3):
            O_Atoms[O_ls[ind]] = ([O_ls[ind].position[0], O_ls[ind].position[1], O_ls[ind].position[2]], [
                                  non_O_atoms[i], non_O_atoms[i+1], non_O_atoms[i+2]], 0, False, False)
            ind += 1

    elif (Args.w == 'spc' or Args.w == 'spce' or Args.w == 'tip3p'):
        for i in range(0, len(non_O_atoms), 2):
            O_Atoms[O_ls[ind]] = ([O_ls[ind].position[0], O_ls[ind].position[1], O_ls[ind].position[2]], [
                                  non_O_atoms[i], non_O_atoms[i+1]], 0, False, False)
            ind += 1
    elif Args.w == 'tip5p':
        for i in range(0, len(non_O_atoms), 4):
            O_Atoms[O_ls[ind]] = ([O_ls[ind].position[0], O_ls[ind].position[1], O_ls[ind].position[2]], [
                                  non_O_atoms[i], non_O_atoms[i+1], non_O_atoms[i+2], non_O_atoms[i+3]], 0, False, False)
            ind += 1

    log_file.write(
        "Number of solvent molecules in the system is:" + str(len(O_Atoms)) + "\n")
    log_file.write("\n")
    log_file.flush()

    return (O_Atoms)


def Create_grid_hydration(AtomLi, VOXEL_DIMENSION, O_Atoms):  # Creating the voxel grid
    # print('Creating grid')
    log_file.write("Creating Voxel Grid \n")
    log_file.write("\n")
    log_file.flush()
    # Finding the minimum coordinates of the protein atoms to translate protein to the first octant
    AtomList = mda.core.groups.AtomGroup(list(AtomLi.keys()))
    O_List = mda.core.groups.AtomGroup(list(O_Atoms.keys()))

    min_x, min_y, min_z = np.min(AtomList.positions, axis=0)
    max_x, max_y, max_z = np.max(AtomList.positions, axis=0)

    
    # print("min : ", min_x, min_y, min_z)
    AtomList.positions -= [min_x, min_y, min_z]
    AtomList.positions = AtomList.positions/VOXEL_DIMENSION
    AtomList.positions = np.round(AtomList.positions, 3)

    # Setting the dimensions of the voxel grid
    max_x = (max_x - min_x) / VOXEL_DIMENSION
    max_y = (max_y - min_y) / VOXEL_DIMENSION
    max_z = (max_z - min_z) / VOXEL_DIMENSION
    Min_x = min_x  # Assiging the coordinates used for translation to dummy variables to be returned
    Min_y = min_y
    Min_z = min_z
    min_x = 0.0
    min_y = 0.0
    min_z = 0.0

    # Defining the number of voxels in each dimension
    num_voxel_x = math.ceil(max_x)+2
    num_voxel_y = math.ceil(max_y)+2
    num_voxel_z = math.ceil(max_z)+2
    # The 3D array that contains each voxel at each grid point x,y,z
    voxel_list = np.empty((num_voxel_x, num_voxel_y, num_voxel_z), dtype=voxel)
    # print("dimension of grid : ",num_voxel_x,num_voxel_y,num_voxel_z)
    # print(time.process_time())
    
    log_file.write("Dimensions of the voxels grid is: " + str(num_voxel_x) + "Angstroms x " + str(num_voxel_y) + "Angstroms x " + str(num_voxel_z) + "Angstroms \n")
    log_file.write("\n")
    log_file.flush()    

    for i in range(0, num_voxel_x):
        for j in range(0, num_voxel_y):
            for k in range(0, num_voxel_z):
                voxel_list[i][j][k] = voxel(i, j, k)

    # print("part 1")
    # print(time.process_time())
    voxel_list = voxel_list.tolist()
    for atom in AtomList:

        x_ind, y_ind, z_ind = math.floor(atom.position[0]), math.floor(
            atom.position[1]), math.floor(atom.position[2])

        voxel_list[x_ind][y_ind][z_ind].contained_atoms.append(
            atom)  # Appending atoms to the voxel
        # Marking voxels with atoms in it as non empty
        voxel_list[x_ind][y_ind][z_ind].isEmpty = False

    # Now add the bound to all the outermost voxels in the voxel_grid
    for x in range(num_voxel_x):
        for y in range(num_voxel_y):
            for z in range(num_voxel_z):
                vxl = voxel_list[x][y][z]

                if y == 0:
                    vxl.left_bound = voxel_list[x-1][0][z] if x > 0 else "END"
                    vxl.right_bound = voxel_list[x +
                                                 1][0][z] if x < num_voxel_x-1 else "END"
                    vxl.height_bound = voxel_list[x][0][z +
                                                        1] if z < num_voxel_z-1 else "END"
                    vxl.depth_bound = voxel_list[x][0][z-1] if z > 0 else "END"
                    vxl.up_bound = voxel_list[x][1][z]
                    vxl.down_bound = "END"

                if y == num_voxel_y-1:
                    vxl.left_bound = voxel_list[x -
                                                1][num_voxel_y-1][z] if x > 0 else "END"
                    vxl.right_bound = voxel_list[x+1][num_voxel_y -
                                                      1][z] if x < num_voxel_x-1 else "END"
                    vxl.height_bound = voxel_list[x][num_voxel_y -
                                                     1][z+1] if z < num_voxel_z-1 else "END"
                    vxl.depth_bound = voxel_list[x][num_voxel_y -
                                                    1][z-1] if z > 0 else "END"
                    vxl.up_bound = "END"
                    vxl.down_bound = voxel_list[x][num_voxel_y-1][z]

                if x == 0:
                    vxl.up_bound = voxel_list[0][y +
                                                 1][z] if y < num_voxel_y-1 else "END"
                    vxl.left_bound = voxel_list[0][y-1][z] if y > 0 else "END"
                    vxl.height_bound = voxel_list[0][y][z +
                                                        1] if z < num_voxel_z-1 else "END"
                    vxl.depth_bound = voxel_list[0][y][z-1] if z > 0 else "END"
                    vxl.right_bound = voxel_list[1][y][z]
                    vxl.left_bound = "END"
                    vxl.right_bound = voxel_list[1][y][z]

                if x == num_voxel_x-1:
                    vxl.up_bound = voxel_list[num_voxel_x -
                                              1][y+1][z] if y < num_voxel_y-1 else "END"
                    vxl.left_bound = voxel_list[num_voxel_x -
                                                1][y-1][z] if y > 0 else "END"
                    vxl.height_bound = voxel_list[num_voxel_x -
                                                  1][y][z+1] if z < num_voxel_z-1 else "END"
                    vxl.depth_bound = voxel_list[num_voxel_x -
                                                 1][y][z-1] if z > 0 else "END"
                    vxl.left_bound = voxel_list[num_voxel_x-2][y][z]
                    vxl.right_bound = "END"

                if z == 0:
                    vxl.left_bound = voxel_list[x-1][y][0] if x > 0 else "END"
                    vxl.right_bound = voxel_list[x +
                                                 1][y][0] if x < num_voxel_x-1 else "END"
                    vxl.up_bound = voxel_list[x][y +
                                                 1][0] if y < num_voxel_y-1 else "END"
                    vxl.down_bound = voxel_list[x][y-1][0] if y > 0 else "END"
                    vxl.height_bound = voxel_list[x][y][1]
                    vxl.depth_bound = "END"

                if z == num_voxel_z-1:
                    vxl.left_bound = voxel_list[x -
                                                1][y][num_voxel_z-1] if x > 0 else "END"
                    vxl.right_bound = voxel_list[x+1][y][num_voxel_z -
                                                         1] if x < num_voxel_x-1 else "END"
                    vxl.up_bound = voxel_list[x][y+1][num_voxel_z -
                                                      1] if y < num_voxel_y-1 else "END"
                    vxl.down_bound = voxel_list[x][y -
                                                   1][num_voxel_z-1] if y > 0 else "END"
                    vxl.height_bound = "END"
                    vxl.depth_bound = voxel_list[x][y][num_voxel_z-2]

                if 0 < x < num_voxel_x-1 and 0 < y < num_voxel_y-1 and 0 < z < num_voxel_z-1:
                    vxl.left_bound = voxel_list[x-1][y][z]
                    vxl.right_bound = voxel_list[x+1][y][z]
                    vxl.up_bound = voxel_list[x][y+1][z]
                    vxl.down_bound = voxel_list[x][y-1][z]
                    vxl.height_bound = voxel_list[x][y][z+1]
                    vxl.depth_bound = voxel_list[x][y][z-1]

    New_Atom_List = dict(zip(AtomList, list(AtomLi.values())))
    
    log_file.write("Voxel Grid has been created!\n")
    log_file.write("\n")
    log_file.flush()
    
    return (New_Atom_List, voxel_list, (num_voxel_x, num_voxel_y,  num_voxel_z), (Min_x, Min_y, Min_z))


def Largest_empty_circle_voronoi_mat(Inner_surface_list):

    # Create a list of point to calculate the convex hull and the voronoi diagram
    Inner_surface_list = mda.core.groups.AtomGroup(Inner_surface_list)

    atom_points = np.array(Inner_surface_list.positions)

    vor = Voronoi(atom_points)
    # voronoi_plot_2d(vor)
    # plt.show()

    hull = ConvexHull(atom_points)

    # convex_hull_plot_2d(hull)

    pts = hull.vertices
    hull_polygon_points = []
    # convex_hull_plot_2d(hull)
    # plt.show()
    for vertex in pts:
        hull_polygon_points.append(
            (atom_points[vertex][0], atom_points[vertex][1]))

    possible_circle_centres = []

    voronoi_vertices = vor.vertices

    voronoi_vertices_points = []
    for vertex in voronoi_vertices:
        voronoi_vertices_points.append(Point(vertex[0], vertex[1]))

    convex_hull_polygon = Polygon(hull_polygon_points)

    global_max_rad = -np.inf

    for possible_centre in voronoi_vertices_points:

        if (convex_hull_polygon.contains(possible_centre)):
            cen_x = possible_centre.coords[0][0]
            cen_y = possible_centre.coords[0][1]
            min_dist = np.inf
            for atom in Inner_surface_list:
                dist = math.sqrt(
                    ((atom.position[0]-cen_x)**2) + ((atom.position[1]-cen_y)**2))
                if (dist < min_dist):
                    min_dist = dist

            if (min_dist > global_max_rad):
                global_max_rad = min_dist

    return global_max_rad


def volume_curr_slice(Inner_atoms):
    coords = []
    cen_x = 0
    cen_y = 0
    for atom in Inner_atoms:
        coords.append([atom.position[0], atom.position[1]])
        cen_x += atom.position[0]
        cen_y += atom.position[1]

    cen_x = cen_x / len(Inner_atoms)
    cen_y = cen_y / len(Inner_atoms)

    right_half = []
    left_half = []

    for coordinate in coords:
        if coordinate[0] < cen_x:
            left_half.append(coordinate)
        else:
            right_half.append(coordinate)

    right_half.sort(key=lambda coord: slope(
        [cen_x, cen_y], coord), reverse=True)
    left_half.sort(key=lambda coord: slope(
        [cen_x, cen_y], coord), reverse=True)

    sorted_coords = right_half
    for coord in left_half:
        sorted_coords.append(coord)

    slice_volume = 0

    for i in range(0, len(sorted_coords)-1):
        slice_volume += Triangle_area([cen_x, cen_y],
                                      sorted_coords[i], sorted_coords[i+1])

    # Now add the slice area for the last and first coordinates
    slice_volume += Triangle_area([cen_x, cen_y],
                                  sorted_coords[-1], sorted_coords[0])

    return slice_volume


# Finding the inner surface atoms and the inner cavity voxels
def Find_surface_inner_hydration(Atom_List, voxel_list, grid_dimensions, rad_vol, z_charge):

    log_file.write("Finding inner surface atoms \n")
    log_file.write("\n")
    log_file.flush()    

    Hydrophobicity_scale_kyte_doolittle = {
        "ILE": 4.5,
        "VAL": 4.2,
        "LEU": 3.8,
        "PHE": 2.8,
        "CYS": 2.5,
        "MET": 1.9,
        "ALA": 1.8,
        "GLY": -0.4,
        "THR": -0.7,
        "SER": -0.8,
        "TRP": -0.9,
        "TYR": -1.3,
        "PRO": -1.6,
        "HIS": -3.2,
        "GLU": -3.5,
        "GLN": -3.5,
        "ASP": -3.5,
        "ASN": -3.5,
        "LYS": -3.9,
        "ARG": -4.5
    }

    Hydrophobicity_scale_keys = Hydrophobicity_scale_kyte_doolittle.keys()

    num_vox_x = grid_dimensions[0]  # Setting the dimensions of the grid
    num_vox_y = grid_dimensions[1]
    num_vox_z = grid_dimensions[2]

    Inner_Empty_Voxels = []
    empty_voxels = []
    # Assuming the centroid of the grid to be in the cavity of the protein and starting the search from this point
    empty_voxels.append(voxel_list[num_vox_x//2][num_vox_y//2][num_vox_z//2])

    for box in empty_voxels:  # Checking the bounds of every box in empty_voxels to traverse all voxels
        box.encountered = True
        if (box.isEmpty == True):
            # Checking whether the bounding face is at the end and whether it is empty or not
            if (box.right_bound != "END" and box.right_bound.isEmpty == True):
                if (box.right_bound.encountered == False):
                    box.right_bound.encountered = True
                    empty_voxels.append(box.right_bound)
            if (box.left_bound != "END" and box.left_bound.isEmpty == True):
                if (box.left_bound.encountered == False):
                    box.left_bound.encountered = True
                    empty_voxels.append(box.left_bound)
            if (box.up_bound != "END" and box.up_bound.isEmpty == True):
                if (box.up_bound.encountered == False):
                    box.up_bound.encountered = True
                    empty_voxels.append(box.up_bound)
            if (box.down_bound != "END" and box.down_bound != None and box.down_bound.isEmpty == True):
                if (box.down_bound.encountered == False):
                    box.down_bound.encountered = True
                    empty_voxels.append(box.down_bound)
            if (box.height_bound != "END" and box.height_bound.isEmpty == True):
                if (box.height_bound.encountered == False):
                    box.height_bound.encountered = True
                    empty_voxels.append(box.height_bound)
            if (box.depth_bound != "END" and box.depth_bound.isEmpty == True):
                if (box.depth_bound.encountered == False):
                    box.depth_bound.encountered = True
                    empty_voxels.append(box.depth_bound)

    for box in empty_voxels:  # Now, to find the atoms on the inner or outer surface, we check the bounds of the empty voxels. If the bounding face has atoms, then it is on the surface of the
        if (box.right_bound != "END"):  # protein
            if (box.right_bound.isEmpty == False):
                for atom in box.right_bound.contained_atoms:
                    Atom_List[atom][1] = True
                    Atom_List[atom][2] = True
        if (box.left_bound != "END"):
            if (box.left_bound.isEmpty == False):
                for atom in box.left_bound.contained_atoms:
                    Atom_List[atom][1] = True
                    Atom_List[atom][2] = True
        if (box.up_bound != "END"):
            if (box.up_bound.isEmpty == False):
                for atom in box.up_bound.contained_atoms:
                    Atom_List[atom][1] = True
                    Atom_List[atom][2] = True
        if (box.down_bound != "END"):
            if (box.down_bound != None and box.down_bound.isEmpty == False):
                for atom in box.down_bound.contained_atoms:
                    Atom_List[atom][1] = True
                    Atom_List[atom][2] = True
        if (box.height_bound != "END"):
            if (box.height_bound.isEmpty == False):
                for atom in box.height_bound.contained_atoms:
                    Atom_List[atom][1] = True
                    Atom_List[atom][2] = True
        if (box.depth_bound != "END"):
            if (box.depth_bound.isEmpty == False):
                for atom in box.depth_bound.contained_atoms:
                    Atom_List[atom][1] = True
                    Atom_List[atom][2] = True

    voxel_grid = voxel_list
    rad_x = []
    rad_y = []
    vol_list = []
    tot_volume = 0
    Inner_list_final = []
    vol_x = []
    this_vol = 0

    # Now perform the Z axis check
    Mean_Dists = []

    Z_Hydrophobicity = np.zeros(grid_dimensions[2], dtype=float)
    Z_Charge = np.zeros(grid_dimensions[2], dtype=float)
    pos_res = 0
    neg_res = 0

    n_hydrophillic = 0
    n_hydrophobic = 0

    n_inner_surf = 0

    # Now, to differentiate between the atoms on the inner and outer surface, we slice the protein along the Z axis,and for each slice the standard deviation
    for z in range(1, grid_dimensions[2]):
        temp_boxes = []  # of the distances of each residue from the mean position of all the surface atoms are found. Atoms found within a radius of 0.7*SD are marked as Inner_Surface true, and
        # the empty voxels found inside this radius is marked as Inner_Cavity=True
        z_slice_empty_voxels = []
        mean_x = 0
        mean_y = 0
        for x in range(0, grid_dimensions[0]):
            for y in range(0, grid_dimensions[1]):
                if (voxel_grid[x][y][z].isEmpty == False):
                    for atom in voxel_grid[x][y][z].contained_atoms:
                        if (Atom_List[atom][1] == True and Atom_List[atom][2] == True):
                            temp_boxes.append(atom)
                else:
                    # Taken the empty voxels in that z-slice
                    z_slice_empty_voxels.append(voxel_grid[x][y][z])

        if (len(temp_boxes) > 0):
            mean_x = 0
            mean_y = 0
            for atom in temp_boxes:
                mean_x += atom.position[0]
                mean_y += atom.position[1]

            mean_x = mean_x/len(temp_boxes)
            mean_y = mean_y/len(temp_boxes)

            mean_dist = 0
            store_dist = []
            for atom in temp_boxes:
                d = math.sqrt(
                    ((atom.position[0] - mean_x)**2) + ((atom.position[1] - mean_y)**2))
                store_dist.append(d)
                mean_dist += d

            mean_dist = mean_dist/len(temp_boxes)

            Mean_Dists.append(mean_dist)

            # Find standar deviation for the inside
            if (mean_dist >= 2):
                diff_square = 0
                num_inside = 0
                i = 0
                for atom in temp_boxes:
                    if (math.sqrt(((atom.position[0] - mean_x)**2) + ((atom.position[1] - mean_y)**2)) <= mean_dist):
                        dist = math.sqrt(
                            ((atom.position[0] - mean_x)**2) + ((atom.position[1] - mean_y)**2))
                        diff_square += (dist - mean_dist)**2
                        num_inside += 1

                if (num_inside > 0):
                    stand_dev_inside_half = math.sqrt(
                        diff_square/(num_inside))*0.70
                else:
                    stand_dev_inside_half = 0

                for atom in temp_boxes:
                    atom_dist = math.sqrt(
                        ((atom.position[0] - mean_x)**2) + ((atom.position[1] - mean_y)**2))
                    if (atom_dist > mean_dist-stand_dev_inside_half):
                        Atom_List[atom][1] = "Outer Surface"

                for box in z_slice_empty_voxels:
                    voxel_dist = math.sqrt(
                        ((box.Position[0]-mean_x)**2)+((box.Position[1]-mean_y)**2))
                    if (voxel_dist < mean_dist-stand_dev_inside_half):
                        box.Inner_Cavity = True

                if (len(temp_boxes) <= 7):  # If the volume is very less, the cavity is ignored
                    for atom in temp_boxes:
                        Atom_List[atom][1] = False
                    for box in z_slice_empty_voxels:
                        box.Inner_Cavity = False

            else:
                for atom in temp_boxes:
                    Atom_List[atom][1] = False

            for atom in temp_boxes:
                if (Atom_List[atom][1] == True):
                    Inner_list_final.append(atom)
                    Z_Hydrophobicity[z] += Hydrophobicity_scale_kyte_doolittle[atom.resname]
                    if Hydrophobicity_scale_kyte_doolittle[atom.resname] < 0:
                        n_hydrophillic += 1
                    elif Hydrophobicity_scale_kyte_doolittle[atom.resname] > 0:
                        n_hydrophobic += 1

            if z_charge == True:
                for atom in temp_boxes:
                    if Atom_List[atom][1] == True:
                        Z_Charge[z] += atom.charge

            for box in z_slice_empty_voxels:
                if box.Inner_Cavity == True:
                    Inner_Empty_Voxels.append(box)

            if (len(Inner_list_final) > 3 and rad_vol == True):
                # try:
                rad = Largest_empty_circle_voronoi_mat(Inner_list_final)
                is_vol = volume_curr_slice(Inner_list_final)
                tot_volume += is_vol
                this_vol += is_vol

                if (rad != -np.inf and z % 3 == 0):
                    rad_x.append(z)
                    tot_volume += math.pi * rad * rad
                    rad_y.append(Largest_empty_circle_voronoi_mat(
                        Inner_list_final)*2-3)
                    vol_list.append(this_vol)
                    vol_x.append(z)
                    this_vol = 0

                # else:
                #     rad_x.append(z)
                #     try:
                #         rad_y.append(rad_y[-1])
                #     except Exception:
                #         rad_y.append(0)

                # print("task_achieved")
                # print(z," ",grid_dimensions[2]," ",len(Inner_list_final))
                # except Exception:
                #     print("Moving Forward")

                Inner_list_final = []

    # rad_x = rad_x[3:-3:3]
    # rad_y = rad_y[3:-3:3]
    if rad_vol == True:
        rad_x = rad_x[:len(rad_x)-1]
        rad_y = rad_y[:len(rad_y)-1]
        vol_x = vol_x[:len(vol_x)-1]
        vol_list = vol_list[:len(vol_list)-1]
#        for i in range(0, len(rad_x)):
#           rad_x[i] = rad_x[i] * 2.5
#      for i in range(0, len(vol_x)):
#         vol_x[i] = vol_x[i] * 2.5
    # for vol in vol_list:
        #   Total_Pore_volume += vol
    # plt.plot(vol_x, vol_list)
    # plt.xlabel("Distance along pore in angstrom")
    # plt.ylabel("Volume of pore in cubic angstroms")
    # plt.savefig(pdb_name + "-volume_profile-"+str(pore_number)+".svg", format="svg")

    # print(rad_y)
    # print(rad_x)

    # files_to_zip.append(pdb_name + "-volume_profile-"+str(pore_number)+str(".svg"))

    # plt.clf()
    # plt.plot(rad_x, rad_y)
    # plt.xlabel("Distance along pore in angstrom")
    # plt.ylabel("Diameter of pore in angstroms")
    # plt.savefig(pdb_name + "-radius_profile-"+str(pore_number)+".svg", format="svg")

    # files_to_zip.append(pdb_name + "-radius_profile-"+str(pore_number)+str(".svg"))

    Outer_Empty_Voxels = []

    for x in range(0, num_vox_x):
        for y in range(0, num_vox_y):
            for z in range(0, num_vox_z):
                if (voxel_list[x][y][z].isEmpty == True and voxel_list[x][y][z].encountered == False):
                    empty_voxels.append(voxel_list[x][y][z])
                if (voxel_list[x][y][z].isEmpty == True and voxel_list[x][y][z].Inner_Cavity == False):
                    Outer_Empty_Voxels.append(voxel_list[x][y][z])


    Tot_Hydrophobicity = np.sum(Z_Hydrophobicity)
    # print('Number of voxels in the cavity is: ', len(Inner_Empty_Voxels))
    # print('Total number of empty voxels is: ',len(empty_voxels))
    # print('Number of positively charged residues on the cavity surface are: ', pos_res)
    # print('Number of negatively charged residues on the cavity surface are: ', neg_res)
    # print('Number of empty voxels outside the cavity is: ', len(Outer_Empty_Voxels))
    # print(len(Outer_Empty_Voxels) + len(Inner_Empty_Voxels))
    
    for atom in Atom_List.keys():
        if Atom_List[atom][1] == True:
            n_inner_surf += 1
    
    log_file.write("Number of atoms identified on the protein inner surface is: " + str(n_inner_surf) + "\n")
    log_file.write("\n")
    log_file.flush()
    
    log_file.write("Number of voxels identified within the cavity is: " + str(len(Inner_Empty_Voxels)) + "\n")
    log_file.write("Volume of the identified cavity is: "+ str(len(Inner_Empty_Voxels)) + " cubic Angstroms \n")
    log_file.write("\n")
    log_file.flush()
    
    if (rad_vol == True):
        return (empty_voxels, Inner_Empty_Voxels, rad_x, rad_y, vol_x, vol_list, Z_Charge, Z_Hydrophobicity, Outer_Empty_Voxels)
    else:
        return (empty_voxels, Inner_Empty_Voxels, Z_Charge, Z_Hydrophobicity, Outer_Empty_Voxels)


# Defining a function to add water to the frame
def Add_Water(O_atoms, Voxel_List, Grid_Dimensions):
    Grid_Water = 0
    Cavity_Water = 0
    # Now adding all the water molecules to the required voxels
    # print('Adding water to voxels')
    O_ATOMS = list(O_atoms.keys())
    for atom in O_ATOMS:  # The water molecule is assigned to a voxel if the cooridnates of the O atom of that particular water molecule lies in the voxel
        if (0 < round(atom.position[0], 3) < Grid_Dimensions[0] and 0 < round(atom.position[1], 3) < Grid_Dimensions[1] and 0 < round(atom.position[2], 3) < Grid_Dimensions[2]):
            x_ind = math.floor(atom.position[0])
            y_ind = math.floor(atom.position[1])
            z_ind = math.floor(atom.position[2])
            O_atoms[atom] = (O_atoms[atom][0], O_atoms[atom][1],
                             O_atoms[atom][2], True, O_atoms[atom][4])
            Grid_Water += 1
            # print(Grid_Dimensions)
            # print((x_ind,y_ind,z_ind))
            # print(atom.water_molecule)
            Voxel_List[x_ind][y_ind][z_ind].OW_atoms.append(atom)
            Voxel_List[x_ind][y_ind][z_ind].water.append(atom)
            Voxel_List[x_ind][y_ind][z_ind].water = Voxel_List[x_ind][y_ind][z_ind].water + O_atoms[atom][0]

            if Voxel_List[x_ind][y_ind][z_ind].Inner_Cavity == True:
                O_atoms[atom] = (O_atoms[atom][0], O_atoms[atom]
                                 [1], O_atoms[atom][2], O_atoms[atom][3], True)
                Cavity_Water += 1

#            if atom.id==179132:
#                if atom.Inner_Cavity==True:
#                    print(atom.PDBLINE)
#                else:
#                    print('Not in cavity')
#                    print(atom.PDBLINE)
            # voxel_list_hydrated=Voxel_List
    # grid_water=open('Grid_Water.pdb','w')

#    for atom in O_ATOMS:
 #       if atom.grid==True:
    #          for line in atom.water_molecule:
   #         	grid_water.write(line)
    # print('Number of water molecules in the grid is',len(grid_water))
    # grid_water.close()

    return (Voxel_List)


# Now, splitting the inner cavity voxels into layers of thickness one voxel for analysis
def Find_Layers(O_Atoms, Atom_List, Inner_Empty_Voxels, Grid_Dimensions):
    # print('Finding layers')
    
    log_file.write("Assigning layer numbers to cavity voxels \n")
    log_file.flush()

    Lining_Voxels = []
    Layered_Voxels = []

    n_water = 0

    for box in Inner_Empty_Voxels:  # Finding the layer of voxels that just line the cavity
        r = box.right_bound
        l = box.left_bound
        u = box.up_bound
        d = box.down_bound
        h = box.height_bound
        de = box.depth_bound

        # Checking the bounds of the voxel to make sure that the bounds are defined
        if (r != 'END' and r is not None):
            if (r.isEmpty == False):
                for atom in r.contained_atoms:  # If the bounding voxel has an atom in it, then the voxel is just adjacent to the cavity wall, and is marked as the first layer
                    if Atom_List[atom][1]:
                        box.layer = 1
                        Lining_Voxels.append(box)
                        Layered_Voxels.append(box)
                        for O in box.OW_atoms:
                            O_Atoms[O] = (O_Atoms[O][0], O_Atoms[O]
                                          [1], 1, O_Atoms[O][3], O_Atoms[O][4])
                            n_water += 1

        if (l != 'END' and l is not None):
            if (l.isEmpty == False):
                for atom in l.contained_atoms:
                    if Atom_List[atom][1]:
                        box.layer = 1
                        Lining_Voxels.append(box)
                        Layered_Voxels.append(box)
                        for O in box.OW_atoms:
                            O_Atoms[O] = (O_Atoms[O][0], O_Atoms[O]
                                          [1], 1, O_Atoms[O][3], O_Atoms[O][4])
                            n_water += 1
        if (u != 'END' and u is not None):
            if (u.isEmpty == False):
                for atom in u.contained_atoms:
                    if Atom_List[atom][1]:
                        box.layer = 1
                        Lining_Voxels.append(box)
                        Layered_Voxels.append(box)
                        for O in box.OW_atoms:
                            O_Atoms[O] = (O_Atoms[O][0], O_Atoms[O]
                                          [1], 1, O_Atoms[O][3], O_Atoms[O][4])
                            n_water += 1
        if (d != 'END' and d is not None):
            if (d.isEmpty == False):
                for atom in d.contained_atoms:
                    if Atom_List[atom][1]:
                        box.layer = 1
                        Lining_Voxels.append(box)
                        Layered_Voxels.append(box)
                        for O in box.OW_atoms:
                            O_Atoms[O] = (O_Atoms[O][0], O_Atoms[O]
                                          [1], 1, O_Atoms[O][3], O_Atoms[O][4])
                            n_water += 1
        if (h != 'END' and h is not None):
            if (h.isEmpty == False):
                for atom in h.contained_atoms:
                    if Atom_List[atom][1]:
                        box.layer = 1
                        Lining_Voxels.append(box)
                        Layered_Voxels.append(box)
                        for O in box.OW_atoms:
                            O_Atoms[O] = (O_Atoms[O][0], O_Atoms[O]
                                          [1], 1, O_Atoms[O][3], O_Atoms[O][4])
                            n_water += 1
        if (de != 'END' and de is not None):
            if (de.isEmpty == False):
                for atom in de.contained_atoms:
                    if Atom_List[atom][1]:
                        box.layer = 1
                        Lining_Voxels.append(box)
                        Layered_Voxels.append(box)
                        for O in box.OW_atoms:
                            O_Atoms[O] = (O_Atoms[O][0], O_Atoms[O]
                                          [1], 1, O_Atoms[O][3], O_Atoms[O][4])
                            n_water += 1

    Lining_Voxels = set(Lining_Voxels)
    Lining_Voxels = list(Lining_Voxels)

    if args.layer_end == None:
        Dist = []
        for z1 in range(0, Grid_Dimensions[2]):
            z_dist = []
            z_voxels = []
            for box in Lining_Voxels:
                if box.Position[2] == z1:
                    z_voxels.append(box)
            if len(z_voxels) != 0:
                x_avg = 0
                y_avg = 0
                z_avg = 0
                for box in z_voxels:
                    x_avg = x_avg + box.Position[0]
                    y_avg = y_avg + box.Position[1]
                    z_avg = z_avg + box.Position[2]

                x_avg = x_avg/len(z_voxels)
                y_avg = y_avg/len(z_voxels)
                z_avg = z_avg/len(z_voxels)

                for box in z_voxels:
                    dist = np.sqrt(
                        (box.Position[0] - x_avg)**2 + (box.Position[1] - y_avg)**2 + (box.Position[2] - z_avg)**2)
                    z_dist.append(dist)

                Dist.append(max(z_dist))

        Cutoff = max(Dist) + 5
    elif args.layer_end != None:
        Cutoff = args.layer_end + 1

    log_file.write("Cutoff for layer assignment is: " + str(Cutoff) + "\n")
    log_file.write("\n")
    log_file.flush()
    
    # print('Cutoff value for layers is:',Cutoff)
    Layered_Voxels = []
    All_Layers_Traversed = False  # Initialisng the condition for the while loop
    i = 1  # Initialising the layer number

    while All_Layers_Traversed == False:  # Finding the voxels with layer=0 adjacent to the voxels marked as layer=i, and marking them as layer=i+1, until all the Inner Cavity Voxels have a layer
        for box in Inner_Empty_Voxels:  # attribute other than 0
            Bounds = []
            if box.layer == 0:
                Bounds.append(box.right_bound)
                Bounds.append(box.left_bound)
                Bounds.append(box.up_bound)
                Bounds.append(box.down_bound)
                Bounds.append(box.height_bound)
                Bounds.append(box.depth_bound)

            for Box in Bounds:
                if (Box != 'END' and Box is not None):
                    if Box.layer == i:
                        box.layer = i+1
                        Layered_Voxels.append(box)
                        for O in box.OW_atoms:
                            O_Atoms[O] = (O_Atoms[O][0], O_Atoms[O]
                                          [1], i+1, O_Atoms[O][3], O_Atoms[O][4])
                            n_water += 1
        a = []
        for box in Inner_Empty_Voxels:
            if box.layer == 0:
                a.append(box)
        if len(a) == 0:
            All_Layers_Traversed = True
        elif i > Cutoff:
            break
        else:
            i += 1
            layer = i

    for box in Lining_Voxels:
        Layered_Voxels.append(box)

    Layered_Voxels = set(Layered_Voxels)
    Layered_Voxels = list(Layered_Voxels)

    Layer = []
    for box in Layered_Voxels:
        Layer.append(box.layer)

    Layer = set(Layer)

    layer = max(Layer)
    
    log_file.write("Number of layers found in the cavity is: " + str(layer))
    
    # print('Number of water after layering is: ', n_water)

    return (Lining_Voxels, Layered_Voxels, layer)

def Find_Cavity_Water(u, t, Args, Layer_True):
    Cavity_Water = {}

    Atoms = Produce_Atom_Lists(u, t, Args.w, Args)

    Atom_List = Atoms[0]
    O_Atoms = Atoms[1]
    Axis = Atoms[2]
    Perp_Vec = Atoms[3]

    Grid = Create_grid_hydration(Atom_List, Args.vx_dim, O_Atoms)

    Atom_List = Grid[0]
    Voxel_List = Grid[1]
    Grid_Dimensions = Grid[2]
    Lin_Trans_Coords = Grid[3]

    O_ls = mda.core.groups.AtomGroup(list(O_Atoms.keys()))

    O_ls.positions -= Lin_Trans_Coords
    val = O_Atoms.values()

    O_Atoms = dict(zip(O_ls, val))
    Inner_surface_ls = Find_surface_inner_hydration(
        Atom_List, Voxel_List, Grid_Dimensions, False, False)

    Inner_Empty_Voxels = Inner_surface_ls[1]
    Outer_Empty_Voxels = Inner_surface_ls[-1]

    Voxel_List_Hydrated = Add_Water(O_Atoms, Voxel_List, Grid_Dimensions)

    del Grid
    del Inner_surface_ls
    del Atoms

    if (Layer_True == True):
        Layers = Find_Layers(O_Atoms, Atom_List,
                             Inner_Empty_Voxels, Grid_Dimensions)
        Bulk_Water = []
        O_ls = O_Atoms.keys()
        for O in O_ls:
            if O_Atoms[O][3]:  # Changed [4] to [3] here.
                Cavity_Water[O] = O_Atoms[O]

            if (O_Atoms[O][3] == False):
                Bulk_Water.append(O)

        Layered_Voxels = Layers[1]
        Layer = Layers[2]
        # print("number of voxels ")
        del Layers
        log_file.write("Number of water molecules found within the cavity is: " + str(len(list(Cavity_Water.keys()))) + "\n")
        log_file.write("\n")
        log_file.flush()
        return (Cavity_Water, Axis, Perp_Vec, Atom_List, O_Atoms, Layered_Voxels, Layer, Voxel_List_Hydrated, Grid_Dimensions, Inner_Empty_Voxels, Bulk_Water, Outer_Empty_Voxels)

    else:
        Bulk_Water = []
        O_ls = O_Atoms.keys()
        for O in O_ls:
            if O_Atoms[O][4]:
                Cavity_Water[O] = O_Atoms[O]

            if (O_Atoms[O][3] == False):
                Bulk_Water.append(O)
        
        log_file.write("Number of water molecules found within the cavity is: " + str(len(list(Cavity_Water.keys()))) + "\n")
        log_file.write("\n")
        log_file.flush()
        return (Cavity_Water, Axis, Perp_Vec, Atom_List, O_Atoms, Voxel_List_Hydrated, Grid_Dimensions, Bulk_Water, Outer_Empty_Voxels)
        # return(Cavity_Water, Axis, Perp_Vec, Atom_List, O_Atoms, Voxel_List_Hydrated, Grid_Dimensions, Inner_Empty_Voxels, Bulk_Water)


# Now, splitting the inner cavity voxels into layers of thickness one voxel for analysis





def Dens_Plot(u, Args):
    log_file.write("Calculating Density Profile!\n")
    log_file.write("\n")
    log_file.flush()
    Info = Find_Cavity_Water(u, 0, Args, True)


    Cavity_Water = Info[0]
    Layered_Voxels = Info[5]
    Layer = Info[6]

    Water_Density = []
    for k in range(1, Layer + 1):
        n_layer_voxels = 0
        n_layer_water = 0
        for box in Layered_Voxels:
            if box.layer == k:
                n_layer_water += len(box.OW_atoms)
                n_layer_voxels += 1

        Layer_Dens = (n_layer_water/(n_layer_voxels *
                      (Args.vx_dim**3)))*(29.9157)
        Water_Density.append(Layer_Dens)

    Layer_Numbers = [i for i in range(1, len(Water_Density) + 1)]
    
    Out_Df = pnd.DataFrame({"Distance From Cavity Wall (Angstroms)" : Layer_Numbers, "Water Density (g cm^-3)" : Water_Density})
    
    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df.to_csv("CICLOP_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".csv")
            log_file.write("Data written to file" + "CICLOP_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            Out_Df.to_csv(Args.to_csv)
            log_file.write("Data written to file" + Args.to_csv + "\n")
            log_file.write("\n")
            log_file.flush()
    
    
    Plot(Layer_Numbers, Water_Density, Args, r"Distance From Cavity Wall $\left(\AA \right)$",
         r"Water Density $\left( g\;cm^{-3} \right)$", r"Water Density Profile", "CICLOP_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png", None)

    return (Water_Density, Layer_Numbers)


def errorbar_multiprocess(args):
    u, i, Args, find_cavity_water = args[0], args[1], args[2], args[3]
    # if Args.outer_surface != True:
    Frame_Info = Find_Cavity_Water(u, i, Args, find_cavity_water)
    # else:
    #     Frame_Info = Find_Outer_Water(u,i, Args)

    Inner_Empty_Voxels = Frame_Info[9]
    Layered_Voxels = Frame_Info[5]
    Layer = Frame_Info[6]

    Frame_Water_Density = []

    for k in range(1, Layer+1):
        n_layer_voxels = 0
        n_layer_water = 0
        for box in Layered_Voxels:
            if box.layer == k:
                n_layer_water += len(box.OW_atoms)
                n_layer_voxels += 1

        Layer_Dens = (n_layer_water/(n_layer_voxels *
                      (Args.vx_dim**3)))*(29.9157)
        Frame_Water_Density.append(Layer_Dens)

    return Layer, Frame_Water_Density


# Now results is a list of tuples, where each tuple contains the Layers and Water_Density_Array for each frame

def Errorbar(u, Args):
    log_file.write("Finding time averaged cavity water density profile \n")
    log_file.write("\n")
    log_file.flush()
    
    # Creating DataFrame for making the errorbar plot
    Layers = []
    Water_Density_Array = []

    dt = round(u.trajectory.dt, 3)
    st = 0
    en = len(u.trajectory)
    step = 1

    if (Args.t0 != None):
        st = int(Args.t0/dt)
    if (Args.tf != None):
        en = int(Args.tf/dt)
    if (Args.step != None):
        step = int(Args.step/dt)
    
    log_file.write("Analysis done between " + str(Args.t0) + "ps and " + str(Args.tf) + "ps with a time-step of " + str(Args.step) + "ps \n")
    log_file.write("\n")
    log_file.flush()
    
    T = list(i for i in range(st, en, step))

    results = []
    
    
    if (Args.multi_process == True):
        # Process the universe into chunks of size frame_storage
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size

        start = 0
        Chunks = []

        for i in range(num_chunks):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))
        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()
        
        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)

        chunk_times = []
        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []
            # print("\n")
            # print("\n")
            # print("Processing chunk number: ", i)
            # print("\n")
            # print("\n")
            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                results += list(executor.map(errorbar_multiprocess,
                                 [(Atom_Groups[j], j, Args, True) for j in range(len(Atom_Groups))]))
                gc.collect()
    else:
        for i in tqdm(T):
            results.append(errorbar_multiprocess((u, i, Args, True)))
    
    '''
    if (Args.multi_process == True):
        chunk_size = 500
        num_chunks = len(fr) // chunk_size + (len(fr) % chunk_size > 0)
        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()
        
        for i in tqdm(range(num_chunks)):
            start = i * chunk_size
            end = min((i+1) * chunk_size, len(fr))
            chunk_T = fr[start:end]

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                results += list(executor.map(errorbar_multiprocess,
                                [(get_frame(u, j), j, Args, True) for j in chunk_T]))
                gc.collect()

    '''

    for res in results:
        Layers.append(res[0])
        Water_Density_Array.append(res[1])

    Min_Layers = min(Layers)
    #print("min layer : ", Min_Layers)
    
    log_file.write("Minimum number of layers identified in the cavity is: " + str(Min_Layers) + "\n")
    log_file.write("\n")
    log_file.flush()
    Resized_Water_Density_Array = []

    for List in Water_Density_Array:
        Dummy = List[:Min_Layers]
        Resized_Water_Density_Array.append(Dummy)

    # print(len(Resized_Water_Density_Array),len(Resized_Water_Density_Array[0]),len(Resized_Water_Density_Array[1]),Min_Layers)

    Water_Density_DataFrame = pnd.DataFrame(Resized_Water_Density_Array)
    

    Mean = []
    Standard_Deviation = []
    for i in range(0, Min_Layers):
        Mean.append(Water_Density_DataFrame[i].mean())
        Standard_Deviation.append(Water_Density_DataFrame[i].std())
    
    Layer_Numbers = [l for l in range(1, Min_Layers + 1)]
    Out_Df = pnd.DataFrame({'Distance From Cavity Wall (Angstroms)':Layer_Numbers, 'Average Density (g cm^-3)': Mean, 'Standard Deviation (g cm^-3)': Standard_Deviation})

    
    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df.to_csv("CICLOP_Avg_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".csv")
            log_file.write("Data written to file" + "CICLOP_Avg_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            Water_Density_DataFrame.to_csv(Args.to_csv)
            log_file.write("Data written to file" + Args.to_csv + "\n")
            log_file.write("\n")
            log_file.flush()
    Plot_Errorbar(Layer_Numbers, Mean, Standard_Deviation, Args, 'Distance From Cavity Wall(Angstrom)',
                  r'Average Water Density($g \; cm^{-3}$)', 'Average Water Number Density', "CICLOP_Avg_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png")

    return (0)


def Radius_or_Volume_Profile(args):
    
    Args, u, t, Rad_True, Vol_True = args[0], args[1], args[2], args[3], args[4]
    u.trajectory[t]
    Atom_Lists = Produce_Atom_Lists(u, t, Args.w, Args)
    All_Atoms = Atom_Lists[0]

    Grid = Create_grid_hydration(All_Atoms, Args.vx_dim, Atom_Lists[1])
    All_Atoms = Grid[0]
    Voxel_List = Grid[1]
    Grid_Dimensions = Grid[2]

    Inner_Surface_Info = Find_surface_inner_hydration(
        All_Atoms, Voxel_List, Grid_Dimensions, True, False)

    if Args.rad_profile == True:
        log_file.write("Calculating Radius Profile! \n")
        log_file.write("\n")
        log_file.flush()
    
    elif Args.vol_profile == True:
        log_file.write("Calculating Volume Profile! \n")
        log_file.write("\n")
        log_file.flush()
    
    Rad_X = Inner_Surface_Info[2]
    Rad_Y = Inner_Surface_Info[3]
    
    Vol_X = Inner_Surface_Info[4]
    Vol_List = Inner_Surface_Info[5]
    
    Out_Df = pnd.DataFrame({"Distance Along Protein Axis (Angstroms)": Rad_X, "Radius (Angstroms)" : Rad_Y})
    
    
    if (Args.rad_profile == True or Rad_True == True):
        Out_Df = pnd.DataFrame({"Distance Along Protein Axis (Angstroms)": Rad_X, "Radius (Angstroms)" : Rad_Y})
        Out_Df.to_csv("CICLOP_Radius_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".png")
        Plot(Rad_X, Rad_Y, Args, "Distance Along Cavity Axis(Angstrom)", "Diameter(Angstroms)",
             "Radius Profile", "CICLOP_Radius_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png", None)

    elif (Args.vol_profile == True or Vol_True == True):
        Plot(Rad_X, Rad_Y, Args, "Distance Along Cavity Axis(Angstrom)", "Diameter(Angstroms)",
             "Radius Profile", "CICLOP_Volume_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png", None)
        
        return Vol_List, Vol_X


def Charge_Profile(Args):
    u = mda.Universe(Args.s, Args.f)

    ag = u.select_atoms("all")

    Atom_Lists = Produce_Atom_Lists(ag, 0, Args.w, Args)
    All_Atoms = Atom_Lists[0]
    O_Atoms = Atom_Lists[1]
    Grid = Create_grid_hydration(All_Atoms, Args.vx_dim, O_Atoms)

    All_Atoms = Grid[0]
    Voxel_List = Grid[1]
    Grid_Dimensions = Grid[2]

    Inner_Surf = Find_surface_inner_hydration(
        All_Atoms, Voxel_List, Grid_Dimensions, False, True)
    
    log_file.write("Calculating Charge Profile!\n")
    log_file.write("\n")
    log_file.flush()
    
    Z_Charge = Inner_Surf[2]
    Z = list(range(1, Grid_Dimensions[2] + 1))
    
    Out_Df = pnd.DataFrame({"Distance Along Protein Axis (Angstroms)" : Z, "Charge (e)" : Z_Charge})
    
    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df.to_csv("CICLOP_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d")+"-"+ time.strftime("%H-%M-%S") + ".csv")
            log_file.write("Data written to file" + "CICLOP_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            Out_Df.to_csv(Args.to_csv)
            log_file.write("Data written to file" + Args.to_csv + "\n")
            log_file.write("\n")
            log_file.flush()
    
    Max_Charge = np.max(Z_Charge)
    Min_Charge = np.min(Z_Charge)
    
    Z_Max = Z[np.argmax(Z_Charge)]
    Z_Min = Z[np.argmin(Z_Charge)]
    
    log_file.write("Maximum and Minimum values of charge along the axes are " + str(Max_Charge) + "e and " + str(Min_Charge) + "e at Z = " + str(Z_Max) + " Angstroms and at Z = " + str(Z_Min) + " Angstroms respectively \n")
    log_file.write("\n")
    log_file.flush()
    
    Plot(Z, Z_Charge, Args, r"Distance Along Z-Axis $\AA$",
         r"Charge $e$", r"Protein Cavity Charge Profile", Args.o, None)
    
    #print(np.array(Z))
    #print(np.array(Z_Charge))


def Charge_Pro_Multiprocess(args):
    ag, Args = args[0], args[1]
    Atom_Lists = Produce_Atom_Lists(ag, 0, Args.w, Args)
    All_Atoms = Atom_Lists[0]
    O_Atoms = Atom_Lists[1]
    Grid = Create_grid_hydration(All_Atoms, Args.vx_dim, O_Atoms)

    All_Atoms = Grid[0]
    Voxel_List = Grid[1]
    Grid_Dimensions = Grid[2]

    Inner_Surf = Find_surface_inner_hydration(
        All_Atoms, Voxel_List, Grid_Dimensions, False, True)

    Z_Charge = Inner_Surf[2]

    return Z_Charge


def Charge_Profile_Errorbar(u, Args):
    
    log_file.write("Calculating Average Charge Profile!\n")
    log_file.write("\n")
    log_file.flush()
    

    dt = round(u.trajectory.dt, 3)

    T = list(range(int(Args.t0/dt), int(Args.tf/dt) +
             int(Args.step/dt), int(Args.step/dt)))
    
    log_file.write("Analysis done between " + str(Args.t0) + "ps and " + str(Args.tf) + "ps with a time-step of " + str(Args.step) + "ps \n")
    log_file.write("\n")
    log_file.flush()
    
    Charge_Array = []

    if (Args.multi_process == True):
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size
        
        
        
        start = 0
        Chunks = []

        for i in tqdm(range(num_chunks)):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))
        
        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()
        
        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)

        chunk_times = []
        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []
            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                Charge_Array += list(executor.map(Charge_Pro_Multiprocess,
                                     [(j, Args) for j in Atom_Groups]))

    else:
        for t in tqdm(T):
            u.trajectory[t]
            ag = u.select_atoms("all")
            Charge_Array.append(Charge_Pro_Multiprocess([ag, Args]))

    Z = [len(arr) for arr in Charge_Array]

    z = min(Z)

    Charge_Array_res = np.array([Arr[: z] for Arr in Charge_Array])

    Charge_Avg = np.mean(Charge_Array_res, axis=0)
    Charge_Std = np.std(Charge_Array_res, axis=0)

    X = list(range(1, z + 1))
    Total_Cav_Charge = np.sum(Charge_Array_res, axis=1)
    T1 = dt*np.array(T)

    Out_Df = pnd.DataFrame(
        {"Distance": X, "Average Charge(e)": Charge_Avg, "Standard Deviation": Charge_Std})
    Out_Df_2 = pnd.DataFrame(
        {"Time (ps)": T1, "Total Cavity Charge (e)": Total_Cav_Charge})

    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df.to_csv("CICLOP_Avg_Charge_Profile-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".csv")
            log_file.write("Data written to file" + "CICLOP_Avg_Charge_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            Out_Df.to_csv(Args.to_csv)
            log_file.write("Data written to file" + Args.to_csv + "\n")
            log_file.write("\n")
            log_file.flush()

    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df_2.to_csv("CICLOP_Charge_Time_Series-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv")
            log_file.write("Data written to file" + "CICLOP_Charge_Time_Series-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            name = Args.to_csv
            a = name.split(".")
            name = a[0] + "_Total_Charge_Timeseries." + a[1]
            Out_Df_2.to_csv(name)
            log_file.write("Data written to file" + name + "\n")
            log_file.write("\n")
            log_file.flush()

    Plot_Errorbar(X, Charge_Avg, Charge_Std, Args, r"Distance Along Z-Axis $\AA$", r"Charge $e$",
                  r"Average Cavity Charge Profile", "CICLOP_Avg_Charge_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png")

    plt.figure(num=1, dpi=Args.dpi)
    plt.xlabel(r"Time (ps)")
    plt.ylabel(r"Total Cavity Charge (e)")
    plt.title(r"Total Cavity Charge Time Series")
    plt.plot(T1, Total_Cav_Charge)
    if Args.o == None:
        plt.savefig("CICLOP_Total_Charge_Timeseries-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png", dpi=Args.dpi)
        log_file.write("Data written to file" + "CICLOP_Total_Charge_Timeseries-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png" + "\n")
        log_file.write("\n")
        log_file.flush()
    elif Args.o != None:
        name = Args.o
        a = name.split(".")
        name = a[0] + "_Total_Charge_Timeseries." + a[1]
        plt.savefig(name, dpi=Args.dpi)
        log_file.write("Data written to file" + name + "\n")
        log_file.write("\n")
        log_file.flush()
            
def water_traj_multiprocess1(args):
    u, t, Args = args[0], args[1], args[2]
    Frame_Info = Find_Cavity_Water(u, t, Args, True)

    Cavity_Water = Frame_Info[0]
    O_Atoms = Frame_Info[4]

    EOI = []#Ensemble of Interest
    To_Cav = []#Towards Cavity    
    To_Center = []#Towards Center
    Out = []#Water outside the cavity
    
    if Args.layer_start != 1:
        for O in Cavity_Water.keys():
            if Args.layer_start <=Cavity_Water[O][2] <= Args.layer_end:
                EOI.append(O.index)
            elif Cavity_Water[O][2] == 0 or Cavity_Water[O][2] > Args.layer_end:
                To_Center.append(O.index)
            elif Cavity_Water[O][2] < Args.layer_start:
                To_Cav.append(O.index)
    
    elif Args.layer_start == 1:
        for O in Cavity_Water.keys():
            if Args.layer_start <=Cavity_Water[O][2] <= Args.layer_end:
                EOI.append(O.index)
            elif Cavity_Water[O][2] == 0 or Cavity_Water[O][2] > Args.layer_end:
                To_Center.append(O.index)
    
    for O in O_Atoms.keys():
        if O_Atoms[O][4] == False:
            Out.append(O.index)
    
    EOI = np.array(EOI)
    To_Cav = np.array(To_Cav)
    To_Center = np.array(To_Center)
    Out = np.array(Out)
    
    return(EOI, To_Cav, To_Center, Out)
    
    
def Water_Trajectory(u, Args):
    
    log_file.write("Finding Water Movement Propensity!\n")
    log_file.write("\n")
    log_file.flush()
    
    dt = round(u.trajectory.dt, 3)

    T = list(range(int(Args.t0/dt), int(Args.tf/dt), int(Args.step/dt)))
    T1 = list(range(int(Args.t0/dt), int(Args.tf/dt), int(Args.step/dt)))

    log_file.write("Analysis done between " + str(Args.t0) + "ps and " + str(Args.tf) + "ps with a time-step of " + str(Args.step) + "ps \n")
    log_file.write("\n")
    log_file.flush()

    for i in range(len(T1)):
        T1[i] = T1[i] * dt

    # Args.le = 12
    
    log_file.write("Water for chosen for analyses lie between " + str(Args.layer_start) + " Angstroms and " + str(Args.layer_end) + " Angstroms from the cavity wall\n")
    log_file.write("\n")
    log_file.flush()
    
    EOI = []
    To_Cav = []
    To_Center = []
    Out = []
    if (Args.multi_process == True):
        # Process the universe into chunks of size frame_storage
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size

        start = 0
        Chunks = []

        for i in range(num_chunks):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))

        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()        

        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)

        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []

            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                results = list(executor.map(water_traj_multiprocess1, [(Atom_Groups[j], j, Args) for j in range(len(Atom_Groups))]))
                for res in results:
                    EOI.append(np.array(res[0]))
                    To_Cav.append(np.array(res[1]))
                    To_Center.append(np.array(res[2]))
                    Out.append(np.array(res[3]))
                gc.collect()

            #time_c2 = time.time()
            # print("Time taken to process chunk is: ", time_c2 - time_c1)

    else:
        for i in tqdm(T):
            u_new = u.copy()
            u_new.trajectory[i]
            ag = u_new.select_atoms("all")
            res = water_traj_multiprocess1((ag, i, Args))
            EOI.append(np.array(res[0]))
            To_Cav.append(np.array(res[1]))
            To_Center.append(np.array(res[2]))
            Out.append(np.array(res[3]))
            #print(type(res[0]), type(res[1]), type(res[2]), type(res[3]), type(res[4]))

    del u_new
    
    '''
    At every point in time, wrt to the original ensemble, plot the fraction of water molecules
    1) Moving towards the cavity wall
    2) Moving towards the cavity center
    3) Moving out of the cavity
    4) Fraction staying within the ensemble
    '''
    
    L = len(EOI)    
    
    Self = np.zeros(L, dtype = float)
    Cav = np.zeros(L, dtype = float)
    Center = np.zeros(L, dtype = float)
    Outer = np.zeros(L, dtype = float)
    Counter = np.zeros(L, dtype = int)
    
    if Args.layer_start != 1:
    
        for i in tqdm(range(L)):
            EOI_Ref = EOI[i]
            N_Ref = len(EOI_Ref)
            for j in range(i, L):
                d_ij = j - i
                
                Self[d_ij] += np.sum(np.isin(EOI_Ref, EOI[j]).astype(int))/N_Ref
                Cav[d_ij] += np.sum(np.isin(EOI_Ref, To_Cav[j]).astype(int))/N_Ref
                Center[d_ij] += np.sum(np.isin(EOI_Ref, To_Center[j]).astype(int))/N_Ref
                Outer[d_ij] += np.sum(np.isin(EOI_Ref, Out[j]).astype(int))/N_Ref
                Counter[d_ij] += 1
        
        Self /= Counter
        Cav /= Counter
        Center /= Counter
        Outer /= Counter
        
        
        
        
        Out_Df = pnd.DataFrame({"Time(ps)":T1, "Selected Ensemble: Layers " + str(Args.layer_start) + " - " + str(Args.layer_end):Self, "Towards Cavity":Cav, "Towards Center":Center, "Leaving Cavity":Outer})
        if Args.no_csv == False:
            if Args.to_csv == None:
                Out_Df.to_csv("CICLOP_Water_Movement_Propensity-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv")
                log_file.write("Data written to file" + "CICLOP_Water_Movement_Propensity-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".csv \n")
                log_file.write("\n")
                log_file.flush()
            elif Args.to_csv != None:
                Out_Df.to_csv(Args.to_csv)
                log_file.write("Data written to file" + Args.to_csv)
                log_file.write("\n")
                log_file.flush()
                
                
        X = [T1, T1, T1, T1]
        Y = [Self, Cav, Center, Outer]
        Legend = ['Fraction Staying In ' + str(Args.layer_start) + r'$\AA$ and ' + str(Args.layer_end) + r'$\AA$', 'Fraction Moving Towards Cavity', 'Fraction Moving Towards Cavity Center','Fraction Leaving Cavity']
        Plot(X, Y, Args, r"Time(ps)", r"Fraction of Water Molecules", r"Water Movement Propensity", time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()) + '_Water_Propensity.png', Legend)
        
    
    elif Args.layer_start == 1:
    
        for i in tqdm(range(L)):
            EOI_Ref = EOI[i]
            N_Ref = len(EOI_Ref)
            for j in range(i, L):
                d_ij = j - i
                
                Self[d_ij] += np.sum(np.isin(EOI_Ref, EOI[j]).astype(int))/N_Ref
                Center[d_ij] += np.sum(np.isin(EOI_Ref, To_Center[j]).astype(int))/N_Ref
                Outer[d_ij] += np.sum(np.isin(EOI_Ref, Out[j]).astype(int))/N_Ref
                Counter[d_ij] += 1
        
        Self /= Counter
        Center /= Counter
        Outer /= Counter
        
        Out_Df = pnd.DataFrame({"Time(ps)":T1, "Selected Ensemble: Layers " + str(Args.layer_start) + " - " + str(Args.layer_end):Self, "Towards Center":Center, "Leaving Cavity":Outer})
        if Args.no_csv == False:
            if Args.to_csv == None:
                Out_Df.to_csv("CICLOP_Water_Movement_Propensity-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv")
                log_file.write("Data written to file" + "CICLOP_Water_Movement_Propensity-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".csv \n")
                log_file.write("\n")
                log_file.flush()
            elif Args.to_csv != None:
                Out_Df.to_csv(Args.to_csv)
                log_file.write("Data written to file" + Args.to_csv)
                log_file.write("\n")
                log_file.flush()
        
        X = [T1, T1, T1]
        Y = [Self, Center, Outer]
        Legend = ['Fraction Staying In ' + str(Args.layer_start) + r'$\AA$ and ' + str(Args.layer_end) + r'$\AA$', 'Fraction Moving Towards Cavity Center','Fraction Leaving Cavity']
        Plot(X, Y, Args, r"Time(ps)", r"Fraction of Water Molecules", r"Water Trajectory", "CICLOP_Water_Movement_Propensity-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png", Legend)        


def diffusion_multiprocess(args):
    u, OW_Indices = args[0], args[1]
    Frame_Ensemble = u.atoms[OW_Indices].positions
    return Frame_Ensemble


def Find_Diffusion_Coefficient(u, Args):
    
    log_file.write("Finding Water Diffusion Coefficient!\n")
    log_file.write("\n")
    log_file.flush()
    #########################################################################################
    if(Args.tf-Args.t0 <= 1.0):
        log_file.write("")
        log_file.flush()
        raise Warning("")
    #########################################################################################
    Sel_IDs = []
    #u = mda.Universe(Args.s, Args.traj)
    dt = round(u.trajectory.dt, 3)
    T = list(range(int(Args.t0/dt), int(Args.tf/dt) +
             int(Args.step/dt), int(Args.step/dt)))
    
    u.trajectory[T[0]]
    
    log_file.write("Analysis done between " + str(Args.t0) + "ps and " + str(Args.tf) + "ps with a time-step of " + str(Args.step) + "ps \n")
    log_file.write("\n")
    log_file.flush()
    
    if Args.all_water == True:
        Info = Find_Cavity_Water(u, 0, Args, False)
        #print("Time: ", u.trajectory.time)
        Cavity_Water = Info[0]
        O_Sel = [O for O in Cavity_Water.keys()]
        
        log_file.write("Entirety of Cavity Water chosen for analysis\n")
        log_file.write("\n")
        log_file.flush()
        
        AtomGroup = mda.core.groups.AtomGroup(O_Sel)
        Sel_IDs = [O.index for O in Cavity_Water.keys()]
    elif Args.layer_start != None and Args.layer_end != None:
        Info = Find_Cavity_Water(u, 0, Args, True)
        Cavity_Water = Info[0]
        AtomGroup = []
        for O in Cavity_Water.keys():
            if Args.layer_start <= Cavity_Water[O][2] <= Args.layer_end:
                AtomGroup.append(O)
                Sel_IDs.append(O.index)

        log_file.write("Water found between " + str(Args.layer_start) + " Angstroms and " + str(Args.layer_end) + " Angstrsoms from the cavity wall chosen for analysis\n")
        log_file.write("\n")
        log_file.flush()        

        # print(Ensemble)
        # print(len(Ensemble))
        # print(Ensemble[0])
        # print(type(Ensemble[0]))

        AtomGroup = mda.core.groups.AtomGroup(AtomGroup)

    print('Number of water molecules selected for analysis is: ', len(AtomGroup))
    
    log_file.write('Number of water molecules selected for analysis is: ' + str(len(AtomGroup)) + "\n")
    log_file.write("\n")
    log_file.flush()

    def f(x, m, c):
        return (m*x + c)

    

    # OW_Indices = AtomGroup.indices

    

    Ensemble = []

    if (Args.multi_process == True):
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size

        start = 0
        Chunks = []

        for i in tqdm(range(num_chunks)):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))
        
        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()
        
        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)
        
        chunk_times = []
        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []
            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                Ensemble += list(executor.map(diffusion_multiprocess,
                                 [(j, Ensemble) for j in Atom_Groups]))

       
    else:
        print("Reading Frames")
        for t in tqdm(T):
            u.trajectory[t]
            atoms = u.atoms[Sel_IDs]
            Ensemble.append(atoms.positions.copy())

    
    L = len(Ensemble)
    MSD = np.zeros(L)
    Counter = np.zeros(L)

    # a = [len(Frame) for Frame in Ensemble]
    # print(np.mean(a), np.sum(a))
    print("Beginning moving time average")
    log_file.write("Performing Moving Time Average\n")
    log_file.write("\n")
    log_file.flush()
    
    for t in tqdm(range(L)):
        Ref_Frame = Ensemble[t]
        Fwd_Coords = Ensemble[t:]

        Disp_Vecs = Fwd_Coords - Ref_Frame

        Disp_Vecs_Squared = np.square(Disp_Vecs)

        R_Squared = np.sum(Disp_Vecs_Squared, axis=2)

        R_Sq_Avg = np.mean(R_Squared, axis=1)

        MSD[:L-t] += R_Sq_Avg
        Counter[:L-t] += 1

    print(Counter)

    MSD_Avg = MSD/Counter

    '''
    Initial_Coords = Ensemble[0]
    
    Disp_Vecs = Ensemble - Initial_Coords
    Disp_Vecs_Sq = np.square(Disp_Vecs)
    R_Sq = np.sum(Disp_Vecs_Sq, axis = 2)
    MSD_Avg = np.mean(R_Sq, axis = 1)
    '''

    '''
    MSD_Avg = []
    for t in range(0, len(Ensemble)):
        MSD_t = 0
        for i in range(0, len(Initial_Coords)):
            Initial_Pos = Initial_Coords[i]
            Current_Pos = Ensemble[t][i]

            Dist = math.dist(Initial_Pos, Current_Pos)
            MSD_t += Dist*Dist

        MSD_t = MSD_t/len(Initial_Coords)
        MSD_Avg.append(MSD_t)
    '''

    T1 = list(range(int(Args.t0/dt), int(Args.tf/dt) +
              int(Args.step/dt), int(Args.step/dt)))

    for i in range(len(T1)):
        T1[i] = T1[i]*dt

    limits = int(len(T1)*Args.fit_limit)

    print("Regression line is made between ",
          T1[limits], " and ", T1[len(T1) - limits-1])

    log_file.write("Regression line is made between " + str(T1[limits]) + " and " + str(T1[len(T1) - limits-1]))
    log_file.write("\n")
    log_file.flush()
    

    Out_Df = pnd.DataFrame(
        {"Time(ps)": T1, "Mean Squared Displacement": MSD_Avg})

    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df.to_csv("CICLOP_Diffusion-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv")
            log_file.write("Data written to file" + "CICLOP_Diffusion-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".csv \n")
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            Out_Df.to_csv(Args.to_csv)
            log_file.write("Data written to file" + Args.to_csv)
            log_file.write("\n")
            log_file.flush()

    m, c, r_value, p_value, std_err = linregress(T1[limits: len(T1) - limits], MSD_Avg[limits: len(MSD_Avg) - limits])
    
    
    
    D = m/6.0
    
    
    log_file.write("Fit parameters are:\n")
    log_file.write("Slope: " + str(m) + " Angstroms squared per picosecond \n")
    log_file.write("Intercept: " + str(c) + "Angstroms squared \n")
    log_file.write("r-value: " + str(r_value) + "\n")
    log_file.write("r squared: " + str(r_value**2) + "\n")
    log_file.write("p-value: " + str(p_value) + "\n")
    log_file.write("Standard Error: " + str(std_err) + "\n")
    log_file.write("\n")
    log_file.flush()

    # Must do a power-law fit as well against t^{alpha} and return alpha

    def g(x, m, c, alpha):
        y = m*(x**alpha) + c
        return (y)

    # params_power, params_power_cov = curve_fit(g, T1[limits : len(T1) - limits], MSD_Avg[limits : len(MSD_Avg) - limits])

    # D_Power_Law = params_power[0]/6
    # Alpha = params_power[1]
    # c_power_law = params_power[2]

    return (T1, MSD_Avg, D, c)


def Find_Rel_multiprocess(args):
    u, t, OW_Indices, HW1_Indices, HW2_Indices = args[0], args[1], args[2], args[3], args[4]

    OW_t = u.atoms[OW_Indices].positions
    HW1_t = u.atoms[HW1_Indices].positions
    HW2_t = u.atoms[HW2_Indices].positions

    Dipoles_t = OW_t - (HW1_t + HW2_t)/2

    for i in range(len(Dipoles_t)):
        Dipoles_t[i] = Dipoles_t[i]/np.linalg.norm(Dipoles_t[i])

    return Dipoles_t


def Find_Relaxation_Times_Final(AtomGroup, u, Args):
    log_file.write("Finding Water Rotational Relaxation Times!\n")
    log_file.write("\n")
    log_file.flush()
    
    dt = round(u.trajectory.dt, 3)

    T = list(range(int(Args.t0/dt), int(Args.tf/dt) +
             int(Args.step/dt), int(Args.step/dt)))
    T1 = list(range(int(Args.t0/dt), int(Args.tf/dt) +
              int(Args.step/dt), int(Args.step/dt)))

    for i in range(len(T1)):
        T1[i] = T1[i] * dt
    
    log_file.write("Analysis done between " + str(Args.t0) + "ps and " + str(Args.tf) + "ps with a time-step of " + str(Args.step) + "ps \n")
    log_file.write("\n")
    log_file.flush()
    
    
    if Args.all_water == True:
        log_file.write("Entirety of Cavity Water chosen for analysis\n")
        log_file.write("\n")
        log_file.flush()
    
    elif Args.layer_start != None and Args.layer_end != None:
        log_file.write("Water found between " + str(Args.layer_start) + " Angstroms and " + str(Args.layer_end) + " Angstrsoms from the cavity wall chosen for analysis\n")
        log_file.write("\n")
        log_file.flush()
    
    # Must select entire water molecule from the AtomGroup that contains only OW.
    OW_Indices = AtomGroup.indices

    # OW = AtomGroup.keys()
    # HW1 = AtomGroup.values()[0][0]
    # HW2 = AtomGroup.values()[0][1]

    HW1_Indices = np.array([i + 1 for i in OW_Indices])
    HW2_Indices = np.array([i + 2 for i in OW_Indices])

    # FO_ACF = np.zeros(len(T))
    # SO_ACF = np.zeros(len(T))

    dt = np.round(u.trajectory.dt, 3)

    Dipole_Array = []
    # counter =0
    if (args.multi_process == True):
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size

        start = 0
        Chunks = []

        for i in range(num_chunks):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))

        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()

        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)
        print("Processing Chunks")
        chunk_times = []
        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []
            
            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                results = list(executor.map(Find_Rel_multiprocess, [
                               (Atom_Groups[j], j, OW_Indices, HW1_Indices, HW2_Indices, Dipole_Array, dt) for j in range(len(Atom_Groups))]))
                gc.collect()
            time_c2 = time.time()
            chunk_times.append(time_c2 - time_c1)
            for res in results:
                Dipole_Array.append(res)

    else:
        print("Reading Frames")
        for t in tqdm(T):
            u.trajectory[t]
            ag = u.select_atoms("all")
            dpt = Find_Rel_multiprocess(
                (ag, t, OW_Indices, HW1_Indices, HW2_Indices))
            Dipole_Array.append(dpt)

    Dipole_Array = np.array(Dipole_Array)

    #print("Shape is:", Dipole_Array.shape)

    FO_ACF = np.zeros(len(T), dtype=float)
    SO_ACF = np.zeros(len(T), dtype=float)
    Counter = np.zeros(len(T), dtype=int)

    L = len(Dipole_Array)

    #print("Number of frames for analysis is: ", L)

    print("Calculating moving time average")
    log_file.write("Beginning Moving Time Average\n")
    log_file.write("\n")
    log_file.flush()
    
    for t in tqdm(range(L)):
        Ref_Frame = Dipole_Array[t]
        Fwd_Dipoles = Dipole_Array[t:]

        Multiplied_Dipoles = np.multiply(Ref_Frame, Fwd_Dipoles)

        Cos_Thetas = np.sum(Multiplied_Dipoles, axis=2)

        Second_Order = 0.5*(3*np.square(Cos_Thetas) - 1)

        Avg_Cos = np.mean(Cos_Thetas, axis=1)
        Avg_SO = np.mean(Second_Order, axis=1)

        FO_ACF[:L - t] += Avg_Cos
        SO_ACF[:L - t] += Avg_SO

        Counter[:L - t] += 1

    FO_ACF = FO_ACF/Counter
    SO_ACF = SO_ACF/Counter

    '''
    Dipole_Array  = np.array(Dipole_Array)
    
    FO_ACF = np.zeros(len(T), dtype = float)
    SO_ACF = np.zeros(len(T), dtype = float)
    Counter = np.zeros(len(T), dtype = int)
    
    L = len(Dipole_Array)
    
    print("Calculating moving time average.")
    
    for t in tqdm(range(L)):
        Ref_Frame = Dipole_Array[t]
        Fwd_Dipoles = Dipole_Array[t : ]
        
        Multiplied_Dipoles = np.multiply(Ref_Frame, Fwd_Dipoles)
        
        Cos_Thetas = np.sum(Multiplied_Dipoles, axis = 2)
        
        Second_Order = 0.5*(3*np.square(Cos_Thetas) - 1 )
        
        Avg_Cos = np.mean(Cos_Thetas, axis = 1)
        Avg_SO = np.mean(Second_Order, axis = 1)
        
        FO_ACF[: L - t] += Avg_Cos
        SO_ACF[: L - t] += Avg_SO
        
        Counter[: L - t] += 1
    
    FO_ACF /= Counter
    SO_ACF /= Counter
    '''

    '''
    for i in range(len(T)):
        D1 = Dipole_Array[i]
        for j in range(i, len(T)):
            d_ij = j - i
            D2 = Dipole_Array[j]
            
            cos_theta = np.sum(D1*D2, axis = 1)
            
            SO = 0.5*(3*cos_theta**2 - 1)
            
            FO_ACF[d_ij] += np.mean(cos_theta)
            SO_ACF[d_ij] += np.mean(SO)
            Counter[d_ij] += 1
    """
    
    FO_ACF = FO_ACF/Counter
    SO_ACF = SO_ACF/Counter        
    '''
    '''
    Initial_Dipoles = Dipole_Array[0]
    Multiplied_Dipoles = np.multiply(Initial_Dipoles, Dipole_Array)
    Cos_Thetas = np.sum(Multiplied_Dipoles, axis = 2)
    Second_Order = 0.5*(3*np.square(Cos_Thetas) - 1)
    
    FO_ACF = np.mean(Cos_Thetas, axis = 1)
    SO_ACF = np.mean(Second_Order, axis = 1)
    '''
    

    t1 = simps(FO_ACF, T1)
    t2 = simps(SO_ACF, T1)
    
    log_file.write("First and Second Order rotational relaxation times are " + str(t1) + " ps and " + str(t2) + " ps respectively \n")
    log_file.write("\n")
    log_file.flush()
    
    #print(FO_ACF, SO_ACF)
    return (T1, t1, t2, FO_ACF, SO_ACF)


def Write_Traj(u, Args):
    dt = round(u.trajectory.dt, 3)
    T = list(range(int(Args.t0/dt), int(Args.tf/dt), int(Args.step/dt)))

    u.trajectory[T[0]]

    Cav = Find_Cavity_Water(u, T[0], Args, True)

    Cavity_Water = Cav[0]

    Sel = []

    for O in Cavity_Water.keys():
        if Args.layer_start <= Cavity_Water[O][2] <= Args.layer_end:
            Sel.append(O.index)

    with XTCWriter(Args.o, len(Sel)) as w:
        for t in tqdm(T):
            u.trajectory[t]
            Sel_Atoms = u.atoms[Sel]

            w.write(Sel_Atoms)


'''
def Write_Traj(u, Args):
    dt = u.trajectory.dt
    T = list(range(int(Args.t0/dt), int(Args.tf/dt), int(Args.step/dt)))
    
    with XTCWriter(Args.o, u.atoms.n_atoms) as w:
        for ts in u.trajectory[int(Args.t0/dt):int(Args.tf/dt):int(Args.step/dt)]:
            Info = Find_Cavity_Water(u, 0, Args, True)
            
            Cavity_Water = Info[0]
            Atom_List = Info[3]
            
            # Create an array to store the temp_factors
            temp_factors = u.atoms.tempfactors.copy()

            # Modify temp_factors for Cavity Water atoms
            for O in Cavity_Water.keys():
                if Args.layer_start <= Cavity_Water[O][2] <= Args.layer_end:
                    O.position = Cavity_Water[O][0]
                    temp_factors[O.index] = 9999
                    for atom in Cavity_Water[O][1]:
                        temp_factors[atom.index] = 9999
            
            # Modify temp_factors for other atoms in Atom_List
            for atom in Atom_List.keys():
                if Atom_List[atom][1] == True:
                    temp_factors[atom.index] = 9999
            
            # Assign the modified temp_factors back to the atoms
            u.atoms.tempfactors = temp_factors
        
            # Write the modified frame to the new trajectory
            w.write(u.atoms)
'''


def get_frame(u, t):
    u_new = u.copy()
    u_new.trajectory[t]
    ag = u_new.select_atoms('all')

    del u_new

    return ag


gc.enable()


def res_time_multiprocess(args):
    Args, u, t, p = args[0], args[1], args[2], args[3]
    # print("Reading frame at time: ", t*Args.step)
    # print(t, p, t - p)
    if Args.multi_process == False:
        u.trajectory[t]
    if Args.all_water != True:
        # print("hey2")
        Frame_Info = Find_Cavity_Water(u, t, Args, True)
        Cavity_Water = Frame_Info[0]

        Frame_Ensemble = set()
        for O in Cavity_Water.keys():
            if Args.layer_start <= Cavity_Water[O][2] <= Args.layer_end:
                Frame_Ensemble.add(O.index)

        del Cavity_Water

    elif Args.all_water == True:
        Frame_Info = Find_Cavity_Water(u, t, Args, False)

        Frame_Ensemble = set([O.index for O in Frame_Info[0].keys()])

    del Frame_Info
    Frame_Ensemble = list(Frame_Ensemble)
    Frame_Ensemble = np.array(Frame_Ensemble)
    # print(len(Frame_Ensemble))
    # print("length")
    # print(len(Frame_Ensemble))
    return Frame_Ensemble

def Calculate_Residence_Times_New(Args):
    log_file.write("Finding Water Rotational Relaxation Times!\n")
    log_file.write("\n")
    log_file.flush()
    
    # print("Calculating residence times")
    u = mda.Universe(Args.s, Args.traj)
    dt = round(u.trajectory.dt, 3)
    print("Universe has been loaded")
    # print(dt)
    # print(Args.step)
    # print(int(Args.step/dt))
    T = list(range(int(Args.t0/dt), int(Args.tf/dt), int(Args.step/dt)))
    T = np.array(T)
    # T1 = list(range(int(Args.t0), int(Args.tf), int(Args.step)))

    T1 = np.zeros(len(T), dtype=float)
    for i in range(0, len(T)):
        T1[i] = round(Args.t0 + round(Args.step*i, 3), 3)

    log_file.write("Analysis done between " + str(Args.t0) + "ps and " + str(Args.tf) + "ps with a time-step of " + str(Args.step) + "ps \n")
    log_file.write("\n")
    log_file.flush()

    Ensemble = []
    
    if Args.all_water == True:
        log_file.write("Entirety of Cavity Water chosen for analysis\n")
        log_file.write("\n")
        log_file.flush()
    
    elif Args.layer_start != None and Args.layer_end != None:
        log_file.write("Water found between " + str(Args.layer_start) + " Angstroms and " + str(Args.layer_end) + " Angstrsoms from the cavity wall chosen for analysis\n")
        log_file.write("\n")
        log_file.flush()
        
    
    # Survival_Probabilities = np.zeros(len(T), dtype = float)
    # Counter = np.zeros(len(T), dtype = float)

    # print(int(u.trajectory[0].frame))
    # results = []
    if (Args.multi_process == True):
        # Process the universe into chunks of size frame_storage
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size

        start = 0
        Chunks = []

        for i in range(num_chunks):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))
        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()
        
        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)

        chunk_times = []
        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []
            # print("\n")
            # print("\n")
            # print("Processing chunk number: ", i)
            # print("\n")
            # print("\n")
            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                Ensemble += list(executor.map(res_time_multiprocess,
                                 [(Args, Atom_Groups[j], j, T[0]) for j in range(len(Atom_Groups))]))
                gc.collect()
            time_c2 = time.time()
            # print("Time taken to process chunk is: ", time_c2 - time_c1)
            chunk_times.append(time_c2 - time_c1)

    else:
        print("Reading Frames")
        for t in tqdm(T):
            Frame_Ensemble = res_time_multiprocess((Args, u, t, T[0]))
            Ensemble.append(Frame_Ensemble)

    l = len(Ensemble)
    #print("Number of frames is: ", l)
    # Defining empty arrays for storing surv. prob. and counter
    SP = np.zeros(l, dtype=float)
    Counter = np.zeros(l, dtype=int)
    time0 = time.time()
    print("Calculating moving time average")
    
    log_file.write("Beginning Moving Time Average\n")
    log_file.write("\n")
    log_file.flush()
    
    WM_t = []
    Prod_t = []
    for i in tqdm(range(l)):  # Performing moving time average
        Ref_Frame = Ensemble[i]  # Choosing reference frame
        L = len(Ref_Frame)
        time1 = time.time()

        Water_Map = np.zeros([l - i, L], dtype=int)

        for k in range(i, l):
            WM = np.isin(Ref_Frame, Ensemble[k])
            WM = WM.astype(int)
            Water_Map[k - i] = WM

        Water_Map = Water_Map.T

        time2 = time.time()
        # print("Time taken to produce the water map is: ", time2 - time1, " s")
        time3 = time.time()
        WM_t.append(time2 - time1)
        Time_Series_Product = np.cumprod(Water_Map, axis=1)

        Number_Present_All = np.sum(Time_Series_Product, axis=0)/L

        SP[: l - i] += Number_Present_All
        Counter[: l - i] += 1

        time4 = time.time()
        # print("Amount of time taken for np.prod is: ", time4 - time3, " s")
        Prod_t.append(time4 - time3)
        '''
        Water_Map = np.zeros([l - i, L], dtype = int)#Array that will store whether water is within region or not. 1 for yes, 0 no.
        
        Fwd_Time_Series = Ensemble[i : l]
        
        Time_Series_Flattened = np.concatenate(Fwd_Time_Series)
        
        Restructure_Indices = np.cumsum([len(row) for row in Fwd_Time_Series])[: -1]
        
        WM_1D = np.in1d(Time_Series_Flattened, Ref_Frame)
        
        WM_Split = np.split(WM_1D, Restructure_Indices)
        
        Water_Map = np.zeros(len(WM_Split), dtype = np.ndarray)
        
        for m in range(len(WM_Split)):
            Water_Map[m] = WM_Split[i].astype(int)
        
        #Water_Map = WM_Split.astype(int)
        
        Water_Map = Water_Map.T
        
        
        for k in range(i, l):
            WM = np.isin(Ref_Frame, Ensemble[k])
            WM = WM.astype(int)
            Water_Map[k -i] = WM
        
        Water_Map = Water_Map.T
        '''

    SP = SP/Counter

    Res_Time = simps(SP, T1)
    #print("\n")
    #print("Survival Probabilities: ")
    #print(SP)
    #print(Counter)
    #print("\n")
    #timef = time.time()
    #print("Time taken for calculating only SP is: ", timef - time0, " s")
    #if Args.multi_process == True:
     #   print("Time taken on average per chunk is: ", np.mean(chunk_times))
      #  print(chunk_times)

    #print("Average time taken to create Water Maps: ", np.mean(WM_t))
    #print("Average time taken to perform products: ", np.mean(Prod_t))
    print("Residence Time for your selected ensemble is: ", Res_Time)

    log_file.write("Residence time for the selected ensemble of water is: " + str(Res_Time) + " ps \n")
    log_file.write("\n")
    log_file.flush()
    

    Plot(T1, SP, Args, r"Time($ps$)", r"Survival Probabilities", r"Survival Probability Plot",
         "CICLOP_Residence_Time-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png", None)

    Out_Df = pnd.DataFrame({"Time (ps)": T1, "Survival Probabilities": SP})

    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df.to_csv("CICLOP_Residence_Time-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".csv")
            log_file.write("Data written to file" + "CICLOP_Residence_Time-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".csv \n")
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            Out_Df.to_csv(Args.to_csv)
            log_file.write("Data written to file" + Args.to_csv)
            log_file.write("\n")
            log_file.flush()


def rad_vol_multiprocess(args):
    u, Args, t, Results_Array, Results_X, p = args[0], args[1], args[2], args[3], args[4], args[5]

    Atom_Lists = Produce_Atom_Lists(u, t, Args.w, Args)
    All_Atoms = Atom_Lists[0]

    Grid = Create_grid_hydration(All_Atoms, Args.vx_dim, Atom_Lists[1])
    All_Atoms = Grid[0]
    Voxel_List = Grid[1]
    Grid_Dimensions = Grid[2]

    Inner_Surface_Info = Find_surface_inner_hydration(
        All_Atoms, Voxel_List, Grid_Dimensions, True, False)

    Rad_X = Inner_Surface_Info[2]
    Rad_Y = Inner_Surface_Info[3]

    Vol_X = Inner_Surface_Info[4]
    Vol_List = Inner_Surface_Info[5]

    if Args.rad_profile == True:
        # Results_Array[t - p] =
        # Results_X.append()
        # print(len(Rad_X))
        return (len(Rad_X), Rad_Y)

    elif Args.vol_profile == True:
        # Results_Array[t - p] = Vol_List
        # Results_X.append(len(Vol_X))
        # print(len(Vol_X))
        return (len(Vol_X), Vol_List)


def Rad_Vol_Errorbar(Args):
    u = mda.Universe(Args.s, Args.traj)

    dt = round(u.trajectory.dt, 3)
    T = list(range(int(Args.t0/dt), int(Args.tf/dt), int(Args.step/dt)))

    Results_Array = []
    Results_X = []
    
    if (Args.multi_process == True):
        # Process the universe into chunks of size frame_storage
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size

        start = 0
        Chunks = []

        for i in range(num_chunks):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))
        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()
        
        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)

        chunk_times = []
        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []
            # print("\n")
            # print("\n")
            # print("Processing chunk number: ", i)
            # print("\n")
            # print("\n")
            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

        
            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                Results = list(executor.map(rad_vol_multiprocess,
                                 [(Atom_Groups[j], Args, Results_Array, Results_X, j, T[0]) for j in range(len(Atom_Groups))]))
                
            for Result in Results:
                Results_Array.append(Result[1])
                Results_X.append(Result[0])
            gc.collect()
    else:
        for t in tqdm(T):
            u.trajectory[t]
            ag = u.select_atoms("all")
            Frame_Info = rad_vol_multiprocess((ag, Args, t, Results_Array, Results_X, T[0]))
            Results_Array.append(Frame_Info[1])
            Results_X.append(Frame_Info[0])
    
    '''
    if (Args.multi_process == True):
        chunk_size = 500
        num_chunks = len(T) // chunk_size + (len(T) % chunk_size > 0)
        counter = 0

        for i in tqdm(range(num_chunks)):
            start = i * chunk_size
            end = min((i+1) * chunk_size, len(T))
            chunk_T = T[start:end]

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                results = executor.map(rad_vol_multiprocess, [(
                    get_frame(u, j), Args, j, Results_Array, Results_X, T[0]) for j in chunk_T])

            for res in results:
                Results_Array[counter] = res[1]
                Results_X.append(res[0])
                counter += 1
            gc.collect()

    else:
        counter = 0
        for t in tqdm(T):
            u.trajectory[t]
            result = rad_vol_multiprocess(
                (get_frame(u, t), Args, t, Results_Array, Results_X, T[0]))
            Results_Array[counter] = result[1]
            Results_X.append(result[0])
            counter += 1
    '''
    min_z_dim = min(Results_X)

    print(min_z_dim)

    Reshaped_Res_Array = []

    for t in range(len(T)):
        ls = Results_Array[t]

        Reshaped_Res_Array.append(ls[:min_z_dim])

    print(len(Reshaped_Res_Array[0]), len(Reshaped_Res_Array[1]))

    Reshaped_Res_Array = np.array(Reshaped_Res_Array)

    print(Reshaped_Res_Array)

    Results_Avg = np.mean(Reshaped_Res_Array, axis=0)
    Results_Std = np.std(Reshaped_Res_Array, axis=0)

    Res_X = [i for i in range(0, 3*min_z_dim, 3)]

    print(len(Res_X))
    print(len(Results_Avg))


    if Args.rad_profile == True:
        Out_Df = pnd.DataFrame({"Distance Along Cavity Axis (Angstroms)" : Res_X, "Average Radius (Angstroms)" : Results_Avg, "Standard Deviation (Angstroms)" : Results_Std})
    elif Args.vol_profile == True:
        Out_Df = pnd.DataFrame({"Distance Along Cavity Axis (Angstroms)" : Res_X, "Average Volume (Cubic Angstroms)" : Results_Avg, "Standard Deviation (Cubic Angstroms)" : Results_Std})

    if Args.no_csv == False:
        if Args.to_csv == None:
            if Args.rad_profile == True:
                Out_Df.to_csv("CICLOP_Avg_Radius_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv")
                log_file.write("Data written to file" + "CICLOP_Avg_Radius_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
            elif Args.vol_profile == True:
                Out_Df.to_csv("CICLOP_Avg_Radius_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv")
                log_file.write("Data written to file" + "CICLOP_Avg_Volume_Profile-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
            
            log_file.write("\n")
            log_file.flush()
        elif Args.to_csv != None:
            Out_Df.to_csv(Args.to_csv)
            log_file.write("Data written to file" + Args.to_csv)
            log_file.write("\n")
            log_file.flush()

    if Args.rad_profile == True:
        Plot_Errorbar(Res_X, Results_Avg, Results_Std, Args, r"Distance Along Z Axis $\AA$", r"Radius $\AA$",
                      r"Average Radius Profile Plot", "CICLOP_Avg_Radius_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png")
    elif Args.vol_profile == True:
        Plot_Errorbar(Res_X, Results_Avg, Results_Std, Args, r"Distance Along Z Axis $\AA$", r"Volume $\AA^{3}$", r"Average Volume Profile Plot", "CICLOP_Avg_Radius_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png")

    return ()

def error_handler(Args):
    if((Args.t0 != None) ^ (Args.tf != None)):
        log_file.write("Either -t0 or -tf has not been provided.\n")
        log_file.write('\n')
        log_file.flush()
        raise Exception('Either -t0 or -tf has not been provided.')
    
    # if((Args.layer_start != None) ^ (Args.layer_end != None)):
    #     raise Exception('Either -ls or -le has not been provided. ')
    
    if((Args.layer_start != None) and (Args.layer_end != None)):
        if(Args.layer_start >= Args.layer_end):
            log_file.write('Value of -ls should be smaller then value of -le\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('Value of -ls should be smaller then value of -le')
    
    if((Args.t0 != None) and (Args.tf != None)):
        if(Args.t0 >= Args.tf):
            log_file.write('Value of -t0 should be smaller then value of -tf\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('Value of -t0 should be smaller then value of -tf')
        
    #redo
    '''
    if(Args.o != None):
        acceptable_output = ['pdb','psf']
        if(Args.o[-3:]):
            log_file.write('The output file format provided is not accepted. Accepted set of output formats are ' + ', '.join(acceptable_output) + "\n")
            log_file.write('\n')
            log_file.flush()
            raise Exception('The output file format provided is not accepted. Accepted set of output formats are ' + ', '.join(acceptable_output))
    '''
    if(Args.to_csv != None):
        #######################################################################################################################################
        #pandas.to_csv
        #['csv','xlsx','txt']
        if(Args.to_csv[-3:] != 'csv'):
            log_file.write('The acceptable input for this argumnet is a csv file. Eg: (File_name.csv)\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('The acceptable input for this argumnet is a csv file. Eg: (File_name.csv)')
    
    
    if(Args.s != None): 
        acceptable_structure_files = ['.pdb','.gro','.top', '.prmtop', '.parm7','.tpr']
        fileformat = os.path.splitext(Args.s)[1]
        if(fileformat not in acceptable_structure_files):
            log_file.write('The acceptable formats for the structure file to be provided in -s are, '+ ', '.join(acceptable_structure_files)+'\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('The acceptable formats for the structure file format to be provided in -s are, '+ ', '.join(acceptable_structure_files))

        if(Args.traj == None and Args.charge_plot == False):
            log_file.write("With argument -s, -traj is also required (if you don't have a trajectory use -f instead)\n")
            log_file.write('\n')
            log_file.flush()
            raise Exception("With argument -s, -traj is also required (if you don't have a trajectory use -f instead)")
        
    if(Args.f != None): 
        acceptable_structure_files = ['.pdb','.gro','.top', '.prmtop', '.parm7','.tpr']
        fileformat = os.path.splitext(Args.f)[1]
        if(fileformat not in acceptable_structure_files):
            log_file.write('The acceptable structure file format in -f are, '+ str(acceptable_structure_files)+'\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('The acceptable structure file format in -f are, '+ str(acceptable_structure_files))
    
    if(Args.traj != None):
        acceptable_trajectories = ['xtc','trj']
        fileformat = Args.traj.split('.')[-1]
        if(fileformat not in acceptable_trajectories):
            log_file.write('The acceptable trajectory file format in -traj are, '+ ', '.join(acceptable_trajectories) +'\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('The acceptable trajectory file format in -traj are, '+ ', '.join(acceptable_trajectories))
    if(Args.step != None):
        if(Args.step == 0):
            log_file.write('-step can not be zero.\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('-step can not be zero')
    
        
def Z_Dens_Profile(u, Args):
    Info = Find_Cavity_Water(u, 0, Args, False)
    
    Grid = Info[5]
    Grid_Dimensions = Info[6]
    
    Z_Dens = []
    Z = []
    print(Grid_Dimensions)
    
    for z in tqdm(range(0, Grid_Dimensions[2])):
        Z_Voxels = 0
        Z_Water = 0
        for x in range(0, Grid_Dimensions[0]):
            for y in range(0, Grid_Dimensions[1]):
                vox = Grid[x][y][z]
                if vox.Inner_Cavity == True:
                    Z_Voxels += 1
                    Z_Water += len(vox.OW_atoms)
        if Z_Voxels != 0:
            dens = (Z_Water*29.9157)/(Z_Voxels)
            
            Z_Dens.append(dens)
            Z.append(z)
        
    Plot(Z, Z_Dens, Args, r"Distance Along Protein Axis ($\AA$)", r"Water Density ($g\:cm^{-3}$)", r"Water density profile along protein axis", "CICLOP_Z_Dens_Profile-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png", None)


"""
def errorbar_multiprocess(args):
    u, i, Args, find_cavity_water = args[0], args[1], args[2], args[3]
    # if Args.outer_surface != True:
    Frame_Info = Find_Cavity_Water(u, i, Args, find_cavity_water)
    # else:
    #     Frame_Info = Find_Outer_Water(u,i, Args)

    Inner_Empty_Voxels = Frame_Info[9]
    Layered_Voxels = Frame_Info[5]
    Layer = Frame_Info[6]

    Frame_Water_Density = []

    for k in range(1, Layer+1):
        n_layer_voxels = 0
        n_layer_water = 0
        for box in Layered_Voxels:
            if box.layer == k:
                n_layer_water += len(box.OW_atoms)
                n_layer_voxels += 1

        Layer_Dens = (n_layer_water/(n_layer_voxels *
                      (Args.vx_dim**3)))*(29.9157)
        Frame_Water_Density.append(Layer_Dens)

    return Layer, Frame_Water_Density
"""


def Axis_Dens_Multiprocess(args):
    u, i, Args, find_cavity_water = args[0], args[1], args[2], args[3]
    
    Frame_Info = Find_Cavity_Water(u, i, Args, find_cavity_water)
    
    Grid = Frame_Info[5]
    Grid_Dimensions = Frame_Info[6]
    
    Z_Dens = []
    Z = []
    print(Grid_Dimensions)
    
    for z in tqdm(range(0, Grid_Dimensions[2], 3)):
        Z_Voxels = 0
        Z_Water = 0
        for x in range(0, Grid_Dimensions[0]):
            for y in range(0, Grid_Dimensions[1]):
                vox = Grid[x][y][z]
                if vox.Inner_Cavity == True:
                    if len(vox.OW_atoms) != 0:
                        Z_Voxels += 1
                        Z_Water += len(vox.OW_atoms)
        
        if Z_Voxels != 0:
            dens = (Z_Water*29.9157)/(Z_Voxels)
            
            Z_Dens.append(dens)
            Z.append(z)
    
    return(len(Z), Z_Dens)
    

def Axis_Dens_Errorbar(u, Args):
    log_file.write("Finding time averaged water density profile along protein axis\n")
    log_file.write("\n")
    log_file.flush()
    
    # Creating DataFrame for making the errorbar plot
    Z_Max = []
    Water_Density_Array = []

    dt = round(u.trajectory.dt, 3)
    st = 0
    en = len(u.trajectory)
    step = 1

    if (Args.t0 != None):
        st = int(Args.t0/dt)
    if (Args.tf != None):
        en = int(Args.tf/dt)
    if (Args.step != None):
        step = int(Args.step/dt)
    
    log_file.write("Analysis done between " + str(Args.t0) + "ps and " + str(Args.tf) + "ps with a time-step of " + str(Args.step) + "ps \n")
    log_file.write("\n")
    log_file.flush()
    
    T = list(i for i in range(st, en, step))

    results = []
    
    
    if (Args.multi_process == True):
        # Process the universe into chunks of size frame_storage
        chunk_size = Args.frame_storage
        num_chunks = len(T)//chunk_size
        remainder = len(T) % chunk_size

        start = 0
        Chunks = []

        for i in range(num_chunks):
            Chunks.append(np.array(T[start: start + chunk_size]))
            start += chunk_size

        if remainder != 0:
            Chunks.append(np.array(T[start:]))
        
        log_file.write("Number of nodes used for multiprocessing are: " + str(Args.frame_storage + 1) + "\n")
        log_file.write("Number of chunks to be processed are: " + str(num_chunks))
        log_file.write("\n")
        log_file.flush()
        
        print("Number of chunks of size " + str(Args.frame_storage) +
              " to be processed are: ", num_chunks)

        chunk_times = []
        for i in tqdm(range(len(Chunks))):
            time_c1 = time.time()
            chunk = Chunks[i]
            Atom_Groups = []
            # print("\n")
            # print("\n")
            # print("Processing chunk number: ", i)
            # print("\n")
            # print("\n")
            for t in chunk:
                u_new = u.copy()
                u_new.trajectory[t]
                ag = u_new.select_atoms("all")
                Atom_Groups.append(ag)

            with concurrent.futures.ProcessPoolExecutor(max_workers=Args.frame_storage) as executor:
                results += list(executor.map(Axis_Dens_Multiprocess,
                                 [(Atom_Groups[j], j, Args, False) for j in range(len(Atom_Groups))]))
                gc.collect()
    else:
        for i in tqdm(T):
            results.append(Axis_Dens_Multiprocess((u, i, Args, False)))
    
    for result in results:
        for res in results:
            Z_Max.append(res[0])
            Water_Density_Array.append(res[1])

    Z_Dens_Array = []    

    Z_min = np.min(Z_Max)    

    for List in Water_Density_Array:
        Dummy = List[:Z_min]
        Z_Dens_Array.append(Dummy)
    
    Z_Dens_Array = np.array(Z_Dens_Array)
    
    Z_Dens_Avg = np.mean(Z_Dens_Array, axis = 0)
    Z_Dens_Std = np.std(Z_Dens_Array, axis = 0)
    
    Z = [i for i in range(0, 3*Z_min, 3)]
    
    Out_Df = pnd.DataFrame({"Distance Along Cavity Axis (Angstroms)" : Z, "Average Water Density (grams per cubic centimeters)" : Z_Dens_Avg, "Standard Deviation (grams per cubic centimeters)" : Z_Dens_Std})
    
    if Args.no_csv == False:
        if Args.to_csv == None:
            Out_Df.to_csv("CICLOP_Avg_Axis_Density-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".csv")
        elif Args.to_csv != None:
            Out_Df.to_csv(Args.to_csv)
    
    Plot_Errorbar(Z, Z_Dens_Avg, Z_Dens_Std, Args, r"Distance Along Z Axis $\AA$", r"Average Water Density $(g\:cm^{-3})$", r"Water Density Profile Along Protein Axis", "CICLOP_Avg_Axis_Density-" + datetime.now().strftime("%Y-%m-%d") + time.strftime("%H-%M-%S") + ".png")

def Water_Layer_Density_Check(u, Args):
    Info = Find_Cavity_Water(u,0, Args, True)
    
    Cavity_Water = Info[0]
    Layers = Info[6]
    Atom_List = Info[3]
    Grid = Info[7]
    Grid_Dimensions = Info[8]
    
    Inner_Surface_Atoms = {}
    
    L = list(range(1, Layers + 1))
    Layer_Dist = np.array(Layers, dtype = float)
    Counter = np.array(Layers, dtype = int)
    
    print(Layers, len(Layer_Dist), len(Counter))
    
    for atom in Atom_List.keys():
        if Atom_List[atom][1] == True:
            Inner_Surface_Atoms[atom] = Atom_List[atom]
    
    for z in tqdm(range(0, Grid_Dimensions[2])):
        Z_Atoms = {}
        Z_O = {}
        for atom in Inner_Surface_Atoms.keys():
            if int(atom.position[2]) == z:
                Z_Atoms[atom] = Inner_Surface_Atoms[atom]
        for O in Cavity_Water.keys():
            if int(O.position[2]) == z:
                Z_O[atom] = Cavity_Water[O]
        
        if len(Z_Atoms.keys()) != 0:
            Z_IS = mda.core.groups.AtomGroup(list(Z_Atoms.keys()))
            
            Cav_Pos = Z_IS.positions
            
            for O in Z_O.keys():
                for l in range(1, Layers + 1):
                    if Z_O[O][2] == l:
                        Disp = Cav_Pos - O.position
                        Disp_sq = np.square(Disp)
                        rsq = np.sum(Disp_sq, axis = 1)
                        Dist = np.sqrt(rsq)
                        
                        Dist_Min = np.min(Dist)
                        
                        Layer_Dist[l - 1] += Dist_Min
                        Counter[l - 1] += 1
        
    
    Avg_Dist = Layer_Dist/Counter
    
    #print(len(L), len(Avg_Dist))
    print(Counter)
    print(Avg_Dist)
    Out = pnd.DataFrame({"Layer" : L ,"Average Distance" : Avg_Dist})
    Out.to_csv("3los_Avg_Water_Dist.csv")
    
    plt.figure(num = 0, dpi = 500)
    plt.plot(L, Avg_Dist)
    plt.xlabel("Layer Number")
    plt.ylabel("Average Distance from Cavity Wall")
    plt.savefig("3los_Avg_Water_Dist.png", dpi = 500)
    plt.show()
        


#Execution Function
def Execute(Args):
    # Call error handler over here.
    error_handler(Args)
    
    t0 = time.time()

    if Args.traj != None and Args.residence_time != True:
        u1 = mda.Universe(Args.s, Args.traj)
        dt = u1.trajectory.dt
        
        
        if Args.step < dt:
            log_file.write('The input value of -step is lower than the saving frequency of the given trajectory in picoseconds. Perhaps you have not provided the time-steps in picoseconds?\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('The input value of -step is lower than the saving frequency of the given trajectory in picoseconds. Perhaps you have not provided the time-steps in picoseconds?')
        
        traj_len = u1.trajectory[-1].time
        
        if Args.tf > traj_len:
            log_file.write('The input value given in -tf exceeds the final trajectory time step in picoseconds. Perhaps you have not provided the end time in picoseconds?\n')
            log_file.write('\n')
            log_file.flush()
            raise Exception('The input value given in -tf exceeds the final trajectory time step in picoseconds. Perhaps you have not provided the end time in picoseconds?')
        
        
    elif Args.f != None:
        u2 = mda.Universe(Args.f)
        
    if Args.f != None and Args.o != None and Args.layer == False and Args.layer_range == False and  Args.charge_plot == False and Args.density_plot == False and Args.rad_profile == False and Args.vol_profile == False and Args.axis_dens == False and Args.dist_check == False:
        #print("Doing dat shit")
        Out = open(Args.o, 'w')
        
        Info = Find_Cavity_Water(u2, 0, Args, False)

        Cavity_Water = Info[0]
        Atom_List = Info[3]

        if Args.w == 'no_water':
            print(r"Select Group for Output\n")
            print(r"0: For Inner Surface Atoms")
            print(r"1: For Protein and Inner Surface Atoms")
            
            
            
            Group = int(input(r"Select Group: "))
            
            log_file.write(r"Select Group for Output\n")
            log_file.write(r"0: For Inner Surface Atomsn")
            log_file.write(r"1: For Protein and Inner Surface Atoms\n")
            log_file.write(r"Group selected for output is: " + str(Group) + "\n")
            log_file.write("\n")
            log_file.flush()
            
            
            if Group == 0:
                Inner_Surface_Atoms = {}
                for Atom in Atom_List:
                    if Atom_List[Atom][1] == True:
                        Inner_Surface_Atoms[Atom] = Atom_List[Atom]

                get_pdb_line(Inner_Surface_Atoms, None, Args)
                print("Writing out identified inner surface residues to " + Args.o)
                
                log_file.write("Writing out identified inner surface residues to " + Args.o + "\n")
                log_file.write("\n")
                log_file.flush()
                
            elif Group == 1:
                get_pdb_line(Atom_List, None, Args)
                print(
                    "Writing out protein and identified inner surface residues to " + Args.o)
                log_file.write("Writing out protein and identified inner surface residues to " + Args.o + "\n")
                log_file.write("\n")
                log_file.flush()
                
                
            else:
                print("Selected group " + str(Group) +
                      "is not available for output.")
                log_file.write("Selected group " + str(Group) +
                      "is not available for output\n")
                log_file.wwrite("\n")
                log_file.flushh()
                sys.exit()
            Out.close()
        else:
            print(r"Select Group for Output\n")
            print(r"0: For Inner Surface Atoms")
            print(r"1: For Protein and Inner Surface Atoms")
            print(r"2: For Cavity Water and Inner Surface Atoms")
            print(r"3: For Protein and Cavity Water")
            print(r"4: For Cavity Water\n")

            Group = int(input(r"Select Group: "))

            log_file.write(r"Select Group for Output\n")
            log_file.write(r"0: For Inner Surface Atoms\n")
            log_file.write(r"1: For Protein and Inner Surface Atoms\n")
            log_file.write(r"2: For Cavity Water and Inner Surface Atoms\n")
            log_file.write(r"3: For Protein and Cavity Water\n")
            log_file.write(r"4: For Cavity Water\n")
            log_file.write("Group selected for output is: " + str(Group) + "\n")
            log_file.write("\n")
            log_file.flush()


            if Group == 0:
                Inner_Surface_Atoms = {}
                for Atom in Atom_List:
                    if Atom_List[Atom][1] == True:
                        Inner_Surface_Atoms[Atom] = Atom_List[Atom]

                get_pdb_line(Inner_Surface_Atoms, None, Args)
                print("Writing out identified inner surface residues to " + Args.o)
                log_file.write("Writing out identified inner surface residues to " + Args.o + "\n")
                log_file.write()
                log_file.flush()
            elif Group == 1:
                get_pdb_line(Atom_List, None, Args)

                print(
                    "Writing out protein and identified inner surface residues to " + Args.o)
                log_file.write("Writing out protein and identified inner surface residues to " + Args.o + "\n")
                log_file.write()
                log_file.flush()
            elif Group == 2:
                Inner_Surface_Atoms = {}
                for Atom in Atom_List:
                    if Atom_List[Atom][1] == True:
                        Inner_Surface_Atoms[Atom] = Atom_List[Atom]

                get_pdb_line(Inner_Surface_Atoms, Cavity_Water, Args)
                
                print(
                    "Writing out identified inner surface residues and cavity water to " + Args.o)
                log_file.write("Writing out identified inner surface residues and cavity water to " + Args.o + "\n")
                log_file.write("\n")
                log_file.flush()
            elif Group == 3:
                get_pdb_line(Atom_List, Cavity_Water, Args)

                print(
                    "Writing out protein atoms and identified cavity water to " + Args.o)
                log_file.write("Writing out protein atoms and identified cavity water to " + Args.o + "\n")
                log_file.write("\n")
                log_file.flush()
            elif Group == 4:
                get_pdb_line(None, Cavity_Water, Args)
                print("Writing out identified cavity water molecules to " + Args.o)
                log_file.write("Writing out identified cavity water molecules to " + Args.o + "\n")
                log_file.write()
                log_file.flush()
            else:
                print("Selected group " + str(Group) +
                      "is not available for output.")
                log_file.write("Selected group " + str(Group) +
                      "is not available for output\n")
                log_file.write("\n")
                log_file.flush()
                sys.exit()
            Out.close()

    elif Args.f != None and Args.o != None and Args.layer == True:
        Out = open(Args.o, 'w')

        Info = Find_Cavity_Water(u2, 0, Args, True)
        Cavity_Water = Info[0]
        Atom_List = Info[3]
        O_Atoms = Info[4]
        Layer = Info[6]

        print("The number of layers in the cavity is: ", Layer)
        Select = False
        while Select == False:
            l = int(input("Enter your desired layer value for output: "))
            log_file.write("Selected layer of water at a distance of " + str(l) + " Angstroms from the cavity wall\n")
            log_file.write("\n")
            log_file.flush()
            if l > Layer:
                print("The layer value given as input(" + str(l) +
                      ") exceeds the maximum layer value of "+str(Layer))
                log_file.write("The layer value given as input(" + str(l) +
                      ") exceeds the maximum layer value of "+str(Layer) + "\n")
                log_file.write("\n")
                log_file.flush()
                
            else:
                Select = True
                
        print(r"Select Group for Output\n")
        print(r"0: For Water in Layer " + str(l))
        print(r"1: For Identified Inner Surface Residues and Water in Layer " + str(l))
        print(r"2: For Protein and Water in Layer " + str(l) + "\n")

        Group = int(input(r"Select Group: "))
        
        log_file.write(r"Select Group for Output\n")
        log_file.write(r"0: For Water in Layer " + str(l) + "\n")
        log_file.write(r"1: For Identified Inner Surface Residues and Water in Layer " + str(l) + "\n")
        log_file.write(r"2: For Protein and Water in Layer " + str(l) + "\n")
        log_file.write(r"Group selected for output is: " + str(Group) + "\n")
        log_file.write("\n")
        log_file.flush()
        
        if Group == 0:
            Layer_Water = {}
            for O in Cavity_Water:
                if O_Atoms[O][2] == l:
                    Layer_Water[O] = O_Atoms[O]

            get_pdb_line(None, Layer_Water, Args)
            print("Writing out water in layer " + str(l) + " to " + Args.o)
            log_file.write("Writing out water in layer " + str(l) + " to " + Args.o + "\n")
            log_file.write("\n")
            log_file.flush()

        elif Group == 1:
            Layer_Water = {}
            for O in Cavity_Water:
                if O_Atoms[O][2] == l:
                    Layer_Water[O] = O_Atoms[O]
            Inner_Surface_Atoms = {}
            for Atom in Atom_List:
                if Atom_List[Atom][1] == True:
                    Inner_Surface_Atoms[Atom] = Atom_List[Atom]

            get_pdb_line(Inner_Surface_Atoms, Layer_Water, Args)

            print("Writing out identified inner residues and water in layer " +
                  str(l) + " to " + Args.o)
            
            log_file.write("Writing out identified inner residues and water in layer "  + str(l) + " to " + Args.o + "\n")
            log_file.write("\n")
            log_file.flush()

        elif Group == 2:
            Layer_Water = {}
            for O in Cavity_Water:
                if O_Atoms[O][2] == l:
                    Layer_Water[O] = O_Atoms[O]

            get_pdb_line(Atom_List, Layer_Water, Args)

            print("Writing out protein and water in layer " +
                  str(l) + " to " + Args.o)
            
            log_file.write("Writing out protein and water in layer " + str(l) + " to " + Args.o + "\n")
            log_file.write("\n")
            log_file.flush()

        else:
            print("Selected group " + str(Group) +
                  "is not available for output.")
            log_file.write("Selected group " + str(Group) + " is not available for output\n")
            log_file.write("\n")
            log_file.flush()
            sys.exit()

        Out.close()

    elif Args.f != None and Args.o != None and Args.layer_range == True:
        Out = open(Args.o, 'w')

        Info = Find_Cavity_Water(u2, 0, Args, True)
        Cavity_Water = Info[0]
        Atom_List = Info[3]
        O_Atoms = Info[4]
        Layer = Info[6]

        print("The number of layers in the cavity is: ", Layer)
        
        if Args.layer_end == None and Args.layer_start == None:
            Select = False

            while Select == False:
                ls = int(input("Enter the lower bound for the range of layers: "))
                le = int(input("Enter the upper bound for the range of layers: "))

                if ls > Layer:
                    print("The lower bound for the range of layers given(" +
                          str(ls) + " exceeds the maximum number of layers " + str(Layer))
                elif le > Layer:
                    print("The upper bound for the range of layers given(" +
                          str(ls) + " exceeds the maximum number of layers " + str(Layer))
                elif ls > le:
                    print("The lower bound for the range of layers(" + str(ls) +
                          " cannot exceed the upper bound(" + str(le) + ") given as input")
                else:
                    Select = True

        else:
            ls = Args.layer_start
            le = Args.layer_end

        print("Selecting Water molecules between layers " +
              str(ls) + " and " + str(le))
        
        log_file.write("Selecting water molecules between " + str(ls) + " Angstroms and " + str(le) + " Angstroms from the cavity wall for output \n")
        log_file.write("\n")
        log_file.flush()
        
        
        print(r"Select Group for Output\n")
        print(r"0: For Water in between Layers " + str(ls) + " and " + str(le))
        print(r"1: For Identified Inner Surface Residues and Water in between Layers " +
              str(ls) + " and " + str(le))
        print(r"2: For Protein and Water in between layers " +
              str(ls) + " and " + str(le) + "\n")

        Group = int(input(r"Select Group: "))
        
        log_file.write(r"Select Group for Output\n")
        log_file.write(r"0: For Water in between Layers " + str(ls) + " and " + str(le) + "\n")
        log_file.write(r"1: For Identified Inner Surface Residues and Water in between Layers " +
              str(ls) + " and " + str(le) + "\n")
        log_file.write(r"2: For Protein and Water in between layers " +
              str(ls) + " and " + str(le) + "\n")
        log_file.write(r"Group selected for output is: " + str(Group))
        log_file.write("\n")
        log_file.flush()
        
        
        if Group == 0:
            Layer_Water = {}
            for O in Cavity_Water:
                if ls <= O_Atoms[O][2] <= le:
                    Layer_Water[O] = O_Atoms[O]
            print("Number of water molecules between " + str(Args.layer_start) + " and " +
                  str(Args.layer_end) + " Angstroms are: ", len(list(Layer_Water.keys())))
            get_pdb_line(None, Layer_Water, Args)
            print("Writing out water in between layers " +
                  str(ls) + " and " + str(le) + " to " + Args.o)

            log_file.write("Number of water molecules between " + str(Args.layer_start) + " and " +
                  str(Args.layer_end) + " Angstroms are: " + str(len(list(Layer_Water.keys()))) + "\n")
            log_file.write("Writing out water in between layers " +
                  str(ls) + " and " + str(le) + " to " + Args.o + "\n")
            log_file.write("\n")
            log_file.flush()
        
        elif Group == 1:
            Layer_Water = {}
            for O in Cavity_Water:
                if ls <= O_Atoms[O][2] <= le:
                    Layer_Water[O] = O_Atoms[O]
            Inner_Surface_Atoms = {}
            for Atom in Atom_List:
                if Atom_List[Atom][1] == True:
                    Inner_Surface_Atoms[Atom] = Atom_List[Atom]

            get_pdb_line(Inner_Surface_Atoms, Layer_Water, Args)
            print("Number of water molecules between " + str(Args.layer_start) + " and " +
                  str(Args.layer_end) + " Angstroms are: ", len(list(Layer_Water.keys())))
            print("Writing out identified inner residues and water in between layers " +
                  str(ls) + " and " + str(le) + " to " + Args.o)
            
            log_file.write("Number of water molecules between " + str(Args.layer_start) + " and " +
                  str(Args.layer_end) + " Angstroms are: " + str(len(list(Layer_Water.keys()))) + "\n")
            log_file.write("Writing out identified inner residues and water in between layers " +
                  str(ls) + " and " + str(le) + " to " + Args.o + "\n")
            log_file.write("\n")
            log_file.flush()
        
        elif Group == 2:
            Layer_Water = {}
            for O in Cavity_Water:
                if ls <= O_Atoms[O][2] <= le:
                    Layer_Water[O] = O_Atoms[O]

            get_pdb_line(Atom_List, Layer_Water, Args)
            print("Number of water molecules between " + str(Args.layer_start) + " and " +
                  str(Args.layer_end) + " Angstroms are: ", len(list(Layer_Water.keys())))
            print("Writing out protein and water in between layers " +
                  str(ls) + " and " + str(le) + " to " + Args.o)
            
            log_file.write("Number of water molecules between " + str(Args.layer_start) + " and " +
                  str(Args.layer_end) + " Angstroms are: " + str(len(list(Layer_Water.keys()))) + "\n")
            log_file.write("Writing out protein and water in between layers " +
                  str(ls) + " and " + str(le) + " to " + Args.o + "\n")
            log_file.write("\n")
            log_file.flush()
            
            
        else:
            print("Selected group " + str(Group) +
                  "is not available for output.")
            log_file.write("Selected group " + str(Group) +
                  "is not available for output.")
            log_file.write("\n")
            log_file.flush()
            sys.exit()

        Out.close()
        

    elif Args.f != None and Args.rad_profile == True:
        #Args, u, t, Rad_True, Vol_True = args[0], args[1], args[2], args[3], args[4]
        Inp = [args, u2, 0, True, False]
        
        Radius_or_Volume_Profile(Inp)

    elif Args.f != None and Args.vol_profile == True:
        Inp = [args, u2, 0, False, True]
        Radius_or_Volume_Profile(Inp)

    elif Args.f != None and Args.density_plot == True :
        Dens_Plot(u2, Args)
   
    elif Args.traj != None and Args.s != None and Args.density_plot == True:
        Errorbar(u1, Args)
        
    elif Args.traj != None and Args.s != None and Args.residence_time == True:
        Calculate_Residence_Times_New(Args)
        #tf = time.time()

    elif Args.traj != None and Args.s != None and Args.cavity_diffusion == True:
        # Selecting the ensembles to pass through the function

        # T, MSD_Avg, D, c
        time_period = round(Args.tf - Args.t0, 3)
        
        if time_period < 1.0:
            log_file.write('WARNING!\n')
            log_file.write('The diffusion coefficient of water is being calculated for a time less than 1 picosecond. The diffusion constant hence obtained is most likely from the ballistic regime and is likely to overestimate the value of the diffusion coefficient.\n')
            log_file.write('\n')
            log_file.flush()
            raise Warning('The diffusion coefficient of water is being calculated for a time less than 1 picosecond. The diffusion constant hence obtained is most likely from the ballistic regime and is likely to overestimate the value of the diffusion coefficient.')
        
        
        Diff_Info = Find_Diffusion_Coefficient(u1, Args)

        T = Diff_Info[0]
        MSD_Avg = Diff_Info[1]
        D = Diff_Info[2]
        c = Diff_Info[3]

        print("Diffusion coefficient of cavity water in Angstroms Squared per Picosecond is: ", D)
        Plt_X = [T]
        Plt_Y = [MSD_Avg]
        
        Plot(Plt_X, Plt_Y, Args, r"Time(ps)", r"$\left<r^{2}\right>\left(\AA^{2}\right)$", 'Water Diffusion Profile', "CICLOP_Diffusion-" + datetime.now().strftime("%Y-%m-%d")+"-" + time.strftime("%H-%M-%S") + ".png", None)
        '''   
        if Args.bulk == True:
            Bulk_Water = Info[-1]
            Bulk_Ensemble = mda.core.groups.AtomGroup(Bulk_Water)
            Bulk_Diff_Info = Find_Diffusion_Coefficient(Bulk_Ensemble, Args)
                
            MSD_Avg_Bulk = Bulk_Diff_Info[1]
            D_bulk = Bulk_Diff_Info[2]
            c_bulk = Bulk_Diff_Info[3]
                
            print("Diffusion coefficient of bulk water in Angstroms Squared per Picosecond is: ", D_bulk)
                
            Plt_X.append(T)
            Plt_X.append(T1)
                
            Lsq_Fit_Bulk = [D_bulk*6*x + c_bulk for x in T1]
                
            Plt_Y.append(MSD_Avg_Bulk)
            Plt_Y.append(Lsq_Fit_Bulk)
                
            Legend.append('Bulk Water')
            Legend.append('LSQ Fit for Bulk Water')
            
        '''

        # print(np.array(MSD_Avg))

    elif Args.traj != None and Args.s != None and Args.relaxation_plot == True:
        
        dt = round(u1.trajectory.dt, 3)
        T0 = list(range(int(Args.t0/dt), int(Args.tf/dt), int(Args.step/dt)))
        
        u1.trajectory[T0[0]]
        if Args.all_water == True:
            Info = Find_Cavity_Water(u1, 0, Args, False)
            Cavity_Water = Info[0]
            OW_Indices = list(Cavity_Water.keys())

        elif Args.layer_start != None and Args.layer_end != None:
            Info = Find_Cavity_Water(u1, 0, Args, True)
            Cavity_Water = Info[0]
            OW_Indices = []

            for O in Cavity_Water.keys():
                if Args.layer_start <= Cavity_Water[O][2] <= Args.layer_end:
                    OW_Indices.append(O)

        Ensemble = mda.core.groups.AtomGroup(OW_Indices)

        Rel_Info = Find_Relaxation_Times_Final(Ensemble, u1, Args)

        T = Rel_Info[0]
        t1 = Rel_Info[1]
        t2 = Rel_Info[2]
        FO_ACF = Rel_Info[3]
        SO_ACF = Rel_Info[4]

        print('First Order Relaxation time for the Selected Ensemble is: ' + str(t1) + 'ps')

        Plt_X = [T]
        Plt_Y = [FO_ACF]

        Legend = ['First Order Rotational ACF']

    
        if Args.second_order == True:
            Plt_X.append(T)
            Plt_Y.append(SO_ACF)
            Legend.append('Second Order Rotational ACF')

            print(
                'Second Order Relaxation time for the Selected Ensemble is: ' + str(t2) + 'ps')
        
        Plot(Plt_X, Plt_Y, Args, r"Time (ps)", r"$\left<C_{Rot}\right>$", "Water Reorientational Dynamics", "CICLOP_Rotational_Relaxation_Times-" + datetime.now().strftime("%Y-%m-%d") +"-" + time.strftime("%H-%M-%S") + ".png", Legend)
        
        Out_Df = pnd.DataFrame(
            {"Time(ps)": T, "First Order Relaxation": FO_ACF, "Second Order Relaxation": SO_ACF})

        if Args.no_csv == False:
            if Args.to_csv == None:
                Out_Df.to_csv("CICLOP_Rotational_Relaxation_Times-" + datetime.now().strftime("%Y-%m-%d") +"-" + time.strftime("%H-%M-%S") + ".csv")
                log_file.write("Data written to file" + "CICLOP_Rotational_Relaxation_Times-" + datetime.now().strftime("%Y-%m-%d") +"-"+ time.strftime("%H-%M-%S") + ".csv \n")
                log_file.write("\n")
                log_file.flush()
            elif Args.to_csv != None:
                Out_Df.to_csv(Args.to_csv)
                log_file.write("Data written to file" + Args.to_csv)
                log_file.write("\n")
                log_file.flush()

        # print(np.array(MSD_Avg))

    elif Args.traj != None and Args.s != None and Args.rad_profile == True:
        Rad_Vol_Errorbar(Args)

    elif Args.traj != None and Args.s != None and Args.vol_profile == True:
        Rad_Vol_Errorbar(Args)

    elif Args.s != None and Args.f != None and Args.charge_plot != None:
        Charge_Profile(Args)
        
    elif Args.s != None and Args.traj != None and Args.charge_plot == True:
        Charge_Profile_Errorbar(u1, Args)

    elif Args.traj != None and Args.s != None and Args.water_prop == True:
        Water_Trajectory(u1, Args)

    elif Args.f != None and Args.axis_dens == True:
        Z_Dens_Profile(u2, Args)
    
    elif Args.traj != None and Args.axis_dens == True and Args.s != None:
        Axis_Dens_Errorbar(u1, Args)
    
    elif Args.f != None and Args.dist_check == True:
        Water_Layer_Density_Check(u2, Args)
    
    tf = time.time()
    
    
    print("Time taken is: ", tf - t0, "s")


Execute(args)
