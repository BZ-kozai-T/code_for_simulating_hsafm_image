#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to generate theoritical HS-AFM images from a trajectory file (.dcd or .xtc file) of a Brownian dynamics simulation of the NPC (Kim et al., 2018).
This needs a topology file (.pdb file)
All parameters required to calculate the distance of the FG Nup beads are in .yaml file.
"""

# %%
import argparse
import yaml
import mdtraj as md
import numpy as np
import tifffile

# %%
parser = argparse.ArgumentParser(description="Run calculation of HS-AFM image.")
# expect yaml parameter file
parser.add_argument('-p',
                    dest='prm',
                    required=True,
                    type=argparse.FileType('r'),
                    help="provide a parameter file in yaml format.")

args = parser.parse_args()

# %%
imgp = yaml.safe_load(args.prm)

# with open('hs-afm-analysis-parameters-tk.yaml') as fh:
#     imgp = yaml.safe_load(fh)

print(imgp)

# %%
traj_file = imgp['filenames']['data_dir'] + imgp['filenames']['dcd_file']   # read trajectory file
top_file = imgp['filenames']['data_dir'] + imgp['filenames']['pdb_file']    # read topology file

# %%
traj = md.load(traj_file, top=top_file)

# %%
print('Starting HS-AFM calculation:')
print(f'CGMD nr frames: {len(traj)}')

prms = imgp['hs-afm']
Z_botlim = prms['z_m']  # Z height of the imaging plane
Z_toplim = prms['z_lim'] # Z upper limit
N_xpxl = prms['nxpxl'] # number of pixels in y direction
N_ypxl = prms['nypxl'] # number of pixels in x direction
N_fpp = prms['nfpp']  # frames per pixel in the HS-AFM calculation
L_pxl_size = prms['pxl_size'] # pixel size
R_tip = prms['Rtip'] # tip radius
origin = (prms['x_img_origin'], prms['y_img_origin'])  # coordinates of the imaging plane

_coordx = np.linspace(origin[0] + L_pxl_size / 2,
                      origin[0] + N_xpxl * L_pxl_size - L_pxl_size / 2,
                      N_xpxl)

_coordy = np.linspace(origin[1] - N_ypxl * L_pxl_size + L_pxl_size / 2,
                      origin[1] - L_pxl_size / 2,
                      N_ypxl)

print('x grid: ', _coordx)
print('y grid: ', _coordy)

# end of the inputs

N_cgmd_frames_per_hsafm_frame = N_xpxl * N_ypxl * N_fpp

if N_cgmd_frames_per_hsafm_frame > traj.n_frames:
    raise Exception(
        f'The simulation requires {N_cgmd_frames_per_hsafm_frame} but the are only {traj.n_frames} frames in the MD trajectory')

# %%

N_hsafm_frames = traj.n_frames // N_cgmd_frames_per_hsafm_frame

hsafm_images = np.zeros((N_hsafm_frames, N_xpxl, N_ypxl), dtype=np.float32)

for i_frame in range(N_hsafm_frames):
    hsafm_frame = np.zeros((N_xpxl, N_ypxl), dtype=np.float32)

    for iy in range(N_ypxl):
        y = _coordy[iy]

        for ix in range(N_xpxl):
            x = _coordx[ix]
            #
            before = N_fpp * ix + N_fpp * N_xpxl * iy + N_fpp * N_xpxl * N_ypxl * i_frame
            # print(i_frame, ix, iy, f'{before}:{before+N_fpp}')
            sub_traj = 10 * traj.xyz[before:before + N_fpp]  # relevant_frames

            # iterate the MD frames
            for mdframe in sub_traj:
                # select by Z coords
                z_mask = np.bitwise_and(mdframe[:, 2] > Z_botlim, mdframe[:, 2] < Z_toplim)
                z_mask_indxs = [i for i in range(z_mask.shape[0]) if z_mask[i]]
                sele_z = mdframe[z_mask]

                # calculate the distance to the pixel
                r2 = (sele_z[:, 0] - x) ** 2 + (sele_z[:, 1] - y) ** 2

                # print(sele_z[:, 0].min(), sele_z[:, 0].max())
                r_mask = r2 < R_tip ** 2
                r_mask_indxs = [z_mask_indxs[i] + 1 for i in range(r_mask.shape[0]) if r_mask[i]]
                sele_xyz = sele_z[r_mask]

                hsafm_frame[ix, iy] += np.sum(sele_xyz[:, 2])
                frame_nr = before + 1

                # print command lines for UCSF ChimeraX to save a snapshot of FG beads that contribute to the estimated pixel intensity
                print()
                print("# ix, iy, x, y, calculated hs-afm value", ix, iy, x, y, hsafm_frame[ix, iy])
                print("# frame(s)", frame_nr, before + N_fpp)
                print("coordset #1 ", frame_nr)
                print("transparency @fc 98 target a")
                selection = " @@serial_number=".join([str(i) for i in r_mask_indxs])
                residues = [traj.topology.atom(i-1).residue.resSeq for i in r_mask_indxs]
                print("name mybeads @@serial_number=", selection)
                print('# resids: ', np.unique(residues))
                print("select mybeads residues true")
                print("transparency sel 0 target a")
                print("~select sel")
                print("color mybeads green")
                print("transparency mybeads 0 target a")
                print("transparency @fs 100 target a")
                print('save C:/Users/xogewy72/Desktop/save_images/Frame_' + str(frame_nr) + '.png')

    hsafm_images[i_frame] = hsafm_frame

print('Done.')

# %%
tifffile.imwrite(imgp['filenames']['tiff_file'], hsafm_images)
