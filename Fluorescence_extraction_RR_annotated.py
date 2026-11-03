# The following code is a Python script that implements a full processing pipeline for 1-photon calcium imaging data.
# It uses the CaImAn library to perform motion correction, source extraction and quality evaluation.
# It is based on a pre-existing pipeline, but has been modified to include additional features (produce specific outputs).
# It also optimizes the parameters for 1-photon miniscope data.
# A. Giovannucci et al., (2019), CaImAn an open source tool for scalable calcium imaging data analysis, eLife

import os   # File management
import logging  
import matplotlib.pyplot as plt # Plot the figures
import numpy as np  # Maths/arrays
import caiman   # Module for calcium imaging analysis
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params

def run_vip_calcium_analysis(): # Creates function
    # Progress updates and error messages setup
    logging.basicConfig(
        format="[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s", 
        level=logging.INFO
    ) 

    # Input file path
    video_path = "/Users/noodle/Desktop/Master Neuro UPC /Master 2/Python/Baseline/Calcium video/Baseline_calcium.avi" 
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    fnames = [video_path]
    
    # Specific to our computers (protects MacBook Air's RAM from overloading or freezing during heavy video computations)
    c, dview, n_processes = caiman.cluster.setup_cluster(
        backend='multiprocessing', 
        n_processes=2
    )

    # CNMF-E parameters configuration (specifically modified for this project)
    p = {
        'fnames': fnames,
        'fr': 30,                  # Acquisition frame rate (Hz)
        'decay_time': 0.4,         # Calcium indicator decay time (s)
        'pw_rigid': False,         # Rigid motion correction
        'max_shifts': (5, 5),      # Maximum allowed rigid shifts (pixels)
        'gSig': (3, 3),            # Estimated neuron half-size (pixels)
        'gSiz': (13, 13),          # Bounding box surrounding each neuron
        'method_init': 'corr_pnr', # 1-photon initialization method, optimized to unmix strong background neuropil
        'K': None,                 # Automatic estimation of component count
        'ring_size_factor': 1.4,   # Ring model factor for background extraction
        'merge_thr': 0.7,          # Traces correlation threshold for merging ROIs
        'min_corr': 0.8,           # Minimum correlation threshold for seeding
        'min_pnr': 10,             # Minimum Peak-to-Noise Ratio threshold
        # Memory optimization parameters (patch processing)
        'rf': 30,                  # Half-size of patches (pixels)
        'stride': 15,              # Overlap between adjacent patches
        'ssub': 1,                 # Spatial downsampling factor
        'tsub': 2                  # Temporal downsampling factor during initialization
    }
    opts = params.CNMFParams(params_dict=p)

    # Motion correction : aligns video frames to compensate for the mouse's brain motion artifacts during recording
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    
    # Converts the motion-corrected video into a memory-mapped (.mmap) binary file
    # Allows Python to read and process frames directly from disk chunks
    bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
    fname_new = caiman.save_memmap(
        mc.fname_tot_rig, 
        base_name='memmap_', 
        order='C', 
        border_to_0=bord_px
    )

    # Loads the memory-mapped file back into memory as a 3D matrix (Frames, Height, Width)
    Yr, dims, T = caiman.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')
    opts.change_params(params_dict={'dims': dims, 'border_pix': bord_px})

    # Computes the correlation image (cn_filter) and PNR map (pnr)
    # Highlights where neurons stand out against the background
    cn_filter, pnr = caiman.summary_images.correlation_pnr(
        images[::1], 
        gSig=opts.init['gSig'][0], 
        swap_dim=False
    )

    # Source extraction 
    # Identify active neural components
    # Extract their raw fluorescence traces
    # Model the background neuropil
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
    cnm.fit(images)

    # Quality control
    # Filters out false positives (dust, blood vessels, motion artifacts) 
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    print(f"Total components extracted: {len(cnm.estimates.C)}")
    print(f"Accepted components after QC: {len(cnm.estimates.idx_components)}")

    # Converts raw fluorescence values into relative fluorescence changes to standardize neural activity
    print("Computing ΔF/F traces...")
    cnm.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    # Save results 
    results_filename = "/Users/noodle/Desktop/Master Neuro UPC /Master 2/Python/Baseline/caiman_results.hdf5"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    cnm.save(results_filename)
    print(f"Results successfully saved to: {results_filename}")

    # Stop parallel processing server
    caiman.stop_server(dview=dview)

    # Return CNMF object and summary image for visualization
    return cnm, cn_filter


if __name__ == '__main__':
    cnm, cn_filter = run_vip_calcium_analysis()