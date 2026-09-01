"""
Reference localization utilities for INFORM.

This module intentionally preserves the original batch-based localization logic.
Only package-level imports and hard-coded local paths were cleaned.

Do not refactor the Bayesian localization logic here unless validating against
this reference implementation.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy as scipy
import random
from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI, optimizer_EI
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial
import numpy as np
import seaborn as sns
np.set_printoptions(suppress=True)

colorList = ['#1f78b4',  '#ee7674', '#F6BD60', '#8dd3c7', '#e31a1c', '#cab2d6']
colorList = ['blue',  'orange', 'green', '#8dd3c7', '#e31a1c', '#cab2d6', 'blue',  'orange', 'green', '#8dd3c7', '#e31a1c', '#cab2d6']

my_color = "#83c5be"
random.seed(0)
np.set_printoptions(threshold=sys.maxsize)
from nerve_model.experiment import Experiment
import pickle


def get_data(data):
    """
    Get lead_field_matrix, n_fem_nodes_per_fiber, n_fibers, n_sites, n_internodes_simulations, n_nodes, diameters.
    Params:
    ------
    
    Return:
    ------
    
    """
    lead_field_matrix = data["lead_field_matrix"]
    n_fem_nodes_per_fiber = data["n_fem_nodes_per_fiber"][0].astype(np.int32)
    diameters = data["diameters"]
    n_fibers = n_fem_nodes_per_fiber.shape[0]
    n_sites = lead_field_matrix.shape[1]
    n_fem_nodes_per_fiber = data["n_fem_nodes_per_fiber"][0].astype(np.int32)
    n_internodes = 41 - 1
    n_nodes = n_internodes * 11 + 1

    print(f"LFM: {lead_field_matrix.shape}")
    print(lead_field_matrix.shape)
    print(f"\nNo. of fibers:\t\t`n_fibers` = {n_fibers}")
    print(f"\nNo. of sites:\t\t`n_sites` = {n_sites}")
    return(lead_field_matrix, n_fibers, n_sites, n_nodes, diameters)


def generate_lfm_fiber(lead_field_matrix, n_fibers, n_nodes, n_sites, n_fem_nodes_per_fiber, show=True):
    """
    Generate lfm per fiber.
    Params:
    -------
    
    Return:
    -------
    
    """
    lead_field_per_fiber = np.zeros((n_fibers, n_nodes, n_sites))
    n_max_nodes = np.max(n_fem_nodes_per_fiber)
    current_node = 0
    for i in range(n_fibers):
        selected_nodes = np.arange(current_node, current_node + n_fem_nodes_per_fiber[i])
        n_selected_nodes = selected_nodes.size
        f_i = int((n_max_nodes-n_fem_nodes_per_fiber[i])/2)
        lead_field_per_fiber[i, f_i:(n_selected_nodes+f_i), :] = lead_field_matrix[selected_nodes, :] * 1e3
        current_node += n_fem_nodes_per_fiber[i]
    print(lead_field_per_fiber.shape)
    if show:
        fig, ax = plt.subplots(figsize=(3, 3))
        for i in range(100):
            id_fiber_to_plot = i
            id_site_to_plot = 6
            ax.plot(lead_field_per_fiber[id_fiber_to_plot, :, id_site_to_plot])
            
    return lead_field_per_fiber


def plot_population(pop_base_path, full_experiment, pop_lfm_filename, n_pop, show=True):
    """
    Plot population
    """
    with open(pop_base_path+"Pop"+str(n_pop)+".pkl", "rb") as f:
        true_population, true_identities = pickle.load(f)

    cluster_locs = true_population.cluster_locs 

    true_identities = true_identities.astype(int)
    true_experiment = Experiment(
        fiber_population=true_population,
        nerve_topography=full_experiment.nerve_topography,
        implant=full_experiment.implant
    )


    true_lfm = true_experiment.load_lead_field_matrix(
        hdf5_file_path=pop_base_path+pop_lfm_filename,
        full_experiment=full_experiment,
        identities=true_identities
    )

    if show:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plot_section(experiment=full_experiment, 
                     fiber_population=true_population,
                     pop_clusters=(cluster_locs[:, 0], cluster_locs[:, 1]),
                 ax=ax, 
                 marker_color="black")
        full_experiment.nerve_topography.plot(ax=ax)
        plt.show()
    return true_experiment, true_population, true_identities


def plot_recruitment_extended(true_recruitment_curves, n_clusters, amplitudes, n_sites, colorList=None, title=None):
    """
    Plot recruitment curves for each stimulation site.
    
    Parameters:
    -----------
    
    Return:
    --------


    """

    xsub = np.ceil(np.sqrt(n_sites))
    ysub = np.ceil(n_sites/xsub)
    xsub = xsub.astype(int)
    ysub = ysub.astype(int)

    fig, ax = plt.subplots(xsub,ysub)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1.55,
                    top=0.7,
                    wspace=0.4,
                    hspace=0.5)
    
    for i in range(xsub):
        for j in range(ysub):
            if i*ysub+j>=n_sites:
                fig.delaxes(ax[i,j])
                break
            for k in range(n_clusters):
                if n_sites == 1:
                    ax.plot(amplitudes, true_recruitment_curves[:, k, :].T, color=colorList[k])
                    ax.set_xticks([0, amplitudes[-1]])
                    ax.set_title(f"#{i*ysub+j+1}")
                    ax.set_ylim(0,1)
                else:
                    ax[i, j].plot(amplitudes, true_recruitment_curves[i*ysub+j, k, :].T, color=colorList[k])
                    ax[i, j].set_xticks([0, amplitudes[-1]])
                    ax[i, j].set_title(f"{i*ysub+j+1}")
                    ax[i, j].set_ylim(0, 1)
                    ax[i, j].spines[["top", "right"]].set_visible(False)
   
    if title is not None:
        fig.suptitle(title)
        fig.suptitle(title, y = 0.8)
        ax.grid(axis="y")
        
def recompute_mean_std(full_experiment, true_experiment, true_population, show=True):
    """
    Recompute mean and std of cluster distribution.
    
        
    Parameters:
    -----------
    
    Return:
    --------

    
    """
    
    cluster_locs = true_population.cluster_locs 
    cluster_num = true_population.cluster_num
    n_clusters = len(cluster_num)
    
    mean_locs_x = np.array([0.0]*n_clusters)
    mean_locs_y = np.array([0.0]*n_clusters)
    new_std = np.array([0.0]*n_clusters)
    dist = np.sqrt(true_experiment.fiber_population.locs[:, 0] ** 2 + true_experiment.fiber_population.locs[:, 1] ** 2)
    true_experiment.fiber_population.locs.shape
    
    for i in range(n_clusters):
        idx = np.where(true_experiment.fiber_population.cluster_ids == i)
        mean_locs_x[i] = np.mean(true_experiment.fiber_population.locs[idx, 0])
        mean_locs_y[i] = np.mean(true_experiment.fiber_population.locs[idx, 1])
        new_std[i] = np.std(dist[idx])

    if show:
        fig, ax = plt.subplots(1, 2)
        true_population.plot(ax=ax[0])
        full_experiment.nerve_topography.plot(ax=ax[0])
        full_experiment.implant.plot(ax=ax[0])
        for i in range(n_clusters):
            ax[0].scatter(cluster_locs[i,0], cluster_locs[i,1], color=colorList[i], s=50, edgecolors='black')
        ax[0].set_aspect("equal")

        true_population.plot(ax=ax[1])
        full_experiment.nerve_topography.plot(ax=ax[1])
        full_experiment.implant.plot(ax=ax[1])
        for i in range(n_clusters):
            ax[1].scatter(mean_locs_x[i], mean_locs_y[i], color=colorList[i], s=50, edgecolors='black')
        ax[1].set_aspect("equal")

    return mean_locs_x, mean_locs_y, new_std

def create_loc_candidates(nerve_radius, limCandidateStd, limCandidateNum, nTriesLocs, nTriesStd, nTriesNum):
    """
    Create candidates for localization - grid of location (x,y), std and num.
    
        
    Parameters:
    -----------
    
    Return:
    --------

    """

    limCandidatePosX = [-nerve_radius, nerve_radius]
    limCandidatePosY = [-nerve_radius, nerve_radius]
    
    nTotalCandidates = nTriesLocs[0]*nTriesLocs[1]*nTriesStd*nTriesNum

    xCandidates = np.linspace(limCandidatePosX[0], limCandidatePosX[1], nTriesLocs[0])
    yCandidates = np.linspace(limCandidatePosY[0], limCandidatePosY[1], nTriesLocs[1])
    stdCandidates = np.linspace(limCandidateStd[0], limCandidateStd[1], nTriesStd)
    numCandidates = np.linspace(limCandidateNum[0], limCandidateNum[1], nTriesNum)


    [xCandidates, yCandidates, stdCandidates, numCandidates] = np.meshgrid(xCandidates, yCandidates, stdCandidates, numCandidates)
    xCandidates = np.reshape(xCandidates, [nTotalCandidates, 1])
    yCandidates = np.reshape(yCandidates, [nTotalCandidates, 1])
    stdCandidates = np.reshape(stdCandidates, [nTotalCandidates, 1])
    numCandidates = np.reshape(numCandidates, [nTotalCandidates, 1])

    # exclude candidates that would fall outside the nerve radius
    dist_from_origin = np.sqrt(xCandidates ** 2 + yCandidates ** 2)
    active_locs = (dist_from_origin < nerve_radius)

    candidatesGrid = np.hstack((xCandidates, yCandidates, stdCandidates, numCandidates))
    candidatesGrid = candidatesGrid[active_locs[:, 0],:]
    nTotalCandidates = candidatesGrid.shape[0]
    
    return candidatesGrid

from joblib import Parallel, delayed
import numpy as np
from functools import partial

def performLocalizationClusterParallel(
    experiment_info, refCurves, candidatesGrid, candidatesGridStandardized,
    kernel, tr, maxIter=100, batchSize=1, priorInfoX=None, rcPrior=None,
    tol=1e-3, patience=5, n_jobs=-1
):
    """
    Perform cluster localization with adaptive early stopping and parallel candidate evaluation.
    """

    full_experiment = experiment_info["full_experiment"]
    full_lfm = experiment_info["full_lfm"]
    amp_lims = experiment_info["amp_lims"]
    n_stims_per_site = experiment_info["n_stims_per_site"]
    clf = full_experiment.activation_predictor

    regressor = GaussianProcessRegressor(kernel=kernel, random_state=0)
    max_EI_modified = partial(max_EI, tradeoff=tr)
    optimizer_EI_modified = partial(optimizer_EI, tradeoff=tr)
    optimizer = BayesianOptimizer(estimator=regressor, query_strategy=max_EI_modified)

    X_iter, Y_iter, rc_iter = [], [], []
    X_max, Y_max, acq_funct_iter = [], [], []
    YPrior = []
    countEqual = 1
    best_hist = []
    no_improve = 0

    # Teach prior info if available
    if priorInfoX is not None:
        for i in range(len(priorInfoX)):
            YPrior_temp = []
            for j in range(len(priorInfoX[i])):
                rct = rcPrior[i].recruitment_values[:, j:j+1, :]
                error = rct - refCurves
                YPrior_temp.append(-np.mean(error**2))
            YPrior.append(YPrior_temp)
            optimizer.teach(priorInfoX[i], YPrior[i])

    # Adjust max iterations for later clusters
    if priorInfoX is not None and len(priorInfoX) > 0:
        maxIterCluster = max(5, maxIter // 2)
    else:
        maxIterCluster = maxIter

    # ----------------------
    # Main BO loop
    # ----------------------
    for i in range(maxIterCluster):

        # Select candidates
        if priorInfoX is None:  # first iteration: instruct with 30 samples to shape prior
            idx_batch_iter = np.random.choice(candidatesGridStandardized.shape[0], size=30)
        else:
            idx_batch_iter, _ = optimizer.query(candidatesGridStandardized, n_instances=batchSize)

        X_iter.append(candidatesGridStandardized[idx_batch_iter, :])

        # --------------- Parallel candidate evaluation ----------------
        def evaluate_candidate(idx):
            aux_experiment_temp, identities = Experiment.from_existing_experiment(
                experiment=full_experiment,
                has_struct_info=True,
                cluster_locs=candidatesGrid[idx, 0:2],
                cluster_std=candidatesGrid[idx, 2],
                cluster_num=candidatesGrid[idx, 3].astype(int),
            )

            aux_experiment_temp.load_lead_field_matrix(
                identities=identities,
                lead_field_matrix=full_lfm,
                full_experiment=full_experiment
            )

            aux_experiment_temp._activation_predictor = clf
            rc_temp = aux_experiment_temp.generate_recruitment_curves(
                amp_lims=amp_lims,
                n_steps=n_stims_per_site,
                method="from_self"
            )

            error_list = []
            for bItem in range(1):
                rct = rc_temp.recruitment_values[:, bItem:bItem+1, :]
                error = rct - refCurves
                error_list.append(-np.mean(error**2))
            return rc_temp, error_list[0]

        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_candidate)(idx) for idx in idx_batch_iter)

        # Unpack results
        rc_iter_temp, Y_iter_temp = zip(*results)
        rc_iter.append(list(rc_iter_temp))
        Y_iter.append(list(Y_iter_temp))

        # Teach new observations
        optimizer.teach(X_iter[i], Y_iter[i])

        # Track best
        X_max_temp, y_max_temp = optimizer.get_max()
        X_max.append(X_max_temp)
        Y_max.append(y_max_temp)

        best_hist.append(y_max_temp)

        if i > 0 and abs(best_hist[-1] - best_hist[-2]) < tol:
            no_improve += 1
        else:
            no_improve = 0

        if i > 0 and Y_max[i-1] != y_max_temp:
            countEqual = 1
        else:
            countEqual += 1

        # Acquisition function evaluation
        acq_funct_iter.append(optimizer_EI_modified(optimizer, candidatesGridStandardized))

        # Early stopping conditions
        if max(acq_funct_iter[i]) < 1e-10 or countEqual >= 15 or no_improve >= patience:
            break

    return X_iter, Y_iter, rc_iter, X_max, Y_max, acq_funct_iter, YPrior


def performLocalizationCluster(experiment_info, refCurves, candidatesGrid, candidatesGridStandardized, 
                               kernel, tr=0.01, maxIter=100, batchSize=1, priorInfoX=None, rcPrior=None):
    """
    Perform cluster localization.
    
        
    Parameters:
    -----------
    
    Return:
    --------

    
    """
    full_experiment = experiment_info["full_experiment"]
    full_lfm = experiment_info["full_lfm"]
    amp_lims = experiment_info["amp_lims"]
    n_stims_per_site = experiment_info["n_stims_per_site"]
    clf = full_experiment.activation_predictor
    regressor = GaussianProcessRegressor(kernel=kernel, random_state=0)
    max_EI_modified = partial(max_EI, tradeoff=tr)
    optimizer_EI_modified = partial(optimizer_EI, tradeoff=tr)
    
    optimizer = BayesianOptimizer(
        estimator=regressor,
        query_strategy=max_EI_modified
    )
    
    X_iter = [] # contains the array of variables for the evaluated candidates at each iteration
    Y_iter = []  # containse the MSE for the evaluated candidates at each iter
    rc_iter = []
    X_max = []
    Y_max = []
    acq_funct_iter = []
    
    YPrior = []
    countEqual = 1
    
    if priorInfoX is not None:
        for i in range(len(priorInfoX)):
            YPrior_temp = []
            for j in range(len(priorInfoX[i])):
                rct = rcPrior[i].recruitment_values[:, j:j+1, :]
                error = rct - refCurves
                YPrior_temp.append(-np.mean(error**2))
            YPrior.append(YPrior_temp)
            optimizer.teach(priorInfoX[i], YPrior[i])
                
    for i in range(maxIter):
        if i==0 and priorInfoX is None: #first iteration: instruct with 30 samples to shape a better prior
            idx_batch_iter = np.random.choice(candidatesGridStandardized.shape[0], size=30)
        else:   
            idx_batch_iter, _ = optimizer.query(candidatesGridStandardized, n_instances=batchSize)

        X_iter.append(candidatesGridStandardized[idx_batch_iter, :])  # for the iteration, use the queried candidates

        aux_experiment_temp, identities = Experiment.from_existing_experiment(
            experiment=full_experiment,
            has_struct_info=True,
            cluster_locs=candidatesGrid[idx_batch_iter, 0:2],
            cluster_std=candidatesGrid[idx_batch_iter, 2],
            cluster_num=candidatesGrid[idx_batch_iter, 3].astype(int),
        )
        
        aux_experiment_temp.load_lead_field_matrix(
            identities=identities,
            lead_field_matrix=full_lfm,
            full_experiment=full_experiment
        )

        aux_experiment_temp._activation_predictor = clf
        
        rc_iter_temp = aux_experiment_temp.generate_recruitment_curves(
            amp_lims=amp_lims,
            n_steps=n_stims_per_site,
            method="from_self"
        )
        
        rc_iter.append(rc_iter_temp)
        
        Y_iter_temp = []
        for bItem in range(idx_batch_iter.shape[0]):
            rct = rc_iter_temp.recruitment_values[:, bItem:bItem+1, :]
            error = rct - refCurves
            Y_iter_temp.append(-np.mean(error**2))  # negative sign for maximization  
        
        Y_iter.append(Y_iter_temp)

        # add the new observations of the batch for posterior model fitting
        optimizer.teach(X_iter[i], Y_iter[i])

        # get the temporary best candidate and its relative maximum MSE
        X_max_temp, y_max_temp = optimizer.get_max()
        X_max.append(X_max_temp)
        Y_max.append(y_max_temp)
        
        if Y_max[i-1] != y_max_temp:
            countEqual = 1
        else:
            countEqual = countEqual+1

        acq_funct_iter.append(optimizer_EI_modified(optimizer,candidatesGridStandardized))
    
        if max(acq_funct_iter[i])<1e-8 or countEqual>=15: # stop if acquisition function is low and maximum is the same or n iterations
            print(countEqual)
            print(max(acq_funct_iter[i]))
            print(Y_max[i])
            break
    
    return X_iter, Y_iter, rc_iter, X_max, Y_max, acq_funct_iter, YPrior


def plot_recruitment_superposed(true_recruitment_curves, pred_recruitment_curves, amplitudes, n_sites, n_clusters, colorList=None, title=None):
    xsub = np.ceil(np.sqrt(n_sites)).astype(int)
    ysub = np.ceil(n_sites/xsub).astype(int)

    fig, ax = plt.subplots(xsub,ysub, layout="constrained")
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1.55,
                    top=0.7,
                    wspace=0.4,
                    hspace=0.7)
    
    for i in range(xsub):
        for j in range(ysub):
            if i*ysub+j>=n_sites:
                fig.delaxes(ax[i, j])
                break
            for k in range(n_clusters):
                if n_sites == 1:
                    ax.plot(amplitudes, true_recruitment_curves[:, k, :].T, color=colorList[k])
                    ax.plot(amplitudes, pred_recruitment_curves[:, k, :].T,'--', color=colorList[k])
                    ax.set_xticks([0, amplitudes[-1]])
                    ax.set_title(f"{i*ysub+j+1}")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(axis="y")
                    ax.set_ylim(0, 1)
                else:
                    ax[i, j].plot(amplitudes, true_recruitment_curves[i*ysub+j, k, :].T, color=colorList[k])
                    ax[i, j].plot(amplitudes, pred_recruitment_curves[i*ysub+j, k, :].T,'--', color=colorList[k])
                    ax[i, j].set_xticks([0, amplitudes[-1]])
                    ax[i, j].set_title(f"{i*ysub+j+1}")
                    ax[i, j].grid(axis="y")
                    ax[i, j].set_ylim(0, 1.1)
                    ax[i, j].spines[["top", "right"]].set_visible(False)
                    if i!=xsub-1:
                        ax[i, j].set_xticks([])
   
    if title is not None:
        fig.suptitle(title, size=20)

def plot_section(experiment, ax, colors=None, act=False, activation=None, pop_clusters=None, fiber_population=None, topographic=True, radius=2, marker_color="red"):
    """
    Plot section
    """
    fill_color = "#f3f5f7"
    
    if colors:
        all_colors=colors
    else:
        all_colors = ['#1f78b4', '#ee7674', '#F6BD60', '#8dd3c7', '#e31a1c', '#cab2d6']
        
    nerve_section = plt.Circle((0., 0.), radius, fill=True, edgecolor='none', facecolor=fill_color, label="Nerve and Fascicle sections")
    ax.set_aspect(1)
    ax.add_artist(nerve_section)
    
    if not fiber_population:
        fiber_population = experiment.fiber_population
    n_fibers = len(fiber_population.locs)
    
    fiber_color = [all_colors[fiber_population.cluster_ids[i].astype(int)] for i in range(n_fibers)]
    if act:
        fiber_colors = [fiber_color[i_val] if val else "#ced4da" for i_val, val in enumerate(activation)]
    else:
        fiber_colors = fiber_color
        
    ax.scatter(fiber_population.locs[:, 0], fiber_population.locs[:, 1], c=fiber_colors, alpha=0.3, zorder=1)
    ax.scatter(experiment.implant.site_locs[:, 0], experiment.implant.site_locs[:, 1], c=marker_color, edgecolor="white", marker="^", s=100, label="Sites")
    experiment.nerve_topography.plot(ax=ax)
    
    # Plot clusters
    if pop_clusters:
        n_clusters = len(pop_clusters[0])
        for i in range(n_clusters):
            ax.scatter(pop_clusters[0][i], pop_clusters[1][i], c=all_colors[i], label=f"Cluster {i+1}", s=40, edgecolors='black')


    ax.set_ylim(-2.3, 2.3)
    ax.set_xlim(-2.3, 2.3)
    ax.tick_params(labelsize=20)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, width=2)
    #ax.axhline(-radius, color='black', linewidth=2)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_aspect("equal")
    
def params_to_selectivity_tf(params, n_sites, n_active_sites, true_population, experiment, musc_selective, batch_size, return_recruitment=False):

    params = np.reshape(params, [batch_size, n_sites*2])
    stimulation_protocol = np.zeros((batch_size, n_sites))
    
    for j in range(batch_size):
        x = np.argsort(params[j, n_sites:])[::-1]
        #print(x.shape)
        
        for i in range(n_active_sites): #only the "n_acive_sites" highest ranked electrodes are considered in the stimulation pattern, other electrodes are at 0 amplitude
            stimulation_protocol[j, x[i]] = params[j, x[i]]
    
    recruitment_curves = experiment.compute_recruitment_patterns(
        stimulation_protocols=stimulation_protocol,
        method="from_self"
    )

    selectivity = np.zeros(batch_size)
    for i in range(batch_size):
        if np.sum(recruitment_curves[i, :]) == 0:
            selectivity[i] = 0
        else:
            selectivity[i] = recruitment_curves[i, musc_selective]**2/np.sum(recruitment_curves[i, :])

    if return_recruitment==True:
        return stimulation_protocol, recruitment_curves, selectivity
 
    return -selectivity  #negative as ps, ga and de are minimizing the variable 

def off_diagonal_frobenius_norm(A):
    full_frobenius_norm = np.linalg.norm(A)
    off_diagonal_mask = np.ones(A.shape, dtype=bool)
    np.fill_diagonal(off_diagonal_mask, 0)
    off_diagonal_elements = A[off_diagonal_mask]
    off_diag_frobenius_norm = np.linalg.norm(off_diagonal_elements)
    off_diagonal_ratio = off_diag_frobenius_norm / full_frobenius_norm
    
    return off_diag_frobenius_norm, off_diagonal_ratio 

def plot_matrix(rc, size, title): 
    fro_norm, fro_ratio = off_diagonal_frobenius_norm(rc)
    mask = np.triu(np.ones_like(rc, dtype=bool))
    colorList = ['#1f78b4', '#ee7674', '#F6BD60', '#8dd3c7']

    # Plot setup
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor('white')
    sns.heatmap(rc, mask=mask, cmap=["white"], cbar=False, annot=False, annot_kws={"size": 13}, linewidths=2, linecolor='white')

    # Add color patches on each cell, with intensity proportional to activation
    for i in range(size):
        for j in range(size):
            intensity = rc[j, i] 
            color = colorList[i]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, alpha=intensity, ec='white', lw=2))

    # Annotate each cell with its value
    for i in range(size):
        for j in range(size):
            plt.text(j + 0.5, i + 0.5, f"{rc[j, i]:.2f}", ha='center', va='center', color='black')

    # Configure ticks and labels
    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_xticklabels(np.arange(1, size + 1))
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_yticklabels(np.arange(1, size + 1))
    ax.set_title(title+f"{fro_ratio:.3f}", size=15)
    return fig