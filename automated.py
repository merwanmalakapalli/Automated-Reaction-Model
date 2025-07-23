print("Importing modules")
import jax.numpy as jnp
import pandas as pd
import arviz as az
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import math
from pprint import pprint
import re
from collections import defaultdict
import time
import sys
from skopt import Optimizer
from openpyxl.utils import column_index_from_string
import threading


# symbolic ODE solver from SKAI
from BI.utilities.reaction_solver import SymbolicRXNSolver

# bayesian inference from SKAI
from BI.models.Inference import BayesianKineticAI

print("Finished importing modules!")

def get_init_conditions_excel(file):
    """
    Load and return the initial conditions stacked in a matrix from an Excel file.

    Parameters
    ----------
    file : str or file-like
        Path to the Excel file containing initial condition data.

    Returns
    -------
    numpy.ndarray
        Matrix of initial conditions stacked by relevant columns.
    """
    df_scenarios = pd.read_excel(file, sheet_name="Scenarios", header=2)  # Row 3 → header=2 (0-indexed)

    columns = ["Temperature", "Substrate", "Reagent", "Base"]
    # columns = [
    #     col for col in df_scenarios.select_dtypes(include="number").columns
    #     if "volume" not in col.lower()
    # ]
    scenario_data = df_scenarios[columns]

    # Keep only numeric rows and drop any rows with NaNs
    scenario_data_clean = scenario_data.apply(pd.to_numeric, errors='coerce').dropna()

    # Convert to a NumPy matrix with column stacking
    scenario_matrix = np.vstack([scenario_data_clean[col].values for col in columns])
    return scenario_matrix

def get_X_measured_excel(input_matrix, t):
    """
    Stack the entire initial conditions matrix to match the size of experimental data timepoints.

    Parameters
    ----------
    input_matrix : array-like
        Matrix of initial conditions.
    t : array-like
        Timepoints of the experimental data.

    Returns
    -------
    numpy.ndarray
        Stacked matrix of initial conditions expanded to align with experimental timepoints.
    """
    input_matrix_T = input_matrix.T 
    num_duplications = len(t)
    duplicated_rows = []
    time_duplicated = []
    for i in range(len(input_matrix_T)):
        num_rows = num_duplications
        duplicated = np.tile(input_matrix_T[i], (num_rows, 1))
        duplicated_rows.append(duplicated)
        time_duplicated = np.concatenate([time_duplicated, t])
    time_duplicated = time_duplicated.reshape(-1,1)
    final_matrix = np.vstack(duplicated_rows)
    final_matrix = np.hstack([final_matrix, time_duplicated])
    final_matrix = jnp.array(final_matrix.astype(np.float32))
    return final_matrix


def get_data_time(df_raw):
    """
    Extract timepoints of a single experiment from raw experimental data.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        DataFrame containing the raw experimental data, typically read from an Excel file.

    Returns
    -------
    jnp.ndarray
        JAX numpy array containing the extracted timepoints for the experiment.
    """
    t = df_raw.iloc[:, 0].to_numpy(dtype=np.float32)

    t = jnp.array(t)
    zero_indices = jnp.where(t == 0.0)[0]

    start_idx = zero_indices[0]
    end_idx = zero_indices[1]

    
    return t[start_idx:end_idx]

def prepare_data_excel(df, weights):
    """
    Process experimental concentration data from Excel, apply weights, remove NaNs, and add noise.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing experimental concentration data with columns like 'Substrate', 'Product', 'Impurity1'.
    weights : list or array-like
        List of weights to scale contributions of different species in the data.

    Returns
    -------
    numpy.ndarray
        1D NumPy array suitable for model fitting, containing weighted and noise-added concentration data.
    """
    SM_measured  = pd.to_numeric(df["Substrate"], errors="coerce")
    P_measured   = pd.to_numeric(df["Product"], errors="coerce")
    IMP_measured = pd.to_numeric(df["Impurity1"], errors="coerce")

    mask = ~(SM_measured.isna() & P_measured.isna() & IMP_measured.isna())
    SM_measured  = SM_measured[mask]
    P_measured   = P_measured[mask]
    IMP_measured = IMP_measured[mask]

    SM_measured  = jnp.array(SM_measured)
    P_measured   = jnp.array(P_measured)
    IMP_measured = jnp.array(IMP_measured)

    measured_Y_without_Nan = (
        weights[3] * P_measured +
        weights[0] * SM_measured +
        weights[4] * IMP_measured
    )

    nan_mask = jnp.isnan(measured_Y_without_Nan)
    y_without_nan = measured_Y_without_Nan[~nan_mask]

    noise_level = 0.0001
    noise = np.random.normal(0, noise_level, y_without_nan.shape)
    y_exp_data = y_without_nan + noise

    return y_exp_data

def get_mechanism():
    """
    Retrieve the best mechanism from a CSV file containing all ranked mechanisms.

    The function reads 'res_case study 1.csv', extracts the top 3 mechanisms 
    per experiment based on probability, averages these probabilities across
    all experiments, and returns the mechanism with the highest average probability.

    Returns
    -------
    tuple
        A tuple containing:
        - best_mechanism (str): The mechanism with the highest average probability.
        - best_avg_probability (float): The corresponding average probability value.
    """
    df = pd.read_csv("res_case study 1.csv")

    # Get top 3 mechanisms per experiment
    top_mechs = (
        df.sort_values(['Experiment #', 'Probability'], ascending=[True, False])
        .groupby('Experiment #')
        .head(3)
    )

    # Sum probabilities across experiments
    mech_avg = (
        top_mechs.groupby('Mechanism')['Probability']
                .mean()
                .sort_values(ascending=False)
    )

    # Get the mechanism with the highest average probability
    best_mechanism = mech_avg.idxmax()
    best_avg_probability = mech_avg.max()
    return best_mechanism, best_avg_probability

def parse_species(species_str):
    """
    Parse a species string into a dictionary of species counts for reaction order testing.

    Given a string like 'SM + B + R', this function returns a dictionary mapping 
    species names to their counts. Some shorthand species are normalized to standard keys.

    Parameters
    ----------
    species_str : str
        A string representing species separated by '+' signs, e.g. 'SM + B + R'.

    Returns
    -------
    dict
        A dictionary where keys are species names (normalized) and values are counts (always 1).
        Example: {'SM': 1, 'Base': 1, 'Reagent': 1}
    """
    species = [s.strip() for s in species_str.split('+')]
    counts = defaultdict(int)
    for s in species:
        if s == 'R':
            s = 'Reagent'
        elif s == 'P':
            s = 'product'
        elif s == 'B':
            s = 'Base'
        elif s == 'SM':
            s = 'SM'
        elif s == 'INT1':
            s = 'INT1'
        elif re.fullmatch(r'IMP\d+', s):
            s = "IMP1"
        elif s == 'C':
            s = 'Catalyst'
        counts[s] = 1
    return dict(counts)


def mechanism_to_rxn_template(
    mechanism_str,
    SM_order, reagent_order, base_order,
    params
):
    """
    Convert a mechanism string into a reaction template with specified reaction orders.

    Parameters
    ----------
    mechanism_str : str
        Mechanism represented as a string.
    SM_order : int or float
        Reaction order of the starting material (SM).
    reagent_order : int or float
        Reaction order of the reagent.
    base_order : int or float
        Reaction order of the base.

    Returns
    -------
    reaction_template : [type]
        The reaction template generated based on the mechanism and orders.
    """
    reactions = [r.strip() for r in mechanism_str.split(',')]
    rxn_template = {}

    for i, reaction in enumerate(reactions):
        lhs, rhs = [side.strip() for side in reaction.split('->')]
        lhs_dict = parse_species(lhs)
        rhs_dict = parse_species(rhs)   

        # For some reason, base needs to be at the end of the dictionary or else everything breaks
        base_value = lhs_dict.pop('Base', None)
        if base_value is not None:
            lhs_dict['Base'] = base_value
        # rhs_dict.pop('base', None)

        rxn_key = f'rxn{i+1}'
        k_f = params['k1_f'] if i == 0 else params['k2_f']
        E_f = params['E1_f'] if i == 0 else params['E2_f']

        rxn_orders_f = {}
        
        for species in lhs_dict:
            if species == "Reagent" and reagent_order != 0:
                rxn_orders_f[species] = reagent_order
            elif species == "Base" and base_order != 0:
                rxn_orders_f[species] = base_order
            elif species == "SM" and SM_order != 0:
                rxn_orders_f[species] = SM_order
            elif species == "product":
                rxn_orders_f[species] = 1


        rxn_template[rxn_key] = {
            'reactants': lhs_dict,
            'products': rhs_dict,
            'k_f': k_f,
            'E_f': E_f,
            'reversible': False,
            'T_ref': 273.15 + 81.9,
            'Rxn_orders_f': rxn_orders_f,
            'Rxn_orders_b': '',
        }

    return rxn_template



def get_init_species(filepath):
    """
    Retrieve the list of initial species.

    Parameters
    ----------
    None

    Returns
    -------
    list
        A list containing the initial species.
    """
    df = pd.read_excel(filepath, sheet_name="Scenarios", header=None)

    start_col = column_index_from_string('E') - 1

    row_values = df.iloc[2, start_col:].tolist()
    row_values = ["SM" if val == "Substrate" else val for val in row_values]
    print(row_values)
    return row_values

def run_bayesian_inference_with_timeout(BI_reaction, reaction, priors, x_input, y_input, timeout_seconds=300):
    """
    Run Bayesian inference with a timeout
    
    Args:
        timeout_seconds: Timeout in seconds (default 5 minutes)
    """
    print(f"Starting Bayesian inference with {timeout_seconds} second timeout...")
    
    result = [None]  # Use list to store result from thread
    exception = [None]  # Store any exceptions
    
    def target():
        try:
            result[0] = BI_reaction.run_bayesian_inference(
                function=reaction.simulate_symbolic_ode_for_numpyro,
                priors=priors,
                X=x_input[12:24],
                Y=y_input[12:24],
                num_samples=200,
                num_warmup=100
            )
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print("Timeout! Bayesian inference is taking too long...")
        print("Note: The underlying computation may still be running in the background")
        return None
    
    if exception[0]:
        raise exception[0]
    
    return result[0]



def trainModels(reactions_list, priors, x_input, y_input, time, initial_species, weights):
    """
    Train a Bayesian kinetic AI model for each reaction order in the list.

    Parameters
    ----------
    reactions_list : list
        List of reaction objects/models to train.
    priors : function
        Function defining prior parameter distributions for Bayesian inference.
    x_input : array-like
        Training input data.
    y_input : array-like
        Training output data.
    time : array-like
        List or array of spaced timepoints corresponding to the training data.
    initial_species : array-like
        Initial species concentrations or states used for model initialization.
    weights : list or array-like
        List of weights for the species to scale their contributions.

    Returns
    -------
    tuple
        - modelList : list
            List of trained MCMC objects for each reaction model.
        - KAIList : list
            List of Bayesian kinetic AI model objects corresponding to each trained model.
    """
    modelList = []
    KAIList = []
    # Loop through the different reaction orders and train models to each of them. Add to model list
    for reaction in reactions_list:
        reaction.initilize_ODE_solver_for_numpyro_input(x_input, time, initial_species, weights)
        BI_reaction = BayesianKineticAI()
        # The experiment that you choose to train with can greatly affect the speed of the ODE solver
            # Slower reactions with less change/lower conversion will have mild gradients and will be faster to train
            # Using the correct mechanism and reaction order will be faster regardless of the experiment because
            # the ODE trajectories will be smoother and the gradient will be stable. If mechanism is incorrect,
            # solver will fit data using incorrect dynamics, which will make the inference model take tiny steps and get stuck
            # So for this data, the first experiment was smooth.
        BI_model = BI_reaction.run_bayesian_inference(
                function=reaction.simulate_symbolic_ode_for_numpyro,
                priors=priors,
                X=x_input[12:24],
                Y=y_input[12:24],
                num_samples=200,
                num_warmup=100)
        KAIList.append(BI_reaction)
        modelList.append(BI_model)
        # try:
        #     BI_model = run_bayesian_inference_with_timeout(
        #         BI_reaction, 
        #         reaction, 
        #         priors, 
        #         x_input, 
        #         y_input, 
        #         timeout_seconds=400  #400 seconds
        #     )
        #     if BI_model is not None:
        #         KAIList.append(BI_reaction)
        #         modelList.append(BI_model)
        #     else:
        #         print("Training timed out")
        #         continue
        # except Exception as e:
        #     print(f"Error during Bayesian inference: {e}")
        #     continue
        
    return modelList, KAIList



def chooseModel(modelList, KAIList, reactions_list):
    """
    Select the best kinetic AI model from a list of trained models.

    Parameters
    ----------
    modelList : list
        List of trained MCMC model objects containing posterior samples.
    KAIList : list
        List of Bayesian kinetic AI objects corresponding to the trained models.
    reactions_list : list
        List of reaction objects corresponding to the models.

    Returns
    -------
    tuple
        - best_model : MCMC object
            The best-performing trained MCMC model.
        - best_KAI : Bayesian kinetic AI object
            The Bayesian kinetic AI object associated with the best model.
        - best_reaction : reaction object
            The reaction corresponding to the best model.
    """
    model_dict = {i: item for i, item in enumerate(modelList)}

    # If only one model, choose that. Otherwise, use arviz to rank models
    if len(model_dict) > 1:
        compare_df = az.compare(model_dict, ic="waic")
        print(compare_df)
        best_model_name = compare_df.index[0]
    else:
        best_model_name = 0

    BI_model = modelList[best_model_name]
    BI_reaction1 = KAIList[best_model_name]
    reaction1_1 = reactions_list[best_model_name]

    return BI_model, BI_reaction1, reaction1_1


def get_reaction_mechanism_list(mechanismlist, params):
    """
    Generate a list of reaction templates from mechanism strings with default reaction orders.

    Parameters
    ----------
    mechanismlist : list of str
        List of mechanism strings to convert.
    params : dict
        Parameters required by the mechanism_to_rxn_template function.

    Returns
    -------
    list
        List of reaction templates generated from the mechanisms.
    """
    reactionlist = []
    for mech in mechanismlist:
        rxn = mechanism_to_rxn_template(mech, 1, 1, 1, params)
        rxn_solver = SymbolicRXNSolver(rxn)
        reactionlist.append(rxn_solver)
    return reactionlist


def plot_data(time, C_out, df_raw, y_predicted_all, t):
    """
    Plot mean predictions from Bayesian inference alongside experimental data points.

    Parameters
    ----------
    time : array-like
        Array of time points used for plotting and prediction.
    C_out : array-like
        Output from the ODE model representing compound concentrations.
    df_raw : pandas.DataFrame
        Experimental data containing measured concentrations.
    y_predicted_all : list of arrays
        List of Bayesian inference prediction arrays.
    t : array-like
        Timepoints corresponding to experimental measurements.

    Returns
    -------
    None
        Displays and saves the plot comparing model predictions to experimental data.
    """
    plt.figure(figsize=(10, 10))

    # subplot_index = 1
    Sanofi_color = ['#23004C', '#268500', '#7A00E6']

    subplot_index_2 = 1
    end = 0
    for i in range(5):
        plt.subplot(3, 2, subplot_index_2)
        subplot_index_2 += 1

        # Slice rows for this experiment
        start = end
        end = start + len(t)

        df_exp = df_raw.iloc[start:end]

        # plt.plot(time, C_out[:, i, -1], color=Sanofi_color[1], linewidth=2, label='ODE model - IMP')
        # plt.plot(time, C_out[:, i, 0], color=Sanofi_color[0], linewidth=2, label='ODE model - SM')
        # plt.plot(time, C_out[:, i, -2], color=Sanofi_color[2], linewidth=2, label='ODE model - P')

        plt.scatter(t, df_exp["Product"], label='Product-Exp', s=60, color=Sanofi_color[2], zorder=10)
        plt.scatter(t, df_exp["Substrate"], label='Starting Material-Exp', s=60, color=Sanofi_color[0], zorder=10)
        plt.scatter(t, df_exp["Impurity1"], label='Impurity-Exp', s=60, color=Sanofi_color[1], zorder=10)

        plt.xlabel('Time [min]')
        plt.ylabel('Concentration [mM]')

    subplot_index = 1
    Sanofi_color = ['#23004C', '#268500', '#7A00E6']
    for i in range(0, 5):
        plt.subplot(3, 2, subplot_index)
        subplot_index += 1
        mean_predicted = np.mean([y_predicted_all[j][:, i] for j in range(len(y_predicted_all))], axis=0)


        plt.plot(time, mean_predicted[:, 0], color='red', linewidth=2, linestyle='--', label='Mean BI prediction')
        plt.plot(time, mean_predicted[:, -1], color='red', linewidth=2, linestyle='--')
        plt.plot(time, mean_predicted[:, -2], color='red', linewidth=2, linestyle='--')



    plt.subplots_adjust(right=0.75)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    plt.savefig('static/my_plot.png')


def objective(x, imp_tolerance, reaction1_1, exp_means, initial_species, time):
    """
    Objective function to minimize for scikit-optimize, balancing product yield and impurity.

    This function runs a simulation with given initial conditions and kinetic parameters,
    returning a value that penalizes high impurity relative to product concentration.

    Parameters
    ----------
    x : array-like
        Input array representing initial conditions (e.g., temperature, concentrations).
    imp_tolerance : float
        Maximum allowed impurity percentage; exceeding this incurs a large penalty.
    reaction1_1 : object
        Reaction model used for simulation and optimization.
    exp_means : dict or array-like
        Expected kinetic parameter values obtained from Bayesian inference.
    initial_species : list or array-like
        Initial species concentrations or states for the simulation.
    time : jnp array
        Array of timepoints at which to predict values

    Returns
    -------
    float
        Objective value computed as negative product concentration plus impurity concentration.
        A large penalty (e.g., 1e6) is returned if impurity exceeds the tolerance.
    """
    # Run the simulation
    C_out, names = reaction1_1.simulate_symbolic_ode_(x, exp_means, time, initial_species)

    # Extract the final product concentration
    final_product_conc = C_out[-1,0,3]
    impurity = jnp.sum(C_out[-1,0,4:])
    difference = -final_product_conc + impurity
    # SM, base, reagent, product, impurity
    total_conc = jnp.sum(C_out[-1, 0])

    if total_conc > 0 and (impurity / total_conc) >= imp_tolerance:
        return 1e6
    # Return negative to maximize (BO minimizes by default)
    return float(difference)








def run_automated(file1, weights, final, interval, k1, k2, E1, E2, concentration_dict, lowTemp, highTemp, imp_tolerance):
    """
    Perform automated kinetic hypothesis testing, prediction, and optimization.

    This function orchestrates the end-to-end process of loading data, training Bayesian
    kinetic models, evaluating mechanisms, and optimizing reaction conditions to maximize
    product yield while controlling impurity levels.

    Parameters
    ----------
    file1 : str or DataFrame
        File containing time-series experimental concentration data.
    file2 : str or DataFrame
        File containing initial experimental conditions for scenarios.
    weights : list or array-like
        Weights used to scale contributions of different species during data preparation.
    final : float
        Final timepoint to predict in the kinetic simulation.
    interval : float
        Time interval between prediction steps.
    k1, k2, E1, E2 : float
        Estimated initial values for kinetic parameters.
    concentration_dict : dict
        Bounds for initial concentrations of species for Bayesian Optimization.
    lowTemp, highTemp : float
        Lower and upper temperature bounds for the optimization process.
    imp_tolerance : float
        Impurity threshold used to penalize optimization objectives.

    Returns
    -------
    None
        All output is handled internally (e.g., plots, saved files, logs).
    """
    ######################################################## FUNCTION DEFINITIONS ########################################################

    '''
    Method: _new_priors2
    Description: Samples prior parameter from distributions
    Parameters: None
    Returns: Sampled parameters
    '''
    def _new_priors2():
        k1_f = numpyro.sample("k1_f", dist.LogNormal(abs(math.log(ground_truth_params['k1_f'])), 1))
        E1_f = numpyro.sample("E1_f", dist.Normal(abs(math.log(ground_truth_params['E1_f'])), 1))
        k2_f = numpyro.sample("k2_f", dist.LogNormal(abs(math.log(ground_truth_params['k2_f'])), 1))
        E2_f = numpyro.sample("E2_f", dist.Normal(abs(math.log(ground_truth_params['E2_f'])), 1))
        return {"k1_f": k1_f,
                "E1_f": E1_f,
                "k2_f": k2_f,
                "E2_f": E2_f}
    '''
    Method: prepare_data
    Description: Processes experimental concentration data of multiple species, applies specified weighting, removes NaN values, adds noise.
    Parameters:
        weights: A list of weights used to scale contributions of different species
    Returns: returns a 1D NumPy array suitable for model fitting
    '''
    def prepare_data(df_raw, weights):
        # Preparing the experimental data and save them in a 1D array as y_exp_data
        SM_measured = []
        P_measured = []
        IMP_measured = []

        for i in range(0, 5):
            SM_measured.append(df_raw.iloc[:, i * 4 + 1])
            P_measured.append(df_raw.iloc[:, i * 4 + 2])
            IMP_measured.append(df_raw.iloc[:, i * 4 + 3])


        non_nan_values_SM  = np.concatenate(SM_measured)
        non_nan_values_P   = np.concatenate(P_measured)
        non_nan_values_IMP = np.concatenate(IMP_measured)


        # # Calculate the measured_Y_without_Nan using non-NaN values arrays   
        # ******NOTE ******this is where we apply weighting factor
        # measured_Y_without_Nan =  non_nan_values_P +  non_nan_values_IMP
        measured_Y_without_Nan = weights[3]* non_nan_values_P + weights[0]*non_nan_values_SM + weights[4]*non_nan_values_IMP
        nan_mask = jnp.isnan(measured_Y_without_Nan)
        y_without_nan = measured_Y_without_Nan[~nan_mask]

        # adding noise
        noise_level = 0.0001
        noise = np.random.normal(0, noise_level, y_without_nan.shape)
        y_exp_data = y_without_nan + noise
        return y_exp_data
    
    
    ######################################################## MAIN PART STARTS HERE ########################################################
    
    # experimental data
    yield "Reading files:\n"
    df_raw = pd.read_excel(file1, sheet_name="Data1", header=2)
    columns_to_keep = ["Time", "Substrate", "Product", "Impurity1"]
    df_raw = df_raw[columns_to_keep]

    # Convert all values to numeric, coercing errors (non-numeric → NaN)
    df_raw = df_raw.apply(pd.to_numeric, errors="coerce")

    # Drop rows where all selected columns are NaN (optional)
    df_raw = df_raw.dropna(how='all')

    # t is the timepoints of the experimental data
    t = get_data_time(df_raw)
    
    yield "Read files! \n"

    input_matrix = get_init_conditions_excel(file1)
    
    X_dummy = get_X_measured_excel(input_matrix, t)
    Y_dummy = prepare_data_excel(df_raw, weights)

    ground_truth_params = {
    "k1_f": k1,
    "E1_f": E1,
    "k2_f": k2,
    "E2_f": E2,
    }

    yield "Choosing mechanism..."
    initial_species = get_init_species(file1)

    mechanism_list = ["SM + B + R -> P + B,  P + B + R -> IMP2 + B"
                    #   , "SM + B + R -> P + B,  SM + P + B -> IMP2 + B"
                      ]
    for i, mech in enumerate(mechanism_list):
        yield f"Mechanism {i+1}: {mech}\n"
    reaction_mechanismList = get_reaction_mechanism_list(mechanism_list, ground_truth_params)
    model_mech, KAImech = trainModels(reaction_mechanismList, _new_priors2, X_dummy, Y_dummy, t, initial_species, weights)
    BI_mech, BI_reaction_mech, reaction1 = chooseModel(model_mech, KAImech, reaction_mechanismList)
    index = reaction_mechanismList.index(reaction1)
    mechanism = mechanism_list[index]



    # mechanism, probability = get_mechanism()
    yield f"Mechanism is\n{mechanism} \n"

    # Different reaction orders
    rxn1 = mechanism_to_rxn_template(mechanism, 1, 1, 0, ground_truth_params)
    rxn2 = mechanism_to_rxn_template(mechanism, 1, 1, 1, ground_truth_params)
    rxn3 = mechanism_to_rxn_template(mechanism, 0, 1, 0, ground_truth_params)
    
    yield "Preparing data...\n"
    weights = weights

    # Include all the reaction orders that you want to test
    reactions = [rxn3, rxn1, rxn2]
    reactions = [rxn1]


    # initial_species = ['SM', 'Base', 'Reagent']
    reactions_list = []

    for i, reaction_data in enumerate(reactions, start=1):
        reaction_solver = SymbolicRXNSolver(reaction_data)
        # reaction_solver.reaction_table()
        reactions_list.append(reaction_solver)



    # For prediction
    time = jnp.arange(0.0, final, interval)

    input_new = jnp.concatenate([input_matrix.T, np.zeros((5, 1))], axis=1)



    yield "Prepared data!\nChoosing best reaction order. This will take a couple minutes...\n"

    modelList, KAIList = trainModels(reactions_list, _new_priors2, X_dummy, Y_dummy, t, initial_species, weights)

    BI_model, BI_reaction1, reaction1_1 = chooseModel(modelList, KAIList, reactions_list)

    yield "Reaction order chosen!\nTraining model...\n"

    # Pick the best reaction order to train the model
    reaction1_1.initilize_ODE_solver_for_numpyro_input(X_dummy, t, initial_species, weights)
    BI_reaction = BayesianKineticAI()
    BI_model = BI_reaction.run_bayesian_inference(function = reaction1_1.simulate_symbolic_ode_for_numpyro,
        priors = _new_priors2,
        X= X_dummy[12:24],
        Y= Y_dummy[12:24],
        num_samples=2000,
        num_warmup=1000)
    # BI_model_holder = {}

    # for update in BI_reaction.run_bayesian_inference_with_progress(
    #     shared_result_dict=BI_model_holder,
    #     function=reaction1_1.simulate_symbolic_ode_for_numpyro,
    #     priors=_new_priors2,
    #     X=X_dummy[12:24],
    #     Y=Y_dummy[12:24],
    #     num_samples=2000,
    #     num_warmup=1000
    # ):
    #     # print("Update: ", update)
    #     yield update

    # BI_model = BI_model_holder["mcmc"]


    C_out, names = reaction1_1.simulate_symbolic_ode_(input_new, ground_truth_params, time, initial_species)
    reaction1_1.initilize_ODE_solver_for_numpyro_input(input_new, time, initial_species, weights)
    y_predicted_all = BI_reaction1.bayesian_inference_predict(BI_model,
                                            function = reaction1_1.simulate_symbolic_ode_for_numpyro_C_out,
                                            priors = _new_priors2,
                                            X= input_new,
                                            num_samples=2000,
                                            num_warmup=1000)

    yield "Ran predictions!\nPlotting data...\n\n"
    plot_data(time, C_out, df_raw, y_predicted_all, t)


    # Mean values for parameters
    hmc_samples = BI_model.get_samples().items()
    def summary(samples):
        site_stats = {}
        for site_name, values in samples:
            marginal_site = pd.DataFrame(values)
            describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
            site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats
    
    yield "Mean Values for Parameters:\n"
    for site, values in summary(hmc_samples).items():
        if site == "sigma":
            continue
        mean_val = values["mean"].values[0]
        yield_val = -mean_val if site.startswith("k") else mean_val
        rounded = round(jnp.exp(float(yield_val)), 3)
        yield f"{site}: mean = {rounded:.3f}\n"




    # Optimizer
    site_stats = summary(hmc_samples)
    exp_means = {}
    for site, stats in site_stats.items():
        if site == "sigma":
            continue 
        
        mean = stats.loc[0, "mean"]
        
        if site.startswith("k"):  # apply negative exponent to k values
            exp_means[site] = jnp.exp(-mean)
        else:
            exp_means[site] = jnp.exp(mean)


    yield "\nCalculating optimal conditions...\n"
    search_space = [(float(lowTemp), float(highTemp))]
    search_space.extend([(0.0, float(val)) for val in concentration_dict.values()])


    param_keys = list(concentration_dict.keys())
    param_keys = ["Temperature"] + param_keys + ["Time"]

    opt = Optimizer(search_space, base_estimator='GP', acq_func='EI') 
 
    for i in range(100):
        x_orig = opt.ask()
        x = jnp.array(x_orig)
        x_with_zero = jnp.append(x, 0)
        x = x_with_zero[None, :]
        y = objective(x, imp_tolerance, reaction1_1, exp_means, initial_species, time)
        opt.tell(x_orig, y)
        yield f"Progress: {i}%"

    best_input = opt.Xi[np.argmin(opt.yi)]

    x = best_input
    x = jnp.array(x)
    x_with_zero = jnp.append(x, 0)
    x = x_with_zero[None, :]
    C_out, names = reaction1_1.simulate_symbolic_ode_(x, exp_means, time, initial_species)
    final_product = C_out[-1, 0, 3]
    impurity = C_out[-1, 0, 4]
    total_conc = jnp.sum(C_out[-1, 0])
    # print("Optimal conditions: ", best_input)
    # print("Product: ", final_product)
    # print("Impurity: ", impurity)
    yield f"Recommended conditions:\n"
    for i in range(len(best_input)):
        name = param_keys[i]
        value = round(best_input[i], 3)
        yield f"    {name}: {value}\n"
    yield f"Total Concentration: {total_conc:.3f}\n"
    yield f"Final product concentration: {final_product:.3f}\n"
    yield f"Final impurity concentration: {impurity:.3f}\n"
    yield f"Difference between product and impurity concentrations: {(final_product-impurity):.3f}\n"
    percentage = (100*(impurity/total_conc))
    yield f"Percentage of Impurity: {percentage:.3f}%\n"

    # Checking sensitivity of parameters
    
    
    epsilon = 1e-4

    for i in range(x.shape[1] - 1):
        x_plus = x.at[0, i].set(x[0, i] + epsilon)
        x_minus = x.at[0, i].set(x[0, i] - epsilon)

        C_plus, _ = reaction1_1.simulate_symbolic_ode_(x_plus, exp_means, time, initial_species)
        C_minus, _ = reaction1_1.simulate_symbolic_ode_(x_minus, exp_means, time, initial_species)

        product_plus = C_plus[-1, 0, 3]
        product_minus = C_minus[-1, 0, 3]

        derivative = (product_plus - product_minus) / (2 * epsilon)
        relative_sensitivity = (derivative * x[0, i]) / final_product

        # If relative sensitivity is 1.5, then 1% increase in parameter will lead to 1.5% increase in product
        yield f"{param_keys[i]}: Relative sensitivity = {relative_sensitivity:.3f}\n"

    # for i in range(x.shape[1]):
    # x_modified = x.at[0, i].set(0)  # set element at row 0, column i to 0

    # C_out_mod, _ = reaction1_1.simulate_symbolic_ode_(
    #     x_modified, ground_truth_params, time, initial_species
    # )
    
    # modified_product = C_out_mod[-1, 0, 3]
    # diff = modified_product - base_final_product

    # print(f"Set x[0, {i}] = 0 --> Final product: {modified_product:.4f} | difference = {diff:.4f}")




# SM, reagent, base, product, IMP1
# run_automated('example_2_experimental_data2.csv', 'Initial_conditions_example2.csv', [0.1,0,0,0.9,0.1], 1380, 60, 0.011, 0.002, 70.2E3, 78.92E3)
# run_automated('simple_batch_reaction.xlsx', [0.1,0,0,0.9,0.1], 1380, 60, 0.011, 0.002, 70.2E3, 78.92E3)
# print("done")
