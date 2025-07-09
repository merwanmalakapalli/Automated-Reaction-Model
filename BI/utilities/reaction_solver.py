from prettytable import PrettyTable
import jax.numpy as jnp
from jax.experimental.ode import odeint
from collections import OrderedDict
import jax 

class SymbolicRXNSolver():
    
    """
    Reaction  

    Args:
        reactions (dict): A dictionary containing system of reactions. Example:
        
        reactions = {
                'rxn1': {
                    'reactants': {'SM': 1, 'B': 2, 'R': 2},       
                    'products': {'A': 1, 'C': 2},              # symbolic reactants and products
                    'k_f': 0.1,                                # Forward reaction pre-exponential factor 
                    'E_f': 20,                                 # Forward reaction activation energy
                    'k_b': 0.05,                               # Reverse reaction pre-exponential factor 
                    'E_b': 25,                                 # Reverse reaction activation energy
                    'reversible': True,
                    'T_ref': 300.15,                           # T_ref in Arrhenius rate equation
                    'Rxn_orders_f': {'SM':1, 'B':1}, 
                    'Rxn_orders_b' :'',
                },
                'rxn_2': {
                    'reactants': {'R': 2, 'INT1': 2},
                    'products': {'D': 1, 'E': 2},
                    'k_f': 0.2,
                    'E_f': 30,
                    'reversible': False
                }
            }

    Methods:
        reaction_table(): Prints reactions in a table. 
        
    """

    def __init__(self, reactions):
        self.reactions                     = reactions
        self.num_reactions                 = len(self.reactions)
        self.num_species, self.all_species = self._num_species()
        self.rate_expressions              = self.write_reaction_rate_expressions()
        self.ode_functions, self.ode_k_type_matrix_f, self.ode_k_type_matrix_b = self.write_ODE_system()
        self.reaction_order_dict_f, self.reaction_order_dict_b = self.get_reaction_order_matrix()
        self.T_ref                         = self._get_reference_T()


    def _num_species(self):
        """
        Calculates the number of unique species in the system of reactions.

        Returns:
            int: Number of unique species.
        """
        # all_species = set(species for reaction_info in self.reactions.values() for species in reaction_info['reactants'])
        # all_species.update(species for reaction_info in self.reactions.values() for species in reaction_info['products'])
        # all_species = ['SM', 'reagent', 'base', 'IMP1', 'product']
        all_species = list(dict.fromkeys(
            species
            for reaction_info in self.reactions.values()
            for species in list(reaction_info['reactants']) + list(reaction_info['products'])
        ))
        return len(all_species), all_species
    
    def get_all_species(self):
        return self.all_species

    def _get_reference_T(self):
        self.T_ref = [reaction['T_ref'] for reaction in self.reactions.values()]
        return self.T_ref
        
        
    def reaction_table(self):
        
        """
        Prints reactions in a table using PrettyTable.
        
        """
        table = PrettyTable()
        table.field_names = ["Reaction", "Reactants", " ", "Products", "k (f)", "E (f)", "k (b)", "E (b)", "Reversible?"]
        table.align = 'l'
        table.border = True

        for reaction_name, reaction_info in self.reactions.items():
            reactants     = ' + '.join([f"{count}{species}" for species, count in reaction_info['reactants'].items()])
            products      = ' + '.join([f"{count}{species}" for species, count in reaction_info['products'].items()])
            k_forward     = reaction_info.get('k_f', 0)
            E_forward     = reaction_info.get('E_f', 0)
            k_reverse     = reaction_info.get('k_b', 0)
            E_reverse     = reaction_info.get('E_b', 0)
            reversible    = reaction_info.get('reversible', False)  # if not determined we assume is not reversible
            reaction_sign = "<>" if reaction_info.get('reversible', False) else "-->"

            table.add_row([reaction_name, reactants, reaction_sign, products, k_forward, E_forward, k_reverse, E_reverse, reversible])

        print("Reaction:")
        print()
        print(table)
        
        
    def write_reaction_rate_expressions(self):
        """
        Generates reaction rate expressions based on rate constants and reaction coefficients.

        Returns:
            dict: A dictionary where keys are reaction names and values are rate expressions.
        """
        rate_expressions = {}

        for reaction_name, reaction_info in self.reactions.items():  # need to make sure all these fields are available or get default values! this needs to be added
            reactant_factors              = ' '.join([f"{species}^{count}" for species, count in reaction_info['Rxn_orders_f'].items()])
            product_factors               = ' '.join([f"{species}^{count}" for species, count in reaction_info['Rxn_orders_b'].items()]) if reaction_info.get('reversible', False) else ""

            product_factors              = ' '.join([f"{species}^{count}" for species, count in reaction_info['products'].items()])
            rate_expression_forward      = f"R{reaction_name}_f = k{reaction_name}_f * {reactant_factors}"
            rate_expression_backward     = f"R{reaction_name}_b = k{reaction_name}_b * {product_factors}" if reaction_info.get('reversible', False) else ""

            rate_expressions[reaction_name] = {'forward': rate_expression_forward, 'backward': rate_expression_backward}

        return rate_expressions

    def get_reaction_order_matrix(self):
        """
        Creates a matrix for reaction orders if given

        Returns:
            dictionary : 
        """
        reaction_order_dict_f  = {}
        reaction_order_dict_b  = {}
        for species in self.all_species:
            reaction_order_f = []
            reaction_order_b = []

            for reaction_name, reaction_info in self.reactions.items():
                if species in reaction_info['Rxn_orders_f']:
                    reaction_order_f.append(reaction_info['Rxn_orders_f'][species])
                
                else:
                    reaction_order_f.append(0)

                
                if species in reaction_info['Rxn_orders_b'] and reaction_info.get('reversible', True):
                    reaction_order_b.append(reaction_info['Rxn_orders_b'][species])
                else:
                 
                    reaction_order_b.append(0)
            
            reaction_order_dict_f[species] = reaction_order_f
            reaction_order_dict_b[species] = reaction_order_b
        

        return reaction_order_dict_f, reaction_order_dict_b


    def write_ODE_system(self):
        """
        Generates ODE functions for each species.

        Returns:
            dict: A dictionary where keys are species names and values are ODE functions.
        """
        ode_functions           = {}
        ode_k_type_dict_f       = {}
        ode_k_type_dict_b       = {}


        for species in self.all_species:            
            ode_expression = []
            ode_k_type_matrix_f = []
            ode_k_type_matrix_b = []

            for reaction_name, reaction_info in self.reactions.items():
                rate_expression = f"(R_{reaction_name}_f - R_{reaction_name}_b)" if reaction_info.get('reversible', False) else f"Rate_{reaction_name}_forward"

                if species in reaction_info['reactants']:
                    ode_expression.append(f"-{reaction_info['reactants'][species]} * {rate_expression}")
                    ode_k_type_matrix_f.append(f"-{reaction_info['reactants'][species]}")
                    ode_k_type_matrix_b.append(f"{reaction_info['reactants'][species]}") if reaction_info.get('reversible', True) else ode_k_type_matrix_b.append("0")


                elif species in reaction_info['products']:
                    ode_expression.append(f"{reaction_info['products'][species]} * {rate_expression}")
                    ode_k_type_matrix_f.append(f"{reaction_info['products'][species]}")
                    ode_k_type_matrix_b.append(f"-{reaction_info['products'][species]}") if reaction_info.get('reversible', True) else ode_k_type_matrix_b.append("0")

                else:
                    ode_expression.append("0")
                    ode_k_type_matrix_f.append("0")
                    ode_k_type_matrix_b.append("0")
  

            # Combine the ODE expressions and format the final ODE for the species
            ode_function = f"d{species}/dt = " + " + ".join(ode_expression)
            ode_functions[species]   = ode_function
            ode_k_type_dict_f[species] = ode_k_type_matrix_f
            ode_k_type_dict_b[species] = ode_k_type_matrix_b




        return ode_functions, ode_k_type_dict_f, ode_k_type_dict_b
    
    
    def sys_odes_(self, C:jnp.array, time:jnp.array, params: dict, T:jnp.array)->jnp.array:
        """
        generates dC/dt arrays for ODE intergator
        Args: 
            C:jnp.array intial concentrations for ODE integrator
            time:jnp.array  time poitns
            params: dict   kinetics params
            T:jnp.array    tempeerature broadcasted
        Returns:
            jnp.array : dC/dt
        """
      
        
        # column_names  = self.initial_species
        # name_combined = list(set(OrderedDict.fromkeys(column_names + list(self.all_species - set(column_names)))))
        T_broadcasted = T.reshape((-1, 1))


        k_f_params = jnp.array(list(map(lambda i: params.get(f'k{i + 1}_f', 0.0), range(self.num_reactions))))
        k_b_params = jnp.array(list(map(lambda i: params.get(f'k{i + 1}_b', 0.0), range(self.num_reactions))))
        E_f_params = jnp.array(list(map(lambda i: params.get(f'E{i + 1}_f', 0.0), range(self.num_reactions))))
        E_b_params = jnp.array(list(map(lambda i: params.get(f'E{i + 1}_b', 0.0), range(self.num_reactions))))
        


        k_f_params = jnp.exp(-k_f_params)
        k_b_params = jnp.exp(-k_b_params)
        E_f_params = jnp.exp(E_f_params)
        E_b_params = jnp.exp(E_b_params)


        R = 8.314
        k_f = k_f_params * jnp.exp((-E_f_params /R) * ((1/(T_broadcasted + 273.15))-(1/(self.T_ref[0]))))
        k_b = k_b_params * jnp.exp((-E_b_params /R) * ((1/(T_broadcasted + 273.15))-(1/(self.T_ref[0]))))

 
        rate_constant_matrix = jnp.concatenate((k_f, k_b), axis=0)
        rate_constant_matrix = jnp.column_stack([k_f, k_b])
        



        k_matrix_coeff        = self.k_matrix_coeff 
        reaction_order_matrix = self.reaction_order_matrix
        

        # rate_helper_0 = jnp.ones_like(reaction_order_matrix)
        # helper_new_1  = C[:, jnp.newaxis] * rate_helper_0
        helper_new_1  = C[:, jnp.newaxis] 
        helper_new_2  = jnp.transpose(helper_new_1, (0, 2, 1))
        helper_new_3  = jnp.exp(reaction_order_matrix * jnp.log(helper_new_2 + 1e-20))
        helper_new_4  = jnp.prod(helper_new_3, axis=1)
        helper_new_5  = rate_constant_matrix * helper_new_4
    
        dC_dt = jnp.sum(jnp.transpose(helper_new_5[:, :, None], (0, 2, 1)) * k_matrix_coeff[None, :, :], axis=-1)

        return dC_dt

    """
    Prepares and stores static data for simulating symbolic ODEs with NumPyro.

    This function initializes model state by:
    - Extracting unique initial concentration conditions from X
    - Computing species padding for ODE solving
    - Building stoichiometric and reaction order matrices for forward and backward reactions
    - Computing repeat counts for each unique condition
    - Mapping species names to their indices and acquisition weights

    Args:
        X (jnp.array): Input array of shape (N, D) or (D,) where each row (or vector) consists of:
                       [temperature, concentrations of initial species, time].
        t (jnp.array): Array of time points at which to solve the ODEs.
        initial_species (list of str): Names of species whose concentrations are provided in X.
        acq_weight (list of float): Weights applied to each species output in the acquisition function.

    Side Effects:
        Sets the following attributes on the object:
            - self.initial_species
            - self.time, self.time_t
            - self.acq_weight
            - self.unique_rows
            - self.unique_intial_conditions_num
            - self.list_n_values
            - self.k_matrix_coeff
            - self.reaction_order_matrix
            - self.indices

    Notes:
        This is a JAX-compatible setup step designed for use with NumPyro inference.
        It assumes the system has already been configured with:
            - self.num_species
            - self.all_species
            - self.ode_k_type_matrix_f / b (stoichiometric matrices)
            - self.reaction_order_dict_f / b (reaction orders)
    """
    def initilize_ODE_solver_for_numpyro_input(self, X:jnp.array, t:jnp.array, initial_species:list, acq_weight: list):
        """
        Saves static values for JAX (temporary solution)
        Args: 
            X : jnp.array of [temperature, C concetration vectors, time]
            t : jnp.array of time
            initial_species: list 


        """

        self.initial_species                =  initial_species
        self.time                           =  t
        self.time_t                         =  t.tolist()
        self.acq_weight                     = acq_weight
        if X.ndim == 1:
            initial_condition_C     = X[1:-1]
            T                       = X[0].astype(int)
            num_species_padding     = self.num_species - len(initial_condition_C)
            initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((num_species_padding))])[jnp.newaxis, :]



        else:
            initial_condition_C    = X[:, 1:-1]
            T                      = X[:,0]
            num_species_padding = self.num_species - initial_condition_C.shape[1]
            initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((initial_condition_C.shape[0], num_species_padding))], axis=1)
        
        
        unique_rows, indices, counts          = jnp.unique(initial_condition_C_all, axis=0, return_index=True, return_counts=True)
        self.unique_rows                      = unique_rows
        self.unique_intial_conditions_num     = len(counts)

        # print(self.unique_intial_conditions_num)



        sorted_indicies  = jnp.sort(indices)
        
        indices_2  = sorted_indicies
        m          = indices_2.shape[0]
        n_values   = jnp.diff(indices_2)
        n_values   = jnp.append(n_values, initial_condition_C_all.shape[0]-indices_2[-1])
        self.list_n_values   = n_values.tolist()

        column_names  = self.initial_species
        # name_combined = list(set(OrderedDict.fromkeys(column_names + list(self.all_species - set(column_names)))))
        # name_combined = initial_species + [name for name in self.all_species if name not in initial_species]
        name_combined = self.all_species
        name_combined = list(dict.fromkeys(name_combined))
        # print(name_combined)
        # create k type matrix based on the initial names provdided

        k_matrix_coeff_f = jnp.array(list(map(lambda col: self.ode_k_type_matrix_f.get(col, 0), name_combined)), dtype=int)
        k_matrix_coeff_b = jnp.array(list(map(lambda col: self.ode_k_type_matrix_b.get(col, 0), name_combined)), dtype=int)
        

        reaction_order_f = jnp.array(list(map(lambda col: self.reaction_order_dict_f.get(col, 0), name_combined)), dtype=int)
        reaction_order_b = jnp.array(list(map(lambda col: self.reaction_order_dict_b.get(col, 0), name_combined)), dtype=int)
        
        # Combine values and assign to the matrix
        self.k_matrix_coeff        = jnp.concatenate([k_matrix_coeff_f,  k_matrix_coeff_b], axis=1)
        self.reaction_order_matrix = jnp.concatenate([reaction_order_f,  reaction_order_b], axis=1)

        name_to_index = {name: index for index, name in enumerate(name_combined)}
        indices = [name_to_index[name] for name in initial_species if name in name_to_index]
        self.indices = indices
        self.initial_species = initial_species
        # print(f'The list of indices used to solve the ODE system is: {name_to_index}')
        name_to_index = {key: weight for key, weight in zip(name_to_index.keys(), self.acq_weight)}
        # print(f'Acq function weights: {name_to_index}')

    """
        Simulates a system of ODEs and returns a scalar output based on weighted final species concentrations.
        Used to compare model output to observed data

        This method is optimized for use with NumPyro

        If multiple samples are passed, it efficiently solves only for the unique initial conditions,
        then maps the solution to the original input structure using repeat logic.
        
        Used for probabilistic analysis and running Bayesian inference (training)

        Args:
            X (jnp.array): Input array. If 1D: [T, C0_1, C0_2, ..., tf]; 
                          if 2D: shape (N, M), where columns are [T, C0_1, ..., C0_k, tf].
            params (dict): Dictionary of kinetic parameters (e.g., rate constants, activation energies).

        Returns:
            jnp.array: A scalar or vector of objective function values computed using `acq_weight` applied to
                       concentrations at the target time `tf`. The output is suitable for probabilistic modeling.
        """
    def simulate_symbolic_ode_for_numpyro(self, X:jnp.array, params: dict)->jnp.array:
        """
        Generates a system of ODEs and solves it.
        Args: 
            X : jnp.array of [temperature, C concetration vectors, time]
            params : a dict of all the kinetic parameters

        Returns:
            jnp.array : output of the ODE solver | objective function needs to be editted here 
        """


     
        time = self.time


        if X.ndim == 1:
            initial_condition_C     = X[1:-1]
            T                       = X[0].astype(int)
            num_species_padding     = self.num_species - len(initial_condition_C)
            initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((num_species_padding))])[jnp.newaxis, :]
        else:
            initial_condition_C    = X[:, 1:-1]
            T                      = X[:,0]
            num_species_padding = self.num_species - initial_condition_C.shape[1]
            initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((initial_condition_C.shape[0], num_species_padding))], axis=1)
        


        # only run ODE intergator once for each unique initial conditions
            # unique_rows, indices, counts      = jnp.unique(initial_condition_C_all, axis=0, return_index=True, return_counts=True)
            unique_rows, indices, counts      = jnp.unique(initial_condition_C_all, axis=0, return_index=True, return_counts=True, size=self.unique_intial_conditions_num)

            unique_intial_conditions          = unique_rows[jnp.argsort(indices)]
            unique_T                          =  T[indices[jnp.argsort(indices)]]
        


        # sorted_unique_intial_conditions = jnp.zeros_like(unique_intial_conditions)

        # for i in range(0, len(self.initial_species)):
        #    sorted_unique_intial_conditions =  sorted_unique_intial_conditions.at[:,self.indices[i]].set(unique_intial_conditions[:,i])


        # print(self.indices)
        # print(sorted_unique_intial_conditions)
        C_solutions = odeint(self.sys_odes_, unique_intial_conditions, time, params, unique_T)

        C_solutions1 = C_solutions
        # C_solutions = odeint(self.sys_odes_, sorted_unique_intial_conditions, time, params, unique_T)


        combined_array = jnp.zeros((len(time), 0, self.num_species))


        for i in range(self.unique_intial_conditions_num):
            array_helper = jnp.repeat(C_solutions[:,i, :], self.list_n_values[i], axis=0).reshape((self.time.shape[0], self.list_n_values[i], self.num_species))
            combined_array = jnp.concatenate((combined_array, array_helper), axis=1)


        C_solutions  = combined_array

        if X.ndim == 1:
            out = jnp.array([0])
        else:
            index = jnp.searchsorted(time, X[:,5])
        
            C_masked_list = []
            for i, j in zip(index, range(0,X.shape[0])):
                C_masked_list.append (C_solutions[i,j,:])

            
            C_masked = jnp.stack(C_masked_list)
            # end_product_out =  (C_masked[:,-1]) 
            end_product_out = sum(weight * C_masked[:, i] for i, weight in enumerate(self.acq_weight))

            if jnp.ndim(X) == 0:
                out = jnp.array(end_product_out[0])
            else:
                out = jnp.array(end_product_out)

        return jnp.array(out)

    """
        Simulates a system of ODEs and returns the full concentration profiles over time.

        This variant is useful when you want access to all time points for every species,
        instead of a single weighted output. It otherwise uses the same efficient logic
        for batching unique initial conditions as `simulate_symbolic_ode_for_numpyro`.
        
        Used for probabilistic analysis and Bayesian inference predictions (experimental data) 

        Args:
            X (jnp.array): Input array. If 1D: [T, C0_1, C0_2, ..., tf]; 
                          if 2D: shape (N, M), where columns are [T, C0_1, ..., C0_k, tf].
            params (dict): Dictionary of kinetic parameters (e.g., rate constants, activation energies).

        Returns:
            jnp.array: Array of shape (time_points, N, num_species), representing the full
                       simulation results for each initial condition over time.
        """
    def simulate_symbolic_ode_for_numpyro_C_out(self, X:jnp.array, params: dict)->jnp.array:
            """
            Generates a system of ODEs and solves it.
            Args: 
                X : jnp.array of [temperature, C concetration vectors, time]
                params : a dict of all the kinetic parameters

            Returns:
                jnp.array : output of the ODE solver | objective function needs to be editted here 
            """


        
            time = self.time


            if X.ndim == 1:
                initial_condition_C     = X[1:-1]
                T                       = X[0].astype(int)
                num_species_padding     = self.num_species - len(initial_condition_C)
                initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((num_species_padding))])[jnp.newaxis, :]
            else:
                initial_condition_C    = X[:, 1:-1]
                T                      = X[:,0]
                num_species_padding = self.num_species - initial_condition_C.shape[1]
                initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((initial_condition_C.shape[0], num_species_padding))], axis=1)

            # only run ODE integrator once for each unique initial conditions; collapses identical rows
            unique_rows, indices, counts      = jnp.unique(initial_condition_C_all, axis=0, return_index=True, return_counts=True, size=self.unique_intial_conditions_num)
            # unique_rows, indices, counts      = jnp.unique(initial_condition_C_all, axis=0, return_index=True, return_counts=True)

            unique_intial_conditions          = unique_rows[jnp.argsort(indices)]
            unique_T                          =  T[indices[jnp.argsort(indices)]]

 
            sorted_unique_intial_conditions = jnp.zeros_like(unique_intial_conditions)

            # for i in range(0, len(self.initial_species)):
            #     sorted_unique_intial_conditions =  sorted_unique_intial_conditions.at[:,self.indices[i]].set(unique_intial_conditions[:,i])

            # solves ODE, sys_odes_ is rate equation (dy/dt = f(y,t))
            C_solutions = odeint(self.sys_odes_, unique_intial_conditions, time, params, unique_T)
            # C_solutions = odeint(self.sys_odes_, sorted_unique_intial_conditions, time, params, unique_T)

            combined_array = jnp.zeros((len(time), 0, self.num_species))

            # expand back to full dataset
            for i in range(self.unique_intial_conditions_num):
                array_helper = jnp.repeat(C_solutions[:,i, :], self.list_n_values[i], axis=0).reshape((self.time.shape[0], self.list_n_values[i], self.num_species))
                combined_array = jnp.concatenate((combined_array, array_helper), axis=1)


            C_solutions  = combined_array

            
            return C_solutions

    """
        General-purpose ODE simulation for symbolic chemical kinetics, returning full time-course results.

        This method is designed for symbolic model building or diagnostics. It computes forward and
        backward rate constants using Arrhenius equations, constructs rate matrices based on a
        specified ordering of species, and integrates the system using `odeint`.

        Unlike the previous two methods, this function does not use the de-duplication trick
        for repeated initial conditions. It focuses on clarity and flexibility for simulation.

        Args:
            X (jnp.array): Input array of shape (N, M) or (M,), where columns/entries are [T, C0_1, ..., C0_k].
            params (dict): Dictionary of kinetic parameters, including k_f, k_b, E_f, E_b for each reaction.
            time (jnp.array): Array of time points to evaluate the system over.
            initial_species (list): List of species provided in `X` (used to align with internal species order).

        Returns:
            Tuple[jnp.array, list]: 
                - Array of shape (time_points, N, num_species) with ODE solutions.
                - List of species names ordered as in the output concentrations.
        """
    def simulate_symbolic_ode_(self, X: jnp.array, params: dict, time: jnp.array, initial_species: list)-> jnp.array:
        """
        Generates a system of ODEs and solves it.
        Args: 
            X : jnp.array of [temperature, C concetration vectors, time]
            params : a dict of all the kinetic parameters
            time : jnp.array of time points 

        Returns:
            jnp.array : output of the ODE solver | objective function needs to be editted here 
        """


        # i don't like this "if", we need to re-do it! - YJ
        if X.ndim == 1:
            initial_condition_C     = X[1:]
            T                       = X[0].astype(int)
            num_species_padding     = self.num_species - len(initial_condition_C)
            initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((num_species_padding))])[jnp.newaxis, :]



        else:
            initial_condition_C    = X[:, 1:]
            T                      = X[:,0]
            num_species_padding = self.num_species - initial_condition_C.shape[1]
            initial_condition_C_all = jnp.concatenate([initial_condition_C, jnp.zeros((initial_condition_C.shape[0], num_species_padding))], axis=1)

        column_names  = initial_species



        # name_combined = list(set(OrderedDict.fromkeys(column_names + list(self.all_species - set(column_names)))))
        name_combined = initial_species + [name for name in self.all_species if name not in initial_species]
        name_combined = list(dict.fromkeys(name_combined))
        T_broadcasted = T.reshape((-1, 1))


        k_f_params = jnp.array(list(map(lambda i: params.get(f'k{i + 1}_f', 0.0), range(self.num_reactions))))
        k_b_params = jnp.array(list(map(lambda i: params.get(f'k{i + 1}_b', 0.0), range(self.num_reactions))))
        E_f_params = jnp.array(list(map(lambda i: params.get(f'E{i + 1}_f', 0.0), range(self.num_reactions))))
        E_b_params = jnp.array(list(map(lambda i: params.get(f'E{i + 1}_b', 0.0), range(self.num_reactions))))

        k_f_params = k_f_params
        k_b_params = k_b_params
        E_f_params = E_f_params
        E_b_params = E_b_params



        R = 8.314   #unit of gas constant needs to be adjusted if the E is in different unit ! 
        k_f = k_f_params * jnp.exp((-E_f_params /R) * ((1/(T_broadcasted + 273.15))-(1/(self.T_ref[0]))))
        k_b = k_b_params * jnp.exp((-E_b_params /R) * ((1/(T_broadcasted + 273.15))-(1/(self.T_ref[0]))))

   

        rate_constant_matrix = jnp.column_stack([k_f, k_b])


        k_matrix_coeff_f = jnp.array(list(map(lambda col: self.ode_k_type_matrix_f.get(col, 0), name_combined)), dtype=int)
        k_matrix_coeff_b = jnp.array(list(map(lambda col: self.ode_k_type_matrix_b.get(col, 0), name_combined)), dtype=int)
        reaction_order_f = jnp.array(list(map(lambda col: self.reaction_order_dict_f.get(col, 0), name_combined)), dtype=int)
        reaction_order_b = jnp.array(list(map(lambda col: self.reaction_order_dict_b.get(col, 0), name_combined)), dtype=int)
        k_matrix_coeff        = jnp.concatenate([k_matrix_coeff_f,  k_matrix_coeff_b], axis=1)
        reaction_order_matrix = jnp.concatenate([reaction_order_f,  reaction_order_b], axis=1)
        
        
        name_to_index = {name: index for index, name in enumerate(name_combined)}
        indices = [name_to_index[name] for name in initial_species if name in name_to_index]
        sorted_initial_condition_C_all = jnp.zeros_like(initial_condition_C_all)

        # for i in range(0, len(initial_species)):
        #    sorted_initial_condition_C_all =  sorted_initial_condition_C_all.at[:,indices[i]].set(initial_condition_C_all[:,i])




        C_solutions = odeint(self.sys_odes_ODE_solver_, initial_condition_C_all, time, k_matrix_coeff, rate_constant_matrix, reaction_order_matrix)

        # C_solutions = odeint(self.sys_odes_ODE_solver_, sorted_initial_condition_C_all, time, k_matrix_coeff, rate_constant_matrix, reaction_order_matrix)

        return C_solutions, name_combined
    

    def sys_odes_ODE_solver_(self, C: jnp.array, time: jnp.array, k_matrix_coeff: jnp.array, rate_constant_matrix: jnp.array, reaction_order_matrix:jnp.array)->jnp.array:
        """
        generates dC/dt arrays for ODE intergator
        
        ***This is not recommended for NumPyro funciton defintion. 



        Returns:
            Args:
                C : jnp.array - initial concentrations
                time : jnp.array 
                k_matrix_coeff : 2D jnp.array of k_matrix for reactions
                rate_constat_matrix: 2D jnp.array of stoichiometric coefficients
                reaction_order_matrix: jnp.array of reaction rate constants 

            Out: 
                dC/dt : jnp.array
        """
        dC_dt          = jnp.zeros_like(C)

        helper_new_1   = C[:, jnp.newaxis]


        helper_new_2   = jnp.transpose(helper_new_1, (0, 2, 1))
        helper_new_3   = jnp.power(helper_new_2, reaction_order_matrix)  

        helper_new_4   = jnp.prod(helper_new_3, axis=1)
        helper_new_5   = rate_constant_matrix * helper_new_4
        dC_dt = jnp.sum(jnp.transpose(helper_new_5[:, :, None], (0, 2, 1)) * k_matrix_coeff[None, :, :], axis=-1)
        
        return dC_dt