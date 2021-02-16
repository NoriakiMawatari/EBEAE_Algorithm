# Extended Blind Endmembers and Abundances Extraction (EBEAE) Algorithm
import logging as log
import numpy as np
import time
from scipy import linalg


def performance(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        stop = time.time()
        # log.info(f'Function {function.__name__} took {stop - start} secs.')
        return stop - start, result
    return wrapper


def args_consistency(kwargs: dict) -> dict:
    given_args = list(kwargs.keys())
    default_args = {
        'y_matrix': np.array([]),
        'n_order': 2,
        'parameters': {
            'initcond': 1,
            'rho': 0.1,
            'lambda_var': 0,
            'epsilon': 1e-3,
            'maxiter': 20,
            'downsampling': 0.5,
            'parallel': 0,
            'normalization': 1,
            'display': 0},
        'oae': 0
    }
    if len(given_args) in [4, 5]:
        default_args['Po'] = np.array([])
    for arg in given_args:
        if arg in default_args.keys():
            default_args[arg] = kwargs[arg]
    return default_args


def hyper_parameters_inspection(parameters: dict) -> dict:
    if parameters['initcond'] not in [1, 2, 3, 4]:
        print("The initialization procedure of endmembers matrix is 1,2,3 or 4!")
        print("The default value is considered!")
        parameters['initcond'] = 1
    if parameters['rho'] < 0:
        print("The regularization weight rho cannot be negative")
        print("The default value is considered!")
        parameters['rho'] = 0.1
    if parameters['lambda_var'] < 0 or parameters['lambda_var'] >= 1:
        print("The entropy weight lambda is limited to [0,1)")
        print("The default value is considered!")
        parameters['lambda_var'] = 0
    if parameters['epsilon'] < 0 or parameters['epsilon'] > 0.5:
        print("The threshold epsilon can't be negative or > 0.5")
        print("The default value is considered!")
        parameters['epsilon'] = 1e-3
    if parameters['maxiter'] < 0 or parameters['maxiter'] > 100:
        print("The upper bound maxiter can't be negative or > 100")
        print("The default value is considered!")
        parameters['maxiter'] = 20
    if parameters['downsampling'] < 0 or parameters['downsampling'] > 1:
        print("The downsampling factor cannot be negative or > 1")
        print("The default value is considered!")
        parameters['downsampling'] = 0.5
    if parameters['parallel'] not in [0, 1]:
        print("The parallelization parameter is 0 or 1")
        print("The default value is considered!")
        parameters['parallel'] = 0
    if parameters['normalization'] not in [0, 1]:
        print("The normalization parameter is 0 or 1")
        print("The default value is considered!")
        parameters['normalization'] = 1
    if parameters['display'] not in [0, 1]:
        print("The display parameter is 0 or 1")
        print("The default value is considered")
        parameters['display'] = 0
    return parameters


def display_details(n_order: int, oae: int, initcond: int) -> None:
    log.info("EBEAE Linear Unmixing")
    log.info(f"Model Order = {n_order}")
    if oae == 1:
        log.info("Only the abundances are estimated from <p_origin>\n")
    elif oae == 0 and initcond == 0:
        log.info("The endmembers matrix is initialized externally by matrix <p_origin>\n")
    elif oae == 0 and initcond == 1:
        log.info("<p_origin> is constructed based on the maximum cosine difference from mean measurement\n")
    elif oae == 0 and initcond == 2:
        log.info(
            "<p_origin> is constructed based on the maximum and minimum energy, and largest difference from them\n")
    elif oae == 0 and initcond == 3:
        log.info("<p_origin> is constructed based on the PCA selection + Rectified Linear Unit\n")
    elif oae == 0 and initcond == 4:
        log.info("<p_origin> is constructed based on the ICA selection (FOBI) + Rectified Linear Unit\n")


@performance
def abundance(y_matrix: np.ndarray, p_matrix: np.ndarray, lambda_var: float, parallel: int) -> np.ndarray:
    """
    Estimation of Optimal Abundances in Linear Mixture Model
    INPUTS:
    y_matrix = matrix of measurements
    p_matrix = matrix of end-members
    lambda_var =  entropy weight in abundance estimation in (0,1)
    parallel = implementation in parallel of the estimation
    OUTPUTS:
    abundance_matrix = abundances matrix
    - Daniel U. Campos-Delgado (September/2020)
    """
    try:
        # Check arguments dimensions
        num_error = 0
        m_rows, n_cols = y_matrix.shape
        p_n_cols = p_matrix.shape[1]
        abundance_matrix = np.zeros((p_n_cols, n_cols))
        if p_matrix.shape[0] != m_rows:
            log.error("The number of rows in <y_matrix> and <p_matrix> does not match")
            num_error = 1
    except Exception as error:
        log.error(f"ERROR [Abundance function | arguments dimensions]: {error}")
    try:
        # Compute fixed vectors and matrices
        c_vector = np.ones((p_n_cols, 1))
        d_lagrange = 1  # Lagrange Multiplier for equality restriction
        g_o = p_matrix.T @ p_matrix
        w, v = np.linalg.eig(g_o)
        l_min = np.amin(w)
        g = g_o - np.eye(p_n_cols) * l_min * lambda_var
        while (1 / np.linalg.cond(g, 1)) < 1e-6:
            lambda_var = lambda_var / 2
            g = g_o - np.eye(p_n_cols) * l_min * lambda_var
            if lambda_var < 1e-6:
                log.error("Unstable numerical results in abundances estimation, update rho!!")
                num_error = 1
        g_i = np.linalg.pinv(g)
        term_1 = g_i @ c_vector
        term_2 = c_vector.T @ term_1
    except Exception as error:
        log.error(f"ERROR [Abundance function | computed fixed vectors and matrices]: {error}")
    try:
        # Start Computation of Abundances
        for k_pos in range(n_cols):
            y_k = np.c_[y_matrix[:, k_pos]]
            b_y_k = float(y_k.T @ y_k)
            b_k = p_matrix.T @ y_k
            # Compute Optimal Unconstrained Solution
            d_k = np.divide((b_k.T @ term_1) - 1, term_2)
            a_k = g_i @ (b_k - c_vector @ d_k)
            # Check for Negative Elements
            if float(sum(a_k >= 0)) != p_n_cols:
                index_set = np.zeros((1, p_n_cols))
                while float(sum(a_k < 0)) != 0:
                    index_set = np.where(a_k < 0, 1, index_set.T).reshape(1, p_n_cols)
                    index_length = len(np.where(index_set == 1)[1])
                    q_pos = p_n_cols + 1 + index_length
                    gamma_matrix = np.zeros((q_pos, q_pos))
                    beta_vector = np.zeros((q_pos, 1))
                    gamma_matrix[:p_n_cols, :p_n_cols] = g / b_y_k
                    gamma_matrix[:p_n_cols, p_n_cols] = c_vector.T
                    gamma_matrix[p_n_cols, :p_n_cols] = c_vector.T
                    cont = 0
                    if p_n_cols >= 2:
                        if bool(index_set[:, 0] != 0):
                            cont += 1
                            gamma_matrix[0, p_n_cols + cont] = 1
                            gamma_matrix[p_n_cols + cont, 0] = 1
                        if bool(index_set[:, 1] != 0):
                            cont += 1
                            gamma_matrix[1, p_n_cols + cont] = 1
                            gamma_matrix[p_n_cols + cont, 1] = 1
                        if p_n_cols >= 3:
                            if bool(index_set[:, 2] != 0):
                                cont += 1
                                gamma_matrix[2, p_n_cols + cont] = 1
                                gamma_matrix[p_n_cols + cont, 2] = 1
                            if p_n_cols == 4:
                                if bool(index_set[:, 3] != 0):
                                    cont += 1
                                    gamma_matrix[3, p_n_cols + cont] = 1
                                    gamma_matrix[p_n_cols + cont, 3] = 1
                    beta_vector[:p_n_cols, :] = b_k / b_y_k
                    beta_vector[p_n_cols, :] = d_lagrange
                    delta = np.linalg.solve(gamma_matrix, beta_vector)
                    a_k = delta[:p_n_cols]
                    a_k = np.where(abs(a_k) < 1e-9, 0, a_k)
            abundance_matrix[:, k_pos] = np.c_[a_k].T
    except Exception as error:
        log.error(f"ERROR [Abundance funtion | computation of abundances]: {error}")
    return abundance_matrix, num_error


@performance
def endmember(y_matrix: np.ndarray, a_matrix: np.ndarray, rho: int, normalization: int) -> np.ndarray:
    """
    p_matrix = endmember(Y,A,rho,normalization)
    Estimation of Optimal End-members in Linear Mixture Model
    Input Arguments:
    Y = Matrix of measurements
    A =  Matrix of abundances
    rho = Weighting factor of regularization term
    normalization = normalization of estimated profiles (0=NO or 1=YES)
    Output Argument:
    p_matrix = Matrix of end-members
    Daniel U. Campos-Delgado
    September/2020
    """
    try:
        # Check arguments dimensions
        a_rows, a_cols = a_matrix.shape
        y_rows, y_cols = y_matrix.shape
        num_error = 0
        r = sum(a_rows - np.array(range(1, a_rows)))
        w = np.tile((1 / y_cols / sum(y_matrix ** 2)), [a_rows, 1]).T
        if y_matrix.shape[1] != a_cols:
            log.error("The number of columns in <y_matrix> and <a_matrix> does not match")
            num_error = 1
        o_matrix = (a_rows * np.eye(a_rows) - np.ones((a_rows, a_rows)))
        n1 = (np.ones((a_rows, 1)))
        m1 = (np.ones((y_rows, 1)))
    except Exception as error:
        log.error(f"ERROR [Endmember function | arguments dimensions]: {error}")
    try:
        # Construct Optimal Endmembers Matrix
        term_0 = (a_matrix @ (w * a_matrix.T) + rho * np.divide(o_matrix, r))
        while 1 / np.linalg.cond(term_0, 1) < 1e-6:
            rho = rho / 10
            term_0 = (a_matrix @ (w * a_matrix.T) + rho * np.divide(o_matrix, r))
            if rho < 1e-6:
                log.error("Unstable numerical results in endmembers estimation, update rho!!")
                num_error = 1
        v = (np.eye(a_rows) @ np.linalg.pinv(term_0))
        term_2 = (y_matrix @ (w * a_matrix.T) @ v)
        if normalization == 1:
            term_1 = (np.eye(y_rows) - (1 / y_rows) * (m1 @ m1.T))
            term_3 = ((1 / y_rows) * m1 @ n1.T)
            p_estimation = term_1 @ term_2 + term_3
        else:
            p_estimation = term_2
    except Exception as error:
        log.error(f"ERROR [Endmember function | optimal endmember matrix construction]: {error}")
    try:
        # Evaluate and Project Negative Elements
        p_estimation = np.where(p_estimation < 0, 0, p_estimation)
        p_estimation = np.where(np.isnan(p_estimation), 0, p_estimation)
        p_estimation = np.where(np.isinf(p_estimation), 0, p_estimation)
        # Normalize Optimal Solution
        if normalization == 1:
            p_sum = np.sum(p_estimation, 0)
            p_matrix = p_estimation / np.tile(p_sum, [y_rows, 1])
        else:
            p_matrix = p_estimation
    except Exception as error:
        log.error(f"ERROR [Endmember function | negative elements evaluation and solution normalization]: {error}")
    return p_matrix, num_error


@performance
def ebeae(**kwargs: [np.ndarray, int, dict, np.ndarray, int]):
    """
    p_matrix, a_matrix, abundances_normalized, yh_matrix, a_Time, p_Time = ebeae(Yo, n, parameters, p_origin, oae)
    Estimation of Optimal Endmembers and Abundances in Linear Mixture Model
    Input Arguments:
      y_matrix = matrix of measurements (MxN)
      n_order = order of linear mixture model
      parameters = 9x1 vector of hyperparameters in EBEAE methodology
                 = [initicond, rho, lambda_var, epsilon, maxiter, downsampling, parallel, normalization, display]
          initcond = initialization of endmembers matrix {1,2,3,4}
                                    (1) Maximum cosine difference from mean measurement (default)
                                    (2) Maximum and minimum energy, and largest distance from them
                                    (3) PCA selection + Rectified Linear Unit
                                    (4) ICA selection (FOBI) + Rectified Linear Unit
          rho = regularization weight in endmember estimation (default rho=0.1)
          lambda_var = entropy weight in abundance estimation in [0,1) (default lambda_var=0)
          epsilon = threshold for convergence in ALS method (default epsilon=1e-3)
          maxiter = maximum number of iterations in ALS method (default maxiter=20)
          downsampling = percentage of random downsampling in endmember estimation [0,1) (default downsampling=0.5)
          parallel = implement parallel computation of abundances (0->NO or 1->YES) (default parallel=0)
          normalization = normalization of estimated end-members (0->NO or 1->YES) (default normalization=1)
          display = show progress of iterative optimization process (0->NO or 1->YES) (default display=0)
      p_origin = initial end-member matrix (Mxn)
      oae = only optimal abundance estimation with p_origin (0 -> NO or 1 -> YES) (default oae = 0)
    Output Arguments:
      p_matrix  = matrix of endmembers (Mxn)
      a_matrix  = scaled abundances matrix (nxN)
      abundances_normalized = abundances matrix normalized (nxN)
      yh_matrix = estimated matrix of measurements (MxN)
      a_Time = estimated time in abundances estimation
      p_Time = estimated time in endmembers estimation
    Daniel U. Campos Delgado
    July/2020
    """
    # Checking consistency of input arguments
    try:
        num_error = 0
        num_input_args = len(kwargs)
        if num_input_args == 0:
            raise Exception("Measurement matrix y_matrix_scaled <y_matrix> has to be used as argument.")
        args = args_consistency(kwargs)
        if args['n_order'] < 2:
            log.error("The order of the linear mixture model has to be greater than 2!")
            log.info("The default value n=2 is considered!")
            args['n_order'] = 2
        if len(args['parameters']) != 9:
            log.error("The length of parameters vector is not 9 !!")
            log.info("Default values of hyper-parameters are used instead")
        else:
            hyper_parameters = hyper_parameters_inspection(args['parameters'])
        if num_input_args in [4, 5]:
            if type(args['p_origin']) != np.ndarray:
                log.error("The initial end-members p_origin must be a matrix !!")
                log.info("The initialization is considered by the maximum cosine difference from mean measurement")
                hyper_parameters['initcond'] = 1
            else:
                if args['p_origin'].shape == (args['y_matrix'].shape[0], args['n_order']):
                    hyper_parameters['initcond'] = 0
                else:
                    log.error("The size of p_origin must be num_measurements x n!!")
                    log.info("The initialization is considered based on the input dataset")
                    hyper_parameters['initcond'] = 1
            if args['oae'] not in [0, 1]:
                log.error("The assignment of oae is incorrect!!")
                log.info("The initial end-members p_origin will be improved iteratively from a selected sample")
                args['oae'] = 0
            elif args['oae'] == 1 and hyper_parameters['initcond'] != 0:
                log.error("The initial end-members p_origin is not defined properly!")
                log.info("p_origin will be improved iteratively from a selected sample")
                args['oae'] = 0
        if num_input_args >= 6:
            log.error("The number of input arguments is 5 maximum")
            raise Exception("Please check the docstrings.")
    except Exception as error:
        log.error(f"ERROR [Consistency of input arguments step]: {error}")
    try:
        # Random downsampling
        if type(args['y_matrix']) != np.ndarray:
            log.error("The measurements matrix y_matrix_scaled has to be a matrix")
        num_measurements, num_time_samples = args['y_matrix'].shape
        if num_measurements > num_time_samples:
            raise Exception("The number of spatial measurements has to be larger to the number of time samples!")
        num_ts_downsampled = round(num_time_samples * (1 - hyper_parameters['downsampling']))
        samples_index = np.random.choice(num_time_samples, num_ts_downsampled, replace=False)
        y_matrix_scaled = args['y_matrix'][:, samples_index - 1]
    except Exception as error:
        log.error(f"ERROR [Random downsampling step]: {error}")
    try:
        # Normalization
        if hyper_parameters['normalization'] == 1:
            m_ym = np.sum(y_matrix_scaled, 0)
            m_ymo = np.sum(args['y_matrix'], 0)
        else:
            m_ym = np.ones((1, num_ts_downsampled), dtype=int)
            m_ymo = np.ones((1, num_time_samples), dtype=int)
        ym = y_matrix_scaled / np.tile(m_ym, [num_measurements, 1])
        ym_origin = args['y_matrix'] / np.tile(m_ymo, [num_measurements, 1])
        norm_ym = np.linalg.norm(ym, 'fro')
    except Exception as error:
        log.error(f"ERROR [Normalization step]: {error}")
    try:
        # Selection of Initial Endmembers Matrix
        if hyper_parameters['initcond'] in [1, 2]:
            if hyper_parameters['initcond'] == 1:
                index = 1
                p_origin = np.zeros((num_measurements, 1))
                p_max = np.mean(args['y_matrix'], axis=1)
                y_copy = args['y_matrix']
                p_origin[:, index - 1] = p_max
            elif hyper_parameters['initcond'] == 2:
                index = 1
                y1m = np.sum(abs(args['y_matrix']), 0)
                y_max, y_min = np.max(y1m), np.min(y1m)
                index_max, index_min = np.argwhere(y1m == y_max)[0][0], np.argwhere(y1m == y_min)[0][0]
                p_max, p_min = args['y_matrix'][:, index_max], args['y_matrix'][:, index_min]
                k_positions = args['y_matrix'].shape[1]
                i_i = np.arange(1, k_positions + 1)
                condition = np.logical_and(i_i != i_i[index_max], i_i != i_i[index_min])
                i_i = np.extract(condition, i_i)
                y_copy = args['y_matrix'][:, i_i - 1]
                p_origin = p_max
                index += 1
                p_origin = np.c_[p_origin, p_min]
            while index < args['n_order']:
                y_max = np.zeros((1, index))
                index_max = np.zeros((1, index), dtype=int)
                for j in range(index):
                    if j == 0:
                        for i in range(index):
                            e1m = np.around(np.sum(y_copy * np.tile(p_origin[:, i], [y_copy.shape[1], 1]).T, 0) /
                                            np.sqrt(np.sum(y_copy ** 2, 0)) / np.sqrt(np.sum(p_origin[:, i] ** 2, 0)),
                                            4)
                            y_max[j][i] = np.around(np.amin(abs(e1m)), 4)
                            index_max[j][i] = np.where(e1m == y_max[j][i])[0][0]
                ym_max = np.amin(y_max)
                im_max = np.where(y_max == ym_max)[1][0]
                i_imax = index_max[0][im_max]
                p_max = y_copy[:, i_imax]
                index += 1
                p_origin = np.c_[p_origin, p_max]
                i_i = np.arange(1, y_copy.shape[1] + 1)
                i_i = np.extract(i_i != i_imax + 1, i_i)
                y_copy = y_copy[:, list(i_i - 1)]
        elif hyper_parameters['initcond'] == 3:
            uu, s, vv = np.linalg.svd(ym.T, full_matrices=False)
            w = vv.T[:, :args['n_order']]
            p_origin = w * np.tile(np.sign(w.T @ np.ones((num_measurements, 1))).T, [num_measurements, 1])
        elif hyper_parameters['initcond'] == 4:
            yo_mean = np.mean(ym, axis=1)
            yo_norm = ym - np.tile(yo_mean, [num_ts_downsampled, 1]).T
            uu, s, vv = np.linalg.svd(yo_norm.T, full_matrices=False)
            s_diagonal = np.diag(s)
            yo_w = np.linalg.pinv(linalg.sqrtm(s_diagonal)) @ vv @ ym
            v, s, u = np.linalg.svd((np.tile(sum(yo_w * yo_w), [num_measurements, 1]) * yo_w) @ yo_w.T,
                                    full_matrices=False)
            w = vv.T @ linalg.sqrtm(s_diagonal) @ v[:args['n_order'], :].T
            p_origin = w * np.tile(np.sign(w.T @ np.ones((num_measurements, 1))).T, [num_measurements, 1])
        p_origin = np.where(p_origin < 0, 0, p_origin)
        p_origin = np.where(np.isnan(p_origin), 0, p_origin)
        p_origin = np.where(np.isinf(p_origin), 0, p_origin)
        if hyper_parameters['normalization'] == 1:
            m_po = np.sum(p_origin, 0)
            p_matrix = p_origin / np.tile(m_po, [num_measurements, 1])
        else:
            p_matrix = p_origin
    except Exception as error:
        log.error(f"ERROR [Selection of initial endmembers matrix step]: {error}")
    try:
        # Alternated Least Squares Procedure
        iteration, j, jp, a_time, p_time = [1, 1e5, 1e6, 0, 0]
        tic = time.time()
        if hyper_parameters['display'] == 1:
            display_details(args['n_order'], args['oae'], hyper_parameters['initcond'])
        while (jp - j) / jp >= hyper_parameters['epsilon'] and iteration < hyper_parameters['maxiter'] and \
                        args['oae'] == 0 and num_error == 0:
            time_a, outputs_a = abundance(ym, p_matrix, hyper_parameters['lambda_var'], hyper_parameters['parallel'])
            a_matrix, num_error = outputs_a
            a_time += time_a
            init_p_matrix = p_matrix
            if num_error == 0:
                time_e, outputs_e = endmember(ym, a_matrix, hyper_parameters['rho'], hyper_parameters['normalization'])
                p_matrix, num_error = outputs_e
                p_time += time_e
            jp = j
            j = np.linalg.norm(ym - p_matrix @ a_matrix, 'fro')
            if j > jp:
                p_matrix = init_p_matrix
                break
            if hyper_parameters['display'] == 1:
                log.info(f"Number of iteration = {iteration}")
                log.info(f"Percentage Estimation Error = {(100 * j) / norm_ym} %")
                log.info(f"Abundance estimation took {time_a}")
                log.info(f"Endmember estimation took {time_e}")
            iteration += 1
        if num_error == 0:
            time_a, outputs_a = abundance(ym_origin, p_matrix, hyper_parameters['lambda_var'], hyper_parameters['parallel'])
            a_matrix, num_error = outputs_a
            a_time += time_a
            toc = time.time()
            elap_time = toc - tic
            if hyper_parameters['display'] == 1:
                log.info(f"Elapsed Time = {elap_time} seconds")
            abundances_normalized = a_matrix
            a_scaled = a_matrix * np.tile(m_ymo, [args['n_order'], 1])
            yh_matrix = p_matrix @ a_scaled
        else:
            log.error("Please review the problem formulation, not reliable results")
            p_matrix = np.array([])
            a_scaled = np.array([])
            abundances_normalized = np.array([])
            yh_matrix = np.array([])
    except Exception as error:
        log.error(f"ERROR [Alternated Least Squares step]: {error}")
    return p_matrix, a_scaled, abundances_normalized, yh_matrix, a_time, p_time
