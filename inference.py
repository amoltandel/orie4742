import numpy as np
import sys

#done
def compute_messages(G, F):
    '''
    create messages from each node to other
    :param stack_order:
    :param G: Graph matrix
    :param F: Factor matrix (Potential functions)
    :return: factor-to-variable(fv) and variable-to-factor(vf) messages
    '''
    num_vars, num_facts = G.shape

    fv = np.ones((num_facts, num_vars, 2))
    vf  =np.ones((num_vars, num_facts, 2))

    v_sent = np.zeros((num_vars, num_facts)).astype(int)
    f_sent = np.zeros((num_facts, num_vars)).astype(int)

    leaf_facts = np.argwhere(np.sum(G, axis=0).flatten() == 1).flatten()
    leaf_vars = np.argwhere(np.sum(G, axis=1).flatten() == 1).flatten()

    for elf in leaf_facts:
        neighbors = get_fact_neighbors(G, elf)
        for each_n in neighbors:
            fv[elf, each_n, :] = F[elf, each_n, :]
            f_sent[elf, each_n] = 1

    for elv in leaf_vars:
        neighbors = get_var_neighbors(G, elv)
        for each_n in neighbors:
            v_sent[elv, each_n] = 1
    condition = np.any(v_sent != G) or np.any(f_sent != G.T)
    while condition:
        for each_factor in range(num_facts):
            for each_var in range(num_vars):
                if each_factor in get_var_neighbors(G, each_var) and v_sent[each_var, each_factor] != 1 and dependencies_vf(each_var, each_factor, f_sent, G):
                    vf[each_var, each_factor, :] = compute_vf_message(each_var, each_factor, fv, G)
                    v_sent[each_var, each_factor] = 1

                if each_var in get_fact_neighbors(G, each_factor) and f_sent[each_factor, each_var] != 1 and dependencies_fv(each_factor, each_var, v_sent, G):
                    fv[each_factor, each_var, :] = compute_fv_message(each_factor, each_var, vf, G, F)
                    f_sent[each_factor, each_var] = 1
        condition = np.any(v_sent != G) or np.any(f_sent != G.T)
    print "Variable to factor"
    print vf
    print "Factor to variable"
    print fv
    return fv, vf

#done
def compute_fv_message(factor, var, vf, G, F):
    all_neighbors = np.argwhere(G[:, factor] == 1).flatten()
    incoming_message_vars = set(all_neighbors) - set([var])
    incoming_message_vars = np.array(list(incoming_message_vars)).astype(int)
    product_of_messages = np.ones((2, ))

    for imv in incoming_message_vars:
        product_of_messages *= vf[imv, factor, :]
    result = None
    if var > incoming_message_vars[0]:
        answer = product_of_messages * (np.array(F[factor, :, :]).T)
        answer = answer.T
        result = np.sum(answer, axis=0)
    else:
        answer = product_of_messages * (np.array(F[factor, :, :]))
        result = np.sum(answer, axis=1)

    return result

# done
def compute_vf_message(each_var, each_factor, fv, G):
    neighbors = set(np.argwhere(G[each_var, :] == 1).flatten())
    incoming_message_fact = neighbors - set([each_factor])
    incoming_message_fact = np.array(list(incoming_message_fact)).astype(int)
    answer = np.ones((2, ))
    for imf in incoming_message_fact:
        answer *= fv[imf, each_var]
    return answer


# done
def dependencies_fv(each_fact=0, each_var=0, v_sent=np.array([]), G=np.array([])):
    neighbors = set(np.argwhere(G[:, each_fact] == 1).flatten())
    incoming_message_vars = neighbors - set([each_var])
    incoming_message_vars = np.array(list(incoming_message_vars)).astype(int)
    deps = v_sent[incoming_message_vars, each_fact].flatten()
    return np.all(deps == 1)


# done
def dependencies_vf(each_var=0, each_fact=0, f_sent=np.array([]), G=np.array([])):
    neighbors = set(np.argwhere(G[each_var, :] == 1).flatten())
    incoming_message_facts = neighbors - set([each_fact])
    incoming_message_facts = np.array(list(incoming_message_facts)).astype(int)
    deps = f_sent[incoming_message_facts, each_var].flatten()
    return np.all(deps == 1)

# done
def get_fact_neighbors(G, elf):
    return np.argwhere(G[:, elf] == 1).flatten()

# done
def get_var_neighbors(G, elv):
    return np.argwhere(G[elv, :] == 1).flatten()


# Done
def product_messages(G, fv, var_index, var_val):
    '''
    Compute the product of messages from adjacent factors for a variable 'var_index' at value 'var_val'
    Note: These values are unnormalized
    :param G: Graph matrix
    :param fv: Messages from factor to variable
    :param var_index: index of the variable
    :param var_val: value of the given variable
    :return: floating point value equivalent to product of messages from adjacent factors
    '''
    temp = 1.0
    neighbor_factors = np.argwhere(G[var_index, :] == 1).astype(int).flatten()
    for factor in neighbor_factors:
        temp *= fv[factor, var_index, var_val]
    return temp


# Done
def computeMarginals(G,F):
    '''
    Computes the marginal for each variable
    :param G: Graph matrix
    :param F: Factor values (Potential function)
    :return: a numpy array of shape (G.shape[0], 2) containing marginals for each variable
    '''
    B = np.ones((G.shape[0], 2))
    fv, vf = compute_messages(G, F)
    for var_index in range(G.shape[0]):
        for var_val in [0, 1]:
            B[var_index, var_val] = product_messages(G, fv, var_index, var_val)

    row_sum = np.sum(B, axis=1)
    B = B / row_sum.reshape((n, 1))
    return B


# Done - Perfect
def bruteForce(G, F):
    '''
    Bruteforce calculation for each unnormalized marginal
    :param G: Graph matrix
    :param F: Factor matrix
    :return: a tuple of size G.shape[0] x 2 containing marginals for each setting of parameters
    '''
    n, k = G.shape
    B = np.zeros((n, 2))

    # for each marginal variable
    for marg_variable in range(n):
        # initializations
        index_values = -1 * np.ones((n,))
        index_values = index_values.astype(int)

        #for each value of marginal variable
        for marg_var_val in [0, 1]:
            #set the value of the marginal variable in the corresponding index
            index_values[marg_variable] = marg_var_val

            # call the summation function over variables other than marg_variable
            B[marg_variable][marg_var_val] = summation(G=G, F=F,index_vals=index_values)

    row_sum = np.sum(B, axis=1)
    B = B / row_sum.reshape((n, 1))
    return B


# Done - Perfect
def product_factors(G, F, variable_values=np.array([])):
    '''
    Product of factors
    :param G: Graph matrix
    :param F: Factor matrix
    :param variable_values: a numpy array of size G.shape[0] which contains variables values for each variable
    Eg. variable_values = np.array([0, 1, 1, 0]) contains x_0 = 0, x_1 = 1 and so on
    :return: product of factors for a given setting of values
    '''
    num_var, num_factors = G.shape
    answer = 1.0
    variable_values = variable_values.astype(int)
    for i in range(num_factors):
        # find out the indices of the variables involved in factor i for G graph matrix
        variable_indices = np.argwhere(G[:, i] == 1).flatten()

        # get the values of those indices
        indices_values = variable_values[variable_indices]

        # create an index compatible to access F
        chosen_indices = tuple([i]) + tuple(indices_values)

        # update the answer by considering the currect factor setting
        answer *= F[chosen_indices]

    return answer


# Done - Perfect
def summation(G, F, index_vals):
    '''
    Calculates summation over indices whos values are -1
    :param G: Graph matrix
    :param F: Factor matrix (Potential functions)
    :param index_vals: An array contains values of variables. These variables are specified as indices
    :return: Computes the summation over the values in index_vals
    '''
    # create a duplicate of index values
    index_values = np.array(index_vals)

    # if all the variables are set to specific value, then compute the product
    if np.all(index_values > -1):
        return product_factors(G=G, F=F, variable_values=index_values)
    else:

        temp = 0.0
        # find an element whose values haven't been set. This means we are yet to compute summation over that variable
        unset_var = np.argwhere(index_values == -1).astype(int).flatten()[0]

        # for each summation variable in lst
        for bin_val in [0, 1]:
            # set the indices of the corresponding summation variable
            index_values[unset_var] = bin_val

            # calculate the sum
            temp += summation(G=G, F=F, index_vals=index_values)
        return temp


if __name__ == '__main__':
    n = 6
    k = 5
    G = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])
    F = np.ones((k, 2, 2))
    F[0, 1, 1] = 5
    F[1, 0, 1] = 0.5
    F[2, 0, 0] = 0
    F[3, 0, 0] = 2
    sys.setrecursionlimit(6 ** 6)
    B = computeMarginals(G=G, F=F)
    print "Sum-product result\n", B
    B_prime = bruteForce(G, F)
    print "Brute force result\n", B_prime
    print "B==B_prime? ", np.all(B==B_prime)
