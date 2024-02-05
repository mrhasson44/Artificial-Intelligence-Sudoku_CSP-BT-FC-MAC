def read_file(row):
    """
    Read a specific row from a CSV file containing Sudoku data.

    Args:
        row (int): The row number to be read from the file.

    Returns:
        list: A 2D list representing a Sudoku grid.
    """
                #Here put the full path of sudoku_samples.csv in order to read it
    with open('C:\\Users\\user\\Desktop\\Computer Science\\Level 6\\AI\\Project\\sudoku_samples.csv', 'r', encoding='utf-8-sig') as file:
        for i, line in enumerate(file):
            if i == row:
                row = line.strip().split(',')
                table = row[0]
                break

    # Initialize an empty 9x9 grid (2D list)
    grid = []

    # Loop through rows (9 rows)
    for row_index in range(9):
        # Initialize an empty list for the row
        row_list = []

        # Loop through columns (9 columns)
        for col_index in range(9):
            char_index = row_index * 9 + col_index
            char = table[char_index]
            row_list.append(int(char))

        # Append the row to the 2D grid
        grid.append(row_list)
    # print(grid)
    return grid

def generate_CSP(sudoku_grid):
    """
    Generate a Constraint Satisfaction Problem (CSP) for solving Sudoku.
    Variables: We have 81 variables, one for each cell in the 9x9 grid.
    Domains: Each variable has a domain {1, 2, 3, 4, 5, 6, 7, 8, 9}, as it can take on any of these values.
    Constraints: The constraints are formulated as follows:
        Row Constraints: For each row, we have a constraint that ensures that all variables in that row have distinct values.
        Column Constraints: Similarly, for each column.
        Sub-Grid Constraints: For each 3x3 sub-grid, we have a constraint to ensure distinct values within that sub-grid.

    Args:
        sudoku_grid (list): A 2D list representing a Sudoku grid.

    Returns:
        dict: A CSP representation containing variables, domains, and neighbors.
    """

    N = 9  # Sudoku grid size (9x9)

    def get_neighbors(row, col):
        neighbors = []
        for i in range(N):
            if i != col:
                neighbors.append((row, i))
            if i != row:
                neighbors.append((i, col))
        for i in range((row // 3) * 3, (row // 3 + 1) * 3):
            for j in range((col // 3) * 3, (col // 3 + 1) * 3):
                if i != row and j != col:
                    neighbors.append((i, j))
        return neighbors

    # Create the variable list (9x9 grid)
    variables = [(row, col) for row in range(N) for col in range(N)]

    # Create the domain for each variable
    domains = {(row, col): list(range(1, 10)) for row in range(N) for col in range(N)}

    # Find the neighbors for each variable
    neighbors = {(row, col): get_neighbors(row, col) for row in range(N) for col in range(N)}

    # Initialize the CSP dictionary
    csp = {
        "variables": variables,
        "domains": domains,
        "neighbors": neighbors
    }

    for row in range(N):
        for col in range(N):
            given_value = sudoku_grid[row][col]
            if given_value != 0:
                csp["domains"][row, col] = [given_value]

    return csp

def find_unassigned_variable(assignments):
    """
    Finds an unassigned variable in a Constraint Satisfaction Problem (CSP).

    Args:
        assignments (dict): A dictionary representing the current assignments in a CSP.

    Returns:
        tuple or None: A tuple (row, col) representing the unassigned variable, or None if all variables are assigned.

    This function iterates through the variables in a CSP and returns the first unassigned variable found.
    If all variables are assigned, it returns None.
    """
    for var in assignments:
        if assignments[var] == 0:
            return var
    return None

def is_valid_assignment(assignments, row, col, num):
    """
    Checks if assigning a number 'num' to a cell at position (row, col) is a valid assignment.

    Args:
        assignments (dict): A dictionary representing the current assignments in a Sudoku puzzle.
        row (int): The row of the cell to be checked.
        col (int): The column of the cell to be checked.
        num (int): The number to be assigned to the cell.

    Returns:
        bool: True if the assignment is valid, False otherwise.

    This function checks if the assignment is valid by ensuring that 'num' is not already present in the same row,
    column, or the 3x3 subgrid containing the cell (row, col).
    """
    for i in range(9):
        if assignments.get((row, i), 0) == num:
            return False
        if assignments.get((i, col), 0) == num:
            return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if assignments.get((i, j), 0) == num:
                return False

    return True


def copy_csp(csp):
    """
    Create a deep copy of a Constraint Satisfaction Problem (CSP).

    Args:
        csp (dict): The original CSP to be copied.

    Returns:
        dict: A new CSP with copied variables, domains, and neighbors.

    This function creates a deep copy of the input CSP by creating a new dictionary with copied variables, domains, and neighbors.
    It ensures that the original CSP is not modified when making changes to the copied CSP.
    """
    return {
        "variables": csp["variables"][:],
        "domains": {var: csp["domains"][var][:] for var in csp["variables"]},
        "neighbors": {var: csp["neighbors"][var][:] for var in csp["variables"]},
    }

#----------------------------------------------------------------------------------
#-------------------------------Backtrack Algorithm--------------------------------
#----------------------------------------------------------------------------------

def solve_backtrack_csp(csp):
    """
    Solve a Sudoku CSP using a backtracking algorithm.

    Args:
        csp (dict): A CSP representation containing variables, domains, and neighbors.

    Returns:
        tuple: A tuple containing the solved Sudoku grid and the number of expanded nodes.
    """

    def find_unassigned_variable(assignments, csp):
        """
        Finds an unassigned variable in a Constraint Satisfaction Problem (CSP) using MRV heuristic.

        Args:
            assignments (dict): A dictionary representing the current assignments in a CSP.
            csp (dict): The Constraint Satisfaction Problem represented as a dictionary with "neighbors" and "domains" attributes.

        Returns:
            tuple or None: A tuple (row, col) representing the unassigned variable, or None if all variables are assigned.

        This function iterates through the variables in a CSP and returns the unassigned variable with the minimum remaining values (MRV).
        If all variables are assigned, it returns None.
        """
        unassigned_vars = [var for var in csp["variables"] if assignments[var] == 0]
        if not unassigned_vars:
            return None
        return min(unassigned_vars, key=lambda var: len(csp["domains"][var]))

    expanded_nodes = 0 # Initialize the expanded nodes count as a regular variable
    
    def backtrack(assignments):
        nonlocal expanded_nodes

        if all(assignments[var] != 0 for var in csp["variables"]):
            return assignments

        var = find_unassigned_variable(assignments, csp)

        if var is None:
            return assignments
        row, col = var

        for num in csp["domains"][var]:
            expanded_nodes += 1  # Increment the count of expanded nodes
            if is_valid_assignment(assignments, row, col, num):
                assignments[var] = num
                result = backtrack(assignments)
                if result:
                    return result
                assignments[var] = 0

        return None

    N = 9  # Sudoku grid size (9x9)
    assignments = {(row, col): 0 for row in range(N) for col in range(N)}
    
    solution = backtrack(assignments)

    if solution:
        result = [[0 for _ in range(N)] for _ in range(N)]
        for var, value in solution.items():
            result[var[0]][var[1]] = value
        return result, expanded_nodes
    else:
        return "No solution", expanded_nodes

#----------------------------------------------------------------------------------
#--------------------------Forward Checking Algorithm------------------------------
#----------------------------------------------------------------------------------

def solve_forward_csp(csp):
    """
    Solve a Sudoku CSP using a backtracking algorithm with forward checking.

    Args:
        csp (dict): A CSP representation containing variables, domains, and neighbors.

    Returns:
        tuple: A tuple containing the solved Sudoku grid and the number of expanded nodes.
    """

    expanded_nodes = 0  # Initialize the expanded nodes count as a regular variable

    def forward_checking(assignments, csp, var, value):
        """
        Apply forward checking to update the domains of neighboring variables
        after assigning a value to a variable in a Constraint Satisfaction Problem (CSP).

        Args:
            assignments (dict): A dictionary of variable assignments in the CSP.
            csp (dict): The Constraint Satisfaction Problem represented as a dictionary
                        with "neighbors" and "domains" attributes.
            var (str): The variable for which a value is being assigned.
            value: The value being assigned to the variable.

        Returns:
            bool: True if forward checking is successful (no domain wipeout), False otherwise.
        """
        neighbors = csp["neighbors"][var]
        for neighbor in neighbors:
            if assignments[neighbor] == 0:
                if value in csp["domains"][neighbor]:
                    csp["domains"][neighbor].remove(value)
                    if len(csp["domains"][neighbor]) == 0:
                        return False
        return True


    def backtrack_with_forward_checking(assignments, csp):
        nonlocal expanded_nodes  # Declare expanded_nodes as non-local

        if all(assignments[var] != 0 for var in csp["variables"]):
            return assignments

        var = find_unassigned_variable(assignments)

        if var is None:
            return assignments
        row, col = var

        for num in csp["domains"][var][:]:
            expanded_nodes += 1  # Increment the expanded nodes count
            if is_valid_assignment(assignments, row, col, num):
                assignments[var] = num

                # Create a copy of the CSP
                csp_copy = copy_csp(csp)

                if forward_checking(assignments, csp_copy, var, num):
                    result = backtrack_with_forward_checking(assignments, csp_copy)
                    if result:
                        return result

                assignments[var] = 0

        return None

    N = 9  # Sudoku grid size (9x9)
    assignments = {(row, col): 0 for row in range(N) for col in range(N)}

    solution = backtrack_with_forward_checking(assignments, csp)

    if solution:
        result = [[0 for _ in range(N)] for _ in range(N)]
        for var, value in solution.items():
            result[var[0]][var[1]] = value
        return result, expanded_nodes  # Return the expanded nodes count as a regular variable
    else:
        return "No solution", expanded_nodes

#----------------------------------------------------------------------------------
#------------------------Maintaining Arc-Consistency Algorithm---------------------
#----------------------------------------------------------------------------------

def solve_mac_csp(csp):
    """
    Solve a Constraint Satisfaction Problem (CSP) using the MAC (Maintaining Arc-Consistency) algorithm.

    Parameters:
    csp (dict): The CSP to be solved, represented as a dictionary.

    Returns:
    tuple: A solution grid or "No solution" and the number of expanded nodes.

    The MAC algorithm enforces arc-consistency to solve CSPs. It iteratively prunes inconsistent values from domains.
    If a solution is found, it returns the grid. Otherwise, it returns "No solution" and the expanded nodes count.
    """

    expanded_nodes = 0  # Initialize expanded nodes count

    def revise(csp):
        """
        Apply the revise operation to enforce arc-consistency in a Constraint Satisfaction Problem (CSP).

        Parameters:
        csp (dict): The CSP to be revised.

        Returns:
        bool: True if any domain is revised, indicating progress; False otherwise.

        This function checks each variable's domain and eliminates values that are inconsistent with their neighbors' domains.
        It updates the CSP in place and returns True if any revisions are made, indicating progress.
        """

        nonlocal expanded_nodes  # Declare expanded_nodes as non-local
        revised = False
        for var in csp["variables"]:
            if len(csp["domains"][var]) != 1:
                continue
            value = csp["domains"][var][0]
            for neighbor in csp["neighbors"][var]:
                if value in csp["domains"][neighbor]:
                    csp["domains"][neighbor].remove(value)
                    revised = True
                    expanded_nodes += 1  # Increment expanded nodes
        return revised

    def add_arcs_to_queue(csp, queue):
        """
        Add variable pairs to the queue for further consistency checks in a CSP.

        Parameters:
        csp (dict): The CSP containing variables and their neighbors.
        queue (list): The queue to which variable pairs are added.

        This function populates the queue with pairs of variables that need to be checked for arc-consistency.
        """

        for var in csp["variables"]:
            for neighbor in csp["neighbors"][var]:
                queue.append((var, neighbor))

    queue = []  # Initialize the queue

    add_arcs_to_queue(csp, queue)  # Add initial arcs to the queue

    while queue:
        var, neighbor = queue.pop(0)

        if revise(csp):
            if len(csp["domains"][var]) == 0:
                return "No solution", expanded_nodes

            for neighbor2 in csp["neighbors"][var]:
                if neighbor2 != neighbor:
                    queue.append((neighbor2, var))  # Reversed the order to ensure that we recheck the revised neighbor

    # Check if all variables are assigned
    if all(len(csp["domains"][var]) == 1 for var in csp["variables"]):
        solved_grid = [[csp["domains"][(row, col)][0] for col in range(9)] for row in range(9)]
        return solved_grid, expanded_nodes
    else:
        return "No solution", expanded_nodes



#----------------------------------------------------------------------------------

grid = read_file(14) # choose from 0 to 19 (there are 20 sudoku puzzle on the file)

csp = generate_CSP(grid)

backtrack_result, backtrack_expanded_nodes = solve_backtrack_csp(csp)
forward_result, forward_expanded_nodes = solve_forward_csp(csp)
mac_result, mac_expanded_nodes = solve_mac_csp(csp)

print("Backtrack Algorithm:\n")
if backtrack_result != "No solution":
    for row in backtrack_result:
        print(f"     {row}")
    print(f"     Backtrack expanded nodes: {backtrack_expanded_nodes}")
else:
    print("Backtrack No Solution")

print()

print("Forward Checking Algorithm:\n")
if forward_result != "No solution":
    for row in forward_result:
        print(f"     {row}")
    print(f"     Forward checking expanded nodes: {forward_expanded_nodes}")
else:
    print("Forward checking No Solution")

print()

print("Maintaining Arc-Consistency (MAC) Algorithm:\n")
if mac_result != "No solution":
    for row in mac_result:
        print(f"     {row}")
    print(f"     MAC expanded nodes: {mac_expanded_nodes}")
else:
    print("MAC No Solution")