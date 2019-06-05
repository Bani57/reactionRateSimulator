from re import split
import numpy as np


class ChemicalEquation:

    def __init__(self):
        self.reactants = []
        self.products = []
        self.reactant_formulas = []
        self.product_formulas = []
        self.reactant_coefficients = []
        self.product_coefficients = []

    def __parse_reagent_formula(self, reagent_formula):
        elements = split('[0-9]+', reagent_formula)
        elements.remove('')
        indexes = split('[A-Za-z]+', reagent_formula)
        indexes.remove('')
        reagent_dict = {}
        reagent_formula_cleaned = ""
        for element, index in zip(elements, indexes):
            reagent_dict[element] = int(index)
            reagent_formula_cleaned += element
            if int(index) > 1:
                reagent_formula_cleaned += index
        return reagent_dict, reagent_formula_cleaned

    def add_reactant(self, reactant_formula):
        reactant_dict, reactant_formula_cleaned = self.__parse_reagent_formula(reactant_formula)
        self.reactants.append(reactant_dict)
        self.reactant_formulas.append(reactant_formula_cleaned)

    def add_product(self, product_formula):
        product_dict, product_formula_cleaned = self.__parse_reagent_formula(product_formula)
        self.products.append(product_dict)
        self.product_formulas.append(product_formula_cleaned)

    def __str__(self):
        if len(self.reactant_coefficients) > 0 and len(self.product_coefficients) > 0:
            weighted_reactants = [str(coefficient) + " " + reactant_formula if coefficient != 1 else reactant_formula
                                  for coefficient, reactant_formula in
                                  zip(self.reactant_coefficients, self.reactant_formulas)]
            weighted_products = [str(coefficient) + " " + product_formula if coefficient != 1 else product_formula
                                 for coefficient, product_formula in
                                 zip(self.product_coefficients, self.product_formulas)]
            return " + ".join(weighted_reactants) + " = " + " + ".join(weighted_products)
        else:
            return " + ".join(self.reactant_formulas) + " = " + " + ".join(self.product_formulas)

    def balance_equation(self, verbose=False):
        reagents = self.reactants + self.products
        num_reagents = len(reagents)
        all_elements = set()
        for reagent in reagents:
            all_elements.update(reagent.keys())
        all_elements = list(all_elements)
        num_elements = len(all_elements)
        num_extra_variables = num_reagents - num_elements

        reaction_matrix = np.zeros((num_elements, num_reagents), dtype="int")
        target_vector = np.zeros(num_elements, dtype="int")
        for j, reagent in enumerate(reagents):
            for element, index in reagent.items():
                i = all_elements.index(element)
                reaction_matrix[i][j] = index if reagent in self.reactants else (-1) * index

        for i in range(num_extra_variables):
            column_to_delete = reaction_matrix[:, i]
            reaction_matrix = np.delete(reaction_matrix, i, axis=1)
            target_vector = target_vector - column_to_delete

        k = 0
        if num_extra_variables == 0:
            while np.linalg.det(reaction_matrix) == 0:
                column_data_to_subtract = reaction_matrix[1:, k]
                reaction_matrix = np.delete(reaction_matrix, k, 0)
                reaction_matrix = np.delete(reaction_matrix, k, 1)
                target_vector = target_vector[1:] - column_data_to_subtract
                k += 1
            num_extra_variables = k

        coefficients = np.ones(num_reagents, dtype="float")
        coefficients[num_extra_variables:] = np.matmul(np.linalg.inv(reaction_matrix), target_vector)

        if all(coefficient == 0 for coefficient in coefficients):
            coefficients = np.ones(num_reagents, dtype="float")

        if verbose:
            print("A = \n", reaction_matrix)
            print("b = ", target_vector)

        self.reactant_coefficients = [round(coefficient, 2) for coefficient in coefficients[:len(self.reactants)]]
        self.product_coefficients = [round(coefficient, 2) for coefficient in coefficients[len(self.reactants):]]
