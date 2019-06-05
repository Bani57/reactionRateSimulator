from chem_equation import *
from matplotlib import pyplot as plt


class ChemicalReaction:

    def __init__(self, equation, forward_rate, reverse_rate=0.0):
        self.equation = equation
        self.forward_rate = forward_rate
        self.reverse_rate = reverse_rate

    def calculate_simple_linear_reaction_model(self, initial_reactant_amounts, step_size=0.01, max_time=1):
        time = np.arange(0, max_time, step_size)
        reactant_coefficients = self.equation.reactant_coefficients
        product_coefficients = self.equation.product_coefficients
        num_reactants = len(reactant_coefficients)
        num_products = len(product_coefficients)
        k_r = self.forward_rate

        reactant_amounts = [
            [initial_reactant_amounts[i] - reactant_coefficients[i] * k_r * t for t in time]
            for i in range(num_reactants)]
        product_amounts = [
            [product_coefficients[i] * k_r * t for t in time]
            for i in range(num_products)]

        return time, reactant_amounts, product_amounts

    def calculate_simple_exponential_reaction_model(self, initial_reactant_amounts, step_size=0.01, max_time=1):
        time = np.arange(0, max_time, step_size)
        reactant_coefficients = self.equation.reactant_coefficients
        product_coefficients = self.equation.product_coefficients
        num_reactants = len(reactant_coefficients)
        num_products = len(product_coefficients)
        k_r = self.forward_rate

        min_reactant_ratio = min([initial_reactant_amounts[i] / reactant_coefficients[i] for i in range(num_reactants)])

        final_reactant_amounts = [initial_reactant_amounts[i] - reactant_coefficients[i] * min_reactant_ratio
                                  for i in range(num_reactants)]
        final_product_amounts = [product_coefficients[j] * min_reactant_ratio
                                 for j in range(num_products)]

        reactant_amounts = [
            [final_reactant_amounts[i] + reactant_coefficients[i] * min_reactant_ratio *
             np.exp((-1) * reactant_coefficients[i] * k_r * t) for t in time] for i in range(num_reactants)]
        product_amounts = [
            [final_product_amounts[j] * (1 - np.exp((-1) * product_coefficients[j] * k_r * t)) for t in time]
            for j in range(num_products)]

        return time, reactant_amounts, product_amounts

    def calculate_equilibrium_exponential_reaction_model(self, initial_reactant_amounts, step_size=0.01, max_time=1):
        time = np.arange(0, max_time, step_size)
        reactant_coefficients = self.equation.reactant_coefficients
        product_coefficients = self.equation.product_coefficients
        num_reactants = len(reactant_coefficients)
        num_products = len(product_coefficients)
        k_r = self.forward_rate
        k_p = self.reverse_rate

        min_reactant_ratio = min([initial_reactant_amounts[i] / reactant_coefficients[i] for i in range(num_reactants)])
        min_product_ratio = 0
        current_reactant_amounts = list(initial_reactant_amounts)
        current_product_amounts = [0, ] * num_products
        reactant_amounts = [[] for _i in range(num_reactants)]
        product_amounts = [[] for _j in range(num_products)]

        for t in time:
            final_reactant_amounts = [initial_reactant_amounts[i]
                                      + reactant_coefficients[i] * (min_product_ratio - min_reactant_ratio)
                                      for i in range(num_reactants)]
            final_product_amounts = [product_coefficients[j] * (min_reactant_ratio - min_product_ratio)
                                     for j in range(num_products)]

            current_reactant_amounts = [final_reactant_amounts[i]
                                        - (final_reactant_amounts[i] - current_reactant_amounts[i])
                                        * np.exp((-1) * reactant_coefficients[i] * (k_r + k_p) * t)
                                        for i in range(num_reactants)]
            current_product_amounts = [final_product_amounts[j]
                                       - (final_product_amounts[j] - current_product_amounts[j])
                                       * np.exp((-1) * product_coefficients[j] * (k_r + k_p) * t)
                                       for j in range(num_products)]

            for i in range(num_reactants):
                reactant_amounts[i].append(current_reactant_amounts[i])
            for j in range(num_products):
                product_amounts[j].append(current_product_amounts[j])

            if k_p > 0:
                min_reactant_ratio = min([current_reactant_amounts[i] / reactant_coefficients[i]
                                          for i in range(num_reactants)])
                min_product_ratio = min([current_product_amounts[j] / product_coefficients[j]
                                         for j in range(num_products)])

        return time, reactant_amounts, product_amounts

    def plot_reaction(self, time, reactant_amounts, product_amounts):
        plt.figure(figsize=(16, 9), dpi=1280 // 16)
        plt.title("Plot of reaction flow", fontsize=24)
        linestyles = ['-', '--', '-.', ':']
        for i, reactant_amount in enumerate(reactant_amounts):
            plt.plot(time, reactant_amount, color="orange", linestyle=linestyles[i % 4])
        for j, product_amount in enumerate(product_amounts):
            plt.plot(time, product_amount, color="lightblue", linestyle=linestyles[j % 4])
        plt.xlabel("Time")
        plt.ylabel("Amount of substance")
        plt.legend(self.equation.reactant_formulas + self.equation.product_formulas, shadow=True, fontsize='x-large')
        plt.show()
