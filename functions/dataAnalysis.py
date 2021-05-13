from pyteomics import mass
import matplotlib.pyplot as plt
import pandas as pd

'''
The calculateMassUncertainty function takes a dataframe containing a processed spectrum with the following columns: formula, formula_mz, observed_mz.
It returns a dictionary with the uncertainty for each element calculated as the average of all the uncertainties for wich the element was present.
By default it will average the uncertainty based solely on the presence of the element.
If weighted is set to True, it will perform a weighted averged based on the number of elements in the compounds/ions.
Parameters:
processedSpectrum: pandas.DataFrame
    processed information about the spectrum containing formula, formula_mz, observed_mz.
weighted: boolean
    Defines  wether to perform weighted average (on the element composition) or not.
dfOutput: boolean
    if True the function returns a dataframe, a dictionary otherwise
shoe: boolean
    mainly used for debugging purposes. It will show the result in graph form
'''
def calculateMassUncertainty(processedSpectrum, weighted=False, dfOutput=True, show=False):
    data = [processedSpectrum["formula"],processedSpectrum["formula_mz"]-processedSpectrum["observed_mz"]]
    headers = ["formula", "uncertainty"]
    instance = pd.concat(data, axis=1, keys=headers)
    elements = {}
    for index, row in instance.iterrows():
        ion = row["formula"]
        ion = ion if ion[-1] != '-' else ion[:-1]
        tmp = mass.Composition(formula=ion)
        v = row["uncertainty"]
        total = sum(tmp.values())
        for e in tmp.keys():
            f = 1
            if weighted:
                f = tmp[e] / total
            if e not in elements:
                elements[e] = [v*f]
            else:
                elements[e].append(v*f)
    for e in elements.keys():
        elements[e] = sum(elements[e]) / len(elements[e])
    if show:
        keys = elements.keys()
        values = elements.values()
        plt.figure(1)
        plt.bar(keys, values)
        plt.ylabel('Error')
        plt.xlabel('Elements')
    if dfOutput:
        df = pd.DataFrame(elements.items(), columns=['Element', 'Uncertainty'])
        return df
    else:
        return elements
