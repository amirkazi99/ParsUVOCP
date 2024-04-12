import ast

import pandas as pd

if __name__ == '__main__':
    commentsAspects = pd.read_excel('results/comments_aspects2.xlsx')
    commentsAspects = commentsAspects.where(pd.notnull(commentsAspects), '[]')

    predAspects = commentsAspects['aspects'].values.tolist()
    trueAspects = commentsAspects['true_aspects'].values.tolist()

    predAspects = [set(ast.literal_eval(predAspect)) if predAspect != float('nan') else set() for predAspect in predAspects]
    trueAspects = [set(ast.literal_eval(trueAspect)) if trueAspect != float('nan') else set() for trueAspect in trueAspects]

    TP, FP, FN, TN = 0, 0, 0, 0
    for predAspect, trueAspect in zip(predAspects, trueAspects):
        # if len(predAspect) == 0 or len(trueAspect) == 0:
        #     continue
        TP += len(predAspect & trueAspect)
        FP += len(predAspect - trueAspect)
        FN += len(trueAspect - predAspect)

    if TP + FN == 0 or TP + FP == 0:
        print('Bad Data!')
    else:
        R = TP / (TP + FN)
        P = TP / (TP + FP)
        F = 2 * P * R / (P + R)

        print(f'TP = {TP:5d} ,   FP = {FP:5d}')
        print(f'FN = {FN:5d} ,   TN = {TN:5d}')
        print(f'Recall     = {100 * R:5.2f} %')
        print(f'Precision  = {100 * P:5.2f} %')
        print(f'F1-score   = {100 * F:5.2f} %')

