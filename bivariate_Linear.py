X = [2, 3, 5, 4, 6, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Y = [50, 60, 80, 70, 88, 45, 95, 100, 105, 110, 115, 120, 125, 130, 135]
n=len(X)
X_matrix=[[1,X[i]] for i in range(n)]
Y_matrix=[[Y[i]] for i in range(n)]
X_transpose=[[X_matrix[j][i] for j in range(n)] for i in range(2)]
XTX=[[sum(X_transpose[i][k]*X_matrix[k][j] for k in range(n)) for j in range(2)] for i in range(2)]
XTY=[[sum(X_transpose[i][k]*Y_matrix[k][0] for k in range(n))] for i in range(2)]
det=XTX[0][0]*XTX[1][1]-XTX[0][1]*XTX[1][0]
if det==0:
    raise Exception("Matrix is not invertible!")
inv_XTX = [
    [ XTX[1][1] / det, -XTX[0][1] / det],
    [-XTX[1][0] / det,  XTX[0][0] / det]
]
beta=[[inv_XTX[0][0]*XTY[0][0] +inv_XTX[0][1]*XTY[1][0]],[inv_XTX[1][0]*XTY[0][0]+inv_XTX[1][1]*XTY[1][0]]]
intercept=beta[0][0]
slope=beta[1][0]
print("\nX matrix:")
for row in X_matrix:
    print(row)

print("\nY matrix:")
for row in Y_matrix:
    print(row)

print("\nX^T * X matrix (XTX):")
for row in XTX:
    print(row)

print("\nX^T * Y matrix (XTY):")
for row in XTY:
    print(row)

print("\nInverse of X^T * X matrix:")
for row in inv_XTX:
    print(row)

print("\nBeta (θ) matrix [intercept, slope]:")
for row in beta:
    print(row)

print(f"\nIntercept: {intercept:.2f}")
print(f"Slope: {slope:.2f}")

print(f"\nRegression line: y = {intercept:.2f} + {slope:.2f} * x")
