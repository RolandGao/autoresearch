so there are a few things:

1. instead of computing dx and dw at the same time from dy, do dw first, recalculate dy, and then do dx. 
2. use normal equation for dw
3. use pseudo inverse for dx
4. cross entropy loss needs special handling cuz it's different from MSE. 