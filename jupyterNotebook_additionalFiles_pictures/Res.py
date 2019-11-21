#     if L % 2 == 0:
#         print("Even number of layers in Fully Connected net (including output layer)")
#         start_res = 2 # first taken activation value for resnet is first hidden layer activation value.
#     else:
#         print("Odd number of layers in Fully Connected net (including output layer)")
#         start_res = 1 # first taken activation value for resnet is input of fully connected net.

        if suma != 0:
            a = Z.shape[0]
            b = A_prev_prev.shape[0]
            ones = np.ones((a,b))
            Z = Z + np.dot(ones, A_prev_prev)


#         print("A"+str(l)+" "+str(A.shape)+" = \n"+str(A)+"\n")
#         print("W"+str(l)+" "+str(parameters['W' + str(l)].shape)+" = \n"+str(parameters['W' + str(l)])+"\n")
#         print("Max in A"+str(l-1)+" je :"+str(np.max(A_prev)))
#         maxx = np.max(A_prev) 
#         standardize = A_prev / maxx
#         print("Max in A"+str(l-1)+" je :"+str(np.max(standardize)))
#         A_prev_prev_resNet.append(standardize)
#         A_zero = np.zeros((1,1))
        
#         if start_res == 2:      #if there is even number of hidden layers (including output layer)
#             if l % 2 == 0:      #if index of layer is even number apply residual operation
#                 print("\nApplying Residual block in hidden layer %i" %(l))
#                 A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], A_prev_prev_resNet[l-2], activation = "relu")
#                 caches.append(cache)
#             else:               #if index of layer is odd number do not apply residual operation
#                 A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], A_zero, activation = "relu")
#                 caches.append(cache)
#         if start_res == 1 and l > 2: #if there is odd number of hidden layers (including output layer), start res. op. from third layer
#             if l % 2 > 0:       #if index of layer is odd number do apply residual operation
#                 print("\nApplying Residual block in hidden layer %i" %(l))
#                 A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], A_prev_prev_resNet[l-2], activation = "relu")
#                 caches.append(cache)
#             else:               #if index of layer is even number do not apply residual operation
#                 A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], A_zero, activation = "relu")
#                 caches.append(cache)
#         if start_res == 1 and l < 3: 
#             A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], A_zero, activation = "relu")
#             caches.append(cache)
#         a1,b1 = A.shape
#         p1 = a1*b1
#         print("\n++Sloj "+str(l))
#         print("Vektor A"+str(l)+" "+str(A.shape)+" ima ukupno " +str(p1)+" vrednosti za sve piksele")
#         print("Nula vrednosti ima: " +str(p1-np.count_nonzero(A)))

#     # Implement LINEAR -> SOFTMAX. Add "cache" to the "caches" list.
#    #     Apply residual operation allways in the last, output layer
#     print("\nApplying Residual block in output layer %i" %(L))
#     AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], A_zero,  activation = "softmax")
# #     caches.append(cache)
#     print("A"+str(L)+" "+str(AL.shape)+" = \n"+str(AL)+"\n")
#     print("w"+str(L)+" "+str(parameters['W' + str(L)].shape)+" = \n"+str(parameters['W' + str(L)])+"\n")
#     a1,b1 = AL.shape
#     p1 = a1*b1
#     print("\n++Sloj "+str(L))
#     print("Vektor Al ima ukupno " +str(p1)+" vrednosti za sve piksele")
#     print("Nula vrednosti ima: " +str(p1-np.count_nonzero(AL)))