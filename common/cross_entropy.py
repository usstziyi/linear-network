import torch



# y_hat(B,G)
# y(B,1)
def cross_entropy(y_hat, y):
    # y_hat(B,G)
    # y(B,1)
    # Pc(B,1)
    Pc= y_hat[range(len(y_hat)), y]
     # return(B,1)
    return - torch.log(Pc)


