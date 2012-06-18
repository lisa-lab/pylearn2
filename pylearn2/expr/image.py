""" Mathematical expressions related to image processing. """

def color_to_gray(color):
    """ Standard conversion from color to luma

    Y' = W_R * red_channel + W_G * green_channel + W_B * blue_channel

    with
    W_R = 0.299
    W_G = 0.587
    W_B = 0.114

    Source: http://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB

    Arguments:
        color: a 4-tensor with the channel index last
               may be either a numpy or a theano tensor
    Returns:
        a tensor of the same number of dimensions, but with the final
        dimension changed to 1.
    """

    W_R = 0.299
    W_B = 0.114
    W_G = 0.587

    red_channel = color[:,:,:,0:1]
    blue_channel = color[:,:,:,1:2]
    green_channel = color[:,:,:,2:3]

    Y_prime = W_R * red_channel + W_G * green_channel + W_B * blue_channel

    return Y_prime

