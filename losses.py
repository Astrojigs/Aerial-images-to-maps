# discriminator loss
def discriminator_loss(real,generated):
    '''
    Function description:
    real = the output of discriminator when real image is fed in
    generated = the output of discriminator when fake image is fed in

    Discriminator loss will determine whether it is able to detect real from real and fake/generated from fake/generated.
    '''
    real_loss = loss_obj(tf.ones_like(real),real)

    # generated image loss
    generated_loss = loss_obj(tf.zeros_like(generated),generated)

    # add them
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


# generator loss
def generator_loss(generated):
    '''
    input: output of a discriminator when supplied a generated/fake image

        It implies that this loss gets smaller if the discriminator marks the generated image as the real image.
        In other words this loss helps to fool the discriminator into thinking that the generated image is the real image,
        by making the generator improve by each epoch.
    '''
    return loss_obj(tf.ones_like(generated),generated)


def calc_cycle_loss(real_image, cycle_image):
    loss1 = tf.reduce_mean(tf.abs(real_image-cycle_image))
    return LAMBDA*loss1


def identity_loss(real_image, same_image):
    ''' generator_g is responsible for translating image X to image Y. Identity
    loss says that, if you fed image Y to generator G, it should yield the real
    image Y or something close to image Y.
        identity_loss = |G(Y)-Y|+|F(X)-X|
        '''
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA*loss
