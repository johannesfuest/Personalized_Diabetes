import torch


def xi(x, a, epsilon):
    """
    xi function from gMSE paper: 2/epsilon * (x - a - epsilon/2)
    Only use PyTorch operations to ensure that the gradient is computed correctly and efficiently
    """
    two = torch.tensor(2.0, dtype=torch.float32)
    two_over_epsilon = two / epsilon
    a_plus_epsilon_over_two = a + epsilon / two
    return two_over_epsilon * (x - a_plus_epsilon_over_two)


def sigmoid(x, a, epsilon):
    """
    sigmoid function from gMSE paper
    """
    XI = xi(x, a, epsilon)
    zero = torch.tensor(0.0, dtype=torch.float32)
    half = torch.tensor(0.5, dtype=torch.float32)
    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    epsilon_over_two = epsilon / two
    three = torch.tensor(3.0, dtype=torch.float32)
    four = torch.tensor(4.0, dtype=torch.float32)

    calc = (- (XI ** three) + XI + half)
    term1 = (-(half) * (XI ** four)) + calc
    term2 = ((half) * (XI ** four)) + calc

    return torch.where(
        x <= a,
        zero,
        torch.where(
            x <= (a + epsilon_over_two),
            term1,
            torch.where(
                x <= (a + epsilon),
                term2,
                one
            )
        )
    )


def xi_bar(x, a, epsilon):
    """
    xi_bar function from gMSE paper: -2/epsilon * (x - a + epsilon/2)
    """
    minus_two = torch.tensor(-2.0, dtype=torch.float32)
    m_two_over_epsilon = minus_two / epsilon
    a_plus_epsilon_over_m_two = a + (epsilon / minus_two)
    return m_two_over_epsilon * (x - a_plus_epsilon_over_m_two)


def sigmoid_bar(x, a, epsilon):
    """
    sigmoid_bar function from gMSE paper
    """
    XI_BAR = xi_bar(x, a, epsilon)
    zero = torch.tensor(0.0, dtype=torch.float32)
    half = torch.tensor(0.5, dtype=torch.float32)
    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    three = torch.tensor(3.0, dtype=torch.float32)
    four = torch.tensor(4.0, dtype=torch.float32)
    epsilon_over_two = epsilon / two

    calc = (- (XI_BAR ** three) + XI_BAR + half)
    term1 = (-(half) * (XI_BAR ** four)) + calc
    term2 = ((half) * (XI_BAR ** four)) + calc

    return torch.where(
        x <= (a - epsilon),
        one,
        torch.where(
            x <= (a - epsilon_over_two),
            term2,
            torch.where(
                x <= a,
                term1,
                zero
            )
        )
    )


# Define constants from original gMSE paper
alpha_L = torch.tensor(1.5, dtype=torch.float32)
alpha_H = torch.tensor(1.0, dtype=torch.float32)
beta_L = torch.tensor(30.0, dtype=torch.float32)
beta_H = torch.tensor(100.0, dtype=torch.float32)
gamma_L = torch.tensor(10.0, dtype=torch.float32)
gamma_H = torch.tensor(20.0, dtype=torch.float32)
t_L = torch.tensor(85.0, dtype=torch.float32)
t_H = torch.tensor(155.0, dtype=torch.float32)


def Pen(
    g,
    g_hat,
    alpha_L=alpha_L,
    alpha_H=alpha_H,
    beta_L=beta_L,
    beta_H=beta_H,
    gamma_L=gamma_L,
    gamma_H=gamma_H,
    t_L=t_L,
    t_H=t_H,
):
    """Penalty function from gMSE paper"""
    one = torch.tensor(1.0, dtype=torch.float32)
    return one + alpha_L * sigmoid_bar(g, t_L, beta_L) * sigmoid(g_hat, g, gamma_L) \
        + alpha_H * sigmoid(g, t_H, beta_H) * sigmoid_bar(g_hat, g, gamma_H)


def gSE(
    g,
    g_hat,
    alpha_L=alpha_L,
    alpha_H=alpha_H,
    beta_L=beta_L,
    beta_H=beta_H,
    gamma_L=gamma_L,
    gamma_H=gamma_H,
    t_L=t_L,
    t_H=t_H,
):
    """gMSE function from gMSE paper: (g - g_hat)^2 * Pen(g, g_hat) = MSE * Pen(g, g_hat)"""
    g = g.float()
    g_hat = g_hat.float()
    return (g - g_hat) ** 2 * Pen(g, g_hat, alpha_L, alpha_H, beta_L, beta_H, gamma_L, gamma_H, t_L, t_H)


def gMSE(
    g,
    g_hat,
    alpha_L=alpha_L,
    alpha_H=alpha_H,
    beta_L=beta_L,
    beta_H=beta_H,
    gamma_L=gamma_L,
    gamma_H=gamma_H,
    t_L=t_L,
    t_H=t_H,
):
    """Mean aggregated gMSE function from gMSE paper: mean(gMSE) = mean(MSE * Pen(g, g_hat))"""
    return torch.mean(gSE(g, g_hat, alpha_L, alpha_H, beta_L, beta_H, gamma_L, gamma_H, t_L, t_H))
