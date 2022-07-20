from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import addcopyfighandler

class mixture:

    def __init__(self,Y_vec, W):
        self.weights = W
        self.components = Y_vec
        self.a = 0.005
        self.x_pdf = self.make_x(self.a)
        self.mix_pdf = self.pdf(self.x_pdf)
        self.mix_cdf = self.cdf(self.x_pdf)

    def ppf_lower(self, Y):
        return Y.ppf(self.a)

    def ppf_upper(self,Y):
        return Y.ppf(1-self.a)

    def make_x(self, a):
        a = 0.005
        x_lowers = list(map(self.ppf_lower, self.components))
        x_uppers = list(map(self.ppf_upper, self.components))
        x = np.linspace(min(x_lowers), max(x_uppers), 1000)
        return x

    def pdf(self,x):

        component_pdfs = np.array([Y.pdf(x) for Y in self.components])
        weighted_pdfs = component_pdfs * np.array([self.weights]).transpose()
        return np.sum(weighted_pdfs, axis=0)


    def plot_pdf(self, ax):
        ax.plot(self.x_pdf, self.mix_pdf, label = "$f_{mix}$")
        return ax

    def plot_cdf(self, ax):
        ax.plot(self.x_pdf, self.mix_cdf)
        return ax

    def cdf(self,x):
        component_cdfs = np.array([Y.cdf(x) for Y in self.components])
        weighted_cdfs = component_cdfs * np.array([self.weights]).transpose()
        return np.sum(weighted_cdfs, axis=0)

    def ppf(self,p):
        def continuous_bisect_fun_left(f, v, lo, hi):
            val_range = [lo, hi]
            k = 0.5 * sum(val_range)
            for i in range(32):
                val_range[int(f(k) > v)] = k
                next_k = 0.5 * sum(val_range)
                if next_k == k:
                    break
                k = next_k
            return k

        # Return the function that is the cdf of the mixture distribution
        def get_mixture_cdf(component_distributions, ps):
            return lambda x: sum(component_dist.cdf(x) * p for component_dist, p in zip(component_distributions, ps))

        # Return the pth quantile of the mixture distribution given by the component distributions and their probabilities

        def mixture_quantile(p):
            mixture_cdf = get_mixture_cdf(self.components, self.weights)

            # We can probably be a bit smarter about how we pick the limits
            lo = np.min([dist.ppf(p) for dist in self.components])
            hi = np.max([dist.ppf(p) for dist in self.components])

            return continuous_bisect_fun_left(mixture_cdf, p, lo, hi)


        return mixture_quantile(p)

    def plot_icdf(self, ax):
        p = np.linspace(0,1,100)
        icdf = np.array([self.ppf(i) for i in p])
        ax.plot(p, icdf)
        return ax





def plot_pdf(Y, ax):
    a = 0.005
    x = np.linspace(Y.ppf(a), Y.ppf(1-a), 1000)
    ax.plot(x, Y.pdf(x), label = f"N({Y.args[0]}, {Y.args[1]**2})" )
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$f(x)$')

    #ax.set_title('Probability Distribution Function')
    return ax

def plot_cdf(Y,ax):
    a = 0.005
    x = np.linspace(Y.ppf(a), Y.ppf(1-a), 1000)
    ax.plot(x, Y.cdf(x))
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$F_{Y_i}(y)$')
    ax.set_title("Cumulative Distribution Function")
    return ax

def plot_icdf(Y,ax):
    x = np.linspace(0,1)
    ax.plot(x, Y.ppf(x))
    ax.set_xlabel(r'$F_{Y_i}(y)$')
    ax.set_ylabel(r'$y$')
    ax.set_title("Inverse CDF or Quantile Function")
    return ax
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

Y1 = stats.norm(3,1)
Y2 = stats.norm(5,1)
Y3 = stats.norm(7,1)
Y4 = stats.norm(9,1)


Ymix = mixture([Y1, Y2, Y3, Y4], [0.25, 0.25, 0.25, 0.25])

test = Ymix.ppf(0.9)

fig1, ax1 = plt.subplots()
ax1 = plot_pdf(Y1, ax1)
ax1 = plot_pdf(Y2, ax1)
ax1 = plot_pdf(Y3, ax1)
ax1 = plot_pdf(Y4, ax1)
ax1.legend()
ax1 = Ymix.plot_pdf(ax1)
ax1.vlines(test,0,0.4, linestyles = "dashed", label = '$F_{mix}^{-1}(0.9)$')
ax1.legend()
fig1.show()


# fig2, ax2 = plt.subplots()
# ax2 = plot_cdf(Y1, ax2)
# ax2 = plot_cdf(Y2, ax2)
# ax2 = plot_cdf(Y3, ax2)
# ax2 = plot_cdf(Y4, ax2)
# ax2 = Ymix.plot_cdf(ax2)
# fig2.show()
#
# fig3, ax3 = plt.subplots()
# ax3 = plot_icdf(Y1, ax3)
# ax3 = plot_icdf(Y2, ax3)
# ax3 = plot_icdf(Y3, ax3)
# ax3 = plot_icdf(Y4, ax3)
# ax3 = Ymix.plot_icdf(ax3)
# fig3.show()



## Plotting
# fig, axes = plt.subplots(1,3)
# axes[0] = plot_pdf(Y1, axes[0])
# axes[0] = plot_pdf(Y2, axes[0])
#
# axes[1] = plot_cdf(Y1, axes[1])
# axes[1] = plot_cdf(Y2, axes[1])
#
# axes[2] = plot_icdf(Y1, axes[2])
# axes[2] = plot_icdf(Y2, axes[2])


## Individual Plots
# Plot pdf
# fig, ax = plt.subplots()
# ax = plot_pdf(Y1, ax)
# ax = plot_pdf(Y2, ax)
# fig.show()
#
# # Plot cdf
# fig2, ax2 = plt.subplots()
# ax2 = plot_cdf(Y1, ax2)
# ax2 = plot_cdf(Y2, ax2)
# fig2.show()
#
# # Plot inverse cdf (quantile function)
# fig3, ax3 = plt.subplots()
# ax3 = plot_icdf(Y1, ax3)
# ax3 = plot_icdf(Y2, ax3)
# fig3.show()