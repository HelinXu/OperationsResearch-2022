from numpy import maximum
from sd import *
import click

@click.command()
@click.option('--eps', default=1e-4, help='tolerance')
@click.option('--max_iter', default=50, help='max iteration')
@click.option('--ord', default='2', help='gradient order', type=str)
@click.option('--plot', default=False, help='plot the result')
def main(eps, max_iter, ord, plot, x0=[0,0]):
    print(ord)
    print(type(ord))
    xmesh, ymesh = np.mgrid[-1:2:50j,-1:2:50j]
    fmesh = f(np.array([xmesh, ymesh]))

    guesses = sd(f=f, df=df, x0=np.array(x0), eps=eps, max_iter=max_iter, ord=ord)

    # plot the search path
    plt.axis("equal")
    plt.contour(xmesh, ymesh, fmesh, 100)
    it_array = np.array(guesses)
    plt.plot(it_array.T[0], it_array.T[1], "x-")
    plt.plot([1], [1] , ".", color="red")
    plt.savefig("sd_L{}_{}.png".format(ord, max_iter))
    if plot:
        plt.show()
    plt.clf()
    
    # plot the value along iterations
    plt.plot(range(max_iter), [f(x) for x in guesses])
    plt.savefig("sd_L{}_curve.png".format(ord, max_iter))


if __name__ == '__main__':
    main()