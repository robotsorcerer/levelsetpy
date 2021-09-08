import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show3D(title='Zero Level Set', g3=None, fig=None, ax=None, winsize=(16, 9),
            labelsize=18, linewidth=6, fontdict={'fontsize':12, 'fontweight':'bold'},
            block=False, savedict=None, mesh=None):

    """
        Visualize the 3D levelset of an implicit function

        Lekan Molu, September 07, 2021
    """
    fig = plt.figure(figsize=winsize)

    ax = fig.add_subplot(111, projection='3d')

    if g3:
        ax.plot3D(g3.xs[0].flatten(), g3.xs[1].flatten(), g3.xs[2].flatten(), color='cyan')
    if isinstance(mesh, list):
        for m in mesh:
            ax.add_collection3d(m)
    else:
        ax.add_collection(mesh)
    ax.set_xlabel("X-axis", fontdict = fontdict)
    ax.set_ylabel("Y-axis", fontdict = fontdict)
    ax.set_zlabel("Z-axis", fontdict = fontdict)
    ax.set_title(title, fontdict = fontdict)


    if savedict["save"]:
        fig.savefig(join(savedict["savepath"],savedict["savename"]),
                    bbox_inches='tight',facecolor='None')

    plt.show()
