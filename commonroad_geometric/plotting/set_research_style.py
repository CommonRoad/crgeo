import matplotlib.pyplot as plt
import matplotlib

def set_research_style(
    size_multiplier: float = 1.0,
    latex: bool = False
):
    import seaborn
    matplotlib.rcParams['hatch.linewidth'] = 0.7

    params = {
        'font.size'   : 39*size_multiplier,
        'font.family': "Helvetica"
    }
    if latex:
        params.update({
        'font.size': 39*size_multiplier,
        'text.usetex' : True,
    })
    matplotlib.rcParams.update(params)

    if latex:
        plt.style.use('ggplot')
        matplotlib.rc('text', usetex=True)
    plt.rc('xtick', labelsize=int(39*size_multiplier))
    plt.rc('ytick', labelsize=int(39*size_multiplier))
    plt.rc('axes', labelsize=int(47*size_multiplier))
    plt.rc('legend',fontsize=int(25*size_multiplier))
